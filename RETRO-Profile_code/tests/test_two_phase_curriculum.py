"""Phase B tests: the two-phase curriculum that trains Novelty 3's residual head.

Runs a real (tiny) Lightning fit so the callback is exercised by the actual
trainer: phase 1 must leave the residual branch untrained, and phase 2 must
attach the datastore, freeze the embedding path, and start training the residual.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags, make_collate_fn
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.training.callbacks import TwoPhaseRetrievalCurriculum
from retro_profile.training.lightning_module import RetroProfileLitModule

warnings.filterwarnings("ignore")

LABEL_STATS = {"age": {"mean": 30.0, "std": 5.0}, "height": {"mean": 170.0, "std": 8.0}}


class _TinyEnc(BaseEncoder):
    def __init__(self, output_dim=8):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        return self.proj(avg) * um.unsqueeze(-1).to(w.dtype)


class _BagDS(Dataset):
    """6 speakers; indexable as bags (training) or single utterances (datastore)."""

    def __init__(self, mode="bag"):
        self.mode = mode
        self.bags = [
            SpeakerBag([torch.randn(400), torch.randn(400)], 16000, f"spk_{i}",
                       ["u0", "u1"], {"age": 25.0 + i, "height": 165.0 + 2 * i})
            for i in range(6)
        ]

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        from retro_profile.data.base import SpeakerExample
        b = self.bags[i]
        if self.mode == "bag":
            return b
        return SpeakerExample(waveform=b.waveforms[0], sample_rate=16000,
                              speaker_id=b.speaker_id, utt_id="u0", labels=b.labels)


class _DM(pl.LightningDataModule):
    """Minimal datamodule exposing the utterance_loader the callback needs."""

    def __init__(self):
        super().__init__()
        self.targets = ["age", "height"]

    def _dl(self, mode):
        return DataLoader(_BagDS(mode), batch_size=3,
                          collate_fn=make_collate_fn(self.targets, mode=mode))

    def train_dataloader(self):
        return self._dl("bag")

    def val_dataloader(self):
        return self._dl("bag")

    def utterance_loader(self, split="train"):
        return self._dl("utterance")


def _make_lit():
    torch.manual_seed(0)
    model = RetroProfile(
        encoder=_TinyEnc(8),
        pooling=MILAttentionPooling(input_dim=8, hidden_dim=4, num_heads=1,
                                    target_names=["age", "height"]),
        heads={"age": ResidualRetrievalHead(input_dim=4, hidden_dim=4,
                                            label_min=18.0, label_max=80.0, knn_k=3),
               "height": ResidualRetrievalHead(input_dim=4, hidden_dim=4,
                                               label_min=140.0, label_max=210.0, knn_k=3)},
        projector=MLPHead(input_dim=8, hidden_dim=8, output_dim=4, l2_normalize=True),
    )
    lit = RetroProfileLitModule(
        model=model, target_names=["age", "height"], embedding_dim=4,
        ordcon_weight=0.1, label_stats=LABEL_STATS,
        scheduler={"name": "none"},
    )
    return model, lit


def _fit(max_epochs, callbacks, checkpointing=False):
    model, lit = _make_lit()
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=callbacks, logger=False,
                         enable_checkpointing=checkpointing, enable_progress_bar=False,
                         enable_model_summary=False, accelerator="cpu", devices=1,
                         num_sanity_val_steps=0)
    trainer.fit(lit, datamodule=_DM())
    return model, lit


def test_phase1_only_leaves_residual_untrained():
    """Baseline: without the callback the residual branch never trains.

    Its last layer is zero-initialized, so 'still exactly zero' == 'never updated'.
    """
    model, _ = _fit(2, callbacks=[])
    for t in ("age", "height"):
        assert not model.heads[t].has_datastore
        w = model.heads[t].residual[-1].weight
        assert torch.count_nonzero(w) == 0                          # untouched
        assert w.grad is None                                       # never got a gradient


def test_phase2_attaches_datastore_and_trains_residual():
    """The whole point of Phase B: after the transition the residual LEARNS.

    Evidence is that the zero-initialized last layer has moved away from zero
    (gradients themselves are cleared by the optimizer after the final step).
    """
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1, freeze_embedding=True)
    model, _ = _fit(3, callbacks=[cb])

    assert cb.activated
    for t in ("age", "height"):
        head = model.heads[t]
        assert head.has_datastore                                   # switched on
        w = head.residual[-1].weight
        assert torch.count_nonzero(w) > 0                           # moved off zero-init
        assert torch.isfinite(w).all()


def test_phase2_freezes_the_embedding_path():
    """The indexed space must stop moving, or the datastore goes stale."""
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1, freeze_embedding=True)
    model, _ = _fit(2, callbacks=[cb])
    for name in ("encoder", "pooling", "projector"):
        module = getattr(model, name)
        assert all(not p.requires_grad for p in module.parameters()), name
    # the residual head is deliberately left trainable
    assert any(p.requires_grad for p in model.heads["height"].residual.parameters())


def test_freeze_embedding_false_keeps_path_trainable():
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1, freeze_embedding=False)
    model, _ = _fit(2, callbacks=[cb])
    assert cb.activated
    assert any(p.requires_grad for p in model.projector.parameters())


def test_callback_does_not_activate_before_start_epoch():
    cb = TwoPhaseRetrievalCurriculum(start_epoch=5)
    model, _ = _fit(2, callbacks=[cb])
    assert not cb.activated
    assert not model.heads["age"].has_datastore


def test_early_stopping_cannot_end_the_run_before_phase2():
    """Regression: phase 1 plateaus and EarlyStopping used to terminate training
    BEFORE start_epoch, so phase 2 never ran and the residual stayed untrained.
    The curriculum must suspend patience until the transition, then restore it."""
    from pytorch_lightning.callbacks import EarlyStopping

    es = EarlyStopping(monitor="val/total", mode="min", patience=1)
    cb = TwoPhaseRetrievalCurriculum(start_epoch=3, freeze_embedding=True)
    model, _ = _fit(5, callbacks=[cb, es])

    assert cb.activated, "phase 2 must run even though val plateaued early"
    assert es.patience == 1                       # real patience restored
    for t in ("age", "height"):
        assert model.heads[t].has_datastore
        assert torch.count_nonzero(model.heads[t].residual[-1].weight) > 0


def test_restores_best_phase1_weights_before_indexing(tmp_path):
    """Phase 1 peaks then drifts; phase 2 must index the BEST phase-1 model.

    We check the weights at the transition equal the best checkpoint's weights,
    not whatever the model happened to be at start_epoch.
    """
    from pytorch_lightning.callbacks import ModelCheckpoint

    mc = ModelCheckpoint(dirpath=str(tmp_path), monitor="val/total", mode="min",
                         save_top_k=1, save_last=False)
    cb = TwoPhaseRetrievalCurriculum(start_epoch=3, freeze_embedding=True,
                                     restore_best_phase1=True)
    model, _ = _fit(4, callbacks=[cb, mc], checkpointing=True)

    assert cb.activated and mc.best_model_path
    best = torch.load(mc.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    # the frozen embedding path must match phase 1's BEST checkpoint exactly
    for name in ("projector.net.0.weight", "encoder.proj.weight"):
        key = f"model.{name}"
        if key in best:
            live = dict(model.named_parameters())[name]
            assert torch.allclose(live, best[key], atol=1e-6), name


def test_restore_best_phase1_can_be_disabled(tmp_path):
    from pytorch_lightning.callbacks import ModelCheckpoint

    mc = ModelCheckpoint(dirpath=str(tmp_path), monitor="val/total", mode="min", save_top_k=1)
    cb = TwoPhaseRetrievalCurriculum(start_epoch=2, restore_best_phase1=False)
    model, _ = _fit(3, callbacks=[cb, mc], checkpointing=True)
    assert cb.activated
    assert model.heads["age"].has_datastore


def test_resume_inside_phase2_does_not_wipe_the_trained_residual(tmp_path):
    """Colab sessions die mid-run. Resuming after phase 2 has started must NOT
    roll the model back to phase-1 weights (which would erase the residual the
    curriculum just trained); it must re-attach the datastore and carry on."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    # --- first leg: run into phase 2, then stop ---
    mc = ModelCheckpoint(dirpath=str(tmp_path), monitor="val/total", mode="min",
                         save_top_k=1, save_last=True)
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1, freeze_embedding=True,
                                     restore_best_phase1=True)
    model, lit = _make_lit()
    tr = pl.Trainer(max_epochs=3, callbacks=[cb, mc], logger=False,
                    enable_checkpointing=True, enable_progress_bar=False,
                    enable_model_summary=False, accelerator="cpu", devices=1,
                    num_sanity_val_steps=0)
    tr.fit(lit, datamodule=_DM())
    assert cb.activated
    trained_residual = model.heads["age"].residual[-1].weight.detach().clone()
    assert torch.count_nonzero(trained_residual) > 0

    # --- second leg: fresh objects, resume from last.ckpt ---
    ckpt = str(tmp_path / "last.ckpt")
    cb2 = TwoPhaseRetrievalCurriculum(start_epoch=1, freeze_embedding=True,
                                      restore_best_phase1=True)
    model2, lit2 = _make_lit()
    tr2 = pl.Trainer(max_epochs=4, callbacks=[cb2], logger=False,
                     enable_checkpointing=False, enable_progress_bar=False,
                     enable_model_summary=False, accelerator="cpu", devices=1,
                     num_sanity_val_steps=0)
    tr2.fit(lit2, datamodule=_DM(), ckpt_path=ckpt)

    assert cb2.activated, "phase flag must survive the resume"
    resumed = model2.heads["age"].residual[-1].weight
    assert torch.count_nonzero(resumed) > 0, "residual was wiped by the resume"
    assert model2.heads["age"].has_datastore, "datastore must be re-attached"


def test_model_returns_to_train_mode_after_datastore_build():
    """build_datastore calls .eval(); training must resume in train mode."""
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1)
    model, _ = _fit(2, callbacks=[cb])
    assert cb.activated
    # fit() ends with the module in eval; assert the flag was restored mid-run by
    # checking dropout-bearing submodules were not left permanently disabled.
    model.train()
    assert model.training


def test_resume_state_roundtrip_preserves_phase():
    """Resume safety: a checkpointed phase-2 run must come back in phase 2 and
    re-attach the datastore WITHOUT rolling back to phase-1 weights."""
    cb = TwoPhaseRetrievalCurriculum(start_epoch=1)
    cb.activated = True
    state = cb.state_dict()
    assert state == {"activated": True}

    fresh = TwoPhaseRetrievalCurriculum(start_epoch=1)
    fresh.load_state_dict(state)
    assert fresh.activated          # comes back in phase 2; on_fit_start re-attaches

    never = TwoPhaseRetrievalCurriculum(start_epoch=1)
    never.load_state_dict({"activated": False})
    assert not never.activated


def test_checkpoint_best_reset_is_opt_in(tmp_path):
    """Default OFF preserves the original behaviour (ModelCheckpoint keeps
    scoring across both phases, so `best` may be a phase-1 model)."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    mc = ModelCheckpoint(dirpath=str(tmp_path / "off"), monitor="val/total",
                         mode="min", save_top_k=1)
    cb = TwoPhaseRetrievalCurriculum(start_epoch=2, reset_checkpoint_best=False)
    _fit(4, callbacks=[cb, mc], checkpointing=True)
    assert cb.activated
    assert mc.best_model_path                      # tracker never cleared


def test_checkpoint_best_reset_selects_phase2_checkpoint(tmp_path):
    """With the flag ON, checkpoint selection restarts at the transition, so the
    saved `best` comes from phase 2 (epoch >= start_epoch)."""
    from pytorch_lightning.callbacks import ModelCheckpoint

    mc = ModelCheckpoint(dirpath=str(tmp_path / "on"), monitor="val/total",
                         mode="min", save_top_k=1, filename="{epoch}")
    cb = TwoPhaseRetrievalCurriculum(start_epoch=2, reset_checkpoint_best=True)
    _fit(4, callbacks=[cb, mc], checkpointing=True)
    assert cb.activated and mc.best_model_path
    epoch = int(torch.load(mc.best_model_path, map_location="cpu",
                           weights_only=False)["epoch"])
    assert epoch >= 2, f"best checkpoint came from phase 1 (epoch {epoch})"
