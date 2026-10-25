"""The per-epoch diagnostics callback.

Runs a real (tiny) Lightning fit so the callback is exercised by the actual
trainer. What matters:

  * it writes one row per (epoch, target) with accuracy AND attention columns,
  * it reads the VALIDATION loader, never the test split — evaluating test each
    epoch would let the test curve influence which epoch we keep,
  * it leaves the model in training mode (bug §6.8 was exactly this),
  * a failure inside it never kills the run.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import make_collate_fn
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.training.callbacks import PerEpochDiagnostics
from retro_profile.training.lightning_module import RetroProfileLitModule

warnings.filterwarnings("ignore")

LABEL_STATS = {"age": {"mean": 30.0, "std": 5.0}, "height": {"mean": 170.0, "std": 8.0}}
UTT_IDS = ["SA1", "SI1573", "SX133", "SA2"]      # real TIMIT sentence-type ids


class _TinyEnc(BaseEncoder):
    def __init__(self, output_dim=8):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        return self.proj(avg) * um.unsqueeze(-1).to(w.dtype)


class _BagDS(Dataset):
    def __init__(self):
        self.bags = [
            SpeakerBag([torch.randn(400) for _ in UTT_IDS], 16000, f"spk_{i}",
                       list(UTT_IDS), {"age": 25.0 + i, "height": 165.0 + 2 * i})
            for i in range(6)
        ]

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, i):
        return self.bags[i]


class _DM(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.targets = ["age", "height"]
        self.val_calls = 0

    def _dl(self):
        return DataLoader(_BagDS(), batch_size=3,
                          collate_fn=make_collate_fn(self.targets, mode="bag"))

    def train_dataloader(self):
        return self._dl()

    def val_dataloader(self):
        self.val_calls += 1
        return self._dl()

    def test_dataloader(self):            # must never be touched by the callback
        raise AssertionError("diagnostics must not read the test split")


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
    return model, RetroProfileLitModule(
        model=model, target_names=["age", "height"], embedding_dim=4,
        ordcon_weight=0.1, label_stats=LABEL_STATS, scheduler={"name": "none"},
    )


def _run(tmp_path, epochs=3, every_n_epochs=1):
    model, lit = _make_lit()
    cb = PerEpochDiagnostics(tmp_path / "diagnostics.csv", every_n_epochs=every_n_epochs)
    dm = _DM()
    pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, logger=False,
               enable_checkpointing=False, enable_progress_bar=False,
               enable_model_summary=False, callbacks=[cb]).fit(lit, datamodule=dm)
    return model, cb, dm


def test_writes_a_row_per_epoch_and_target(tmp_path):
    _, _, _ = _run(tmp_path, epochs=3)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    assert set(df["target"]) == {"age", "height"}
    assert sorted(df["epoch"].unique()) == [0, 1, 2]
    assert len(df) == 6                                   # 3 epochs x 2 targets


def test_records_accuracy_and_attention_columns(tmp_path):
    _run(tmp_path, epochs=2)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    for col in ("mae", "rmse", "pearson", "n",
                "attn_entropy", "attn_max", "n_eff",
                "w_SA", "w_SI", "w_SX", "anchor_share", "retrieval_active"):
        assert col in df.columns, f"missing {col}"
    assert df["mae"].notna().all()
    assert (df["n"] == 6).all()                           # 6 val speakers


def test_n_eff_is_within_bag_size(tmp_path):
    """n_eff must lie in [1, K]: it is an effective count of averaged clips."""
    _run(tmp_path, epochs=2)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    assert df["n_eff"].between(1.0, len(UTT_IDS) + 1e-6).all()


def test_sentence_type_columns_are_populated(tmp_path):
    """The bags carry real TIMIT ids, so SA/SI/SX must all be resolved."""
    _run(tmp_path, epochs=1)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    assert df[["w_SA", "w_SI", "w_SX"]].notna().all().all()


def test_uses_validation_never_test(tmp_path):
    """_DM.test_dataloader raises; reaching it would fail the fit."""
    _, _, dm = _run(tmp_path, epochs=2)
    assert dm.val_calls > 0


def test_leaves_the_model_in_training_mode(tmp_path):
    """Bug §6.8: a diagnostic that leaves eval() on silently disables dropout."""
    model, _, _ = _run(tmp_path, epochs=2)
    assert model.training is True


def test_every_n_epochs_thins_the_log(tmp_path):
    _run(tmp_path, epochs=4, every_n_epochs=2)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    assert sorted(df["epoch"].unique()) == [0, 2]


def test_failure_does_not_kill_the_run(tmp_path, monkeypatch):
    """Diagnostics are never worth losing a training run over."""
    import retro_profile.evaluation.evaluator as ev

    def boom(*a, **k):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(ev, "evaluate_with_explanations", boom)
    _run(tmp_path, epochs=2)          # must complete without raising


def test_detects_retrieval_when_a_head_has_a_datastore(tmp_path):
    """model.heads is an nn.ModuleDict, which has no .get().

    An earlier version called heads.get("age"), which silently returned None and
    reported retrieval as never active for every run -- suppressing the phase
    marker on every diagnostic plot. Detection must go through .values().
    """
    import numpy as np

    from retro_profile.retrieval.builder import attach_datastore
    from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore

    model, lit = _make_lit()
    # has_datastore is a read-only property, so simulate phase 2 the way the
    # curriculum does: build a datastore and attach it to the heads.
    rng = np.random.RandomState(0)
    ds = RetrievalDatastore(embedding_dim=4, metric="cosine", temperature=0.1)
    for i in range(6):
        v = rng.randn(4).astype(np.float32)
        ds.add(DatastoreEntry(speaker_id=f"spk_{i}", embedding=v / np.linalg.norm(v),
                              labels={"age": 25.0 + i, "height": 165.0 + 2 * i}))
    ds.build()
    attach_datastore(model, ds)
    assert any(h.has_datastore for h in model.heads.values())

    cb = PerEpochDiagnostics(tmp_path / "d.csv")
    pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
               enable_checkpointing=False, enable_progress_bar=False,
               enable_model_summary=False, callbacks=[cb]).fit(lit, datamodule=_DM())

    df = pd.read_csv(tmp_path / "d.csv")
    assert df["retrieval_active"].all(), "retrieval was active but not reported"


def test_reports_retrieval_inactive_before_the_switch(tmp_path):
    """No datastore attached -> phase 1 -> retrieval must read as inactive."""
    _run(tmp_path, epochs=1)
    df = pd.read_csv(tmp_path / "diagnostics.csv")
    assert not df["retrieval_active"].any()
