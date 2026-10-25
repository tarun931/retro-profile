"""Tests for training.checkpoint.per_target (2026-10-10).

Age keeps improving late in training while height peaks early, so the single
combined checkpoint is saved at a compromise point. per_target adds one extra
best-by-validation copy per trait. These tests pin that the extra copies never
displace the main checkpoint, are named so no selector mistakes them for it,
and are refused in two-phase runs (whose reset would orphan them).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_script_pt", ROOT / "scripts" / "train.py")
train_script = importlib.util.module_from_spec(_spec)
sys.modules["train_script_pt"] = train_script
_spec.loader.exec_module(train_script)

_sel_spec = importlib.util.spec_from_file_location("sel_pt", ROOT / "scripts" / "select_checkpoint.py")
sel = importlib.util.module_from_spec(_sel_spec)
_sel_spec.loader.exec_module(sel)


def _cfg(per_target=None, two_phase=False):
    ck = {"monitor": "val/combined_mae", "mode": "min", "save_top_k": 3, "save_last": True}
    if per_target is not None:
        ck["per_target"] = per_target
    return OmegaConf.create({"training": {
        "checkpoint": ck, "diagnostics": {"enabled": False}, "early_stopping": {"enabled": False},
        "two_phase": {"enabled": two_phase, "start_epoch": 30}, "datastore_rebuild": {"every_n_epochs": 0},
    }})


def _ckpts(cfg, tmp_path):
    return [c for c in train_script._build_callbacks(cfg, tmp_path) if isinstance(c, ModelCheckpoint)]


def test_without_per_target_there_is_exactly_the_one_main_checkpoint(tmp_path):
    cks = _ckpts(_cfg(), tmp_path)
    assert len(cks) == 1 and cks[0].monitor == "val/combined_mae"


def test_per_target_copies_come_after_the_main_checkpoint(tmp_path):
    cks = _ckpts(_cfg(["age", "height"]), tmp_path)
    # Lightning's trainer.checkpoint_callback is the FIRST ModelCheckpoint; it
    # must stay the combined one (best_checkpoint.txt, final test pass).
    assert [c.monitor for c in cks] == ["val/combined_mae", "val/age/mae", "val/height/mae"]
    assert cks[0].save_last and cks[0].save_top_k == 3
    for c in cks[1:]:
        assert c.save_top_k == 1 and not c.save_last and c.mode == "min"


def test_per_target_filenames_are_exact_and_unmistakable(tmp_path):
    age = _ckpts(_cfg(["age", "height"]), tmp_path)[1]
    name = Path(age.format_checkpoint_name({"epoch": 46, "val/age/mae": torch.tensor(3.73)})).name
    assert name == "best_age-epoch=46-val_age=3.7300.ckpt"


def test_two_phase_runs_refuse_per_target_copies(tmp_path):
    with pytest.raises(ValueError, match="two_phase"):
        _ckpts(_cfg(["age"], two_phase=True), tmp_path)


def test_the_main_selector_never_picks_a_per_target_copy(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / ".hydra").mkdir()
    (run / ".hydra" / "config.yaml").write_text("training: {two_phase: {enabled: false}}\n")
    for f in ["epoch=59-step=5880-val=1.3108.ckpt", "epoch=40-step=3920-val=1.3200.ckpt",
              "best_age-epoch=17-val_age=0.5000.ckpt",       # lower number, wrong metric
              "best_height-epoch=50-val_height=0.4000.ckpt", "last.ckpt"]:
        (run / "checkpoints" / f).write_bytes(b"")
    assert sel.select_checkpoint(run).name == "epoch=59-step=5880-val=1.3108.ckpt"


def test_real_training_writes_one_copy_per_trait(tmp_path):
    """End to end: a tiny 3-epoch fit with the real callbacks writes the main
    top-k, last.ckpt, and exactly one correctly named file per trait."""
    pytest.importorskip("faiss")
    import pytorch_lightning as pl
    import torch.nn as nn
    from pytorch_lightning.loggers import CSVLogger

    from retro_profile.data.base import SpeakerBag
    from retro_profile.data.collate import collate_bags
    from retro_profile.models.encoders.base import BaseEncoder
    from retro_profile.models.heads.mlp import MLPHead
    from retro_profile.models.heads.residual import ResidualRetrievalHead
    from retro_profile.models.pooling import MILAttentionPooling
    from retro_profile.models.retro_profile import RetroProfile
    from retro_profile.training.lightning_module import RetroProfileLitModule

    class _Enc(BaseEncoder):
        def __init__(self):
            super().__init__(output_dim=8)
            self.proj = nn.Linear(1, 8)

        def forward(self, w, sm, um):
            m = sm.to(w.dtype)
            avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
            return self.proj(avg) * um.unsqueeze(-1).to(w.dtype)

    torch.manual_seed(0)
    model = RetroProfile(
        encoder=_Enc(),
        pooling=MILAttentionPooling(input_dim=8, hidden_dim=8, num_heads=1, target_names=["age", "height"]),
        heads={"age": ResidualRetrievalHead(input_dim=8, hidden_dim=8, label_min=18.0, label_max=80.0,
                                            use_retrieval=False),
               "height": ResidualRetrievalHead(input_dim=8, hidden_dim=8, label_min=140.0, label_max=210.0,
                                               use_retrieval=False)},
        projector=MLPHead(input_dim=8, hidden_dim=8, output_dim=8, l2_normalize=True))
    stats = {"age": {"mean": 30.0, "std": 8.0}, "height": {"mean": 175.0, "std": 9.0}}
    lit = RetroProfileLitModule(model=model, target_names=["age", "height"], embedding_dim=8,
                                ordcon_weight=1.0, label_stats=stats,
                                scheduler={"name": "warmup_cosine", "warmup_steps": 1, "total_steps": 12})

    def batches(n, seed):
        g = torch.Generator().manual_seed(seed)
        return [collate_bags([SpeakerBag(waveforms=[torch.randn(800, generator=g) for _ in range(2)],
                                         sample_rate=16000, speaker_id=f"s{seed}{b}{i}", utt_ids=["a", "b"],
                                         labels={"age": 20.0 + 3 * i + b, "height": 160.0 + 4 * i + b})
                              for i in range(4)], targets=("age", "height")) for b in range(n)]

    cfg = _cfg(["age", "height"])
    trainer = pl.Trainer(max_epochs=3, accelerator="cpu", logger=CSVLogger(str(tmp_path)),
                         callbacks=train_script._build_callbacks(cfg, tmp_path),
                         enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    from torch.utils.data import DataLoader

    def loader(items):
        return DataLoader(items, batch_size=None, collate_fn=lambda b: b)

    trainer.fit(lit, train_dataloaders=loader(batches(4, 1)), val_dataloaders=loader(batches(2, 2)))

    files = sorted(p.name for p in (tmp_path / "checkpoints").glob("*.ckpt"))
    assert "last.ckpt" in files
    assert len([f for f in files if f.startswith("best_age-epoch=")]) == 1
    assert len([f for f in files if f.startswith("best_height-epoch=")]) == 1
    assert len([f for f in files if f.startswith("epoch=")]) == 3          # main top-3
    assert trainer.checkpoint_callback.monitor == "val/combined_mae"   # main one stays first
