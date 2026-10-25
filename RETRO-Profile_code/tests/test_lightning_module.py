"""Integration tests for the LightningModule.

The bug fixed here was subtle: the residual head outputs predictions in REAL
label units (cm, years), but the old LitModule:
  - normalized the labels to z-scores before computing the loss
  - did NOT normalize the predictions
This made the loss huge (off by a factor of ~std) and the gradients wrong.
Combined with the bias-zero init of the direct MLP, training collapsed to
``label_min`` because the model couldn't learn to escape the clamp.

These tests pin down the fixed behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.retrieval.builder import attach_datastore
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore
from retro_profile.training.lightning_module import RetroProfileLitModule


class _TinyEnc(BaseEncoder):
    """Stub encoder for fast tests. Projects waveform-mean into a fixed dim."""

    def __init__(self, output_dim=16):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        emb = self.proj(avg)
        return emb * um.unsqueeze(-1).to(emb.dtype)


def _make_model_and_litmodule(label_stats, with_datastore=False):
    torch.manual_seed(0)
    enc = _TinyEnc(output_dim=16)
    pool = MILAttentionPooling(
        input_dim=16, hidden_dim=8, num_heads=1, target_names=["age", "height"]
    )
    proj = MLPHead(input_dim=16, hidden_dim=16, output_dim=8, l2_normalize=True)
    heads = {
        "age":    ResidualRetrievalHead(input_dim=8, hidden_dim=8,
                                        label_min=18.0,  label_max=80.0),
        "height": ResidualRetrievalHead(input_dim=8, hidden_dim=8,
                                        label_min=140.0, label_max=210.0),
    }
    model = RetroProfile(encoder=enc, pooling=pool, heads=heads, projector=proj)

    if with_datastore:
        rng = np.random.RandomState(0)
        ds = RetrievalDatastore(embedding_dim=8, metric="cosine", temperature=0.1)
        for i in range(10):
            v = rng.randn(8).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            ds.add(DatastoreEntry(
                speaker_id=f"spk_{i}",
                embedding=v,
                labels={"age": 30.0 + i, "height": 165.0 + 2 * i},
            ))
        ds.build()
        attach_datastore(model, ds)

    lit = RetroProfileLitModule(
        model=model,
        target_names=["age", "height"],
        embedding_dim=8,
        regression_kind="smooth_l1",
        regression_weight=1.0,
        ordcon_weight=0.3,
        label_stats=label_stats,
        normalize_targets=True,
        optimizer={"name": "adamw", "lr": 1e-3},
        scheduler={"name": "warmup_cosine", "warmup_steps": 5, "total_steps": 100},
    )
    return model, lit


def _make_batch(B=4):
    bags = [
        SpeakerBag(
            waveforms=[torch.randn(1600), torch.randn(1200)],
            sample_rate=16000,
            speaker_id=f"s{i}",
            utt_ids=["u0", "u1"],
            labels={"age": 25.0 + 5 * i, "height": 168.0 + 4 * i},
        )
        for i in range(B)
    ]
    return collate_bags(bags, targets=("age", "height"))


LABEL_STATS = {
    "age":    {"mean": 40.0,  "std": 12.0, "min": 18.0,  "max": 80.0,  "count": 100},
    "height": {"mean": 175.0, "std": 9.0,  "min": 145.0, "max": 200.0, "count": 100},
}


# ──────────────────────────────────────────────────────────────────────────
# Regression tests
# ──────────────────────────────────────────────────────────────────────────


def test_litmodule_loss_is_finite_and_small_at_init():
    """Before fix: loss was huge (~hundreds) because predictions were in real
    units and labels were in z-score units. After fix: loss is in the
    z-score regime (typically < 5)."""
    _, lit = _make_model_and_litmodule(LABEL_STATS, with_datastore=False)
    batch = _make_batch()
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
    # The smooth-L1 loss in z-score space at init should be very modest.
    # If we accidentally compare real-unit preds (~midpoint) to z-score labels
    # (~ ±2), the loss would be > 100.
    assert loss.item() < 20.0, (
        f"Loss too large ({loss.item():.2f}); likely a unit mismatch between "
        f"predictions and labels."
    )


def test_litmodule_predictions_are_in_real_units_after_step():
    """After a training step, the predictions used for metrics must still be
    in real units, not normalized."""
    model, lit = _make_model_and_litmodule(LABEL_STATS, with_datastore=False)
    batch = _make_batch()
    lit.training_step(batch, 0)
    # Now do a manual forward and check ranges
    model.eval()
    with torch.no_grad():
        preds = model(batch)
    # Real-unit predictions must lie inside the head's label ranges.
    assert torch.all(preds["age"] >= 18.0) and torch.all(preds["age"] <= 80.0)
    assert torch.all(preds["height"] >= 140.0) and torch.all(preds["height"] <= 210.0)
    # And the means should be near the label-range midpoints (49 and 175)
    # at init — NOT at the lower bounds.
    assert preds["age"].mean().item()    > 35.0
    assert preds["height"].mean().item() > 160.0


def test_litmodule_with_datastore_anchor_in_real_units():
    """The kNN anchor read from the datastore is in REAL units (because we
    stored real-unit labels). The full prediction (anchor + delta) must
    stay in real units, not get mixed with the z-score scale by the loss."""
    model, lit = _make_model_and_litmodule(LABEL_STATS, with_datastore=True)
    batch = _make_batch()
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert loss.item() < 20.0, (
        f"Loss too large with datastore attached ({loss.item():.2f}); the "
        f"anchor units may not match the loss-side normalization."
    )
    # Predictions stay in real units
    model.eval()
    with torch.no_grad():
        comps = model.forward_with_components(batch)
    for t, target_range in [("age", (18.0, 80.0)), ("height", (140.0, 210.0))]:
        anchor = comps[t]["anchor"]
        pred   = comps[t]["pred"]
        # Anchor is a kNN-weighted mean of stored labels, must be inside the
        # observed range (which is a subset of the head's label range).
        assert torch.all(anchor >= target_range[0])
        assert torch.all(anchor <= target_range[1])
        assert torch.all(pred >= target_range[0])
        assert torch.all(pred <= target_range[1])


def test_litmodule_metrics_logged_in_real_units():
    """The MAE reported by val/test metrics must be in real label units
    (years, cm) — not normalized. A height MAE of ~5 cm is plausible;
    a height MAE of ~0.5 (z-score units) is the bug signature."""
    model, lit = _make_model_and_litmodule(LABEL_STATS, with_datastore=False)
    batch = _make_batch()
    lit.eval()
    with torch.no_grad():
        lit.validation_step(batch, 0)
    metrics = lit.val_metrics.compute()
    # height MAE at random init is ~|midpoint - true_height|, which is some
    # number of cm — definitely > 1 cm. If it's <0.1, predictions are
    # normalized (the bug).
    assert metrics["height/mae"] > 1.0, (
        f"height/mae={metrics['height/mae']:.3f} suggests metrics are in "
        f"normalized units, not real cm."
    )
    assert metrics["age/mae"] > 1.0, (
        f"age/mae={metrics['age/mae']:.3f} suggests metrics are in "
        f"normalized units, not real years."
    )


def test_litmodule_works_without_normalization():
    """Sanity check: with normalize_targets=False, the loss must still be
    finite and predictions still in real units."""
    torch.manual_seed(0)
    enc = _TinyEnc(output_dim=16)
    pool = MILAttentionPooling(
        input_dim=16, hidden_dim=8, num_heads=1, target_names=["age", "height"]
    )
    proj = MLPHead(input_dim=16, hidden_dim=16, output_dim=8, l2_normalize=True)
    heads = {
        "age":    ResidualRetrievalHead(input_dim=8, hidden_dim=8,
                                        label_min=18.0,  label_max=80.0),
        "height": ResidualRetrievalHead(input_dim=8, hidden_dim=8,
                                        label_min=140.0, label_max=210.0),
    }
    model = RetroProfile(encoder=enc, pooling=pool, heads=heads, projector=proj)
    lit = RetroProfileLitModule(
        model=model,
        target_names=["age", "height"],
        embedding_dim=8,
        ordcon_weight=0.0,                # disable OrdCon for simplicity
        label_stats=LABEL_STATS,
        normalize_targets=False,
    )
    batch = _make_batch()
    loss = lit.training_step(batch, 0)
    assert torch.isfinite(loss)
