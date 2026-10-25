"""Tests for the residual retrieval head — Component 3 of RETRO-Profile."""

from __future__ import annotations

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore


def _populate_datastore(D=4, n=10):
    rng = np.random.RandomState(0)
    ds = RetrievalDatastore(embedding_dim=D, metric="cosine", temperature=0.1)
    for i in range(n):
        v = rng.randn(D).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        ds.add(DatastoreEntry(
            speaker_id=f"spk_{i}",
            embedding=v,
            labels={"height": 160.0 + 2.0 * i},
        ))
    ds.build()
    return ds


def test_head_falls_back_to_direct_when_no_datastore():
    head = ResidualRetrievalHead(input_dim=4, hidden_dim=8, label_min=140.0,
                                 label_max=210.0, use_retrieval=True)
    z = torch.randn(3, 4)
    out = head(z)
    assert out.shape == (3,)
    assert torch.all(out >= 140.0) and torch.all(out <= 210.0)
    # Components flag returns dict
    comp = head(z, return_components=True)
    assert "pred" in comp


def test_head_uses_anchor_plus_delta_when_datastore_set():
    ds = _populate_datastore()
    head = ResidualRetrievalHead(input_dim=4, hidden_dim=8, label_min=140.0,
                                 label_max=210.0, use_retrieval=True, knn_k=5,
                                 delta_clip=3.0)
    head.set_datastore(ds, target_name="height")
    assert head.has_datastore

    z = torch.randn(2, 4)
    comp = head(z, return_components=True)
    assert comp["anchor"].shape == (2,)
    assert comp["delta"].shape == (2,)
    # Residual must be tanh-clipped.
    assert torch.all(comp["delta"].abs() <= 3.0 + 1e-5)
    # Final prediction must be inside the label range.
    assert torch.all(comp["pred"] >= 140.0) and torch.all(comp["pred"] <= 210.0)


def test_head_use_retrieval_false_disables_kNN():
    ds = _populate_datastore()
    head = ResidualRetrievalHead(input_dim=4, hidden_dim=8, use_retrieval=False)
    head.set_datastore(ds, target_name="height")
    # has_datastore must be False because use_retrieval is False.
    assert not head.has_datastore
    out = head(torch.randn(3, 4))
    assert out.shape == (3,)


# ──────────────────────────────────────────────────────────────────────────
# Regression tests for fixed bugs (don't remove)
# ──────────────────────────────────────────────────────────────────────────

def test_head_NOT_stuck_at_label_min_at_init():
    """At random init the direct MLP must produce predictions near the midpoint
    of the label range, NOT clamped to label_min.

    Regression test for the bug where:
        nn.Linear default bias ~0 + clamp(label_min=140, label_max=210, 0) = 140
    causing the model to start training from a degenerate state."""
    torch.manual_seed(0)
    head = ResidualRetrievalHead(
        input_dim=8, hidden_dim=16,
        label_min=140.0, label_max=210.0,
    )
    z = torch.randn(16, 8)
    out = head(z)
    # Mean prediction at init should be near the midpoint (175 cm), NOT 140 cm.
    midpoint = 0.5 * (140.0 + 210.0)
    assert abs(out.mean().item() - midpoint) < 1.0, (
        f"Direct head not centered at midpoint: mean={out.mean().item():.2f}, "
        f"expected ~{midpoint}. Predictions may be stuck at label_min."
    )
    # And no prediction should equal label_min exactly at init.
    assert not torch.any(out == 140.0)


def test_residual_branch_starts_at_zero_so_pred_equals_anchor():
    """With a populated datastore, predictions at random init should equal the
    kNN anchor (delta starts at 0 because we zero-init the residual MLP)."""
    ds = _populate_datastore()
    torch.manual_seed(0)
    head = ResidualRetrievalHead(
        input_dim=4, hidden_dim=8,
        label_min=140.0, label_max=210.0,
        use_retrieval=True, knn_k=5, delta_clip=5.0,
    )
    head.set_datastore(ds, target_name="height")
    z = torch.randn(3, 4)
    comp = head(z, return_components=True)
    # At init, delta is exactly 0 (zero-initialized last layer + tanh(0)=0)
    assert torch.allclose(comp["delta"], torch.zeros(3), atol=1e-6)
    # So pred == anchor (modulo the final clamp, which won't trigger here).
    assert torch.allclose(comp["pred"], comp["anchor"], atol=1e-6)


def test_head_output_is_in_real_label_units():
    """The head's output must be on the same scale as the label range (real
    units like cm/years), NOT on a normalized scale.

    Regression test for the unit-mismatch bug where the head produced
    normalized predictions while clamping/anchors were in real units."""
    ds = _populate_datastore()
    head = ResidualRetrievalHead(
        input_dim=4, hidden_dim=8,
        label_min=140.0, label_max=210.0,
        use_retrieval=True, knn_k=5,
    )
    head.set_datastore(ds, target_name="height")
    z = torch.randn(8, 4)
    out = head(z)
    # In real units the prediction must be inside [140, 210] cm.
    assert torch.all(out >= 140.0)
    assert torch.all(out <= 210.0)
    # And clearly NOT on a normalized scale (which would put values in [-3, +3]).
    assert out.mean().item() > 50.0
