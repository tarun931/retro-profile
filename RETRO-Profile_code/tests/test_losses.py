"""Tests for losses, especially the OrdCon novelty."""

from __future__ import annotations

import torch

from retro_profile.losses.multitask import MultitaskLoss
from retro_profile.losses.ordcon import OrdConLoss
from retro_profile.losses.regression import MaskedRegressionLoss


# ---------- regression ----------------------------------------------------


def test_masked_regression_ignores_nan_labels():
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, float("nan"), 5.0])
    loss = MaskedRegressionLoss(kind="l1")
    val = loss(pred, tgt).item()
    # Only positions 0 and 2 contribute: |1-1| + |3-5| over 2 = 1.0
    assert abs(val - 1.0) < 1e-6


def test_masked_regression_zero_when_perfect():
    pred = torch.tensor([2.0, 3.0])
    tgt = torch.tensor([2.0, 3.0])
    assert MaskedRegressionLoss(kind="mse")(pred, tgt).item() < 1e-12


# ---------- OrdCon --------------------------------------------------------


def test_ordcon_zero_when_already_perfectly_ordered():
    """If embeddings are perfectly ordered along a fixed direction by the label,
    direction loss should be zero (with margin=0). The magnitude term may still
    contribute slightly, so we just check the direction component implicitly
    by setting magnitude_gain very high so that it saturates to zero too."""
    torch.manual_seed(0)
    target_names = ["height"]
    D = 4
    loss = OrdConLoss(embedding_dim=D, target_names=target_names, margin=0.0,
                      magnitude_gain=10.0, magnitude_scale=1.0)
    # Force the learned direction to be the standard basis vector e0.
    with torch.no_grad():
        loss.directions["height"].fill_(0.0)
        loss.directions["height"][0] = 1.0
    # Embeddings perfectly ordered along axis 0 by label.
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    embeddings = torch.zeros(4, D)
    embeddings[:, 0] = labels                         # perfectly aligned
    label_mask = torch.ones(4, dtype=torch.bool)
    out = loss(embeddings, {"height": labels}, {"height": label_mask})
    # direction loss must be 0; magnitude should also be 0 since dist(i,j)=|y_i-y_j|.
    assert out["ordcon/total"].item() == 0.0


def test_ordcon_positive_when_anti_correlated():
    torch.manual_seed(0)
    D = 4
    loss = OrdConLoss(embedding_dim=D, target_names=["height"], margin=0.5)
    with torch.no_grad():
        loss.directions["height"].fill_(0.0)
        loss.directions["height"][0] = 1.0
    # Labels increase but embeddings DECREASE on axis 0 — completely wrong order.
    labels = torch.tensor([1.0, 2.0, 3.0, 4.0])
    embeddings = torch.zeros(4, D)
    embeddings[:, 0] = -labels
    label_mask = torch.ones(4, dtype=torch.bool)
    out = loss(embeddings, {"height": labels}, {"height": label_mask})
    assert out["ordcon/total"].item() > 0.0


def test_ordcon_skips_targets_with_too_few_labels():
    """If only one valid label is present for a target, OrdCon for that target
    must be zero (no pairs to compare)."""
    D = 4
    loss = OrdConLoss(embedding_dim=D, target_names=["age", "height"])
    embeddings = torch.randn(4, D)
    labels = {
        "age":    torch.tensor([30.0, 40.0, 50.0, 60.0]),
        "height": torch.tensor([float("nan")] * 4),
    }
    mask = {
        "age":    torch.ones(4, dtype=torch.bool),
        "height": torch.zeros(4, dtype=torch.bool),
    }
    out = loss(embeddings, labels, mask)
    assert out["ordcon/height"].item() == 0.0
    # Age component should be a scalar tensor (could be > 0 or 0 depending on init).


# ---------- Multitask -----------------------------------------------------


def test_multitask_total_is_finite_and_components_are_logged():
    target_names = ["age", "height"]
    loss = MultitaskLoss(target_names=target_names, embedding_dim=16,
                         regression_weight=1.0, ordcon_weight=0.5)
    preds = {"age": torch.randn(4), "height": torch.randn(4)}
    labels = {"age": torch.tensor([30.0, 40.0, 50.0, 60.0]),
              "height": torch.tensor([170.0, 175.0, 180.0, 185.0])}
    mask = {"age": torch.ones(4, dtype=torch.bool),
            "height": torch.ones(4, dtype=torch.bool)}
    embeddings = torch.randn(4, 16)
    out = loss(preds, labels, mask, embeddings=embeddings)
    assert torch.isfinite(out["total"])
    for k in ("reg/total", "reg/age", "reg/height", "ordcon/total"):
        assert k in out
