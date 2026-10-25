"""Regression tests for `val/combined_mae`, the checkpoint-selection metric.

Background (SESSION_HANDOFF.md §7.2). Selection on `val/height/mae` alone cost
three Gupta runs (rule R5). Selection on `val/total` -- the uncertainty-weighted
training loss -- replaced that with a subtler failure: age's dynamic range over
a run is ~10x height's, so the sum tracks age almost alone, and on E11 it chose
epoch 48 (val age 4.54 / height 5.97) over epoch 26 (4.13 / 5.29), which was
better on *both* targets.

`combined_mae` sums the reported MAEs, each divided by its chance-level MAE, so
the quantity being selected on is the quantity being reported. These tests pin
the two properties that make it safe to monitor across corpora.
"""

from __future__ import annotations

import torch

from retro_profile.training.metrics import (
    DEFAULT_COMBINED_NORMALISERS,
    RegressionMetrics,
)


def _full_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x, dtype=torch.bool)


def _update(m: RegressionMetrics, preds: dict, labels: dict) -> None:
    m.update(preds, labels, {k: _full_mask(v) for k, v in labels.items()})


def test_combined_mae_is_the_normalised_sum_of_reported_maes():
    m = RegressionMetrics(["age", "height"])
    labels = {"age": torch.tensor([30.0, 40.0]), "height": torch.tensor([170.0, 180.0])}
    preds = {"age": torch.tensor([32.0, 40.0]), "height": torch.tensor([174.0, 180.0])}
    _update(m, preds, labels)
    out = m.compute()

    assert out["age/mae"] == 1.0        # (2 + 0) / 2
    assert out["height/mae"] == 2.0     # (4 + 0) / 2
    expected = 1.0 / 5.91 + 2.0 / 7.40
    assert abs(out["combined_mae"] - expected) < 1e-6


def test_combined_mae_skips_targets_with_no_labels():
    """HeightCeleb has no age column -- an all-NaN target must drop out.

    CLAUDE.md §3: a single model trains across corpora with different available
    targets. If the absent target contributed anything (0.0 included) the metric
    would silently reward a model for a target it never predicted.
    """
    m = RegressionMetrics(["age", "height"])
    labels = {
        "age": torch.tensor([float("nan"), float("nan")]),
        "height": torch.tensor([170.0, 180.0]),
    }
    preds = {"age": torch.tensor([30.0, 40.0]), "height": torch.tensor([174.0, 180.0])}
    _update(m, preds, labels)
    out = m.compute()

    assert "age/mae" not in out
    assert abs(out["combined_mae"] - 2.0 / 7.40) < 1e-6


def test_combined_mae_excludes_targets_without_a_normaliser():
    """`gender` is a classification target; its 'MAE' must never be summed in."""
    m = RegressionMetrics(["age", "height", "gender"])
    labels = {
        "age": torch.tensor([30.0, 40.0]),
        "height": torch.tensor([170.0, 180.0]),
        "gender": torch.tensor([0.0, 1.0]),
    }
    preds = {
        "age": torch.tensor([32.0, 40.0]),
        "height": torch.tensor([174.0, 180.0]),
        "gender": torch.tensor([1.0, 0.0]),
    }
    _update(m, preds, labels)
    out = m.compute()

    assert "gender/mae" in out and out["gender/mae"] == 1.0
    assert "gender" not in DEFAULT_COMBINED_NORMALISERS
    assert abs(out["combined_mae"] - (1.0 / 5.91 + 2.0 / 7.40)) < 1e-6


def test_combined_mae_prefers_the_epoch_better_on_both_targets():
    """The E11 failure, reduced to two epochs.

    val/total chose epoch 48 over epoch 26 despite epoch 26 being better on age
    AND height. Any metric worth monitoring must not be able to do that.
    """
    def combined(age_mae: float, height_mae: float) -> float:
        m = RegressionMetrics(["age", "height"])
        labels = {"age": torch.tensor([0.0]), "height": torch.tensor([0.0])}
        preds = {"age": torch.tensor([age_mae]), "height": torch.tensor([height_mae])}
        _update(m, preds, labels)
        return m.compute()["combined_mae"]

    epoch_26 = combined(4.13, 5.29)
    epoch_48 = combined(4.54, 5.97)
    assert epoch_26 < epoch_48


def test_custom_normalisers_override_the_timit_defaults():
    m = RegressionMetrics(["age", "height"], {"age": 1.0, "height": 1.0})
    labels = {"age": torch.tensor([30.0]), "height": torch.tensor([170.0])}
    preds = {"age": torch.tensor([31.0]), "height": torch.tensor([172.0])}
    _update(m, preds, labels)
    assert abs(m.compute()["combined_mae"] - 3.0) < 1e-6
