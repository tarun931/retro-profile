"""Tests for the Kendall uncertainty loss used by the Gupta 2022 baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.losses.uncertainty import UncertaintyLoss


def test_log_vars_are_learnable_and_zero_init():
    loss = UncertaintyLoss(["age", "height", "gender"])
    assert set(loss.log_vars.keys()) == {"age", "height", "gender"}
    for name, p in loss.log_vars.items():
        assert isinstance(p, nn.Parameter)
        assert p.requires_grad
        assert float(p) == 0.0                       # init 0 -> weight exp(0)=1


def test_total_equals_sum_of_raw_losses_at_init():
    """At init (log_var=0), exp(0)*L + 0 = L, so total == sum of raw losses."""
    loss = UncertaintyLoss(["age", "height", "gender"])
    raw = {"age": torch.tensor(2.0), "height": torch.tensor(3.0),
           "gender": torch.tensor(0.5)}
    out = loss(raw)
    assert torch.isclose(out["total"], torch.tensor(5.5), atol=1e-6)


def test_total_is_finite_and_backprops_to_log_vars():
    loss = UncertaintyLoss(["age", "height"])
    raw = {"age": torch.tensor(1.5, requires_grad=True),
           "height": torch.tensor(2.5, requires_grad=True)}
    out = loss(raw)
    assert torch.isfinite(out["total"])
    out["total"].backward()
    for p in loss.log_vars.values():
        assert p.grad is not None and torch.isfinite(p.grad)


def test_gradient_downweights_noisy_task():
    """A high-loss (noisy) task should get its log-variance pushed UP by the
    optimizer, i.e. its gradient is negative (since s <- s - lr*grad). A
    low-loss task gets the opposite. Grad wrt s at s=0 is (1 - L)."""
    loss = UncertaintyLoss(["noisy", "clean"])
    raw = {"noisy": torch.tensor(4.0), "clean": torch.tensor(0.25)}
    loss(raw)["total"].backward()
    assert loss.log_vars["noisy"].grad.item() < 0.0    # -> s increases -> down-weight
    assert loss.log_vars["clean"].grad.item() > 0.0    # -> s decreases -> up-weight


def test_missing_task_is_skipped():
    """Tasks absent from the losses dict (e.g. gender missing) don't contribute."""
    loss = UncertaintyLoss(["age", "height", "gender"])
    out = loss({"age": torch.tensor(1.0), "height": torch.tensor(1.0)})
    assert "uncertainty/gender" not in out
    assert torch.isclose(out["total"], torch.tensor(2.0), atol=1e-6)
