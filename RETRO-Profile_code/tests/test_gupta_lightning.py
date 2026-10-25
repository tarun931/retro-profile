"""Tests for the Gupta 2022 LightningModule (uncertainty loss + gender + LR groups)."""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.models.baselines.gupta2022 import Gupta2022Baseline
from retro_profile.models.encoders.frame_encoders import _HFFrameEncoder
from retro_profile.training.gupta_lightning import Gupta2022LitModule

warnings.filterwarnings("ignore")   # silence Lightning's "no trainer" self.log warning

LABEL_STATS = {"age": {"mean": 40.0, "std": 12.0}, "height": {"mean": 175.0, "std": 9.0}}


class _StubFrameEncoder(nn.Module):
    def __init__(self, output_dim=16, tprime=6):
        super().__init__()
        self.output_dim = output_dim
        self.tprime = tprime
        self.proj = nn.Linear(1, output_dim)

    def extract_frames(self, waveforms, sample_mask):
        x = waveforms[:, : self.tprime].unsqueeze(-1)
        feats = self.proj(x)
        return feats, _HFFrameEncoder._downsample_mask(sample_mask, self.tprime)


def _make_lit():
    torch.manual_seed(0)
    model = Gupta2022Baseline(
        encoder=_StubFrameEncoder(16, 6), feature_dim=16, num_layers=1, nhead=2,
        dim_feedforward=32, dropout=0.0, fc_dim=8, label_stats=LABEL_STATS,
    )
    lit = Gupta2022LitModule(
        model=model, target_names=["age", "height"], label_stats=LABEL_STATS,
        encoder_lr=1e-5, head_lr=1e-4,
        scheduler={"name": "warmup_cosine", "warmup_steps": 2, "total_steps": 10},
    )
    return model, lit


def _batch(B=4):
    bags = [
        SpeakerBag([torch.randn(1600), torch.randn(1200)], 16000, f"s{i}", ["u0", "u1"],
                   {"age": 25.0 + 5 * i, "height": 168.0 + 4 * i, "gender": float(i % 2)})
        for i in range(B)
    ]
    return collate_bags(bags, targets=("age", "height", "gender"))


def test_training_step_finite_and_backprops():
    _, lit = _make_lit()
    loss = lit.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    # the learnable uncertainty log-vars must receive gradients
    assert all(p.grad is not None for p in lit.uncertainty_loss.log_vars.values())


def test_training_step_is_amp_safe():
    """Regression: the gender loss must be autocast/AMP-safe (fp16 on the T4).
    We mimic AMP with CPU bfloat16 autocast; plain binary_cross_entropy on a
    probability raises 'unsafe to autocast' here -- BCE-with-logits does not."""
    _, lit = _make_lit()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = lit.training_step(_batch(), 0)
    assert torch.isfinite(loss)


def test_loss_components_include_gender_and_log_vars():
    _, lit = _make_lit()
    comps, _ = lit._shared_step(_batch(), "train")
    assert "reg/age" in comps and "reg/height" in comps
    assert "bce/gender" in comps                        # gender supervised
    assert "log_var/age" in comps                       # uncertainty weights logged


def test_metrics_are_real_units_and_exclude_gender():
    model, lit = _make_lit()
    lit.eval()
    with torch.no_grad():
        lit.validation_step(_batch(), 0)
    m = lit.val_metrics.compute()
    assert "height/mae" in m and "age/mae" in m
    assert not any(k.startswith("gender") for k in m)   # gender never reported
    assert m["height/mae"] > 1.0 and m["age/mae"] > 1.0  # real cm/years, not z-scored


def test_optimizer_has_head_group_and_trains_log_vars():
    _, lit = _make_lit()                                # stub encoder is trainable
    cfg = lit.configure_optimizers()
    opt = cfg["optimizer"] if isinstance(cfg, dict) else cfg
    assert len(opt.param_groups) >= 1
    # the uncertainty log-vars are in the optimizer (head group)
    opt_params = {id(p) for grp in opt.param_groups for p in grp["params"]}
    assert all(id(p) in opt_params for p in lit.uncertainty_loss.log_vars.values())


# ---- loss-combination / regression-kind options (shared by both baselines) ----


def _lit_with(**kw):
    torch.manual_seed(0)
    model = Gupta2022Baseline(
        encoder=_StubFrameEncoder(16, 6), feature_dim=16, num_layers=1, nhead=2,
        dim_feedforward=32, dropout=0.0, fc_dim=8, label_stats=LABEL_STATS,
    )
    return Gupta2022LitModule(
        model=model, target_names=["age", "height"], label_stats=LABEL_STATS,
        scheduler={"name": "none"}, **kw,
    )


def test_default_is_gupta_recipe_uncertainty_plus_mse():
    lit = _lit_with()
    assert lit.loss_combination == "uncertainty"
    assert lit.uncertainty_loss is not None
    assert lit.reg_loss.kind == "mse"


def test_sum_combination_is_plain_total_of_task_losses():
    """Jaid 2023's recipe: no learnable weighting, just add the task losses."""
    lit = _lit_with(loss_combination="sum", regression_kind="l1")
    assert lit.uncertainty_loss is None
    assert lit.reg_loss.kind == "l1"

    comps, _ = lit._shared_step(_batch(), "train")
    expected = comps["reg/age"] + comps["reg/height"] + comps["bce/gender"]
    assert torch.allclose(comps["total"], expected, atol=1e-6)
    assert not any(k.startswith("log_var/") for k in comps)   # no uncertainty params


def test_sum_combination_still_trains():
    lit = _lit_with(loss_combination="sum", regression_kind="l1")
    loss = lit.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()


def test_invalid_loss_combination_rejected():
    with pytest.raises(ValueError):
        _lit_with(loss_combination="average")


def test_legacy_alias_points_at_the_generic_module():
    from retro_profile.training import Gupta2022LitModule as Alias
    from retro_profile.training import MultiTaskProfilingLitModule
    assert Alias is MultiTaskProfilingLitModule
