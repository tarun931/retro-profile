"""Tests for helpers in scripts/train.py.

Regression: train.py used to assume every model had `projector.output_dim` or
`encoder.output_dim`, which crashed the Jaid 2023 baseline (it has a `frontend`
and a `feature_dim` instead) before a single training step ran.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch.nn as nn

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_script", _SCRIPT)
train_script = importlib.util.module_from_spec(_spec)
sys.modules["train_script"] = train_script
_spec.loader.exec_module(train_script)

embedding_dim_of = train_script.embedding_dim_of


class _WithProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Identity()
        self.projector.output_dim = 256
        self.encoder = nn.Identity()
        self.encoder.output_dim = 768


class _WithEncoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.encoder.output_dim = 768


class _WithFeatureDim(nn.Module):
    """Shape of the Jaid baseline: no projector, no `encoder`."""

    def __init__(self):
        super().__init__()
        self.frontend = nn.Conv1d(1, 64, 3)
        self.feature_dim = 1280


class _WithNothing(nn.Module):
    pass


def test_projector_wins_over_encoder():
    assert embedding_dim_of(_WithProjector()) == 256


def test_falls_back_to_encoder():
    assert embedding_dim_of(_WithEncoderOnly()) == 768


def test_falls_back_to_feature_dim_for_jaid_style_models():
    assert embedding_dim_of(_WithFeatureDim()) == 1280


def test_raises_a_clear_error_when_nothing_matches():
    with pytest.raises(AttributeError, match="Cannot infer an embedding dim"):
        embedding_dim_of(_WithNothing())


def test_real_jaid_baseline_does_not_crash_the_helper():
    from retro_profile.models.baselines.jaid2023 import Jaid2023Baseline
    assert embedding_dim_of(Jaid2023Baseline()) == 1280


def test_sharing_strategy_unchanged_unless_requested():
    """Joint Jaid (job 60955713) died with 'received 0 items of ancdata'; the switch is opt-in only."""
    import torch.multiprocessing as tmp
    before = tmp.get_sharing_strategy()
    assert train_script.apply_sharing_strategy({}) is None
    assert tmp.get_sharing_strategy() == before
    try:
        assert train_script.apply_sharing_strategy({"TORCH_SHARING_STRATEGY": "file_system"}) == "file_system"
        assert tmp.get_sharing_strategy() == "file_system"
    finally:
        tmp.set_sharing_strategy(before)
