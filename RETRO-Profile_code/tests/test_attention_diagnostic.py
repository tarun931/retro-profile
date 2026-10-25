"""Tests for the MIL attention diagnostic in the evaluator.

`attn_entropy` is the evidence for *why* MIL helps or doesn't: 1.0 means the
attention collapsed to uniform (i.e. MIL is behaving like mean pooling), 0.0
means it put all its weight on a single utterance.
"""

from __future__ import annotations

import math

import torch

from retro_profile.evaluation.evaluator import _attention_stats


def _comps(weights):
    return {"age": {"attention": torch.tensor([weights], dtype=torch.float)}}


def test_uniform_attention_scores_entropy_one():
    mask = torch.ones(4, dtype=torch.bool)
    s = _attention_stats(_comps([0.25, 0.25, 0.25, 0.25]), "age", 0, mask)
    assert s["n_clips"] == 4
    assert abs(s["attn_entropy"] - 1.0) < 1e-6      # collapsed == mean pooling
    assert abs(s["attn_max"] - 0.25) < 1e-6


def test_one_hot_attention_scores_entropy_zero():
    mask = torch.ones(4, dtype=torch.bool)
    s = _attention_stats(_comps([1.0, 0.0, 0.0, 0.0]), "age", 0, mask)
    assert s["attn_entropy"] < 1e-3                 # maximally selective
    assert abs(s["attn_max"] - 1.0) < 1e-6


def test_selective_attention_is_between():
    mask = torch.ones(4, dtype=torch.bool)
    s = _attention_stats(_comps([0.7, 0.1, 0.1, 0.1]), "age", 0, mask)
    assert 0.0 < s["attn_entropy"] < 1.0
    assert abs(s["attn_max"] - 0.7) < 1e-6


def test_padded_clips_are_excluded():
    """Only real utterances count toward the diagnostic."""
    mask = torch.tensor([True, True, False, False])
    s = _attention_stats(_comps([0.5, 0.5, 0.0, 0.0]), "age", 0, mask)
    assert s["n_clips"] == 2
    assert abs(s["attn_entropy"] - 1.0) < 1e-6      # uniform over the 2 real clips


def test_missing_attention_returns_nan_not_crash():
    mask = torch.ones(3, dtype=torch.bool)
    for comps in (None, {"age": {"attention": None}}, {"age": {}}):
        s = _attention_stats(comps, "age", 0, mask)
        assert math.isnan(s["attn_entropy"])
