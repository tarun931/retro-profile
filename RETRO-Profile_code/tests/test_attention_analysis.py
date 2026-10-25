"""Attention-by-sentence-type analysis (§3.5 / §7.7).

Entropy tells us *how much* MIL attention concentrates; it cannot tell us *what*
it concentrates on. TIMIT's utterance ids encode sentence type (SA identical
across all speakers, SX shared across groups, SI unique per speaker), so a
content-aware attention should downweight SA. These tests pin the arithmetic
that claim will rest on.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch

_SPEC = importlib.util.spec_from_file_location(
    "analyse_attention", Path(__file__).resolve().parent.parent / "scripts" / "analyse_attention.py"
)
aa = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aa)

from retro_profile.evaluation.evaluator import (  # noqa: E402  (after the spec-loaded module)
    _attention_rows,
    _attention_stats,
)


def test_sentence_type_parsing():
    assert aa.sentence_type("SA1") == "SA"
    assert aa.sentence_type("SA2") == "SA"
    assert aa.sentence_type("SX133") == "SX"
    assert aa.sentence_type("SI1573") == "SI"
    assert aa.sentence_type("weird") == "other"


def test_rel_to_uniform_is_one_when_attention_is_flat():
    """Uniform attention must score exactly 1.0 — that is mean pooling."""
    rows = [{"speaker_id": "s", "target": "age", "utt_id": u,
             "weight": 0.25, "rank": i, "n_clips": 4}
            for i, u in enumerate(["SA1", "SA2", "SI1", "SX1"])]
    g = aa.by_sentence_type(pd.DataFrame(rows))
    assert all(abs(v - 1.0) < 1e-9 for v in g["rel_to_uniform"])


def test_rel_to_uniform_detects_a_preference():
    """SI weighted 4x the others must come out clearly above 1."""
    rows = [
        {"speaker_id": "s", "target": "age", "utt_id": "SA1", "weight": 0.10, "rank": 2, "n_clips": 4},
        {"speaker_id": "s", "target": "age", "utt_id": "SA2", "weight": 0.10, "rank": 3, "n_clips": 4},
        {"speaker_id": "s", "target": "age", "utt_id": "SX1", "weight": 0.20, "rank": 1, "n_clips": 4},
        {"speaker_id": "s", "target": "age", "utt_id": "SI1", "weight": 0.60, "rank": 0, "n_clips": 4},
    ]
    g = aa.by_sentence_type(pd.DataFrame(rows)).set_index("type")
    assert g.loc["SI", "rel_to_uniform"] > 2.0
    assert g.loc["SA", "rel_to_uniform"] < 0.5
    assert g.loc["SI", "top1_rate"] == 1.0


# --- n_eff ------------------------------------------------------------------

def _stats(weights):
    comps = {"age": {"attention": torch.tensor([weights])}}
    return _attention_stats(comps, "age", 0, torch.ones(len(weights), dtype=torch.bool))


def test_n_eff_equals_k_for_uniform_attention():
    """Uniform over 8 clips averages 8 clips — the full sqrt(K) benefit."""
    assert abs(_stats([0.125] * 8)["n_eff"] - 8.0) < 1e-4


def test_n_eff_collapses_to_one_when_all_weight_is_on_one_clip():
    s = _stats([1.0, 0.0, 0.0, 0.0])
    assert abs(s["n_eff"] - 1.0) < 1e-3


def test_n_eff_is_between_one_and_k_when_selective():
    s = _stats([0.5, 0.2, 0.2, 0.1])
    assert 1.0 < s["n_eff"] < 4.0


# --- per-utterance rows -----------------------------------------------------

def test_attention_rows_normalise_and_rank():
    comps = {"age": {"attention": torch.tensor([[0.4, 0.3, 0.2, 0.1]])}}
    rows = _attention_rows(comps, "age", 0, "spk", torch.ones(4, dtype=torch.bool),
                           ["SA1", "SA2", "SI1", "SX1"])
    assert len(rows) == 4
    assert abs(sum(r["weight"] for r in rows) - 1.0) < 1e-6
    assert next(r["utt_id"] for r in rows if r["rank"] == 0) == "SA1"


def test_attention_rows_skip_padded_utterances():
    """A bag padded to 4 with only 2 real clips must yield 2 rows."""
    comps = {"age": {"attention": torch.tensor([[0.6, 0.4, 0.0, 0.0]])}}
    mask = torch.tensor([True, True, False, False])
    rows = _attention_rows(comps, "age", 0, "spk", mask, ["SA1", "SI1"])
    assert [r["utt_id"] for r in rows] == ["SA1", "SI1"]


def test_attention_rows_empty_without_attention():
    assert _attention_rows(None, "age", 0, "s", torch.ones(2, dtype=torch.bool), ["a", "b"]) == []
