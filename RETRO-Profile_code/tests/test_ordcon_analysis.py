"""OrdCon diagnostics.

OrdCon's claim is about the *embedding space*, not MAE: it exists so that kNN
retrieval has an axis-ordered space to interpolate over. These tests pin the two
measurements that claim rests on — whether the space is ordered along the learned
direction, and how good the retrieved anchor is on its own.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

_SPEC = importlib.util.spec_from_file_location(
    "analyse_ordcon", Path(__file__).resolve().parent.parent / "scripts" / "analyse_ordcon.py"
)
ao = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ao)


# --- ordering metrics --------------------------------------------------------

def _space(n=60, noise=0.0, seed=0):
    """Embeddings whose FIRST axis is the label — a perfectly ordered space."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(20, 70, size=n)
    emb = rng.normal(scale=0.01, size=(n, 4))
    emb[:, 0] = (y - y.mean()) / y.std() + rng.normal(scale=noise, size=n)
    return emb, y, np.array([1.0, 0.0, 0.0, 0.0])


def test_perfectly_ordered_space_scores_near_one():
    m = ao.ordering_metrics(*_space())
    assert m["axis_pearson"] > 0.99
    assert m["ordering_accuracy"] > 0.99
    assert m["distance_spearman"] > 0.9


def test_unordered_space_scores_near_chance():
    """A direction orthogonal to the label axis must look like a coin flip."""
    emb, y, _ = _space()
    m = ao.ordering_metrics(emb, y, np.array([0.0, 1.0, 0.0, 0.0]))
    assert abs(m["axis_pearson"]) < 0.4
    assert 0.3 < m["ordering_accuracy"] < 0.7


def test_noise_degrades_ordering_monotonically():
    clean = ao.ordering_metrics(*_space(noise=0.0))
    noisy = ao.ordering_metrics(*_space(noise=3.0))
    assert clean["ordering_accuracy"] > noisy["ordering_accuracy"]


def test_direction_is_normalised_before_projection():
    """Scaling the direction must not change any metric."""
    emb, y, u = _space()
    a = ao.ordering_metrics(emb, y, u)
    b = ao.ordering_metrics(emb, y, u * 137.0)
    assert abs(a["axis_pearson"] - b["axis_pearson"]) < 1e-9
    assert a["ordering_accuracy"] == b["ordering_accuracy"]


def test_nan_labels_are_dropped():
    """HeightCeleb has no age column — NaN labels must not poison the metric."""
    emb, y, u = _space(n=40)
    y = y.copy()
    y[:10] = np.nan
    m = ao.ordering_metrics(emb, y, u)
    assert m["n"] == 30


# --- retrieval quality -------------------------------------------------------

def _preds():
    return pd.DataFrame({
        "target": ["age"] * 4,
        "ground_truth": [30.0, 40.0, 50.0, 60.0],
        "anchor":       [32.0, 38.0, 53.0, 57.0],     # anchor MAE = 2.5
        "delta":        [-1.0, 1.0, -2.0, 2.0],
        "prediction":   [31.0, 39.0, 51.0, 59.0],     # pred  MAE = 1.0
        "neighbour_0_label": [31.0, 41.0, 52.0, 58.0],
        "neighbour_1_label": [35.0, 44.0, 45.0, 66.0],
    })


def test_anchor_mae_is_measured_before_the_residual():
    m = ao.retrieval_quality(_preds())["age"]
    assert abs(m["anchor_mae"] - 2.5) < 1e-9
    assert abs(m["pred_mae"] - 1.0) < 1e-9


def test_residual_gain_is_anchor_minus_prediction():
    """What the learned residual buys on top of pure retrieval."""
    m = ao.retrieval_quality(_preds())["age"]
    assert abs(m["residual_gain_mae"] - 1.5) < 1e-9


def test_neighbour_label_mae_reflects_retrieval_relevance():
    m = ao.retrieval_quality(_preds())["age"]
    assert m["k_neighbours"] == 2
    # mean over both neighbours must be worse than the best-of-two
    assert m["neighbour_label_mae"] >= m["neighbour_label_best"]


def test_missing_anchor_columns_degrade_gracefully():
    """A no-kNN run has no anchor — the script must still report pred_mae."""
    df = _preds().drop(columns=["anchor", "delta",
                                "neighbour_0_label", "neighbour_1_label"])
    m = ao.retrieval_quality(df)["age"]
    assert "pred_mae" in m and "anchor_mae" not in m
