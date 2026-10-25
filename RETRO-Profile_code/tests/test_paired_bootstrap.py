"""Tests for scripts/paired_bootstrap.py -- the difference CI must behave."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "paired_bootstrap", Path(__file__).resolve().parents[1] / "scripts" / "paired_bootstrap.py"
)
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)

SPK = [f"s{i}" for i in range(50)]


def _err(values):
    return pd.Series(values, index=SPK, dtype=float)


def test_uniformly_better_model_has_a_ci_excluding_zero():
    rng = np.random.default_rng(0)
    b = _err(rng.uniform(2, 8, len(SPK)))
    res = pb.paired_bootstrap(b - 1.0, b, n_boot=2000)
    assert res["diff"] == pytest.approx(-1.0)
    assert res["ci_hi"] < 0
    assert res["p_a_better"] == 1.0


def test_identical_models_have_a_zero_difference():
    b = _err(np.linspace(1, 5, len(SPK)))
    res = pb.paired_bootstrap(b, b.copy(), n_boot=500)
    assert res["diff"] == 0.0 and res["ci_lo"] == 0.0 and res["ci_hi"] == 0.0


def test_pairing_is_by_speaker_not_by_row_order():
    rng = np.random.default_rng(1)
    b = _err(rng.uniform(2, 8, len(SPK)))
    a = (b - 0.5).sample(frac=1.0, random_state=3)       # shuffled row order
    assert pb.paired_bootstrap(a, b, n_boot=500)["diff"] == pytest.approx(-0.5)


def test_different_speaker_sets_are_refused():
    b = _err(np.ones(len(SPK)))
    with pytest.raises(ValueError, match="same speakers"):
        pb.paired_bootstrap(b.iloc[:-1], b)


def test_gender_lookup_uses_training_means(tmp_path):
    train = pd.DataFrame({"speaker_id": ["t1", "t2", "t3", "t4"],
                          "gender": ["M", "M", "F", "F"],
                          "height": [180.0, 170.0, 160.0, 150.0]})
    train.to_csv(tmp_path / "train.csv", index=False)
    rows = pd.DataFrame({"gender": ["M", "F"], "ground_truth": [177.0, 151.0]},
                        index=pd.Index(["x", "y"], name="speaker_id"))
    err = pb.gender_lookup_errors(rows, tmp_path / "train.csv", "height")
    assert err.tolist() == [2.0, 4.0]                    # |175-177|, |155-151|
