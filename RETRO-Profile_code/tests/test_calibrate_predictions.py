"""scripts/calibrate_predictions.py: fit 2 numbers per target on VALIDATION predictions, apply to test.

Pins: the fit never sees test labels; a shifted-and-stretched model is corrected; gender rows are
ignored; validation and test must not share speakers; the output is readable by paired_bootstrap; the
leave-one-out validation score (used to choose between runs) never lets a speaker grade itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = _load("calibrate_predictions")
pb = _load("paired_bootstrap")


def _preds(prefix, n, rng):
    """Age truth 20-60; the model ranks well but reads 1.5x the spread and +10 yr."""
    truth = rng.uniform(20, 60, n)
    pred = 1.5 * (truth - 40) + 50 + rng.normal(0, 1, n)
    rows = [{"speaker_id": f"{prefix}{i}", "target": "age", "prediction": p, "ground_truth": t}
            for i, (p, t) in enumerate(zip(pred, truth, strict=True))]
    rows += [{"speaker_id": f"{prefix}{i}", "target": "gender", "prediction": 0.4, "ground_truth": 1.0}
             for i in range(n)]
    return pd.DataFrame(rows)


@pytest.fixture
def run(tmp_path):
    rng = np.random.default_rng(0)
    for sub, prefix, n in (("eval_spk_val", "v", 70), ("eval_spk", "t", 168)):
        (tmp_path / sub).mkdir()
        _preds(prefix, n, rng).to_csv(tmp_path / sub / "predictions_timit.csv", index=False)
    return tmp_path


def test_linear_calibration_removes_shift_and_stretch(run):
    r = cal.calibrate_run(run, "timit", ["age", "height"], "eval_spk_val", "eval_spk", "eval_spk_cal")["age"]
    assert r["a"] == pytest.approx(1 / 1.5, abs=0.02) and r["n_val"] == 70 and r["n_test"] == 168
    assert r["raw"]["mae"] > 8 and r["linear"]["mae"] < 1.2 and abs(r["linear"]["bias"]) < 0.5
    assert r["shift_only"]["mae"] < r["raw"]["mae"]
    assert r["linear"]["rmse"] >= r["linear"]["mae"] and r["raw"]["rmse"] > r["linear"]["rmse"]
    out = pd.read_csv(run / "eval_spk_cal" / "predictions_timit.csv")
    assert set(out.target) == {"age"} and len(out) == 168                      # gender rows ignored
    assert json.loads((run / "calibration_timit.json").read_text())["targets"]["age"]["a"] == r["a"]


def test_fit_never_uses_test_labels(run):
    a = cal.calibrate_run(run, "timit", ["age"], "eval_spk_val", "eval_spk", "eval_spk_cal")["age"]
    test = pd.read_csv(run / "eval_spk" / "predictions_timit.csv")
    test["ground_truth"] = test["ground_truth"] + 100                           # wreck the test labels
    test.to_csv(run / "eval_spk" / "predictions_timit.csv", index=False)
    b = cal.calibrate_run(run, "timit", ["age"], "eval_spk_val", "eval_spk", "eval_spk_cal")["age"]
    assert (a["a"], a["b"], a["shift"]) == (b["a"], b["b"], b["shift"])


def test_shared_speakers_are_refused(run):
    val = pd.read_csv(run / "eval_spk_val" / "predictions_timit.csv")
    val.loc[0, "speaker_id"] = "t0"
    val.to_csv(run / "eval_spk_val" / "predictions_timit.csv", index=False)
    with pytest.raises(ValueError, match="share speakers"):
        cal.calibrate_run(run, "timit", ["age"], "eval_spk_val", "eval_spk", "eval_spk_cal")


def test_paired_bootstrap_reads_calibrated_predictions(run):
    cal.calibrate_run(run, "timit", ["age"], "eval_spk_val", "eval_spk", "eval_spk_cal")
    rows = pb.load_predictions(run, "timit", "age", "calibrated")
    assert len(rows) == 168 and rows.index.str.startswith("t").all()


def _add_height(run, rng):
    """Append height rows (truth 150-195; read +8 cm) to both splits of ``run``."""
    for sub in ("eval_spk_val", "eval_spk"):
        path = run / sub / "predictions_timit.csv"
        df = pd.read_csv(path)
        ids = df.loc[df.target == "age", "speaker_id"]
        truth = rng.uniform(150, 195, len(ids))
        h = pd.DataFrame({"speaker_id": ids.to_numpy(), "target": "height",
                          "prediction": truth + 8 + rng.normal(0, 3, len(ids)), "ground_truth": truth})
        pd.concat([df, h]).to_csv(path, index=False)


def test_leave_one_out_matches_closed_form():
    """LOO error of a least-squares line = in-sample residual / (1 - leverage)."""
    rng = np.random.default_rng(1)
    x = rng.uniform(20, 60, 70)
    y = 0.6 * x + 10 + rng.normal(0, 4, 70)
    val = pd.DataFrame({"prediction": x, "ground_truth": y})
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    lev = 1 / len(x) + (x - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
    np.testing.assert_allclose(y - cal.loo_predictions(val).to_numpy(), resid / (1 - lev), rtol=1e-8)


def test_leave_one_out_is_never_flattering(run):
    r = cal.calibrate_run(run, "timit", ["age"], "eval_spk_val", "eval_spk", "eval_spk_cal")["age"]
    assert r["val_loo"]["mae"] > r["val_in_sample"]["mae"]
    assert r["val_loo"]["rmse"] > r["val_in_sample"]["rmse"]


def test_combined_selection_score(run):
    only_age = cal.calibrate_run(run, "timit", ["age", "height"], "eval_spk_val", "eval_spk", "eval_spk_cal")
    assert cal.combined_score(only_age) is None                                  # a target missing -> no score
    _add_height(run, np.random.default_rng(2))
    res = cal.calibrate_run(run, "timit", ["age", "height"], "eval_spk_val", "eval_spk", "eval_spk_cal")
    want = res["age"]["val_loo"]["mae"] / 5.91 + res["height"]["val_loo"]["mae"] / 7.40
    saved = json.loads((run / "calibration_timit.json").read_text())
    assert saved["val_loo_combined"] == pytest.approx(want)
    assert saved["val_in_sample_combined"] < saved["val_loo_combined"]
