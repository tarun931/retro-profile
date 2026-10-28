"""Post-hoc calibration of a cross-corpus model on the TARGET corpus's validation speakers.

A model trained on other corpora can rank TIMIT speakers well and still put the whole scale off
(HeightCeleb + NISP -> TIMIT: age r 0.71, bias +5.9 yr; SESSION_HANDOFF §0.7.13). Per target, this
fits ``corrected = a * prediction + b`` by least squares on the VALIDATION speakers' bag predictions
(``runs/<run>/eval_spk_val/``, from evaluate.py --split val) and applies it
unchanged to the test speakers (``eval_spk/``). Test labels are never used to fit. Method fixed in
advance (2026-10-12): linear (primary); shift-only (a = 1, b = mean val error) as a check. The network
is not retrained -- two numbers per target.

Choosing BETWEEN runs (agreed 2026-10-13) uses the leave-one-out validation score: each validation speaker is
calibrated by a fit on the other speakers only, so none grades itself; per run, age/5.91 + height/7.40 on those
MAEs (``training.combined_normalisers``), lower is better. The in-sample validation score is kept for reference.

Writes ``runs/<run>/eval_spk_cal/predictions_<corpus>.csv`` (linear; readable by
``paired_bootstrap.py --reading calibrated``) and ``runs/<run>/calibration_<corpus>.json``.

    python scripts/calibrate_predictions.py runs/heightceleb_nisp_l100_s5_timit_s42 runs/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANCE_MAE = {"age": 5.91, "height": 7.40}  # configs/training/default.yaml combined_normalisers


def load(run: Path, sub: str, corpus: str, targets: list[str]) -> pd.DataFrame:
    """Bag predictions with a ground truth, one row per (speaker, target)."""
    path = Path(run) / sub / f"predictions_{corpus}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["target"].isin(targets) & df["ground_truth"].notna()]
    if df.duplicated(["speaker_id", "target"]).any():
        raise ValueError(f"{path}: more than one row per speaker and target")
    return df


def fit(val: pd.DataFrame) -> dict[str, float]:
    """Least-squares a, b (truth ~ a * prediction + b) and the shift-only b, on validation rows."""
    x, y = val["prediction"].to_numpy(float), val["ground_truth"].to_numpy(float)
    a, b = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "shift": float((y - x).mean()), "n_val": int(len(val))}


def loo_predictions(val: pd.DataFrame) -> pd.Series:
    """Each validation row calibrated by a least-squares fit on the OTHER rows only."""
    x, y = val["prediction"].to_numpy(float), val["ground_truth"].to_numpy(float)
    out = np.empty_like(x)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        a, b = np.polyfit(x[keep], y[keep], 1)
        out[i] = a * x[i] + b
    return pd.Series(out, index=val.index)


def combined_score(results: dict[str, dict], field: str = "val_loo") -> float | None:
    """age/5.91 + height/7.40 on ``field`` MAEs; None unless every target in CHANCE_MAE is present."""
    if not set(CHANCE_MAE) <= set(results):
        return None
    return float(sum(results[t][field]["mae"] / CHANCE_MAE[t] for t in CHANCE_MAE))


def summary(pred: pd.Series, truth: pd.Series) -> dict[str, float]:
    err = pred - truth
    return {"mae": float(err.abs().mean()), "rmse": float((err ** 2).mean() ** 0.5),
            "bias": float(err.mean()), "r": float(pred.corr(truth))}


def calibrate_run(run: Path, corpus: str, targets: list[str], val_sub: str, test_sub: str,
                  out_sub: str) -> dict[str, dict]:
    """Fit on ``val_sub``, apply to ``test_sub``, write ``out_sub`` + JSON; returns per-target results."""
    val, test = load(run, val_sub, corpus, targets), load(run, test_sub, corpus, targets)
    if set(val["speaker_id"]) & set(test["speaker_id"]):
        raise ValueError(f"{run}: validation and test share speakers")
    results, rows = {}, []
    for t, g in test.groupby("target"):
        v = val[val["target"] == t]
        p = fit(v)
        raw = g["prediction"]
        lin = p["a"] * raw + p["b"]
        results[t] = {**p, "n_test": int(len(g)),
                      "val_in_sample": summary(p["a"] * v["prediction"] + p["b"], v["ground_truth"]),
                      "val_loo": summary(loo_predictions(v), v["ground_truth"]),
                      "raw": summary(raw, g["ground_truth"]),
                      "shift_only": summary(raw + p["shift"], g["ground_truth"]),
                      "linear": summary(lin, g["ground_truth"])}
        rows.append(g.assign(raw_prediction=raw, prediction=lin))
    out = Path(run) / out_sub
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out / f"predictions_{corpus}.csv", index=False)
    (Path(run) / f"calibration_{corpus}.json").write_text(json.dumps(
        {"method": "least squares on validation speakers; applied unchanged to test",
         "val": val_sub, "test": test_sub, "targets": results,
         "val_loo_combined": combined_score(results),
         "val_in_sample_combined": combined_score(results, "val_in_sample")}, indent=2))
    return results


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--targets", nargs="+", default=["age", "height"])
    ap.add_argument("--val-sub", default="eval_spk_val")
    ap.add_argument("--test-sub", default="eval_spk")
    ap.add_argument("--out-sub", default="eval_spk_cal")
    args = ap.parse_args(argv)
    for run in args.runs:
        res = calibrate_run(run, args.corpus, args.targets, args.val_sub, args.test_sub, args.out_sub)
        for t, r in res.items():
            print(f"{run.name[:40]:40s} {t:6s} a {r['a']:+.2f} b {r['b']:+7.1f} (val n {r['n_val']}) | test MAE "
                  f"raw {r['raw']['mae']:.2f} (bias {r['raw']['bias']:+.1f}) -> shift {r['shift_only']['mae']:.2f}"
                  f" -> linear {r['linear']['mae']:.2f} (bias {r['linear']['bias']:+.1f}) | RMSE raw "
                  f"{r['raw']['rmse']:.2f} -> linear {r['linear']['rmse']:.2f} | r {r['raw']['r']:.2f}")
            print(f"{'':40s} {'':6s} validation, linear: leave-one-out MAE {r['val_loo']['mae']:.2f} RMSE "
                  f"{r['val_loo']['rmse']:.2f} (in-sample MAE {r['val_in_sample']['mae']:.2f})")
        c = combined_score(res)
        if c is not None:
            print(f"{run.name[:40]:40s} SELECTION score (leave-one-out validation, age/5.91 + height/7.40): {c:.4f}"
                  f" (in-sample {combined_score(res, 'val_in_sample'):.4f})")


if __name__ == "__main__":
    main()
