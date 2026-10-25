"""Is the height floor an information limit, or is the model hedging?

Height MAE has sat at ~5 cm across every model and every run in this project,
and it has been described as a "physical floor". That framing has never been
tested. There are two very different explanations and they call for opposite
responses:

  MEAN REVERSION (fixable). The model compresses its predictions toward the
  training mean. The regression slope of true-on-predicted is well below 1, so
  it is hedging: confident predictions are penalised by the loss more than
  cautious ones. Fixes are cheap -- reweight the loss, predict residuals from
  the mean, or calibrate the output affinely after training.

  INFORMATION LIMIT (not fixable by architecture). The slope is near 1 but the
  scatter is wide. The signal simply is not in the audio at this resolution, and
  no pooling, retrieval or loss change will recover it.

The diagnostic is the slope of the least-squares fit of TRUE on PREDICTED. A
perfectly calibrated model gives slope 1.0. A model predicting the mean for
everyone gives slope 0 (and undefined, since predictions have no variance).

Also reported per gender, because height correlates ~0.69-0.72 overall but only
~0.32-0.40 within gender (SESSION_HANDOFF.md §3.1b) -- so a model can look
calibrated overall purely by separating men from women, while being close to
uninformative inside each group. `oracle_gender_mae` makes that explicit: it is
the MAE of predicting each gender's mean height and nothing else. A model that
does not beat it has learned gender and nothing more.

Usage:
    python scripts/analyse_calibration.py --predictions runs/<run>/eval_spk/predictions_timit.csv
    python scripts/analyse_calibration.py --run runs/<run> --target height
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _fit(pred: pd.Series, true: pd.Series) -> dict[str, float]:
    """Least-squares fit of TRUE on PREDICTED, plus error summaries."""
    n = len(pred)
    out: dict[str, float] = {"n": float(n)}
    if n < 2:
        return out
    pv, tv = float(pred.var(ddof=1)), float(true.var(ddof=1))
    err = true - pred
    out["mae"] = float(err.abs().mean())
    out["rmse"] = float(math.sqrt((err**2).mean()))
    out["bias"] = float(err.mean())
    out["pred_std"] = float(pred.std(ddof=1))
    out["true_std"] = float(true.std(ddof=1))
    # std ratio: 1.0 means the model's predictions are as spread out as reality.
    out["std_ratio"] = out["pred_std"] / out["true_std"] if tv > 0 else float("nan")
    if pv <= 1e-12 or tv <= 1e-12:
        out["slope"] = float("nan")
        out["pearson"] = float("nan")
        return out
    cov = float(((pred - pred.mean()) * (true - true.mean())).sum() / (n - 1))
    out["slope"] = cov / pv
    out["pearson"] = cov / math.sqrt(pv * tv)
    return out


def analyse(df: pd.DataFrame, target: str) -> dict:
    # evaluator.py writes LONG format: one row per (speaker, target), with the
    # value in `prediction` / `ground_truth` and the target named in `target`.
    if "target" in df.columns and "prediction" in df.columns:
        d = df[df["target"] == target].copy()
        if d.empty:
            have = ", ".join(sorted(df["target"].dropna().unique()))
            raise SystemExit(f"no rows for target {target!r} (file has: {have})")
        pcol, tcol = "prediction", "ground_truth"
        keep = [pcol, tcol] + (["gender"] if "gender" in d.columns else [])
        d = d[keep]
    else:  # wide format fallback
        pcol, tcol = f"pred_{target}", f"true_{target}"
        if pcol not in df or tcol not in df:
            raise SystemExit(
                f"columns {pcol!r}/{tcol!r} not in predictions "
                f"(have: {', '.join(df.columns[:12])}...)"
            )
        keep = [pcol, tcol] + (["gender"] if "gender" in df.columns else [])
        d = df[keep].copy()
    d = d.dropna(subset=[pcol, tcol])
    report: dict = {"target": target, "overall": _fit(d[pcol], d[tcol])}

    # Predicting the training mean for everyone -- the chance baseline.
    report["overall"]["constant_mae"] = float((d[tcol] - d[tcol].mean()).abs().mean())

    if "gender" in d and d["gender"].notna().any():
        per: dict[str, dict] = {}
        oracle_err = []
        for g, sub in d.groupby("gender"):
            if len(sub) < 2:
                continue
            per[str(g)] = _fit(sub[pcol], sub[tcol])
            # Oracle-gender baseline: this group's own mean, for every member.
            oracle_err.extend((sub[tcol] - sub[tcol].mean()).abs().tolist())
        report["by_gender"] = per
        if oracle_err:
            report["oracle_gender_mae"] = float(sum(oracle_err) / len(oracle_err))
    return report


def verdict(r: dict) -> str:
    o = r["overall"]
    slope, ratio = o.get("slope", float("nan")), o.get("std_ratio", float("nan"))
    lines = []
    if not math.isnan(ratio):
        if ratio < 0.6:
            lines.append(
                f"MEAN REVERSION: predictions vary only {ratio:.0%} as much as the truth. "
                "The model is hedging toward the mean -- a loss/calibration problem, "
                "not an information limit. An affine recalibration is worth trying."
            )
        elif ratio < 0.85:
            lines.append(
                f"PARTIAL MEAN REVERSION: prediction spread is {ratio:.0%} of the truth's. "
                "Some hedging, but not the whole story."
            )
        else:
            lines.append(
                f"WELL SPREAD: prediction spread is {ratio:.0%} of the truth's, so the "
                "model is not simply hedging. The error looks like an information limit."
            )
    if not math.isnan(slope):
        lines.append(f"Slope of true-on-predicted is {slope:.2f} (1.00 = calibrated).")
    if "oracle_gender_mae" in r:
        gm, mae = r["oracle_gender_mae"], o.get("mae", float("nan"))
        gap = gm - mae
        lines.append(
            f"Predicting each gender's mean height alone gives MAE {gm:.2f}; "
            f"the model gives {mae:.2f} -- it adds {gap:+.2f} cm beyond knowing gender."
            + ("  That is almost nothing: the model has largely learned gender."
               if gap < 0.25 else "")
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, help="predictions CSV")
    ap.add_argument("--run", type=Path, help="run dir; uses eval_spk/predictions_timit.csv")
    ap.add_argument("--target", default="height")
    ap.add_argument("--output", type=Path, help="write JSON here")
    args = ap.parse_args()

    path = args.predictions
    if path is None and args.run is not None:
        path = args.run / "eval_spk" / "predictions_timit.csv"
    if path is None or not path.is_file():
        raise SystemExit("give --predictions <csv> or --run <dir>")

    report = analyse(pd.read_csv(path), args.target)
    print(f"# calibration: {args.target}  ({path})\n")
    o = report["overall"]
    print(f"  n            {int(o['n'])}")
    print(f"  MAE          {o.get('mae', float('nan')):.3f}"
          f"   (predict-the-mean: {o.get('constant_mae', float('nan')):.3f})")
    print(f"  bias         {o.get('bias', float('nan')):+.3f}")
    print(f"  pred std     {o.get('pred_std', float('nan')):.3f}"
          f"   true std {o.get('true_std', float('nan')):.3f}"
          f"   ratio {o.get('std_ratio', float('nan')):.3f}")
    print(f"  slope        {o.get('slope', float('nan')):.3f}"
          f"   pearson {o.get('pearson', float('nan')):.3f}")
    if "by_gender" in report:
        print("\n  by gender:")
        for g, m in sorted(report["by_gender"].items()):
            print(f"    {g:>2}  n={int(m['n']):3d}  MAE {m.get('mae', float('nan')):.3f}"
                  f"  slope {m.get('slope', float('nan')):+.3f}"
                  f"  r {m.get('pearson', float('nan')):+.3f}")
    print("\n" + verdict(report))

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
