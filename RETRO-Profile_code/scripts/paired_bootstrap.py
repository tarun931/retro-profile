"""Paired bootstrap over test speakers: is model A's MAE really different from B's?

Both models are scored on the SAME held-out speakers, so the uncertainty that
matters is that of the DIFFERENCE. Resampling speakers with replacement and
recomputing mean(|err_A|) - mean(|err_B|) each time gives a confidence interval
that reflects which 168 (TIMIT) or 135 (NISP) test speakers happened to be
drawn -- something seed-to-seed standard deviation does not capture at all.

With several runs per model (seeds), each speaker's error is averaged over those
runs before resampling. That gives the expected error of a single run, and the
bootstrap then measures test-speaker uncertainty around it. Seed variance is
reported alongside, per run, rather than mixed in.

Readings
    bag       runs/<run>/eval_spk/predictions_<corpus>.csv -- one prediction per
              speaker from a pooled bag (RETRO's method). CAUTION: bag clips are
              drawn from numpy's global RNG at load time (data/base.py), so two
              DIFFERENT architectures can be scored on different 8-of-10 subsets
              of a speaker's clips. Comparisons are still paired by speaker, but
              that clip difference adds noise to every difference.
    averaged  runs/<run>/eval_utt/predictions_<corpus>_speaker.csv -- per-clip
              predictions over ALL of a speaker's clips, averaged. Identical
              audio for every model, but not RETRO's own aggregation.

B may also be the oracle gender-mean lookup for height (``--b-gender-lookup``):
each test speaker is predicted as the mean height of TRAINING speakers of their
gender. It uses true gender, so it is an oracle diagnostic, not a deployable
audio-only baseline.

Usage
    python scripts/paired_bootstrap.py --target age --reading bag \\
        --a runs/retro_age_only_budget_timit_s{42,1,2} \\
        --b runs/gupta2022_finetune_wavlm_timit_s{42,1,2}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

READINGS = {
    "bag": ("eval_spk", "predictions_{corpus}.csv"),
    "averaged": ("eval_utt", "predictions_{corpus}_speaker.csv"),
    # bag predictions after calibrate_predictions.py (2 numbers per target fitted on validation)
    "calibrated": ("eval_spk_cal", "predictions_{corpus}.csv"),
}


def load_predictions(run: Path, corpus: str, target: str, reading: str) -> pd.DataFrame:
    """One row per speaker: speaker_id, prediction, ground_truth, gender."""
    sub, name = READINGS[reading]
    path = Path(run) / sub / name.format(corpus=corpus)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[(df["target"] == target) & df["ground_truth"].notna()]
    if df["speaker_id"].duplicated().any():
        raise ValueError(f"{path}: more than one {target!r} row per speaker")
    cols = ["speaker_id", "prediction", "ground_truth"]
    if "gender" in df.columns:
        cols.append("gender")
    return df[cols].set_index("speaker_id")


def speaker_errors(runs: list[Path], corpus: str, target: str, reading: str
                   ) -> tuple[pd.Series, list[float], pd.DataFrame]:
    """Per-speaker absolute error averaged over runs, each run's MAE, and run 1's rows."""
    frames = [load_predictions(r, corpus, target, reading) for r in runs]
    speakers = frames[0].index
    for r, f in zip(runs, frames, strict=True):
        if not f.index.sort_values().equals(speakers.sort_values()):
            raise ValueError(f"{r} was scored on a different set of speakers")
    errs = pd.concat(
        [(f["prediction"] - f["ground_truth"]).abs().rename(str(r)) for r, f in zip(runs, frames, strict=True)],
        axis=1,
    )
    return errs.mean(axis=1), [float(errs[c].mean()) for c in errs.columns], frames[0]


def gender_lookup_errors(rows: pd.DataFrame, train_manifest: Path, target: str) -> pd.Series:
    """Absolute error of predicting each speaker's training-set gender mean."""
    train = pd.read_csv(train_manifest).drop_duplicates("speaker_id")
    means = train.groupby("gender")[target].mean()
    if "gender" not in rows.columns or rows["gender"].isna().any():
        raise ValueError("gender is needed for every test speaker")
    return (rows["gender"].map(means) - rows["ground_truth"]).abs()


def paired_bootstrap(err_a: pd.Series, err_b: pd.Series, n_boot: int = 10_000,
                     seed: int = 0) -> dict[str, float]:
    """Mean of (err_a - err_b) over speakers, with a percentile 95% CI."""
    joined = pd.concat([err_a.rename("a"), err_b.rename("b")], axis=1, join="inner")
    if len(joined) != len(err_a) or len(joined) != len(err_b):
        raise ValueError("A and B must be scored on the same speakers")
    d = (joined["a"] - joined["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mae_a": float(joined["a"].mean()),
        "mae_b": float(joined["b"].mean()),
        "diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        # Share of resamples in which A has the lower error.
        "p_a_better": float((boots < 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", type=Path, required=True, help="run dirs (seeds) of model A")
    ap.add_argument("--b", nargs="+", type=Path, help="run dirs (seeds) of model B")
    ap.add_argument("--b-gender-lookup", type=Path, metavar="TRAIN_MANIFEST",
                    help="use the oracle gender-mean lookup as B")
    ap.add_argument("--target", required=True)
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--reading", choices=sorted(READINGS), default="bag")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if (args.b is None) == (args.b_gender_lookup is None):
        ap.error("give exactly one of --b or --b-gender-lookup")

    err_a, per_run_a, rows_a = speaker_errors(args.a, args.corpus, args.target, args.reading)
    if args.b is not None:
        err_b, per_run_b, _ = speaker_errors(args.b, args.corpus, args.target, args.reading)
    else:
        err_b = gender_lookup_errors(rows_a, args.b_gender_lookup, args.target)
        per_run_b = [float(err_b.mean())]
    res = paired_bootstrap(err_a, err_b, n_boot=args.n_boot, seed=args.seed)
    res.update(runs_a=len(args.a), runs_b=len(args.b or []),
               per_run_mae_a=per_run_a, per_run_mae_b=per_run_b)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
