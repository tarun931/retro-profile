"""Per-epoch validation metrics for a run, recovered from its TensorBoard log.

Why this exists: `checkpoint.save_top_k: 1` keeps exactly one checkpoint, chosen
by `checkpoint.monitor`. If a *different* epoch was better for one target, that
checkpoint is gone -- but the metrics are not. Lightning's TensorBoardLogger has
been recording every validation epoch all along.

Use it to answer two questions:

  1. Did the selected epoch actually have the best age / height, or did the
     monitored metric pick a compromise? (SESSION_HANDOFF.md §7.2 -- selecting on
     the wrong quantity has already cost this project three runs.)
  2. Is the curve still improving at the end, or did it overfit early?

These are VALIDATION metrics on the 70 held-out dev speakers, not test. Test is
computed once, on the selected checkpoint. Do not quote these as results -- they
are for diagnosing selection.

Usage:
    python scripts/epoch_metrics.py --run runs/retro_finetune_timit_s42
    python scripts/epoch_metrics.py --run runs/<r> --csv epochs.csv --top 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="runs/<run_name>")
    p.add_argument("--csv", type=Path, default=None, help="also write the full table here")
    p.add_argument("--top", type=int, default=10,
                   help="how many rows to show per 'best epoch' ranking")
    return p.parse_args()


WANTED = (
    "val/total",
    "val/age/mae", "val/age/rmse", "val/age/pearson",
    "val/height/mae", "val/height/rmse", "val/height/pearson",
)


def load_scalars(run_dir: Path) -> pd.DataFrame:
    """Read every scalar we care about out of the run's TensorBoard events."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not events:
        raise SystemExit(f"No TensorBoard event files under {run_dir}")

    series: dict[str, dict[int, float]] = {}
    for ev in events:
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for tag in WANTED:
            if tag not in available:
                continue
            # Lightning logs validation once per epoch; `step` is the global step,
            # so index by order of appearance rather than trusting step numbers.
            for i, e in enumerate(acc.Scalars(tag)):
                series.setdefault(tag, {})[i] = e.value

    if not series:
        raise SystemExit("Event files found but none contained val/* scalars.")

    df = pd.DataFrame(series)
    df.index.name = "epoch"
    return df.reset_index()


def best_rows(df: pd.DataFrame, col: str, n: int) -> pd.DataFrame:
    if col not in df:
        return pd.DataFrame()
    keep = [c for c in ("epoch", "val/total", "val/age/mae", "val/height/mae") if c in df]
    return df.nsmallest(n, col)[keep]


def main() -> None:
    args = parse_args()
    df = load_scalars(args.run)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda v: f"{v:8.4f}")

    print(f"\n=== {args.run.name}: {len(df)} validation epochs ===\n")
    print(df.to_string(index=False))

    print("\n=== Which epoch each criterion would have chosen ===")
    for col in ("val/total", "val/age/mae", "val/height/mae"):
        if col in df:
            i = int(df[col].idxmin())
            r = df.loc[i]
            bits = [f"{c.split('/', 1)[1]}={r[c]:.4f}"
                    for c in ("val/total", "val/age/mae", "val/height/mae") if c in df]
            print(f"  best {col:16s} -> epoch {int(r['epoch']):3d}   " + "  ".join(bits))

    print("\nIf these disagree, `save_top_k: 1` discarded the others. The gap between")
    print("them is what selecting on the wrong quantity costs.")

    for col in ("val/age/mae", "val/height/mae"):
        b = best_rows(df, col, args.top)
        if not b.empty:
            print(f"\n=== Top {args.top} epochs by {col} ===")
            print(b.to_string(index=False))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
