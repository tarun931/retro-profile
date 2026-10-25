"""Plot the per-epoch diagnostics written by ``PerEpochDiagnostics``.

Four panels, each answering a question the final metrics cannot:

  1. **Accuracy vs epoch** -- did age genuinely degrade over training, or did the
     selection metric just keep the wrong epoch? A dashed line marks the epoch
     at which retrieval switched on, so any discontinuity is visible.
  2. **n_eff vs epoch** -- the effective number of clips the attention averages
     (uniform over K gives K). Does the attention start uniform and become
     selective, or the reverse, and does it differ per trait?
  3. **Sentence-type preference vs epoch** -- weight given to TIMIT's SA / SI / SX
     relative to uniform. Was the height head's preference for SA (identical text
     across all speakers, so only vocal-tract variation remains) *learned*, or an
     initialisation artefact?
  4. **anchor_share vs epoch** -- how much of a retrieval prediction is the kNN
     anchor rather than the learned residual. Approaching 1.0 means the residual
     has stopped contributing and the model is pure kNN.

All curves are VALIDATION, on the 70 held-out dev speakers. They are for
understanding a run, not for reporting.

Usage:
    python scripts/plot_diagnostics.py --run runs/retro_finetune_timit_s42
    python scripts/plot_diagnostics.py --run runs/<a> --compare runs/<b>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="runs/<run_name>")
    p.add_argument("--compare", type=Path, default=None,
                   help="optional second run to overlay (e.g. the no-kNN control)")
    p.add_argument("--output", type=Path, default=None,
                   help="PNG path (default: <run>/diagnostics.png)")
    return p.parse_args()


def load(run_dir: Path) -> pd.DataFrame:
    csv = run_dir / "diagnostics.csv"
    if not csv.is_file():
        raise SystemExit(
            f"{csv} not found. It is written by PerEpochDiagnostics during training, "
            "so it only exists for runs started after that callback was added."
        )
    return pd.read_csv(csv)


def retrieval_onset(df: pd.DataFrame) -> int | None:
    """First epoch at which the datastore was live — where the curve may jump."""
    if "retrieval_active" not in df:
        return None
    on = df[df["retrieval_active"].astype(str).str.lower().isin(("true", "1"))]
    return int(on["epoch"].min()) if len(on) else None


def summarise(df: pd.DataFrame, label: str) -> list[str]:
    """The text version — useful when matplotlib is unavailable."""
    lines = [f"## {label}", ""]
    onset = retrieval_onset(df)
    lines.append(f"Retrieval became active at epoch {onset}."
                 if onset is not None else "Retrieval never became active.")
    lines.append("")
    lines.append("| target | best epoch (MAE) | best MAE | final MAE | final n_eff | final anchor_share |")
    lines.append("|---|---|---|---|---|---|")
    for t, g in df.groupby("target"):
        g = g.sort_values("epoch")
        b = g.loc[g["mae"].idxmin()]
        f = g.iloc[-1]
        lines.append(
            f"| {t} | {int(b['epoch'])} | {b['mae']:.3f} | {f['mae']:.3f} | "
            f"{f.get('n_eff', float('nan')):.2f} | {f.get('anchor_share', float('nan')):.3f} |"
        )
    lines.append("")
    lines.append("A best epoch far from the final one means training kept going after")
    lines.append("the model stopped improving — and `save_top_k: 1` kept only whatever")
    lines.append("`checkpoint.monitor` preferred, not this.")
    lines.append("")
    return lines


def make_plots(frames: list[tuple[str, pd.DataFrame]], out: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    targets = sorted(frames[0][1]["target"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def onset_line(ax, df):
        o = retrieval_onset(df)
        if o is not None:
            ax.axvline(o, ls="--", lw=1, color="grey")

    # 1. accuracy
    ax = axes[0][0]
    for label, df in frames:
        for t in targets:
            g = df[df["target"] == t].sort_values("epoch")
            ax.plot(g["epoch"], g["mae"], label=f"{t} ({label})")
    onset_line(ax, frames[0][1])
    ax.set_title("Validation MAE vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE (yr / cm)")
    ax.legend(fontsize=8)

    # 2. n_eff
    ax = axes[0][1]
    for label, df in frames:
        for t in targets:
            g = df[df["target"] == t].sort_values("epoch")
            if "n_eff" in g:
                ax.plot(g["epoch"], g["n_eff"], label=f"{t} ({label})")
    onset_line(ax, frames[0][1])
    ax.set_title("Effective bag size (n_eff)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("n_eff")
    ax.legend(fontsize=8)

    # 3. sentence-type preference (first run only — overlaying gets unreadable)
    ax = axes[1][0]
    label, df = frames[0]
    for t in targets:
        g = df[df["target"] == t].sort_values("epoch")
        for st, style in (("SA", "-"), ("SI", "--"), ("SX", ":")):
            col = f"w_{st}"
            if col in g:
                ax.plot(g["epoch"], g[col], style, label=f"{t} {st}")
    ax.axhline(1.0, lw=1, color="k")          # uniform == mean pooling
    onset_line(ax, df)
    ax.set_title(f"Sentence-type preference ({label})\n1.0 = mean pooling")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weight / uniform")
    ax.legend(fontsize=7, ncol=2)

    # 4. anchor share
    ax = axes[1][1]
    for label, df in frames:
        for t in targets:
            g = df[df["target"] == t].sort_values("epoch")
            if "anchor_share" in g:
                ax.plot(g["epoch"], g["anchor_share"], label=f"{t} ({label})")
    onset_line(ax, frames[0][1])
    ax.set_title("Anchor share of the prediction\n1.0 = residual contributes nothing")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|anchor| / (|anchor|+|delta|)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def main() -> None:
    args = parse_args()
    frames = [(args.run.name, load(args.run))]
    if args.compare:
        frames.append((args.compare.name, load(args.compare)))

    report: list[str] = ["# Per-epoch diagnostics (validation)", ""]
    for label, df in frames:
        report += summarise(df, label)
    text = "\n".join(report)
    print(text)

    out = args.output or (args.run / "diagnostics.png")
    (out.with_suffix(".md")).write_text(text)
    png = make_plots(frames, out)
    print(f"\nWrote {out.with_suffix('.md')}")
    print(f"Wrote {png}" if png else "(matplotlib not installed — skipped the figure)")


if __name__ == "__main__":
    main()
