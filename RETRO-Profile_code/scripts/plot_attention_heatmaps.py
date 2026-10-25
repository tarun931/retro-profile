"""MIL attention as heatmaps -- per speaker and clip, and as a trait x sentence-type grid.

Figure 1 (``<prefix>_speakers.png``): one panel per run and target. Rows are the
test speakers sorted by their TRUE value of that target; columns are the clips in
each speaker's bag, grouped by TIMIT sentence type into fixed slots (SA x2, SI x3,
SX x5 -- a bag holds 8 of these 10, the 2 unused slots are hatched). Colour is the
attention weight relative to an even split (weight x clips-in-bag), on a log2
scale: grey = even, red = more attention, blue = less.

Figure 2 (``<prefix>_grid.png``): the same weights averaged into trait quintiles x
sentence type, with the mean relative weight printed in each cell.

Reads runs/<run>/eval_spk/attention_<corpus>.csv and predictions_<corpus>.csv
(bag-mode evaluation; ground truth and gender come from the predictions file).

    python scripts/plot_attention_heatmaps.py --output docs/figures/attn \\
        --run "OrdCon off=runs/retro_ordcon_l000_timit_s42" --run "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

SLOTS = {"SA": 2, "SI": 3, "SX": 5}
TEXT, TEXT2, SURFACE, MID = "#0b0b0b", "#52514e", "#fcfcfb", "#f0efec"
MISSING = "#a9a89f"  # hatch ink for "slot not in this bag"
FEMALE, MALE = "#c3c2b7", "#52514e"
# Diverging blue <-> red with a neutral grey midpoint (equal steps per arm).
CMAP = LinearSegmentedColormap.from_list(
    "attn", ["#104281", "#3987e5", "#9ec5f4", MID, "#f2a8a7", "#e34948", "#9f2524"])
CMAP.set_bad(SURFACE)
LIM = 2.0          # log2 clip: 1/4x .. 4x of an even split
UNITS = {"age": "yr", "height": "cm"}


def load(run: Path, corpus: str) -> pd.DataFrame:
    """Attention rows joined to each speaker's ground truth and gender."""
    att = pd.read_csv(run / "eval_spk" / f"attention_{corpus}.csv")
    pred = pd.read_csv(run / "eval_spk" / f"predictions_{corpus}.csv")
    keep = ["speaker_id", "target", "ground_truth"] + (["gender"] if "gender" in pred else [])
    df = att.merge(pred[keep].drop_duplicates(["speaker_id", "target"]),
                   on=["speaker_id", "target"], how="left")
    df["stype"] = df["utt_id"].str[:2].str.upper()
    df["rel"] = df["weight"] * df["n_clips"]
    return df


def slot_matrix(df: pd.DataFrame, target: str, order: list[str]) -> np.ndarray:
    """(speakers, 10) relative weights in fixed SA/SI/SX slots; NaN = clip not in bag."""
    sub = df[df.target == target]
    cols = [(t, i) for t, n in SLOTS.items() for i in range(n)]
    mat = np.full((len(order), len(cols)), np.nan)
    pos = {s: r for r, s in enumerate(order)}
    for sid, g in sub.groupby("speaker_id"):
        for t, gg in g.groupby("stype"):
            for i, (_, row) in enumerate(gg.sort_values("utt_id").iterrows()):
                mat[pos[sid], cols.index((t, i))] = row["rel"]
    return mat


def n_eff(df: pd.DataFrame, target: str) -> float:
    w = df[df.target == target]
    return float((1.0 / w.groupby("speaker_id")["weight"].apply(lambda x: (x ** 2).sum())).mean())


def speakers_figure(runs: dict[str, pd.DataFrame], out: Path) -> None:
    targets = ["age", "height"]
    first = next(iter(runs.values()))
    fig = plt.figure(figsize=(2.15 * len(runs) + 1.6, 12.4), facecolor=SURFACE)
    outer = fig.add_gridspec(2, 1, hspace=0.22, left=0.07, right=0.90, top=0.91, bottom=0.05)
    for ti, target in enumerate(targets):
        truth = first[first.target == target].drop_duplicates("speaker_id").sort_values("ground_truth")
        order = truth["speaker_id"].tolist()
        inner = outer[ti].subgridspec(1, len(runs) + 1, width_ratios=[0.14] + [1] * len(runs), wspace=0.08)
        # Gender strip, shared by the row (same speaker order in every panel).
        gax = fig.add_subplot(inner[0])
        if "gender" in truth:
            g = (truth["gender"].astype(str).str.upper() == "M").astype(float).to_numpy()[:, None]
            gax.imshow(g, aspect="auto", cmap=LinearSegmentedColormap.from_list("g", [FEMALE, MALE]),
                       interpolation="nearest", vmin=0, vmax=1)
            gax.set_xticks([0], ["F/M"], fontsize=7, color=TEXT2)
        q = np.linspace(0, len(order) - 1, 5).round().astype(int)
        gax.set_yticks(q, [f"{v:.0f}" for v in truth["ground_truth"].to_numpy()[q]], fontsize=8, color=TEXT2)
        gax.set_ylabel(f"speakers sorted by true {target} ({UNITS[target]})", fontsize=9, color=TEXT)
        for s in gax.spines.values():
            s.set_visible(False)
        for ri, (label, df) in enumerate(runs.items()):
            ax = fig.add_subplot(inner[ri + 1])
            mat = slot_matrix(df, target, order)
            val = np.ma.masked_invalid(np.log2(np.clip(mat, 1e-6, None)))
            im = ax.imshow(val, aspect="auto", cmap=CMAP, vmin=-LIM, vmax=LIM, interpolation="nearest")
            # Hatch exactly the slots that held no clip, cell by cell, so "absent"
            # never reads as "even" and no hatch spills onto a real cell.
            holes = [Rectangle((c - 0.5, r - 0.5), 1, 1) for r, c in zip(*np.where(np.isnan(mat)), strict=True)]
            ax.add_collection(PatchCollection(holes, facecolor=SURFACE, edgecolor=MISSING,
                                              hatch="////", linewidths=0))
            for x in (1.5, 4.5):                               # SA | SI | SX gaps
                ax.axvline(x, color=SURFACE, lw=2)
            ax.set_xticks([0.5, 3, 7], ["SA", "SI", "SX"], fontsize=8, color=TEXT2)
            ax.set_yticks([])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{label}\n{target}: n_eff {n_eff(df, target):.2f} of 8", fontsize=8.5, color=TEXT)
    cax = fig.add_axes([0.925, 0.30, 0.015, 0.40])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["1/4x", "1/2x", "even", "2x", "4x"], fontsize=8, color=TEXT2)
    cb.set_label("attention vs an even split", fontsize=9, color=TEXT)
    cb.outline.set_visible(False)
    fig.legend(handles=[Patch(facecolor=SURFACE, edgecolor=MISSING, hatch="////", label="not in bag"),
                        Patch(facecolor=FEMALE, label="female"),
                        Patch(facecolor=MALE, label="male")],
               loc="upper left", bbox_to_anchor=(0.912, 0.285), fontsize=8, frameon=False, labelcolor=TEXT2)
    fig.suptitle("MIL attention per speaker and clip -- TIMIT test, 168 speakers", fontsize=13, color=TEXT, y=0.985)
    fig.text(0.5, 0.945, "Each row is one speaker; each cell one clip in their 8-clip bag, grouped by sentence type "
             "(SA = the 2 sentences every speaker reads; SI, SX = per-speaker). Hatched = slot not in the bag.",
             ha="center", fontsize=8.5, color=TEXT2)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def grid_figure(runs: dict[str, pd.DataFrame], out: Path) -> None:
    targets = ["age", "height"]
    first = next(iter(runs.values()))
    fig, axes = plt.subplots(2, len(runs), figsize=(2.15 * len(runs) + 1.4, 6.6), facecolor=SURFACE,
                             squeeze=False)
    for ti, target in enumerate(targets):
        truth = first[first.target == target].drop_duplicates("speaker_id")
        bins = pd.qcut(truth["ground_truth"], 5, labels=False, duplicates="drop")
        qmap = dict(zip(truth["speaker_id"], bins, strict=True))
        edges = truth.groupby(bins)["ground_truth"].agg(["min", "max"])
        ylabels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in edges.to_numpy()]
        for ri, (label, df) in enumerate(runs.items()):
            sub = df[df.target == target].copy()
            sub["q"] = sub["speaker_id"].map(qmap)
            tab = sub.pivot_table(index="q", columns="stype", values="rel", aggfunc="mean").reindex(columns=list(SLOTS))
            ax = axes[ti, ri]
            im = ax.imshow(np.log2(tab.to_numpy()), cmap=CMAP, vmin=-LIM, vmax=LIM, aspect="auto")
            for (r, c), v in np.ndenumerate(tab.to_numpy()):
                dark = abs(np.log2(v)) > 1.0
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if dark else TEXT)
            ax.set_xticks(range(3), list(SLOTS), fontsize=8, color=TEXT2)
            ax.set_yticks(range(len(ylabels)), ylabels if ri == 0 else [""] * len(ylabels), fontsize=8, color=TEXT2)
            if ri == 0:
                ax.set_ylabel(f"true {target} ({UNITS[target]}), quintiles", fontsize=9, color=TEXT)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{label}\n{target}", fontsize=8.5, color=TEXT)
    fig.subplots_adjust(left=0.08, right=0.90, top=0.84, bottom=0.07, wspace=0.12, hspace=0.38)
    cax = fig.add_axes([0.925, 0.25, 0.015, 0.50])
    cb = fig.colorbar(im, cax=cax, ticks=[-2, -1, 0, 1, 2])
    cb.ax.set_yticklabels(["1/4x", "1/2x", "even", "2x", "4x"], fontsize=8, color=TEXT2)
    cb.set_label("mean attention vs an even split", fontsize=9, color=TEXT)
    cb.outline.set_visible(False)
    fig.suptitle("Where attention goes: trait quintile x sentence type (1.00 = even split)",
                 fontsize=13, color=TEXT, y=0.975)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--output", required=True, type=Path, help="output path prefix")
    args = ap.parse_args()
    runs = {lab: load(Path(d), args.corpus) for lab, d in (r.split("=", 1) for r in args.run)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, draw in [(Path(f"{args.output}_speakers.png"), speakers_figure),
                       (Path(f"{args.output}_grid.png"), grid_figure)]:
        draw(runs, path)
        # Flatten to RGB: an alpha channel makes some PDF renderers paint the
        # page behind an embedded figure black.
        Image.open(path).convert("RGB").save(path)
    print(f"wrote {args.output}_speakers.png and {args.output}_grid.png")


if __name__ == "__main__":
    main()
