"""What is the MIL attention actually selecting on?

`SESSION_HANDOFF.md` §3.5 reports that attention does not collapse (entropy
0.81, max weight 0.35 -- 2.8x concentration) yet MIL still fails to beat mean
pooling on TIMIT. The working hypothesis is that selectivity is *counterproductive*
here: concentrating weight lowers the effective number of averaged utterances and
forfeits the sqrt(K) variance reduction, while TIMIT's homogeneous read speech
offers nothing worth selecting.

That hypothesis has never been tested directly -- entropy says *how much* the
attention concentrates, not *what it concentrates on*. TIMIT lets us ask the
second question, because its utterance ids encode sentence type:

    SA (2 per speaker)  the SAME two sentences for every speaker in the corpus
    SX (5 per speaker)  phonetically compact, shared across groups of speakers
    SI (3 per speaker)  phonetically diverse, UNIQUE to each speaker

If MIL has learned anything content-aware, it should systematically downweight
SA -- identical text across all 630 speakers carries the least speaker-specific
information beyond raw voice quality. If weights are flat across the three
types, the attention is selecting on nothing legible, which is direct support
for §3.5 and a far stronger statement than an entropy number.

Usage:
    python scripts/analyse_attention.py \\
        --attention runs/<run>/eval_utt/attention_timit.csv \\
        --predictions runs/<run>/eval_utt/predictions_timit.csv \\
        --output runs/<run>/eval_utt/attention_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attention", required=True, type=Path,
                   help="attention_<corpus>.csv written by evaluate.py")
    p.add_argument("--predictions", type=Path, default=None,
                   help="predictions_<corpus>.csv, for the n_eff summary")
    p.add_argument("--output", type=Path, default=None,
                   help="Path prefix for the report (.md) and plot (.png)")
    return p.parse_args()


def sentence_type(utt_id: str) -> str:
    """TIMIT utterance ids are SA1/SA2, SX###, SI### -- the prefix is the type."""
    u = str(utt_id).upper()
    for prefix in ("SA", "SX", "SI"):
        if u.startswith(prefix):
            return prefix
    return "other"


TYPE_NOTE = {
    "SA": "identical text for every speaker (2/speaker)",
    "SX": "phonetically compact, shared across groups (5/speaker)",
    "SI": "phonetically diverse, unique to the speaker (3/speaker)",
}


def by_sentence_type(attn: pd.DataFrame) -> pd.DataFrame:
    """Mean attention weight per sentence type, per target, vs the uniform rate.

    ``rel_to_uniform`` is the number that matters: 1.0 means the type receives
    exactly the weight it would under mean pooling. Above 1 = preferred.
    """
    attn = attn.copy()
    attn["type"] = attn["utt_id"].map(sentence_type)
    attn["uniform"] = 1.0 / attn["n_clips"]

    g = attn.groupby(["target", "type"], as_index=False).agg(
        mean_weight=("weight", "mean"),
        uniform=("uniform", "mean"),
        n=("weight", "size"),
        top1_rate=("rank", lambda r: float((r == 0).mean())),
    )
    g["rel_to_uniform"] = g["mean_weight"] / g["uniform"]
    return g.sort_values(["target", "rel_to_uniform"], ascending=[True, False])


def neff_summary(preds: pd.DataFrame) -> pd.DataFrame:
    """How many utterances the bag is *effectively* averaging (§7.7)."""
    cols = [c for c in ("n_eff", "n_clips", "attn_entropy", "attn_max") if c in preds]
    if not cols:
        return pd.DataFrame()
    return preds.groupby("target")[cols].mean().reset_index()


def make_plot(g: pd.DataFrame, path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")            # headless: compute nodes have no display
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    targets = sorted(g["target"].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), squeeze=False)
    for ax, t in zip(axes[0], targets, strict=False):
        sub = g[g["target"] == t].sort_values("type")
        ax.bar(sub["type"], sub["rel_to_uniform"])
        ax.axhline(1.0, ls="--", lw=1, color="k")   # mean pooling
        ax.set_title(f"{t}: attention vs uniform")
        ax.set_ylabel("mean weight / uniform weight")
        ax.set_xlabel("TIMIT sentence type")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main() -> None:
    args = parse_args()
    attn = pd.read_csv(args.attention)
    if attn.empty:
        raise SystemExit("attention csv is empty — was the model an attention-pooling one?")

    g = by_sentence_type(attn)

    lines = ["# MIL attention by TIMIT sentence type", ""]
    lines.append("`rel_to_uniform` = 1.0 means the type gets exactly the weight mean")
    lines.append("pooling would give it. Above 1 = preferred by the attention.")
    lines.append("")
    for t in sorted(g["target"].unique()):
        lines += [f"## {t}", "",
                  "| type | mean weight | uniform | rel_to_uniform | top-1 rate | n | note |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in g[g["target"] == t].iterrows():
            lines.append(
                f"| {r['type']} | {r['mean_weight']:.4f} | {r['uniform']:.4f} | "
                f"**{r['rel_to_uniform']:.3f}** | {r['top1_rate']:.3f} | {int(r['n'])} | "
                f"{TYPE_NOTE.get(r['type'], '')} |"
            )
        lines.append("")

    if args.predictions and args.predictions.is_file():
        ne = neff_summary(pd.read_csv(args.predictions))
        if not ne.empty:
            lines += ["## Effective bag size (§7.7)", ""]
            lines.append("`n_eff` is the effective number of clips averaged "
                         "(uniform over K gives K; all weight on one gives 1).")
            lines.append("")
            lines.append("| target | " + " | ".join(c for c in ne.columns[1:]) + " |")
            lines.append("|---" * len(ne.columns) + "|")
            for _, r in ne.iterrows():
                lines.append("| " + r["target"] + " | "
                             + " | ".join(f"{r[c]:.3f}" for c in ne.columns[1:]) + " |")
            lines.append("")

    report = "\n".join(lines)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        md = args.output.with_suffix(".md")
        md.write_text(report)
        print(f"\nWrote {md}")
        png = make_plot(g, args.output.with_suffix(".png"))
        print(f"Wrote {png}" if png else "\n(matplotlib not installed — skipped the plot)")


if __name__ == "__main__":
    main()
