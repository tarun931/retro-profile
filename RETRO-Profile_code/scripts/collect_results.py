"""Collect eval summaries from several runs into one markdown table.

Reads ``<run>/eval*/summary.json`` for each run directory given and prints a
table of age/height MAE, RMSE and Pearson r. Runs whose names differ only by a
seed suffix (``..._s1``, ``..._s2``) are grouped and reported as mean +/- std.

Usage:
    python scripts/collect_results.py runs/ablation_*            # auto-discover
    python scripts/collect_results.py --labels "WavLM+MLP=runs/ablation_1_*" ...
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METRICS = ("age/mae", "age/rmse", "age/pearson", "height/mae", "height/rmse", "height/pearson")
SEED_SUFFIX = re.compile(r"[_-]s(?:eed)?\d+$")


def find_summaries(run_dir: Path) -> dict[str, Path]:
    """Return {variant_name: summary.json} for a run (eval, eval_nokNN, ...)."""
    out = {}
    for sub in sorted(run_dir.glob("eval*")):
        summary = sub / "summary.json"
        if summary.is_file():
            out[sub.name] = summary
    return out


def read_metrics(summary: Path, corpus: str = "timit") -> dict[str, float] | None:
    """Metrics for the unit this eval actually ran (per its own `<corpus>` block)."""
    try:
        data = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    block = data.get(corpus) or next(iter(data.values()), None)
    if not isinstance(block, dict):
        return None
    return {m: float(block[m]) for m in METRICS if m in block}


def read_all_units(summary: Path, corpus: str = "timit") -> dict[str, dict[str, float]]:
    """Every protocol this eval produced, keyed by unit.

    `evaluate.py --unit utterance` derives per-SPEAKER metrics from the same
    predictions and stores both under `<corpus>__breakdown`. Those derived
    speaker numbers are the ones the RQ1 table is built on -- Gupta predicts each
    clip and averages, which is the aggregation RETRO's `eval_spk` (MIL pooling)
    must be compared against. Reading only the top-level block would report
    Gupta's per-utterance figure instead and silently compare two protocols.
    """
    try:
        data = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, dict[str, float]] = {}
    breakdown = data.get(f"{corpus}__breakdown")
    if isinstance(breakdown, dict):
        for unit, sub in breakdown.items():
            block = sub.get("overall") if isinstance(sub, dict) else None
            if isinstance(block, dict):
                got = {m: float(block[m]) for m in METRICS if m in block}
                if got:
                    out[unit] = got
    if not out:
        got = read_metrics(summary, corpus)
        if got:
            out[str(data.get(f"{corpus}__unit", "overall"))] = got
    return out


def group_name(run_name: str) -> str:
    """Strip a trailing seed suffix so seeds of one config group together."""
    return SEED_SUFFIX.sub("", run_name)


def fmt(values: list[float]) -> str:
    if not values:
        return "--"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories")
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--variant", default=None,
                    help="only report this eval subdir (e.g. 'eval' or 'eval_nokNN')")
    args = ap.parse_args()

    # {(group, variant): [metrics per seed]}
    rows: dict[tuple[str, str], list[dict[str, float]]] = {}
    for run in args.runs:
        if not run.is_dir():
            continue
        for variant, summary in find_summaries(run).items():
            if args.variant and variant != args.variant:
                continue
            for unit, metrics in read_all_units(summary, args.corpus).items():
                # `eval_spk` already IS the speaker protocol; labelling its rows
                # again would duplicate them against eval_utt's derived speaker
                # block, which is a different aggregation (MIL vs averaging).
                label = variant if variant.endswith(unit[:3]) else f"{variant}:{unit}"
                rows.setdefault((group_name(run.name), label), []).append(metrics)

    if not rows:
        print("No summary.json found. Has evaluate.py been run for these runs?")
        return

    header = ("| Run | Variant | n | Age MAE | Age RMSE | Age r | "
              "Height MAE | Height RMSE | Height r |")
    print(header)
    print("|" + "---|" * 9)
    for (group, variant), seeds in sorted(rows.items()):
        cells = [fmt([s[m] for s in seeds if m in s]) for m in METRICS]
        print(f"| {group} | {variant} | {len(seeds)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
