"""Which WavLM/wav2vec2 layers does each trait actually use?

`WavLMEncoder` and `Wav2Vec2Encoder` run with `layer_aggregation: weighted`: they
learn a softmax over all 13 hidden states (1 convolutional + 12 transformer) and
combine them into one representation. Those weights are trained, they sit in
every checkpoint, and nothing has ever looked at them.

Why it is worth looking. Self-supervised speech encoders are known to stratify:
lower layers carry acoustic and speaker-identity information, upper layers carry
phonetic and lexical content. Height is a vocal-tract property, so it should
prefer lower layers; age is bound up with speaking style and articulation, so it
may sit higher. If the learned weights show that split, it is an interpretable,
mechanistic result of the same kind as the SA attention finding (§3.8c) -- and
unlike that one it costs no GPU time and no new runs.

CAVEAT, and it is a real one: the encoder is SHARED across targets. There is one
set of layer weights per model, not one per target, so a two-target run cannot
tell you what height alone would have chosen. The comparison that answers the
question is between the SINGLE-TASK runs -- `retro_age_only_budget` against
`retro_height_only_budget` -- where each model was free to weight layers for its
own trait. Two-target runs are shown for reference only.

Usage:
    python scripts/analyse_layer_weights.py --ckpt runs/<run>/checkpoints/<best>.ckpt
    python scripts/analyse_layer_weights.py --runs runs/retro_age_only_budget_timit_s42 \\
                                                   runs/retro_height_only_budget_timit_s42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _best_ckpt(run: Path) -> Path | None:
    """Lowest val=... in the filename -- the same rule the slurm scripts use.

    They used to take the newest by mtime, which silently evaluated a
    sub-optimal checkpoint in 7 of 10 runs once save_top_k rose above 1
    (SESSION_HANDOFF.md §6.13). Keep this consistent with them.
    """
    ck = run / "checkpoints"
    if not ck.is_dir():
        return None
    best, best_v = None, math.inf
    for f in ck.glob("*.ckpt"):
        if f.name == "last.ckpt":
            continue
        try:
            v = float(f.stem.rsplit("val=", 1)[1])
        except (IndexError, ValueError):
            continue
        if v < best_v:
            best, best_v = f, v
    return best


def layer_weights(ckpt: Path) -> dict[str, list[float]]:
    """Return {parameter_name: softmax-normalised weights} for every layer_weights tensor."""
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    out: dict[str, list[float]] = {}
    for k, v in sd.items():
        if k.endswith("layer_weights") and isinstance(v, torch.Tensor) and v.dim() == 1:
            # The encoder applies softmax at forward time; do the same so the
            # numbers are the mixture actually used, not raw logits.
            out[k] = torch.softmax(v.float(), dim=0).tolist()
    return out


def summarise(w: list[float]) -> dict[str, float]:
    n = len(w)
    idx = list(range(n))
    centre = sum(i * p for i, p in zip(idx, w, strict=True))          # weighted mean layer
    ent = -sum(p * math.log(max(p, 1e-12)) for p in w)
    return {
        "n_layers": float(n),
        "centre_of_mass": centre,          # 0 = conv output, n-1 = top layer
        "top_layer": float(max(idx, key=lambda i: w[i])),
        "max_weight": max(w),
        "entropy_ratio": ent / math.log(n) if n > 1 else float("nan"),
        "lower_half_mass": sum(w[: n // 2]),
    }


def _bar(p: float, width: int = 40) -> str:
    return "#" * int(round(p * width))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, help="a single checkpoint")
    ap.add_argument("--runs", nargs="*", type=Path, default=[],
                    help="run dirs; uses each one's best checkpoint")
    ap.add_argument("--output", type=Path, help="write JSON here")
    args = ap.parse_args()

    targets: list[tuple[str, Path]] = []
    if args.ckpt:
        targets.append((args.ckpt.parent.parent.name or args.ckpt.stem, args.ckpt))
    for run in args.runs:
        c = _best_ckpt(run)
        if c is None:
            print(f"{run}: no checkpoint found (were they pruned?)")
            continue
        targets.append((run.name, c))

    if not targets:
        raise SystemExit("give --ckpt <file> or --runs <dir> ...")

    report: dict[str, dict] = {}
    for name, ckpt in targets:
        weights = layer_weights(ckpt)
        if not weights:
            print(f"\n{name}: no `layer_weights` tensor -- this encoder is not using "
                  f"layer_aggregation: weighted")
            continue
        print(f"\n=== {name}  ({ckpt.name}) ===")
        for param, w in weights.items():
            st = summarise(w)
            print(f"  {param}")
            for i, p in enumerate(w):
                tag = "conv" if i == 0 else f"L{i:02d}"
                print(f"    {tag}  {p:.4f}  {_bar(p)}")
            top = int(st["top_layer"])
            peak = "conv" if top == 0 else f"L{top:02d}"
            print(f"    centre of mass {st['centre_of_mass']:.2f} "
                  f"of {int(st['n_layers']) - 1}"
                  f"   peak {peak}"
                  f"   lower-half mass {st['lower_half_mass']:.2f}"
                  f"   entropy {st['entropy_ratio']:.3f}")
            report.setdefault(name, {})[param] = {"weights": w, **st}

    if len(report) > 1:
        print("\n=== comparison (centre of mass; lower = more acoustic, higher = more phonetic) ===")
        for name, params in report.items():
            for st in params.values():
                print(f"  {name:38s} {st['centre_of_mass']:5.2f}   lower-half {st['lower_half_mass']:.2f}")
        print("\nOnly the SINGLE-TASK runs are interpretable per trait: the encoder is")
        print("shared, so a two-target model has one set of weights serving both.")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
