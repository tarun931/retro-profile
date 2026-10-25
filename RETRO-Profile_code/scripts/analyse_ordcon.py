"""Does OrdCon do what it claims — and does it matter?

OrdCon's stated job (losses/ordcon.py) is not to lower MAE directly. It is to
*organise the embedding space along each trait axis*, because "kNN retrieval only
works if the embedding space is organized along the target axis". So judging it
by MAE alone tests the wrong thing: it could be doing its job perfectly while the
head or the selection metric hides the benefit.

This script asks two separate questions.

**A. Does the space actually get ordered?** (needs `--ckpt`)
OrdCon learns a unit direction `u` per target. Project each speaker embedding
onto it and compare with the label:

    axis_pearson / axis_spearman   correlation of <z, u> with y. If OrdCon works
                                   this should be high — it is the loss's own
                                   objective, measured directly.
    ordering_accuracy              fraction of speaker pairs whose projection
                                   order matches their label order. 0.5 = chance.
    distance_spearman              rho(||z_i - z_j||, |y_i - y_j|) — the
                                   magnitude term. Ordered *and* metric.

**B. Does that ordering help retrieval?** (needs only a predictions CSV)
This is the mechanism OrdCon exists to serve, and it is measurable without any
model:

    anchor_mae                     MAE of the kNN anchor ALONE, before the
                                   residual. A direct read on retrieval-space
                                   quality.
    neighbour_label_mae            mean |label of retrieved neighbour - truth|.
                                   Low means retrieval lands on speakers who
                                   actually share the trait.
    residual_share                 how much work the learned residual does on
                                   top of the anchor.

Run it on an OrdCon run and an `ordcon_weight: 0` run to get the comparison that
matters. Section B alone already works on every existing run.

Usage:
    python scripts/analyse_ordcon.py --run runs/<r>
    python scripts/analyse_ordcon.py --run runs/<r> --ckpt <ckpt> --config-dir <r>/.hydra
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path, help="runs/<run_name>")
    p.add_argument("--predictions", type=Path, default=None,
                   help="predictions CSV (default: <run>/eval_spk/predictions_timit.csv)")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="checkpoint — enables the embedding-space analysis (section A)")
    p.add_argument("--config-dir", type=Path, default=None, help="<run>/.hydra")
    p.add_argument("--split", default="val", choices=("val", "test"),
                   help="split for the embedding analysis; val by default so the "
                        "test set stays untouched")
    p.add_argument("--output", type=Path, default=None, help="write a JSON report here")
    return p.parse_args()


# --- section B: retrieval quality, from a predictions CSV --------------------

def retrieval_quality(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """How good is the kNN anchor, before the residual touches it?"""
    out: dict[str, dict[str, float]] = {}
    nbr_label_cols = [c for c in df.columns if c.startswith("neighbour_") and c.endswith("_label")]

    for target, g in df.groupby("target"):
        g = g.dropna(subset=["ground_truth"])
        if g.empty:
            continue
        row: dict[str, float] = {"n": int(len(g))}

        if "prediction" in g:
            row["pred_mae"] = float((g["prediction"] - g["ground_truth"]).abs().mean())

        if "anchor" in g and g["anchor"].notna().any():
            a = g.dropna(subset=["anchor"])
            err = a["anchor"] - a["ground_truth"]
            row["anchor_mae"] = float(err.abs().mean())
            row["anchor_rmse"] = float((err**2).mean() ** 0.5)
            row["anchor_pearson"] = float(a["anchor"].corr(a["ground_truth"]))
            if "delta" in a:
                d, an = a["delta"].abs(), a["anchor"].abs()
                row["residual_share"] = float((d / (d + an).clip(lower=1e-9)).mean())
                row["mean_abs_delta"] = float(d.mean())
            # What the residual actually buys on top of the anchor.
            if "pred_mae" in row:
                row["residual_gain_mae"] = row["anchor_mae"] - row["pred_mae"]

        if nbr_label_cols:
            gaps = [(g[c] - g["ground_truth"]).abs() for c in nbr_label_cols]
            stacked = pd.concat(gaps, axis=1)
            row["neighbour_label_mae"] = float(stacked.mean(axis=1).mean())
            row["neighbour_label_best"] = float(stacked.min(axis=1).mean())
            row["k_neighbours"] = len(nbr_label_cols)

        out[str(target)] = row
    return out


# --- section A: embedding-space ordering ------------------------------------

def _ordcon_directions(ckpt: dict) -> dict[str, np.ndarray]:
    """Pull the learned per-target unit directions out of a checkpoint."""
    state = ckpt.get("state_dict", ckpt)
    dirs = {}
    for k, v in state.items():
        if "ordcon" in k and "directions" in k:
            dirs[k.rsplit(".", 1)[-1]] = v.detach().float().cpu().numpy()
    return dirs


def ordering_metrics(emb: np.ndarray, y: np.ndarray, direction: np.ndarray,
                     max_pairs: int = 200_000, seed: int = 0) -> dict[str, float]:
    """Is the space ordered along `direction`, and is distance label-metric?"""
    from scipy.stats import spearmanr

    ok = ~np.isnan(y)
    emb, y = emb[ok], y[ok]
    if len(y) < 3:
        return {}

    u = direction / (np.linalg.norm(direction) + 1e-12)
    proj = emb @ u

    res: dict[str, float] = {
        "n": int(len(y)),
        "axis_pearson": float(np.corrcoef(proj, y)[0, 1]),
        "axis_spearman": float(spearmanr(proj, y).statistic),
    }

    i, j = np.triu_indices(len(y), k=1)
    if len(i) > max_pairs:                      # subsample for large splits
        sel = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[sel], j[sel]

    dy = y[i] - y[j]
    keep = dy != 0
    i, j, dy = i[keep], j[keep], dy[keep]
    dproj = proj[i] - proj[j]
    # Sign agreement is literally OrdCon's direction objective. 0.5 = chance.
    res["ordering_accuracy"] = float((np.sign(dproj) == np.sign(dy)).mean())

    dz = np.linalg.norm(emb[i] - emb[j], axis=1)
    res["distance_spearman"] = float(spearmanr(dz, np.abs(dy)).statistic)
    res["n_pairs"] = int(len(i))
    return res


def embedding_analysis(args) -> dict:
    import torch

    from retro_profile.utils import instantiate, load_config

    cfg = load_config(args.config_dir / "config.yaml")
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    dirs = _ordcon_directions(ckpt)
    if not dirs:
        return {"error": "no OrdCon directions in the checkpoint — was ordcon_weight 0?"}

    model = instantiate(cfg.model)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(
        {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")},
        strict=False,
    )
    model.eval()
    if not hasattr(model, "embed_speaker"):
        return {"error": f"{type(model).__name__} has no embed_speaker()"}

    # Built from its _target_, as train.py does -- a lookup by data.name failed for multi-corpus
    # runs (heightceleb_nisp is not a registered name).
    dm = instantiate(cfg.data)
    dm.setup("fit" if args.split == "val" else "test")
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    embs, labels = [], {t: [] for t in dirs}
    with torch.no_grad():
        for batch in loader:
            embs.append(model.embed_speaker(batch).float().cpu().numpy())
            for t in dirs:
                labels[t].append(batch.labels[t].float().cpu().numpy()
                                 if t in batch.labels else np.full(batch.batch_size, np.nan))
    emb = np.concatenate(embs)

    out = {}
    for t, u in dirs.items():
        if u.shape[-1] != emb.shape[-1]:
            out[t] = {"error": f"direction dim {u.shape[-1]} != embedding dim {emb.shape[-1]}"}
            continue
        out[t] = ordering_metrics(emb, np.concatenate(labels[t]), u)
    return out


def main() -> None:
    args = parse_args()
    preds = args.predictions or (args.run / "eval_spk" / "predictions_timit.csv")

    report: dict = {"run": args.run.name}

    print("=== B. Retrieval-space quality (what OrdCon exists to improve) ===\n")
    if preds.is_file():
        report["retrieval"] = retrieval_quality(pd.read_csv(preds))
        for t, m in report["retrieval"].items():
            print(f"  {t}:")
            for k, v in m.items():
                print(f"      {k:22s} {v:.4f}" if isinstance(v, float) else f"      {k:22s} {v}")
        print("\n  anchor_mae is the kNN interpolation alone. If it is close to")
        print("  pred_mae, the residual is contributing little and the result is")
        print("  essentially retrieval over the OrdCon-shaped space.")
    else:
        print(f"  (no predictions CSV at {preds} — run evaluate.py with --unit speaker)")

    if args.ckpt and args.config_dir:
        print("\n=== A. Is the embedding space actually ordered? ===\n")
        report["ordering"] = embedding_analysis(args)
        for t, m in report["ordering"].items():
            print(f"  {t}: " + ("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in m.items())))
        print("\n  ordering_accuracy 0.5 = chance. axis_spearman near 0 means the")
        print("  learned direction carries no label information — OrdCon is not")
        print("  doing its job, regardless of what MAE says.")
    else:
        print("\n(pass --ckpt and --config-dir for the embedding-space analysis)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
