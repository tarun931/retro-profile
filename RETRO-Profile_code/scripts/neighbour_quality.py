"""Table 3 (RQ4): neighbour quality of RETRO's retrieval space, with vs without OrdCon (2026-10-16).

For every test speaker: its top-k training speakers in the SAME lookup table the second stage built
(``runs/<run>/stage2<suffix>/datastore``), queried with the same vector (the frozen model's shared embedding from all
clips). Per trait it reports:

  gap@k              mean |neighbour label - speaker label| over the k nearest neighbours that have the label
                     (k = 1, 5, 20 by default),
                     averaged over test speakers -- low = retrieval lands on speakers who share the trait;
  distance_spearman  Spearman rho between cosine distance and |label difference| over all test x training pairs --
                     positive = nearer speakers have closer labels (how well the space is ordered by the trait);
  knn_only_mae       MAE of the second stage's kNN-alone test predictions (its validation-chosen k).

Before measuring, it recomputes those kNN-alone test predictions at the second stage's k and stops unless they match
``predictions_test_knn.csv``, so Table 3 describes exactly the neighbours behind Table 2. Test labels choose nothing.

Writes ``runs/<run>/stage2<suffix>/neighbour_quality.json``.

    python scripts/neighbour_quality.py --run runs/retro_ordcon_fixed_l100_s5_b10_timit_s42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retro_profile.retrieval.datastore import RetrievalDatastore

_SCRIPTS = Path(__file__).resolve().parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def neighbour_labels(ds: RetrievalDatastore, query: torch.Tensor, target: str, k: int) -> np.ndarray:
    """(N, k) labels of each query's k nearest datastore entries THAT HAVE the label, nearest first.

    Entries without the label are skipped and the search goes further out, so each row holds k labelled neighbours
    (one TIMIT training speaker, FKFB0, has no age; as a speaker's single nearest neighbour it made gap@1 NaN in the
    first run, 16 Oct 2026). kNN itself differs slightly: it drops such an entry from its k without looking further,
    and falls back to the training-mean label when none of its k is labelled (at k 1: one test speaker on seed 3 of
    Table 3). NaN only where fewer than k labelled entries exist.
    """
    labels = ds._labels_df[target].to_numpy(dtype=float)
    n_missing = int(np.isnan(labels).sum())
    _, idx = ds.search(query, min(k + n_missing, len(ds)))
    idx = idx.cpu().numpy()
    got = labels[np.clip(idx, 0, None)]
    got[idx < 0] = np.nan
    out = np.full((len(got), k), np.nan)
    for i, row in enumerate(got):
        row = row[np.isfinite(row)][:k]
        out[i, :len(row)] = row
    return out


def label_gaps(nbr: np.ndarray, y: np.ndarray, ks) -> dict[int, float]:
    """Per k: mean over speakers (with a label) of the mean |neighbour label - own label| over the top k."""
    ok = np.isfinite(y)
    return {int(k): float(np.nanmean(np.abs(nbr[ok, :k] - y[ok, None]), axis=1).mean()) for k in ks}


def distance_spearman(ds: RetrievalDatastore, query: torch.Tensor, target: str, y: np.ndarray) -> float:
    """Spearman rho(cosine distance, |label difference|) over all query x datastore pairs with both labels."""
    emb = ds._index.reconstruct_n(0, ds._index.ntotal)
    q = query.cpu().numpy().astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    dist = 1.0 - q @ e.T                                                     # (N, M)
    lab = ds._labels_df[target].to_numpy(dtype=float)
    diff = np.abs(y[:, None] - lab[None, :])
    ok = np.isfinite(diff)
    return float(pd.Series(dist[ok]).corr(pd.Series(diff[ok]), method="spearman"))


def check_reproduces(ds: RetrievalDatastore, split: dict, target: str, k: int, label_range, reference: Path,
                     tol: float = 0.05) -> float:
    """Max |our kNN-alone prediction - the second stage's|, at its k; stop if above ``tol``."""
    ours = ds.knn_anchor(split["query"], target=target, k=min(k, len(ds)),
                         exclude_speaker_ids=split["speaker_ids"]).clamp(*label_range).cpu().numpy()
    ref = pd.read_csv(reference, dtype={"speaker_id": str})
    ref = ref[ref.target == target].set_index("speaker_id")["prediction"]
    diff = float(np.abs(ours - ref.loc[split["speaker_ids"]].to_numpy()).max())
    if not diff <= tol:
        raise SystemExit(f"{target}: kNN-alone predictions differ from {reference} by up to {diff:.4g} -- these would "
                         "not be the neighbours behind Table 2. Stop.")
    return diff


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--bag-size", type=int, default=10)
    ap.add_argument("--suffix", default="_b10", help="second-stage folder is stage2<suffix>")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)

    fr = _script("fit_residual")
    stage2 = args.run / f"stage2{args.suffix}"
    summary2 = json.loads((stage2 / "summary.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = fr.load_frozen_model(args.run)
    if ckpt.name != summary2["checkpoint"]:
        raise SystemExit(f"checkpoint {ckpt.name} is not the second stage's {summary2['checkpoint']}. Stop.")
    model.to(device)
    data_cfg = fr.corpus_data_cfg(args.corpus, args.bag_size, args.num_workers, args.data_root)
    _, loaders = fr.bag_loaders(args.corpus, data_cfg)
    test = fr.embed(model, loaders["test"], device)
    ds = RetrievalDatastore.load(stage2 / "datastore")

    report: dict = {"run": args.run.name, "checkpoint": ckpt.name, "ks": args.ks, "n_test": len(test["speaker_ids"]),
                    "datastore_size": len(ds), "targets": {}}
    for t, head in model.heads.items():
        rng = (float(head.label_min), float(head.label_max))
        k_used = int(summary2["targets"][t]["k"])
        repro = check_reproduces(ds, test, t, k_used, rng, stage2 / "predictions_test_knn.csv")
        y = test["truth"][t].numpy().astype(float)
        gaps = label_gaps(neighbour_labels(ds, test["query"], t, max(args.ks)), y, args.ks)
        report["targets"][t] = {
            "gap_at_k": gaps, "distance_spearman": distance_spearman(ds, test["query"], t, y),
            "knn_only_mae": summary2["results"]["test/knn"]["overall"][f"{t}/mae"], "knn_k": k_used,
            "reproduction_max_diff": repro}
        print(f"{args.run.name}: {t:6s} gap@k " + " / ".join(f"{gaps[k]:.2f}" for k in args.ks)
              + f" | distance rho {report['targets'][t]['distance_spearman']:+.3f}"
              + f" | kNN-only MAE {report['targets'][t]['knn_only_mae']:.2f} (k {k_used}; reproduces, max diff {repro:.1g})")
    (stage2 / "neighbour_quality.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
