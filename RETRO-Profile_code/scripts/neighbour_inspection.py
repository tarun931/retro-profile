"""Neighbour inspection (RQ3 interpretability): what do RETRO's age neighbours share? (2026-10-22)

The proposal promised a qualitative look at the retrieved neighbours. For one run (the paper uses seed 42 of the final
RETRO) this script asks, for each TIMIT test speaker's k nearest training speakers in the age lookup:

  * gender      share of neighbours with the test speaker's gender, against chance (all training speakers);
  * age gap     mean |neighbour age - speaker age|, against a random training speaker of the SAME gender;
  * accent      share of neighbours from the test speaker's TIMIT dialect region, against same-gender chance;
  * weights     effective number of neighbours 1 / sum(w^2) of the softmax weights (k = all equal);
  * hubs        how often one training speaker is the nearest neighbour of several test speakers;
  * confidence  Spearman between the nearest-neighbour similarity and the kNN age error;
  * coverage    training speakers over 50 / 60, and the error of the test speakers over 50.

Every difference gets a 95 % bootstrap interval over test speakers (10,000 resamples, seed 0), as in the paper.

Inputs are the files an earlier forced-retrieval evaluation of the run's chosen checkpoint saved locally:
``<run>/datastore_test_knn_b10`` (FAISS index; its row order is the first appearance of each speaker in the training
manifest, as ``retrieval/builder.py`` builds it) and ``<run>/eval_spk_test_knn_b10/embeddings_timit.npz``. Before measuring,
the script recomputes each test speaker's five nearest neighbours and STOPS unless their IDs and similarities match the
``neighbour_*`` columns of that evaluation's ``predictions_timit.csv``. These embeddings come from a separate pass of the
same checkpoint, not from the second stage's own pass: because the top neighbours are near-ties, their kNN age MAE differs
slightly from Table 1's (seed 42: 3.83 against 3.79). Test labels choose nothing; k is the second stage's validation choice.

Writes ``<run>/neighbour_inspection.json``.

    python scripts/neighbour_inspection.py --run runs/retro_ordcon_fixed_l100_s5_b10_timit_s42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAU = 0.1                      # the datastore's softmax temperature (configs: retrieval temperature)
N_BOOT, SEED = 10_000, 0


def _boot(x: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean of ``x`` with its 95 % percentile bootstrap interval over test speakers."""
    bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(x.mean()), "ci": [float(lo), float(hi)], "clear": bool(lo > 0 or hi < 0)}


def _weights(s: np.ndarray) -> np.ndarray:
    """Softmax of similarity / TAU over each row (the datastore's weighting)."""
    w = np.exp((s - s[:, :1]) / TAU)
    return w / w.sum(1, keepdims=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--train-manifest", type=Path, default=Path("data/timit/manifests/train.csv"))
    ap.add_argument("--test-manifest", type=Path, default=Path("data/timit/manifests/test.csv"))
    ap.add_argument("--k", type=int, default=None, help="age k; default = the second stage's validation choice")
    args = ap.parse_args(argv)

    k = args.k or json.loads((args.run / "stage2_b10/summary.json").read_text())["targets"]["age"]["k"]
    index = faiss.read_index(str(args.run / "datastore_test_knn_b10/index.faiss"))
    db = index.reconstruct_n(0, index.ntotal)
    tr = pd.read_csv(args.train_manifest)
    order = list(dict.fromkeys(tr.speaker_id))
    if len(order) != index.ntotal:
        raise SystemExit(f"manifest has {len(order)} speakers, datastore {index.ntotal}")
    train = tr.drop_duplicates("speaker_id").set_index("speaker_id").loc[order]
    emb = np.load(args.run / "eval_spk_test_knn_b10/embeddings_timit.npz", allow_pickle=True)
    qid = [str(s) for s in emb["speaker_id"]]
    test = pd.read_csv(args.test_manifest).drop_duplicates("speaker_id").set_index("speaker_id").loc[qid]
    sims = emb["embedding"].astype(np.float32) @ db.T

    # Reproduction check against the stored evaluation: same five nearest neighbours (its list is the raw top five,
    # before speakers without a label are skipped), same similarities.
    stored = pd.read_csv(args.run / "eval_spk_test_knn_b10/predictions_timit.csv")
    stored = stored[stored.target == "age"].set_index("speaker_id").loc[qid]
    top5 = np.argsort(-sims, axis=1)[:, :5]
    ids_ok = all(order[top5[i, j]] == stored[f"neighbour_{j}_speaker"].iloc[i] for i in range(len(qid)) for j in range(5))
    sim_diff = float(np.abs(np.take_along_axis(sims, top5, 1)
                            - stored[[f"neighbour_{j}_similarity" for j in range(5)]].to_numpy()).max())
    if not ids_ok or sim_diff > 1e-4:
        raise SystemExit(f"reproduction failed: neighbour IDs match = {ids_ok}, max similarity difference = {sim_diff}")
    age = train.age.to_numpy()
    has_age = ~np.isnan(age)
    masked = np.where(has_age[None, :], sims, -np.inf)          # a neighbour without an age is skipped, as in the head

    rng = np.random.default_rng(SEED)
    g, dr = train.gender.to_numpy(), train.dialect_region.to_numpy()
    out: dict = {"run": str(args.run), "k_age": int(k), "n_test": len(qid), "n_train": len(order),
                 "reproduction": {"top5_ids_match": ids_ok, "max_similarity_diff": sim_diff}}
    for kk in sorted({k, 10}):
        top = np.argsort(-masked, axis=1)[:, :kk]
        same_g = np.array([(g[top[i]] == test.gender.iloc[i]).mean() for i in range(len(qid))])
        chance_g = np.array([(g[has_age] == test.gender.iloc[i]).mean() for i in range(len(qid))])
        same_dr = np.array([(dr[top[i]] == test.dialect_region.iloc[i]).mean() for i in range(len(qid))])
        chance_dr = np.array([(dr[has_age & (g == test.gender.iloc[i])] == test.dialect_region.iloc[i]).mean()
                              for i in range(len(qid))])
        gap = np.array([np.abs(age[top[i]] - test.age.iloc[i]).mean() for i in range(len(qid))])
        gap_g = np.array([np.abs(age[has_age & (g == test.gender.iloc[i])] - test.age.iloc[i]).mean()
                          for i in range(len(qid))])
        out[f"k{kk}"] = {
            "same_gender": float(same_g.mean()), "gender_chance": float(chance_g.mean()),
            "same_gender_minus_chance": _boot(same_g - chance_g, rng),
            "age_gap": float(gap.mean()), "age_gap_same_gender_random": float(gap_g.mean()),
            "age_gap_minus_random": _boot(gap - gap_g, rng),
            "same_dialect": float(same_dr.mean()), "dialect_chance_same_gender": float(chance_dr.mean()),
            "same_dialect_minus_chance": _boot(same_dr - chance_dr, rng),
        }

    top = np.argsort(-masked, axis=1)[:, :k]
    s_top = np.take_along_axis(sims, top, 1)
    w = _weights(s_top)
    est = (w * age[top]).sum(1)
    err = np.abs(est - test.age.to_numpy())
    s_h = np.sort(sims, 1)[:, ::-1][:, :75]
    rho = spearmanr(s_top[:, 0], err).statistic
    rho_bs = [spearmanr(s_top[ii, 0], err[ii]).statistic
              for ii in (rng.integers(0, len(qid), len(qid)) for _ in range(N_BOOT))]
    low = s_top[:, 0] <= np.percentile(s_top[:, 0], 10)
    old = test.age.to_numpy() > 50
    nearest = pd.Series(np.array(order)[top[:, 0]]).value_counts()
    out["knn_age_mae"] = float(err.mean())
    out["effective_neighbours"] = {"age": float(np.median(1 / (w ** 2).sum(1))),
                                   "height_k75": float(np.median(1 / (_weights(s_h) ** 2).sum(1)))}
    out["similarity_median"] = {"first": float(np.median(s_top[:, 0])), "kth": float(np.median(s_top[:, -1]))}
    out["hubs"] = {"distinct_nearest": int(nearest.size), "max_test_speakers_per_nearest": int(nearest.iloc[0])}
    out["confidence"] = {"spearman_nearest_sim_vs_abs_error": float(rho),
                         "ci": [float(np.percentile(rho_bs, 2.5)), float(np.percentile(rho_bs, 97.5))],
                         "mae_least_similar_10pct": float(err[low].mean()), "mae_rest": float(err[~low].mean()),
                         "mean_age_least_similar_10pct": float(test.age.to_numpy()[low].mean()),
                         "mean_age_rest": float(test.age.to_numpy()[~low].mean())}
    out["coverage"] = {"train_over_50": int((age[has_age] > 50).sum()), "train_over_60": int((age[has_age] > 60).sum()),
                       "test_over_50": int(old.sum()), "mae_test_over_50": float(err[old].mean()),
                       "signed_error_test_over_50": float((est - test.age.to_numpy())[old].mean()),
                       "mae_test_under_50": float(err[~old].mean())}
    examples = {}
    for name, i in zip(("best", "median", "worst"), (np.argsort(err)[0], np.argsort(err)[len(err) // 2], np.argsort(err)[-1]), strict=True):
        examples[name] = {"speaker": qid[i], "gender": test.gender.iloc[i], "dialect_region": test.dialect_region.iloc[i],
                          "age": float(test.age.iloc[i]), "knn_age": float(est[i]), "abs_error": float(err[i]),
                          "neighbours": [{"speaker": order[n], "gender": g[n], "dialect_region": dr[n], "age": float(age[n]),
                                          "similarity": float(sims[i, n]), "weight": float(w[i, j])}
                                         for j, n in enumerate(top[i])]}
    out["examples"] = examples
    path = args.run / "neighbour_inspection.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({key: out[key] for key in out if key != "examples"}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
