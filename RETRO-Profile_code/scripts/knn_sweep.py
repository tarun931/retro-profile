"""Sweep the kNN k (and weighting) OFFLINE: choose on VALIDATION, report on test.

Inputs, from ``reevaluate.slurm`` with WITH_RETRIEVAL=1 FORCE_RETRIEVAL=1
SAVE_EMBEDDINGS=1, run once with EVAL_SPLIT=val and once on test:

    runs/<run>/datastore<suffix>/       the training-speaker datastore (any of the
                                         two jobs' copies; they are identical)
    runs/<run>/eval_spk<val suffix>/    embeddings_<corpus>.npz + predictions_<corpus>.csv
    runs/<run>/eval_spk<test suffix>/   the same for the test speakers

Every setting is scored with ``RetrievalDatastore.knn_anchor`` -- the function the
model's head calls, with the same leave-one-out argument and the same clamp to
the label range -- so a swept number is exactly what the model would predict
with that setting. The script first recomputes the job's own setting and stops
unless it reproduces the predictions the job wrote (kNN forced and residual
untrained, so prediction = clamped anchor).

Selection rule, fixed before looking at any result: per target, the k with the
lowest VALIDATION MAE at the current weighting (the datastore's own softmax
temperature). Test MAE is reported for that k and never used to choose. The other
weightings are a secondary grid, reported for information only. With 70
validation speakers the curve is noisy: the k values within 0.05 of the best are
listed so a flat curve is visible as flat.

    python scripts/knn_sweep.py --run runs/retro_ordcon_fixed_l100_s5_timit_s42 \\
        --datastore datastore_val_knn --val eval_spk_val_knn --test eval_spk_test_knn

``--calibrated`` (cross-corpus setting, 2026-10-14): every setting's kNN output is put through the same
2-number linear calibration as the model's own output (calibrate_predictions.py) before it is scored. Validation
is scored leave-one-out (each speaker calibrated by a fit on the others), and k is chosen on that; test gets the
fit on all validation speakers. Without the flag the behaviour is unchanged (raw MAE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from calibrate_predictions import loo_predictions
from paired_bootstrap import paired_bootstrap

from retro_profile.retrieval.datastore import RetrievalDatastore

K_GRID = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300)
TEMPERATURES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
NORMALISERS = {"age": 5.91, "height": 7.40}      # combined_mae, as in training
CLOSE = 0.05                                     # "within this of the best" on validation


def load_datastore(directory: Path) -> RetrievalDatastore:
    """``RetrievalDatastore.load``, or the same from ``labels.csv`` when the parquet is unreadable.

    the GPU cluster writes labels.parquet with Arrow 25; the Mac's pyarrow 19 cannot read it
    ("Repetition level histogram size mismatch", 2026-10-11). Export a labels.csv
    on the GPU cluster instead. Row order must match index.faiss -- check_reproduces() fails if
    it does not.
    """
    try:
        return RetrievalDatastore.load(directory)
    except (OSError, ValueError) as exc:          # pyarrow raises either, by failure
        csv = directory / "labels.csv"
        if not csv.exists():
            raise SystemExit(f"{directory}/labels.parquet is unreadable here ({exc}) and there is "
                             "no labels.csv beside it -- export one on the GPU cluster.") from exc
    meta = pd.read_json(directory / "meta.json", typ="series").to_dict()
    ds = RetrievalDatastore(embedding_dim=int(meta["embedding_dim"]), metric=str(meta["metric"]),
                            temperature=float(meta["temperature"]), weighting=str(meta["weighting"]))
    ds._index = faiss.read_index(str(directory / "index.faiss"))
    ds._labels_df = pd.read_csv(csv, dtype={"speaker_id": str})
    if len(ds._labels_df) != ds._index.ntotal:
        raise SystemExit(f"{csv}: {len(ds._labels_df)} rows but the index holds {ds._index.ntotal}")
    print(f"(labels read from {csv.name}: the parquet is unreadable with this pyarrow)")
    return ds


def load_split(eval_dir: Path, corpus: str, targets: list[str]) -> dict:
    """Queries (speaker_ids, embeddings) and, per target, ground truth + the job's prediction."""
    z = np.load(eval_dir / f"embeddings_{corpus}.npz")
    sids = [str(s) for s in z["speaker_id"]]
    if len(set(sids)) != len(sids):
        raise ValueError(f"{eval_dir}: expected one embedding per speaker (bag reading)")
    preds = pd.read_csv(eval_dir / f"predictions_{corpus}.csv")
    out = {"speaker_ids": sids, "emb": torch.from_numpy(z["embedding"].astype(np.float32))}
    for t in targets:
        p = preds[preds["target"] == t].set_index("speaker_id").loc[sids]
        out[t] = {"truth": p["ground_truth"].to_numpy(float), "pred": p["prediction"].to_numpy(float)}
    return out


def anchors(ds: RetrievalDatastore, split: dict, target: str, k: int,
            label_range: tuple[float, float]) -> np.ndarray:
    """What the head predicts with this datastore setting (residual untrained)."""
    a = ds.knn_anchor(split["emb"], target=target, k=min(k, len(ds)),
                      exclude_speaker_ids=split["speaker_ids"])
    return a.clamp(*label_range).numpy().astype(float)


def calibrated_errors(val_pred: np.ndarray, val_truth: np.ndarray, test_pred: np.ndarray,
                      test_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|error| after linear calibration: validation leave-one-out; test with the fit on all validation rows."""
    ok = np.isfinite(val_truth)
    v = pd.DataFrame({"prediction": val_pred[ok], "ground_truth": val_truth[ok]})
    ev = np.abs(loo_predictions(v).to_numpy() - v["ground_truth"].to_numpy())
    a, b = np.polyfit(v["prediction"].to_numpy(float), v["ground_truth"].to_numpy(float), 1)
    return ev, np.abs(a * test_pred + b - test_truth)


def errors_on_test(ds: RetrievalDatastore, val: dict, test: dict, target: str, k: int,
                label_range: tuple[float, float], calibrated: bool) -> pd.Series:
    """Per-speaker |error| on test at k: raw, or calibrated with the fit on validation's anchors at the same k."""
    at = anchors(ds, test, target, k, label_range)
    if calibrated:
        e = calibrated_errors(anchors(ds, val, target, k, label_range), val[target]["truth"], at,
                              test[target]["truth"])[1]
    else:
        e = np.abs(at - test[target]["truth"])
    return pd.Series(e, index=test["speaker_ids"])


def check_reproduces(ds: RetrievalDatastore, split: dict, targets: list[str], ks: dict,
                     ranges: dict, name: str, tol: float = 1e-3) -> None:
    """Stop unless the job's own setting reproduces the predictions it wrote."""
    for t in targets:
        diff = np.abs(anchors(ds, split, t, ks[t], ranges[t]) - split[t]["pred"]).max()
        if not diff <= tol:
            raise SystemExit(f"{name}/{t}: recomputed anchors differ from the job's predictions "
                             f"by up to {diff:.4g} -- the sweep would not measure the model. Stop.")


def sweep(ds: RetrievalDatastore, val: dict, test: dict, targets: list[str], ranges: dict,
          weightings: list[tuple[str, float | None]], k_grid=K_GRID, calibrate: bool = False) -> pd.DataFrame:
    """MAE on validation and test for every (weighting, temperature, k, target); with ``calibrate`` also the
    calibrated MAEs (``val_mae_cal`` leave-one-out, ``test_mae_cal``)."""
    rows = []
    base = (ds.weighting, ds.temperature)
    try:
        for weighting, temp in weightings:
            ds.weighting = weighting
            if temp is not None:
                ds.temperature = temp
            for k in sorted({min(k, len(ds)) for k in k_grid}):
                for t in targets:
                    av, at = anchors(ds, val, t, k, ranges[t]), anchors(ds, test, t, k, ranges[t])
                    ev, et = np.abs(av - val[t]["truth"]), np.abs(at - test[t]["truth"])
                    row = {"weighting": weighting, "temperature": temp, "k": k, "target": t,
                           "val_mae": float(np.nanmean(ev)), "test_mae": float(np.nanmean(et))}
                    if calibrate:
                        cv, ct = calibrated_errors(av, val[t]["truth"], at, test[t]["truth"])
                        row.update(val_mae_cal=float(np.nanmean(cv)), test_mae_cal=float(np.nanmean(ct)))
                    rows.append(row)
    finally:
        ds.weighting, ds.temperature = base
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame, weighting: str, temperature: float | None, target: str,
           calibrated: bool = False) -> dict:
    """The k with the lowest VALIDATION MAE (calibrated, if asked) at one weighting, plus the near-ties."""
    vcol, tcol = ("val_mae_cal", "test_mae_cal") if calibrated else ("val_mae", "test_mae")
    sel = table[(table["weighting"] == weighting) & (table["target"] == target)]
    if temperature is not None:
        sel = sel[np.isclose(sel["temperature"].astype(float), temperature)]
    best = sel.loc[sel[vcol].idxmin()]
    close = sel[sel[vcol] <= best[vcol] + CLOSE]["k"].astype(int).tolist()
    return {"k": int(best["k"]), "val_mae": float(best[vcol]),
            "test_mae": float(best[tcol]), "k_within_close": close, "calibrated": calibrated}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--datastore", required=True, help="datastore folder inside the run")
    ap.add_argument("--val", required=True, help="validation eval_spk folder inside the run")
    ap.add_argument("--test", required=True, help="test eval_spk folder inside the run")
    ap.add_argument("--corpus", default="timit")
    ap.add_argument("--output", type=Path, default=None, help="default: <run>/knn_sweep")
    ap.add_argument("--calibrated", action="store_true",
                    help="score and choose after the 2-number linear calibration (validation leave-one-out)")
    args = ap.parse_args()
    vcol, tcol = ("val_mae_cal", "test_mae_cal") if args.calibrated else ("val_mae", "test_mae")

    cfg = yaml.safe_load((args.run / ".hydra" / "config.yaml").read_text())
    heads = cfg["model"]["heads"]
    targets = list(heads)
    ranges = {t: (float(heads[t]["label_min"]), float(heads[t]["label_max"])) for t in targets}
    job_k = {t: int(heads[t]["knn_k"]) for t in targets}

    ds = load_datastore(args.run / args.datastore)
    val = load_split(args.run / args.val, args.corpus, targets)
    test = load_split(args.run / args.test, args.corpus, targets)
    for name, split in (("val", val), ("test", test)):
        check_reproduces(ds, split, targets, job_k, ranges, name)
    print(f"Reproduces the jobs' predictions at k={job_k}, {ds.weighting} T={ds.temperature} "
          f"(datastore {len(ds)} speakers; val n={len(val['speaker_ids'])}, test n={len(test['speaker_ids'])})")

    current = (ds.weighting, ds.temperature if ds.weighting != "uniform" else None)
    weightings = [("softmax_distance", t) for t in TEMPERATURES] + [("uniform", None)]
    if current not in weightings:
        weightings.insert(0, current)
    table = sweep(ds, val, test, targets, ranges, weightings, calibrate=args.calibrated)

    out = args.output or (args.run / ("knn_sweep_cal" if args.calibrated else "knn_sweep"))
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sweep.csv", index=False)

    summary: dict = {"current": {"weighting": current[0], "temperature": current[1], "k": job_k},
                     "calibrated": args.calibrated, "chosen": {}, "secondary_best": {}}
    how = "CALIBRATED VALIDATION (leave-one-out)" if args.calibrated else "VALIDATION"
    print(f"\nPRIMARY: k at the current weighting ({current[0]}, T={current[1]}); chosen on {how}")
    for t in targets:
        c = choose(table, current[0], current[1], t, calibrated=args.calibrated)
        cur = table[(table["weighting"] == current[0]) & (table["target"] == t) & (table["k"] == job_k[t])]
        if current[1] is not None:
            cur = cur[np.isclose(cur["temperature"].astype(float), current[1])]
        cur = cur.iloc[0]
        # Paired bootstrap on TEST: chosen k vs the current k (reported, not used to choose).
        # sweep() restored the datastore's own weighting, which is `current`.
        e_new, e_old = (errors_on_test(ds, val, test, t, k, ranges[t], args.calibrated) for k in (c["k"], job_k[t]))
        bs = paired_bootstrap(e_new, e_old)
        c.update(test_vs_current=bs)
        summary["chosen"][t] = c
        print(f"  {t:<7} chosen k={c['k']:<4} val {c['val_mae']:.3f}  test {c['test_mae']:.3f}   "
              f"(current k={job_k[t]}: val {cur[vcol]:.3f}, test {cur[tcol]:.3f}; "
              f"test diff {bs['diff']:+.3f} [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}])   "
              f"k within {CLOSE} of best on val: {c['k_within_close']}")
        sec = table[table["target"] == t].loc[lambda d: d[vcol].idxmin()]
        summary["secondary_best"][t] = sec.to_dict()
        print(f"          secondary (all weightings): {sec['weighting']} T={sec['temperature']} "
              f"k={int(sec['k'])}  val {sec[vcol]:.3f}  test {sec[tcol]:.3f}")

    # One k for both targets, chosen on validation combined score (for information).
    cur_rows = table[table["weighting"] == current[0]]
    if current[1] is not None:
        cur_rows = cur_rows[np.isclose(cur_rows["temperature"].astype(float), current[1])]
    piv = cur_rows.pivot(index="k", columns="target", values=vcol)
    if all(t in piv for t in NORMALISERS):
        comb = sum(piv[t] / NORMALISERS[t] for t in NORMALISERS)
        summary["single_k_val_combined"] = {"k": int(comb.idxmin()), "val_combined": float(comb.min())}
        print(f"\nOne k for both targets (val combined): k={int(comb.idxmin())} ({comb.min():.4f})")

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {out / 'sweep.csv'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
