"""Offline kNN sweep (scripts/knn_sweep.py) and the evaluation hooks that feed it.

The sweep is only worth running if a swept number is what the model would have
predicted. These tests pin that: the sweep scores settings with the head's own
kNN function, refuses to run when it cannot reproduce the job's predictions, and
chooses on VALIDATION, not on test.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

faiss = pytest.importorskip("faiss")

from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(name):
    sys.path.insert(0, str(SCRIPTS))                     # knn_sweep imports paired_bootstrap
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


ks = _script("knn_sweep")
RANGES = {"age": (18.0, 80.0), "height": (140.0, 210.0)}


def _unit(rng, n, d=6):
    v = rng.randn(n, d).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _datastore(n=40, seed=0):
    rng = np.random.RandomState(seed)
    ds = RetrievalDatastore(embedding_dim=6, metric="cosine", temperature=0.1)
    for i, v in enumerate(_unit(rng, n)):
        ds.add(DatastoreEntry(speaker_id=f"T{i}", embedding=v,
                              labels={"age": 20.0 + i, "height": 150.0 + i}))
    ds.build()
    return ds


def _split(ds, prefix, n, seed, k=20):
    """A split whose 'job predictions' are the datastore's own anchors at k."""
    rng = np.random.RandomState(seed)
    split = {"speaker_ids": [f"{prefix}{i}" for i in range(n)],
             "emb": torch.from_numpy(_unit(rng, n))}
    for t in RANGES:
        pred = ks.anchors(ds, split, t, k, RANGES[t])
        split[t] = {"truth": pred + rng.randn(n) * 3.0, "pred": pred}
    return split


def test_reproduction_check_passes_on_the_jobs_own_setting():
    ds = _datastore()
    ks.check_reproduces(ds, _split(ds, "V", 10, 1), list(RANGES), {"age": 20, "height": 20},
                        RANGES, "val")


def test_reproduction_check_stops_on_a_mismatch():
    ds = _datastore()
    split = _split(ds, "V", 10, 1)
    split["age"]["pred"] = split["age"]["pred"] + 0.5     # not what the model predicted
    with pytest.raises(SystemExit, match="differ"):
        ks.check_reproduces(ds, split, list(RANGES), {"age": 20, "height": 20}, RANGES, "val")


def test_sweep_matches_the_head_function_and_restores_the_datastore():
    ds = _datastore()
    val, test = _split(ds, "V", 12, 1), _split(ds, "S", 12, 2)
    table = ks.sweep(ds, val, test, list(RANGES), RANGES,
                     [("softmax_distance", 0.1), ("uniform", None)], k_grid=(1, 5, 20))
    assert (ds.weighting, ds.temperature) == ("softmax_distance", 0.1)
    row = table[(table.weighting == "uniform") & (table.k == 5) & (table.target == "age")].iloc[0]
    ds.weighting = "uniform"
    expect = np.abs(ds.knn_anchor(val["emb"], "age", 5, val["speaker_ids"]).numpy()
                    - val["age"]["truth"]).mean()
    assert row.val_mae == pytest.approx(expect)


def test_choice_is_made_on_validation_not_test():
    table = pd.DataFrame([
        {"weighting": "uniform", "temperature": None, "k": 5, "target": "age", "val_mae": 4.0, "test_mae": 3.0},
        {"weighting": "uniform", "temperature": None, "k": 20, "target": "age", "val_mae": 3.5, "test_mae": 3.9},
        {"weighting": "uniform", "temperature": None, "k": 50, "target": "age", "val_mae": 3.52, "test_mae": 3.8},
    ])
    c = ks.choose(table, "uniform", None, "age")
    assert c["k"] == 20 and c["test_mae"] == 3.9          # NOT k=5, the best on test
    assert c["k_within_close"] == [20, 50]


def test_k_larger_than_the_datastore_is_capped():
    ds = _datastore(n=8)
    val, test = _split(ds, "V", 4, 1, k=8), _split(ds, "S", 4, 2, k=8)
    table = ks.sweep(ds, val, test, list(RANGES), RANGES, [("uniform", None)], k_grid=(5, 300))
    assert sorted(table.k.unique()) == [5, 8]


def test_unreadable_parquet_falls_back_to_labels_csv(tmp_path):
    """the GPU cluster's Arrow 25 parquet is unreadable by the Mac's pyarrow 19 (2026-10-11)."""
    ds = _datastore(n=10)
    ds.save(tmp_path)
    labels = pd.read_parquet(tmp_path / "labels.parquet")
    (tmp_path / "labels.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(SystemExit, match="labels.csv"):
        ks.load_datastore(tmp_path)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    loaded = ks.load_datastore(tmp_path)
    q = torch.from_numpy(_unit(np.random.RandomState(5), 3))
    assert torch.allclose(loaded.knn_anchor(q, "age", 4), ds.knn_anchor(q, "age", 4))


# ---- the evaluation hooks --------------------------------------------------
def test_evaluate_records_the_split_and_writes_embeddings(tmp_path):
    ev = _script("evaluate")
    cfg = {"model": {"name": "retro_profile", "encoder": {"_target_": "retro_profile.models.WavLMEncoder"}}}
    for split in ("test", "val"):
        args = argparse.Namespace(unit="speaker", split=split, ckpt=Path("runs/x/checkpoints/last.ckpt"))
        meta = ev._run_metadata(cfg, args, torch.nn.Linear(2, 1), {"epoch": 3}, None)
        assert meta["eval"]["split"] == split
    rows = [{"speaker_id": "S0", "utt_id": "", "embedding": np.ones(4, dtype=np.float32)},
            {"speaker_id": "S1", "utt_id": "", "embedding": np.zeros(4, dtype=np.float32)}]
    ev._write_embeddings(rows, tmp_path, "timit")
    z = np.load(tmp_path / "embeddings_timit.npz")
    assert z["speaker_id"].tolist() == ["S0", "S1"] and z["embedding"].shape == (2, 4)


def test_old_results_without_a_split_read_as_test():
    ev = _script("evaluate")
    args = argparse.Namespace(unit="speaker", ckpt=Path("runs/x/checkpoints/last.ckpt"))
    meta = ev._run_metadata({"model": {}}, args, torch.nn.Linear(2, 1), {"epoch": 3}, None)
    assert meta["eval"]["split"] == "test"


def test_calibrated_sweep_scores_after_the_same_calibration_as_the_model():
    """--calibrated (14 Oct 2026): validation leave-one-out, test with the fit on all validation rows."""
    ds = _datastore()
    val, test = _split(ds, "V", 15, 1), _split(ds, "S", 12, 2)
    for split, shift in ((val, 9.0), (test, 9.0)):                         # model reads 9 yr too old
        split["age"]["truth"] = split["age"]["truth"] - shift
    table = ks.sweep(ds, val, test, ["age"], RANGES, [("uniform", None)], k_grid=(5,), calibrate=True)
    row = table.iloc[0]
    ds.weighting = "uniform"
    av = ds.knn_anchor(val["emb"], "age", 5, val["speaker_ids"]).clamp(*RANGES["age"]).numpy().astype(float)
    at = ds.knn_anchor(test["emb"], "age", 5, test["speaker_ids"]).clamp(*RANGES["age"]).numpy().astype(float)
    yv, yt = val["age"]["truth"], test["age"]["truth"]
    loo = []
    for i in range(len(av)):
        keep = np.arange(len(av)) != i
        a, b = np.polyfit(av[keep], yv[keep], 1)
        loo.append(abs(a * av[i] + b - yv[i]))
    a, b = np.polyfit(av, yv, 1)
    assert row.val_mae_cal == pytest.approx(np.mean(loo))
    assert row.test_mae_cal == pytest.approx(np.abs(a * at + b - yt).mean())
    assert row.val_mae > row.val_mae_cal                                    # the 9-yr offset is calibrated away


def test_calibrated_choice_uses_the_calibrated_column_and_raw_mode_is_unchanged():
    table = pd.DataFrame([
        {"weighting": "uniform", "temperature": None, "k": 5, "target": "age", "val_mae": 3.0, "test_mae": 3.0,
         "val_mae_cal": 4.0, "test_mae_cal": 4.1},
        {"weighting": "uniform", "temperature": None, "k": 20, "target": "age", "val_mae": 9.0, "test_mae": 9.0,
         "val_mae_cal": 3.5, "test_mae_cal": 3.6},
    ])
    assert ks.choose(table, "uniform", None, "age", calibrated=True)["k"] == 20
    assert ks.choose(table, "uniform", None, "age")["k"] == 5
    ds = _datastore()
    raw = ks.sweep(ds, _split(ds, "V", 8, 1), _split(ds, "S", 8, 2), ["age"], RANGES, [("uniform", None)], k_grid=(5,))
    assert "val_mae_cal" not in raw.columns
