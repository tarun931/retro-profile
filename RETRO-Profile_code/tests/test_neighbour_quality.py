"""scripts/neighbour_quality.py (Table 3, RQ4): label gaps to the top-k neighbours, how well the space is ordered,
and the guard that the neighbours are the second stage's own (it stops on any mismatch)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("faiss")

from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("neighbour_quality", _ROOT / "scripts" / "neighbour_quality.py")
nq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq)


def _arc(n, labels, jitter=0.0, seed=0):
    """Unit vectors on an arc whose angle follows the label: an ordered space."""
    rng = np.random.RandomState(seed)
    ang = (np.asarray(labels, float) - 20.0) / 60.0 * 1.5 + rng.randn(n) * jitter
    return np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], axis=1).astype(np.float32)


def _datastore(labels, emb):
    ds = RetrievalDatastore(embedding_dim=emb.shape[1], metric="cosine", temperature=0.1)
    for i, (y, e) in enumerate(zip(labels, emb, strict=True)):
        ds.add(DatastoreEntry(speaker_id=f"T{i}", embedding=e, labels={"age": float(y)}))
    ds.build()
    return ds


def test_label_gaps_average_over_neighbours_then_speakers():
    nbr = np.array([[30.0, 34.0, 40.0], [50.0, np.nan, 44.0]])
    y = np.array([32.0, 47.0])
    g = nq.label_gaps(nbr, y, [1, 3])
    assert g[1] == pytest.approx((2 + 3) / 2)
    assert g[3] == pytest.approx(((2 + 2 + 8) / 3 + (3 + 3) / 2) / 2)          # NaN neighbour skipped


def test_neighbours_are_nearest_first_and_an_ordered_space_scores_high():
    labels = np.linspace(20, 80, 60)
    ds = _datastore(labels, _arc(60, labels))
    q_lab = np.array([25.0, 50.0, 75.0])
    q = torch.from_numpy(_arc(3, q_lab))
    nbr = nq.neighbour_labels(ds, q, "age", 5)
    assert nbr.shape == (3, 5) and np.all(np.abs(nbr[:, 0] - q_lab) <= 1.1)       # nearest has the closest age
    assert np.all(np.abs(nbr[:, 0] - q_lab) <= np.abs(nbr[:, 4] - q_lab))
    assert nq.distance_spearman(ds, q, "age", q_lab) > 0.8
    shuffled = _datastore(np.random.RandomState(1).permutation(labels), _arc(60, labels))   # labels unrelated to space
    assert abs(nq.distance_spearman(shuffled, q, "age", q_lab)) < 0.3


def test_reproduction_check_passes_on_the_same_anchors_and_stops_otherwise(tmp_path):
    labels = np.linspace(20, 80, 40)
    ds = _datastore(labels, _arc(40, labels, jitter=0.05))
    split = {"speaker_ids": ["A", "B", "C"], "query": torch.from_numpy(_arc(3, [30.0, 45.0, 70.0]))}
    want = ds.knn_anchor(split["query"], target="age", k=5, exclude_speaker_ids=split["speaker_ids"]).clamp(18, 80)
    ref = tmp_path / "predictions_test_knn.csv"
    pd.DataFrame({"speaker_id": split["speaker_ids"], "target": "age", "prediction": want.numpy()}).to_csv(ref, index=False)
    assert nq.check_reproduces(ds, split, "age", 5, (18.0, 80.0), ref) == pytest.approx(0.0, abs=1e-5)
    pd.DataFrame({"speaker_id": split["speaker_ids"], "target": "age", "prediction": want.numpy() + 1}).to_csv(ref, index=False)
    with pytest.raises(SystemExit, match="differ"):
        nq.check_reproduces(ds, split, "age", 5, (18.0, 80.0), ref)


def test_neighbours_without_the_label_are_skipped_like_knn_skips_them():
    """FKFB0 (TIMIT train) has no age: as someone's nearest neighbour it must not turn gap@1 into NaN."""
    labels = np.linspace(20, 80, 30)
    emb = _arc(30, labels)
    ds = RetrievalDatastore(embedding_dim=3, metric="cosine", temperature=0.1)
    for i, (y, e) in enumerate(zip(labels, emb, strict=True)):
        ds.add(DatastoreEntry(speaker_id=f"T{i}", embedding=e, labels={} if i == 10 else {"age": float(y)}))
    ds.build()
    q = torch.from_numpy(emb[10:11])                                            # nearest entry = the unlabelled one
    nbr = nq.neighbour_labels(ds, q, "age", 3)
    assert np.isfinite(nbr).all() and labels[10] not in nbr[0]
    assert set(nbr[0]) <= {labels[9], labels[11], labels[8], labels[12]}
    assert np.isfinite(nq.label_gaps(nbr, np.array([labels[10]]), [1, 3])[1])
