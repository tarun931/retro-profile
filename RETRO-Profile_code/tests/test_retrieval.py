"""Tests for the FAISS-backed retrieval datastore."""

from __future__ import annotations

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore


def _make_datastore(n_speakers=20, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    ds = RetrievalDatastore(embedding_dim=dim, metric="cosine", temperature=0.1)
    for i in range(n_speakers):
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        ds.add(DatastoreEntry(
            speaker_id=f"spk_{i:03d}",
            embedding=v,
            labels={"height": 160.0 + i, "age": 25.0 + i},
        ))
    ds.build()
    return ds


def test_search_returns_correct_shape():
    ds = _make_datastore()
    queries = torch.randn(4, 8)
    sims, idx = ds.search(queries, k=5)
    assert sims.shape == (4, 5)
    assert idx.shape == (4, 5)


def test_self_search_returns_self_first():
    """Querying with stored vectors should retrieve the same speaker as #1."""
    ds = _make_datastore()
    # Use the entries' own vectors as queries.
    vecs = np.stack([e.embedding for e in ds._entries])
    queries = torch.from_numpy(vecs)
    _, idx = ds.search(queries, k=1)
    assert torch.equal(idx.squeeze(-1), torch.arange(len(ds)))


def test_knn_anchor_uniform_weighting_matches_mean():
    ds = _make_datastore(n_speakers=5)
    queries = torch.from_numpy(np.stack([e.embedding for e in ds._entries]))
    ds.weighting = "uniform"
    anchor = ds.knn_anchor(queries, target="height", k=5)
    expected = sum(160.0 + i for i in range(5)) / 5.0
    assert torch.allclose(anchor, torch.full((5,), expected), atol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    ds = _make_datastore()
    ds.save(tmp_path / "store")
    ds2 = RetrievalDatastore.load(tmp_path / "store")
    assert len(ds2) == len(ds)
    queries = torch.randn(2, 8)
    a1 = ds.knn_anchor(queries, "height", k=3)
    a2 = ds2.knn_anchor(queries, "height", k=3)
    assert torch.allclose(a1, a2, atol=1e-5)
