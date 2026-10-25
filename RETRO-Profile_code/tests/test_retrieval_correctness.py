"""Phase A regression tests for the two retrieval bugs that broke Novelty 3.

1. Self-retrieval leakage: during phase-2 training the query speakers ARE in the
   datastore, so without leave-one-out each speaker retrieves itself at
   similarity 1.0 and the anchor becomes its own ground-truth label -- the
   residual head then has nothing to learn.
2. Query/index space mismatch: the datastore is built from ``embed_speaker``
   (shared, target-averaged embedding) but the head used to query with the
   per-target embedding -- searching a different space than was indexed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore

HEIGHTS = {i: 160.0 + 2.0 * i for i in range(10)}


def _datastore(dim=4, n=10, seed=0):
    rng = np.random.RandomState(seed)
    ds = RetrievalDatastore(embedding_dim=dim, metric="cosine", temperature=0.1)
    vecs = []
    for i in range(n):
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        vecs.append(v)
        ds.add(DatastoreEntry(speaker_id=f"spk_{i}", embedding=v,
                              labels={"height": HEIGHTS[i], "age": 25.0 + i}))
    ds.build()
    return ds, np.stack(vecs)


# ---------- Fix 1: leave-one-out self-exclusion ---------------------------


def test_without_exclusion_a_speaker_retrieves_itself():
    """Documents the bug: a stored speaker retrieves ITSELF at similarity 1.0,
    so its own ground-truth label dominates the softmax-weighted anchor. At
    training time that is label leakage; it disappears at test time (test
    speakers are not in the datastore), so the residual head learns the wrong
    thing. Severity grows as the softmax temperature sharpens."""
    ds, vecs = _datastore()
    q = torch.from_numpy(vecs[0:1])

    sims, idx = ds.search(q, 5)
    assert int(idx[0, 0]) == 0 and sims[0, 0].item() == pytest.approx(1.0, abs=1e-4)

    without = ds.knn_anchor(q, target="height", k=5).item()
    with_excl = ds.knn_anchor(q, target="height", k=5,
                              exclude_speaker_ids=["spk_0"]).item()
    own = HEIGHTS[0]
    # Leakage pulls the anchor toward the speaker's own label.
    assert abs(without - own) < abs(with_excl - own)


def test_exclusion_prevents_self_retrieval():
    """With leave-one-out, the anchor must come from OTHER speakers only."""
    ds, vecs = _datastore()
    q = torch.from_numpy(vecs[0:1])
    anchor = ds.knn_anchor(q, target="height", k=5, exclude_speaker_ids=["spk_0"])
    assert abs(anchor.item() - HEIGHTS[0]) > 0.5      # no longer its own label
    assert 160.0 <= anchor.item() <= 178.0            # still a plausible blend


def test_exclusion_returns_requested_number_of_neighbours():
    """Over-fetch + re-select must still yield exactly k neighbours."""
    ds, vecs = _datastore()
    q = torch.from_numpy(vecs[:3])
    nbrs = ds.neighbours(q, k=4, exclude_speaker_ids=["spk_0", "spk_1", "spk_2"])
    assert len(nbrs) == 3 and all(len(row) == 4 for row in nbrs)
    for i, row in enumerate(nbrs):                    # self never appears
        assert all(n["speaker_id"] != f"spk_{i}" for n in row)


def test_exclusion_length_mismatch_raises():
    ds, vecs = _datastore()
    with pytest.raises(ValueError):
        ds.knn_anchor(torch.from_numpy(vecs[:3]), "height", k=3,
                      exclude_speaker_ids=["spk_0"])          # 3 queries, 1 id


def test_uniform_weighting_also_excludes_self():
    ds, vecs = _datastore(n=5)
    ds.weighting = "uniform"
    anchor = ds.knn_anchor(torch.from_numpy(vecs[0:1]), "height", k=4,
                           exclude_speaker_ids=["spk_0"])
    expected = np.mean([HEIGHTS[i] for i in range(1, 5)])     # speakers 1..4
    assert abs(anchor.item() - expected) < 1e-4


# ---------- Fix 2: the kNN query lives in the datastore's space -----------


class _TinyEncoder(BaseEncoder):
    def __init__(self, output_dim=8):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        return self.proj(avg) * um.unsqueeze(-1).to(w.dtype)


def _model(proj_dim=4):
    torch.manual_seed(0)
    return RetroProfile(
        encoder=_TinyEncoder(8),
        pooling=MILAttentionPooling(input_dim=8, hidden_dim=4, num_heads=1,
                                    target_names=["age", "height"]),
        heads={"age": ResidualRetrievalHead(input_dim=proj_dim, hidden_dim=4,
                                            label_min=18.0, label_max=80.0, knn_k=3),
               "height": ResidualRetrievalHead(input_dim=proj_dim, hidden_dim=4,
                                               label_min=140.0, label_max=210.0, knn_k=3)},
        projector=MLPHead(input_dim=8, hidden_dim=8, output_dim=proj_dim, l2_normalize=True),
    )


def _batch(B=3):
    bags = [SpeakerBag([torch.randn(800), torch.randn(600)], 16000, f"spk_{i}",
                       ["u0", "u1"], {"age": 30.0 + i, "height": 170.0 + i})
            for i in range(B)]
    return collate_bags(bags, targets=("age", "height"))


def test_forward_queries_the_same_space_the_datastore_indexes():
    """The anchor produced inside forward must match one computed directly from
    ``embed_speaker`` -- i.e. the query vector IS the indexed vector."""
    ds, _ = _datastore(dim=4)
    model = _model(proj_dim=4).eval()
    for t in ("age", "height"):
        model.heads[t].set_datastore(ds, target_name=t)

    batch = _batch()
    with torch.no_grad():
        comps = model.forward_with_components(batch)
        expected = ds.knn_anchor(model.embed_speaker(batch), target="height",
                                 k=3, exclude_speaker_ids=batch.speaker_ids)
    assert torch.allclose(comps["height"]["anchor"], expected, atol=1e-5)


def test_forward_applies_leave_one_out_for_training_speakers():
    """Speakers present in the datastore must not retrieve themselves in forward."""
    dim = 4
    ds = RetrievalDatastore(embedding_dim=dim, metric="cosine", temperature=0.1)
    model = _model(proj_dim=dim).eval()
    batch = _batch(3)
    with torch.no_grad():
        embs = model.embed_speaker(batch).numpy()
    # Index the batch's own speakers with wildly distinctive labels.
    for i, sid in enumerate(batch.speaker_ids):
        ds.add(DatastoreEntry(speaker_id=sid, embedding=embs[i].astype(np.float32),
                              labels={"height": 140.0 + 60.0 * i, "age": 20.0 + 20.0 * i}))
    ds.build()
    for t in ("age", "height"):
        model.heads[t].set_datastore(ds, target_name=t)

    with torch.no_grad():
        anchors = model.forward_with_components(batch)["height"]["anchor"]
    own = torch.tensor([140.0 + 60.0 * i for i in range(3)])
    assert not torch.allclose(anchors, own, atol=1.0)     # never its own label
