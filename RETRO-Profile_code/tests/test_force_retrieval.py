"""Tests for evaluate.py --force-retrieval (2026-10-10).

Lets a model trained WITH retrieval off be tested WITH kNN on, so the OrdCon
tuning can measure retrieval quality for every setting without training a
separate two-phase model per setting. For such a model the residual branch was
never trained (zero-initialised last layer), so the prediction must equal the
kNN anchor exactly -- and the result must say it was forced.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

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
from retro_profile.retrieval.builder import attach_datastore
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)

RANGES = {"age": (18.0, 80.0), "height": (140.0, 210.0)}


class _TinyEnc(BaseEncoder):
    def __init__(self, output_dim: int = 16) -> None:
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        return self.proj(avg) * um.unsqueeze(-1).to(w.dtype)


def _retrieval_off_model():
    torch.manual_seed(0)
    return RetroProfile(
        encoder=_TinyEnc(16),
        pooling=MILAttentionPooling(input_dim=16, hidden_dim=8, num_heads=1,
                                    target_names=["age", "height"]),
        heads={t: ResidualRetrievalHead(input_dim=8, hidden_dim=8, label_min=lo,
                                        label_max=hi, use_retrieval=False)
               for t, (lo, hi) in RANGES.items()},
        projector=MLPHead(input_dim=16, hidden_dim=16, output_dim=8, l2_normalize=True),
    ).eval()


def _datastore():
    rng = np.random.RandomState(0)
    ds = RetrievalDatastore(embedding_dim=8, metric="cosine", temperature=0.1)
    for i in range(12):
        v = rng.randn(8).astype(np.float32)
        ds.add(DatastoreEntry(speaker_id=f"train{i}", embedding=v / np.linalg.norm(v),
                              labels={"age": 25.0 + 2 * i, "height": 160.0 + 2 * i}))
    ds.build()
    return ds


def _batch():
    torch.manual_seed(1)
    bags = [SpeakerBag(waveforms=[torch.randn(1600) * (j + 1) for j in range(3)],
                       sample_rate=16000, speaker_id=f"test{i}", utt_ids=["a", "b", "c"],
                       labels={"age": 30.0 + i, "height": 170.0 + i}) for i in range(4)]
    return collate_bags(bags, targets=("age", "height"))


def test_without_forcing_a_retrieval_off_model_ignores_the_datastore():
    model = _retrieval_off_model()
    attach_datastore(model, _datastore())
    assert not any(h.has_datastore for h in model.heads.values())


def test_forcing_makes_the_prediction_the_knn_anchor_exactly():
    model = _retrieval_off_model()
    assert ev._force_retrieval(model) == 2
    attach_datastore(model, _datastore())
    assert all(h.has_datastore for h in model.heads.values())
    with torch.no_grad():
        comps = model.forward_with_components(_batch())
    for t, (lo, hi) in RANGES.items():
        anchor = comps[t]["anchor"]
        assert anchor is not None and torch.isfinite(anchor).all()
        # Residual never trained: zero last layer -> delta exactly 0.
        assert torch.equal(comps[t]["delta"], torch.zeros_like(comps[t]["delta"]))
        assert torch.allclose(comps[t]["pred"], anchor.clamp(lo, hi))


def test_forced_results_are_labelled_in_provenance():
    model = _retrieval_off_model()
    args = argparse.Namespace(unit="speaker", ckpt="x/epoch=1.ckpt", force_retrieval=True)
    meta = ev._run_metadata({}, args, model, {}, _datastore())
    assert meta["eval"]["retrieval"] is True
    assert meta["eval"]["retrieval_forced"] is True
    args.force_retrieval = False
    assert ev._run_metadata({}, args, model, {}, _datastore())["eval"]["retrieval_forced"] is False
