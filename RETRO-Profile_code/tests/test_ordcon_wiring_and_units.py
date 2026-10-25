"""Regression tests for the OrdCon fixes of 2026-10-10.

Three defects, each pinned here:

1. WIRING. The LightningModule fed OrdCon `next(iter(components))["embedding"]`
   -- the FIRST target's projected vector -- under the comment "all targets share
   the same projector output, so any of them works". The projector's weights are
   shared, but with per-target MIL stacks its inputs are not, so both the age and
   height ordering terms shaped the AGE-pooled vector while FAISS searched a third
   vector (`embed_speaker`) that OrdCon never touched. The design
   (models/retro_profile.py) is that OrdCon organises the space the datastore
   indexes.

2. UNITS. `magnitude_scale` was documented in label units (years, cm) but OrdCon
   received z-scored labels, and one scalar served every target.

3. NaN ANCHOR. `knn_anchor` softmaxed a row of -inf into NaN whenever no
   retrieved neighbour carried a label for the target, and read the LAST label
   for a FAISS empty slot (index -1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.losses.ordcon import OrdConLoss
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore
from retro_profile.training.lightning_module import RetroProfileLitModule

LABEL_STATS = {
    "age":    {"mean": 40.0,  "std": 12.0, "min": 18.0,  "max": 80.0,  "count": 100},
    "height": {"mean": 175.0, "std": 9.0,  "min": 145.0, "max": 200.0, "count": 100},
}
HEAD_RANGES = {"age": (18.0, 80.0), "height": (140.0, 210.0)}


class _TinyEnc(BaseEncoder):
    """Stub encoder: projects each utterance's waveform mean into a fixed dim."""

    def __init__(self, output_dim: int = 16) -> None:
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        emb = self.proj(avg)
        return emb * um.unsqueeze(-1).to(emb.dtype)


def _make(targets, *, ordcon_embedding="retrieval", ordcon_kwargs=None,
          normalize_targets=True):
    torch.manual_seed(0)
    model = RetroProfile(
        encoder=_TinyEnc(16),
        pooling=MILAttentionPooling(input_dim=16, hidden_dim=8, num_heads=1,
                                    target_names=list(targets)),
        heads={t: ResidualRetrievalHead(input_dim=8, hidden_dim=8,
                                        label_min=HEAD_RANGES[t][0],
                                        label_max=HEAD_RANGES[t][1])
               for t in targets},
        projector=MLPHead(input_dim=16, hidden_dim=16, output_dim=8, l2_normalize=True),
    )
    lit = RetroProfileLitModule(
        model=model,
        target_names=list(targets),
        embedding_dim=8,
        ordcon_weight=0.3,
        ordcon_kwargs=ordcon_kwargs,
        label_stats={t: LABEL_STATS[t] for t in targets},
        normalize_targets=normalize_targets,
        ordcon_embedding=ordcon_embedding,
    )
    return model, lit


def _batch(targets, n=4):
    torch.manual_seed(1)
    bags = [
        SpeakerBag(
            # Three clips of different loudness so per-target attention differs.
            waveforms=[torch.randn(1600) * (j + 1) + j for j in range(3)],
            sample_rate=16000,
            speaker_id=f"s{i}",
            utt_ids=["u0", "u1", "u2"],
            labels={t: (25.0 + 5 * i if t == "age" else 168.0 + 4 * i) for t in targets},
        )
        for i in range(n)
    ]
    return collate_bags(bags, targets=tuple(targets))


def _ordcon_input_seen_by_loss(lit, batch, monkeypatch):
    """Run one deterministic step and return the embedding handed to the loss."""
    seen = {}
    real = lit.loss_fn.forward

    def spy(*args, **kwargs):
        seen["emb"] = kwargs["embeddings"]
        return real(*args, **kwargs)

    monkeypatch.setattr(lit.loss_fn, "forward", spy)
    lit.model.eval()                          # no dropout -> deterministic
    with torch.no_grad():
        lit._shared_step(batch, "val")
    return seen["emb"]


# ──────────────────────────────────────────────────────────────────────────
# 1. Wiring
# ──────────────────────────────────────────────────────────────────────────

def test_ordcon_receives_the_vector_the_datastore_indexes(monkeypatch):
    model, lit = _make(["age", "height"])
    batch = _batch(["age", "height"])
    emb = _ordcon_input_seen_by_loss(lit, batch, monkeypatch)
    with torch.no_grad():
        indexed = model.embed_speaker(batch)
    assert torch.allclose(emb, indexed, atol=1e-6)


def test_two_target_mil_embeddings_genuinely_differ(monkeypatch):
    """Guards the test above against passing vacuously.

    If the age and height vectors happened to coincide, the wiring bug would be
    invisible and test 1 would prove nothing.
    """
    model, lit = _make(["age", "height"])
    batch = _batch(["age", "height"])
    emb = _ordcon_input_seen_by_loss(lit, batch, monkeypatch)
    with torch.no_grad():
        comps = model.forward_with_components(batch)
    assert (emb - comps["age"]["embedding"]).abs().max() > 1e-4
    assert (comps["age"]["embedding"] - comps["height"]["embedding"]).abs().max() > 1e-4


def test_first_target_mode_reproduces_pre_fix_behaviour(monkeypatch):
    model, lit = _make(["age", "height"], ordcon_embedding="first_target")
    batch = _batch(["age", "height"])
    emb = _ordcon_input_seen_by_loss(lit, batch, monkeypatch)
    with torch.no_grad():
        age_vec = model.forward_with_components(batch)["age"]["embedding"]
    assert torch.allclose(emb, age_vec, atol=1e-6)


def test_single_target_is_unchanged_by_the_fix(monkeypatch):
    """Single-task runs (age-only, height-only) must be bit-identical."""
    for t in ("age", "height"):
        model, lit = _make([t])
        batch = _batch([t])
        emb = _ordcon_input_seen_by_loss(lit, batch, monkeypatch)
        with torch.no_grad():
            own = model.forward_with_components(batch)[t]["embedding"]
        assert torch.allclose(emb, own, atol=1e-6)


def test_unknown_ordcon_embedding_is_rejected():
    with pytest.raises(ValueError, match="ordcon_embedding"):
        _make(["age"], ordcon_embedding="age")


# ──────────────────────────────────────────────────────────────────────────
# 2. Units
#
# Two identical embeddings sit at distance 0, so each off-diagonal pair's
# magnitude hinge equals its target distance exactly, and the logged
# `ordcon/<t>/magnitude` IS the target distance.
# ──────────────────────────────────────────────────────────────────────────

def _magnitude(loss, labels):
    n = len(next(iter(labels.values())))
    emb = torch.ones(n, loss.embedding_dim) / 2.0
    mask = {t: torch.ones(n, dtype=torch.bool) for t in labels}
    out = loss(emb, labels, mask)
    return {t: out[f"ordcon/{t}/magnitude"].item() for t in labels}, out


def test_label_units_five_year_gap_gives_its_intended_distance():
    loss = OrdConLoss(embedding_dim=4, target_names=["age"],
                      magnitude_scale={"age": 5.0}, magnitude_units="label",
                      label_std={"age": 10.0})
    # A z-gap of 0.5 at std 10 is a 5-year gap -> 5 / 5 = 1.0.
    mag, _ = _magnitude(loss, {"age": torch.tensor([0.0, 0.5])})
    assert mag["age"] == pytest.approx(1.0)


def test_loss_units_reproduce_the_historical_formula():
    """The default must keep every run to date bit-identical: gap_received / 5."""
    loss = OrdConLoss(embedding_dim=4, target_names=["age"])
    mag, _ = _magnitude(loss, {"age": torch.tensor([0.0, 0.5])})
    assert mag["age"] == pytest.approx(0.5 / 5.0)


def test_label_units_without_std_treat_labels_as_real():
    loss = OrdConLoss(embedding_dim=4, target_names=["age"],
                      magnitude_scale=5.0, magnitude_units="label")
    mag, _ = _magnitude(loss, {"age": torch.tensor([30.0, 35.0])})
    assert mag["age"] == pytest.approx(1.0)


def test_per_target_scales_are_independent():
    loss = OrdConLoss(embedding_dim=4, target_names=["age", "height"],
                      magnitude_scale={"age": 5.0, "height": 10.0},
                      magnitude_units="label",
                      label_std={"age": 10.0, "height": 10.0})
    z = torch.tensor([0.0, 0.5])                    # 5 units in both targets
    mag, _ = _magnitude(loss, {"age": z, "height": z})
    assert mag["age"] == pytest.approx(1.0)
    assert mag["height"] == pytest.approx(0.5)


def test_direction_and_magnitude_are_logged_and_sum_to_the_target_term():
    loss = OrdConLoss(embedding_dim=4, target_names=["age"])
    _, out = _magnitude(loss, {"age": torch.tensor([0.0, 0.5, 1.5])})
    assert out["ordcon/age"].item() == pytest.approx(
        out["ordcon/age/direction"].item() + out["ordcon/age/magnitude"].item()
    )


@pytest.mark.parametrize("kwargs, match", [
    ({"magnitude_scale": {"height": 5.0}}, "no entry"),
    ({"magnitude_scale": 0.0}, "positive"),
    ({"magnitude_scale": {"age": -1.0}}, "positive"),
    ({"magnitude_units": "years"}, "magnitude_units"),
])
def test_bad_magnitude_settings_fail_loudly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        OrdConLoss(embedding_dim=4, target_names=["age"], **kwargs)


def test_litmodule_hands_ordcon_the_std_it_normalises_with():
    _, lit = _make(["age", "height"])
    assert lit.loss_fn.ordcon.label_std == {"age": 12.0, "height": 9.0}
    _, lit_raw = _make(["age", "height"], normalize_targets=False)
    assert lit_raw.loss_fn.ordcon.label_std == {}


def test_label_units_end_to_end_through_the_litmodule():
    """A 5-year gap, z-scored by the module, must still read as 5 years."""
    _, lit = _make(["age"], ordcon_kwargs={"magnitude_units": "label",
                                            "magnitude_scale": {"age": 5.0}})
    gap_z = 5.0 / LABEL_STATS["age"]["std"]
    mag, _ = _magnitude(lit.loss_fn.ordcon, {"age": torch.tensor([0.0, gap_z])})
    assert mag["age"] == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────────
# 3. knn_anchor
# ──────────────────────────────────────────────────────────────────────────

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _store(rows, weighting="softmax_distance"):
    """rows: list of (embedding, age). Height is always present."""
    ds = RetrievalDatastore(embedding_dim=4, metric="cosine", temperature=0.1,
                            weighting=weighting)
    for i, (vec, age) in enumerate(rows):
        ds.add(DatastoreEntry(speaker_id=f"spk{i}", embedding=_unit(vec),
                              labels={"age": age, "height": 170.0}))
    ds.build()
    return ds


NAN = float("nan")
# Three age-less speakers right next to the query; one labelled speaker far away.
NEAR_UNLABELLED = [([1, 0, 0, 0], NAN), ([1, 0.05, 0, 0], NAN),
                   ([1, 0, 0.05, 0], NAN), ([-1, 0, 0, 0], 50.0)]
QUERY = torch.tensor([_unit([1, 0.01, 0.01, 0])])


@pytest.mark.parametrize("weighting", ["softmax_distance", "uniform"])
def test_all_invalid_neighbours_fall_back_to_the_datastore_mean(weighting):
    ds = _store(NEAR_UNLABELLED, weighting=weighting)
    anchor = ds.knn_anchor(QUERY, target="age", k=3)    # all 3 retrieved lack age
    assert torch.isfinite(anchor).all()
    # Before the fix: NaN (softmax) or 0.0 years (uniform).
    assert anchor.item() == pytest.approx(50.0)


@pytest.mark.parametrize("weighting", ["softmax_distance", "uniform"])
def test_mixed_neighbours_weight_only_the_labelled_ones(weighting):
    rows = [([1, 0, 0, 0], NAN), ([1, 0.05, 0, 0], 40.0), ([1, 0, 0.05, 0], 40.0)]
    ds = _store(rows, weighting=weighting)
    anchor = ds.knn_anchor(QUERY, target="age", k=3)
    assert anchor.item() == pytest.approx(40.0)


def test_empty_faiss_slots_are_not_read_as_the_last_label():
    # k exceeds the store, so FAISS fills the extra slots with index -1.
    rows = [([1, 0, 0, 0], 30.0), ([1, 0.05, 0, 0], 30.0), ([-1, 0, 0, 0], 70.0)]
    ds = _store(rows, weighting="uniform")
    anchor = ds.knn_anchor(QUERY, target="age", k=5)
    # Three real neighbours: 30, 30, 70. Reading -1 as "the last row" would add
    # 70.0 twice more and give 54.0.
    assert anchor.item() == pytest.approx((30.0 + 30.0 + 70.0) / 3)


def test_a_store_with_no_labels_for_the_target_refuses():
    ds = _store([([1, 0, 0, 0], NAN), ([0, 1, 0, 0], NAN)])
    with pytest.raises(ValueError, match="no 'age' labels"):
        ds.knn_anchor(QUERY, target="age", k=2)


def test_fully_labelled_store_is_unchanged():
    """Every TIMIT/NISP datastore is fully labelled; its anchors must not move."""
    rows = [([1, 0, 0, 0], 30.0), ([1, 0.05, 0, 0], 40.0), ([1, 0, 0.05, 0], 50.0)]
    ds = _store(rows)
    sims, idx = ds.search(QUERY, 3)
    labels = torch.tensor([30.0, 40.0, 50.0])[idx]
    expected = (torch.softmax(sims / 0.1, dim=1) * labels).sum(dim=1)
    assert ds.knn_anchor(QUERY, target="age", k=3).item() == pytest.approx(expected.item())
    assert not math.isnan(expected.item())
