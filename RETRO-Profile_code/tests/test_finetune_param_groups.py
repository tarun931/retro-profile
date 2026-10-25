"""Discriminative learning rates for fine-tuning the encoder.

Every result in SESSION_HANDOFF.md §3 came from a frozen encoder, where a single
optimizer param group is correct. Once the encoder is unfrozen that breaks: a
94.4 M pretrained backbone and a randomly-initialised MIL/projector/head stack
cannot share one LR. High enough to train the heads destroys the pretrained
features; low enough to protect them leaves the heads unlearnt.

These tests pin the split -- and, in particular, that the encoder's *new*
parameters (WavLMEncoder.layer_weights, initialised to zeros) train at the fast
head LR rather than the slow backbone LR.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

faiss = pytest.importorskip("faiss")

from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.training.lightning_module import RetroProfileLitModule

BACKBONE_LR = 1e-5
HEAD_LR = 1e-4


class _StubSSLEncoder(BaseEncoder):
    """Mimics WavLMEncoder's parameter layout.

    The pretrained backbone lives at ``.model`` (so its params are named
    ``model.encoder.model.*``) and ``layer_weights`` sits alongside it as a
    freshly-initialised parameter -- exactly the structure the LR split has to
    tell apart.
    """

    def __init__(self, output_dim: int = 16, freeze: bool = False, frame_pool: bool = False):
        super().__init__(output_dim=output_dim)
        self.model = nn.Sequential(nn.Linear(1, output_dim), nn.Linear(output_dim, output_dim))
        self.layer_weights = nn.Parameter(torch.zeros(3))
        if frame_pool:                  # stands in for WavLMEncoder.frame_pool (frame-level attention)
            self.frame_pool = nn.Linear(output_dim, 1)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, w, sm, um):
        m = sm.to(w.dtype)
        avg = (w * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        emb = self.model(avg)
        return emb * um.unsqueeze(-1).to(emb.dtype)


def _make_lit(encoder_lr: float | None, freeze: bool = False, frame_pool: bool = False,
              frame_attention_lr: float | None = None) -> RetroProfileLitModule:
    torch.manual_seed(0)
    model = RetroProfile(
        encoder=_StubSSLEncoder(output_dim=16, freeze=freeze, frame_pool=frame_pool),
        pooling=MILAttentionPooling(
            input_dim=16, hidden_dim=8, num_heads=1, target_names=["age", "height"]
        ),
        heads={
            "age": MLPHead(input_dim=8, hidden_dim=8, output_dim=1),
            "height": MLPHead(input_dim=8, hidden_dim=8, output_dim=1),
        },
        projector=MLPHead(input_dim=16, hidden_dim=16, output_dim=8, l2_normalize=True),
    )
    return RetroProfileLitModule(
        model=model,
        target_names=["age", "height"],
        embedding_dim=8,
        optimizer={"name": "adamw", "lr": HEAD_LR},
        scheduler={"name": "warmup_cosine", "warmup_steps": 5, "total_steps": 100},
        encoder_lr=encoder_lr,
        frame_attention_lr=frame_attention_lr,
    )


def _groups(lit: RetroProfileLitModule) -> list[dict]:
    """Param groups, with ``lr`` reported as the *configured* rate.

    The warmup scheduler rescales ``group["lr"]`` the moment it is constructed
    (step 0 of a linear warmup is 0.0), so the configured value survives only in
    ``initial_lr``. Asserting on the live ``lr`` would just test the warmup.
    """
    out = lit.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    return [{**g, "lr": g.get("initial_lr", g["lr"])} for g in opt.param_groups]


def test_frozen_default_keeps_a_single_param_group():
    """encoder_lr=None must reproduce the pre-existing behaviour exactly."""
    assert len(_groups(_make_lit(encoder_lr=None))) == 1


def test_finetuning_splits_backbone_from_new_layers():
    groups = _groups(_make_lit(encoder_lr=BACKBONE_LR))
    assert len(groups) == 2
    assert {g["lr"] for g in groups} == {HEAD_LR, BACKBONE_LR}


def test_backbone_group_holds_exactly_the_pretrained_weights():
    lit = _make_lit(encoder_lr=BACKBONE_LR)
    slow = next(g for g in _groups(lit) if g["lr"] == BACKBONE_LR)
    slow_ids = {id(p) for p in slow["params"]}
    expected = {id(p) for p in lit.model.encoder.model.parameters()}
    assert slow_ids == expected


def test_layer_weights_train_at_the_head_lr():
    """layer_weights is new (zeros), not pretrained -- it must not be slowed."""
    lit = _make_lit(encoder_lr=BACKBONE_LR)
    fast = next(g for g in _groups(lit) if g["lr"] == HEAD_LR)
    assert any(p is lit.model.encoder.layer_weights for p in fast["params"])


def test_frozen_backbone_yields_no_slow_group():
    """With the encoder frozen there is nothing to fine-tune, even if a slow LR
    was configured -- the backbone params have requires_grad=False."""
    groups = _groups(_make_lit(encoder_lr=BACKBONE_LR, freeze=True))
    assert len(groups) == 1
    assert groups[0]["lr"] == HEAD_LR


FRAME_LR = 1e-3


def test_frame_attention_gets_its_own_lr_when_set():
    """E26: at the head LR the zero-initialised frame scorer never moved; its own LR is opt-in."""
    lit = _make_lit(encoder_lr=BACKBONE_LR, frame_pool=True, frame_attention_lr=FRAME_LR)
    groups = _groups(lit)
    assert sorted(g["lr"] for g in groups) == sorted([HEAD_LR, BACKBONE_LR, FRAME_LR])
    frame = next(g for g in groups if g["lr"] == FRAME_LR)
    assert {id(p) for p in frame["params"]} == {id(p) for p in lit.model.encoder.frame_pool.parameters()}
    head = next(g for g in groups if g["lr"] == HEAD_LR)
    assert not {id(p) for p in head["params"]} & {id(p) for p in lit.model.encoder.frame_pool.parameters()}


def test_frame_attention_stays_at_the_head_lr_when_unset():
    """No frame_attention_lr = the behaviour of every run before 13 Oct 2026."""
    lit = _make_lit(encoder_lr=BACKBONE_LR, frame_pool=True)
    groups = _groups(lit)
    assert len(groups) == 2
    head = next(g for g in groups if g["lr"] == HEAD_LR)
    assert {id(p) for p in lit.model.encoder.frame_pool.parameters()} <= {id(p) for p in head["params"]}
