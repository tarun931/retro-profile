"""The attention-concentration penalty must be inert by default and correct when on.

Rationale in losses/attention_entropy.py. The term interpolates between MIL
attention (weight 0) and mean pooling (weight -> large) so that "MIL versus
averaging" can be measured as a curve rather than a binary, with bag-mode
evaluation on both sides.

The default weight of 0.0 is load-bearing: every result recorded so far was
produced without this term, and turning it on by accident would silently make
them non-comparable (rule R6).
"""

from __future__ import annotations

import math

import torch

from retro_profile.losses.attention_entropy import AttentionEntropyPenalty
from retro_profile.losses.multitask import MultitaskLoss


def _mask(b: int, k: int, valid: int | None = None) -> torch.Tensor:
    m = torch.zeros(b, k, dtype=torch.bool)
    m[:, : (valid or k)] = True
    return m


def test_uniform_attention_is_not_penalised():
    pen = AttentionEntropyPenalty(weight=1.0)
    w = torch.full((4, 8), 1 / 8)
    out = pen({"height": w}, _mask(4, 8))
    assert abs(float(out["attn_kl/height"])) < 1e-5


def test_fully_concentrated_attention_is_maximally_penalised():
    pen = AttentionEntropyPenalty(weight=1.0)
    w = torch.zeros(4, 8)
    w[:, 0] = 1.0
    out = pen({"height": w}, _mask(4, 8))
    assert float(out["attn_kl/height"]) > 0.99


def test_penalty_increases_as_attention_concentrates():
    """The term must be monotone, or sweeping its weight means nothing."""
    pen = AttentionEntropyPenalty(weight=1.0)
    mask = _mask(1, 8)
    vals = []
    for temp in (4.0, 1.0, 0.25):          # progressively peakier
        logits = torch.linspace(0, 3, 8).unsqueeze(0) / temp
        w = torch.softmax(logits, dim=1)
        vals.append(float(pen({"h": w}, mask)["attn_kl/h"]))
    assert vals[0] < vals[1] < vals[2]


def test_short_bags_are_measured_against_their_own_length():
    """A 3-clip bag spread evenly is uniform -- it must not be penalised.

    `BaseSpeakerDataset` yields variable-length bags, so padded slots would
    otherwise make every short bag look concentrated.
    """
    pen = AttentionEntropyPenalty(weight=1.0)
    w = torch.zeros(1, 8)
    w[0, :3] = 1 / 3
    assert abs(float(pen({"h": w}, _mask(1, 8, valid=3))["attn_kl/h"])) < 1e-5


def test_single_clip_bag_contributes_nothing():
    """With one valid instance there is no distribution to spread; log K = 0."""
    pen = AttentionEntropyPenalty(weight=1.0)
    w = torch.zeros(1, 8)
    w[0, 0] = 1.0
    val = float(pen({"h": w}, _mask(1, 8, valid=1))["attn_kl/h"])
    assert val == 0.0 and not math.isnan(val)


def test_n_eff_relationship_holds():
    """n_eff = exp(H), so penalty 0 must mean n_eff == K -- the reported metric."""
    pen = AttentionEntropyPenalty(weight=1.0)
    w = torch.full((1, 8), 1 / 8)
    kl = float(pen({"h": w}, _mask(1, 8))["attn_kl/h"])
    ent = -(w * torch.log(w)).sum().item()
    assert abs(math.exp(ent) - 8.0) < 1e-4
    assert abs(kl) < 1e-5


def test_zero_weight_leaves_the_total_loss_untouched():
    """The default must reproduce every previously recorded run exactly."""
    torch.manual_seed(0)
    preds = {"height": torch.randn(6)}
    labels = {"height": torch.randn(6)}
    mask = {"height": torch.ones(6, dtype=torch.bool)}
    emb = torch.randn(6, 16)
    attn = torch.softmax(torch.randn(6, 8) * 5, dim=1)     # very peaky

    off = MultitaskLoss(target_names=["height"], embedding_dim=16, ordcon_weight=0.0)
    base = off(preds, labels, mask, emb)["total"]
    with_attn = off(preds, labels, mask, emb, attention={"height": attn},
                    utt_mask=_mask(6, 8))["total"]
    assert torch.allclose(base, with_attn)


def test_nonzero_weight_adds_the_scaled_term():
    torch.manual_seed(0)
    preds = {"height": torch.randn(6)}
    labels = {"height": torch.randn(6)}
    mask = {"height": torch.ones(6, dtype=torch.bool)}
    emb = torch.randn(6, 16)
    attn = torch.softmax(torch.randn(6, 8) * 5, dim=1)
    um = _mask(6, 8)

    off = MultitaskLoss(target_names=["height"], embedding_dim=16, ordcon_weight=0.0)
    on = MultitaskLoss(target_names=["height"], embedding_dim=16, ordcon_weight=0.0,
                       attention_entropy_weight=0.5)
    base = off(preds, labels, mask, emb, attention={"height": attn}, utt_mask=um)
    hit = on(preds, labels, mask, emb, attention={"height": attn}, utt_mask=um)
    assert float(hit["total"]) > float(base["total"])
    expected = float(base["total"]) + 0.5 * float(hit["attn_kl/height"])
    assert abs(float(hit["total"]) - expected) < 1e-5


def test_per_target_weight_can_penalise_height_and_spare_age():
    """Age attention is already near-uniform; the penalty should be targetable."""
    attn = {"age": torch.softmax(torch.randn(2, 8), 1),
            "height": torch.softmax(torch.randn(2, 8) * 5, 1)}
    pen = AttentionEntropyPenalty(weight=0.0, per_target_weights={"height": 1.0})
    out = pen(attn, _mask(2, 8))
    # Both are measured and logged, but only height is charged for.
    assert "attn_kl/age" in out and "attn_kl/height" in out
    assert abs(float(out["attn_kl/total"]) - float(out["attn_kl/height"])) < 1e-6


def test_penalty_is_differentiable():
    pen = AttentionEntropyPenalty(weight=1.0)
    logits = torch.randn(3, 8, requires_grad=True)
    w = torch.softmax(logits, dim=1)
    pen({"h": w}, _mask(3, 8))["attn_kl/total"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
