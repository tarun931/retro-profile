"""Tests for pooling layers — especially the MIL novelty."""

from __future__ import annotations

import torch

from retro_profile.models.pooling import AttentionPooling, MeanPooling, MILAttentionPooling


def _fake_utts(B=2, K=4, D=16):
    return torch.randn(B, K, D)


def test_mean_pooling_ignores_padded_utts():
    utt = _fake_utts()
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    pool = MeanPooling(input_dim=16)
    speaker, attn = pool(utt, mask)
    assert speaker.shape == (2, 16)
    # Speaker 0 should equal mean of first 2 utts.
    expected = utt[0, :2].mean(dim=0)
    assert torch.allclose(speaker[0], expected, atol=1e-5)
    # Attention weights should sum to 1 over real utterances.
    assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-5)


def test_attention_pooling_outputs_valid_distribution():
    utt = _fake_utts()
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    pool = AttentionPooling(input_dim=16, hidden_dim=8)
    speaker, attn = pool(utt, mask)
    assert speaker.shape == (2, 16)
    # Padded positions must have zero attention; real positions sum to 1.
    assert attn[0, 3].item() == 0.0
    assert attn[1, 2:].sum().item() == 0.0
    assert torch.allclose(attn.sum(dim=1), torch.ones(2), atol=1e-5)


def test_mil_per_target_heads_are_distinct():
    """Per-target MIL heads should produce DIFFERENT speaker embeddings for the
    same input — confirming they are separately parameterized."""
    utt = _fake_utts(B=3, K=5, D=32)
    mask = torch.ones(3, 5, dtype=torch.bool)
    pool = MILAttentionPooling(
        input_dim=32, hidden_dim=16, num_heads=2, target_names=["age", "height"]
    )
    age_emb, age_attn = pool(utt, mask, target="age")
    height_emb, height_attn = pool(utt, mask, target="height")
    assert age_emb.shape == (3, 32)
    assert height_emb.shape == (3, 32)
    # With random init the two heads should produce different outputs.
    assert not torch.allclose(age_emb, height_emb, atol=1e-6)
    # Attention sums to 1 per speaker.
    assert torch.allclose(age_attn.sum(dim=1), torch.ones(3), atol=1e-5)


def test_mil_dict_return_when_target_is_none():
    utt = _fake_utts(B=2, K=3, D=16)
    mask = torch.ones(2, 3, dtype=torch.bool)
    pool = MILAttentionPooling(
        input_dim=16, hidden_dim=8, num_heads=1, target_names=["age", "height"]
    )
    pooled, weights = pool(utt, mask, target=None)
    assert isinstance(pooled, dict)
    assert set(pooled) == {"age", "height"}
    assert pooled["age"].shape == (2, 16)


def test_mil_handles_padded_utts():
    """Attention should put zero mass on padded utterances."""
    utt = _fake_utts(B=1, K=4, D=8)
    mask = torch.tensor([[True, True, False, False]])
    pool = MILAttentionPooling(input_dim=8, hidden_dim=4, num_heads=1, target_names=None)
    speaker, attn = pool(utt, mask)
    assert torch.all(attn[0, 2:] == 0.0)
    assert abs(attn[0, :2].sum().item() - 1.0) < 1e-5
