"""Smoke test for the full RETRO-Profile model.

We use a tiny synthetic encoder (random projection) instead of WavLM so the
test runs in milliseconds and doesn't need an internet connection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile


class TinyRandomEncoder(BaseEncoder):
    """Encoder stub: project waveform mean to a random fixed-dim embedding."""

    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, waveforms, sample_mask, utt_mask):
        # waveforms: (B, K, T). Reduce to (B, K, 1) by masked mean over T.
        m = sample_mask.to(waveforms.dtype)
        summed = (waveforms * m).sum(dim=-1, keepdim=True)
        denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        avg = summed / denom
        emb = self.proj(avg)
        emb = emb * utt_mask.unsqueeze(-1).to(emb.dtype)
        return emb


def test_retro_profile_forward_smoke(fake_bag_batch):
    enc = TinyRandomEncoder(output_dim=32)
    pool = MILAttentionPooling(
        input_dim=32, hidden_dim=16, num_heads=2, target_names=["age", "height"]
    )
    proj = MLPHead(input_dim=32, hidden_dim=32, output_dim=16, l2_normalize=True)
    heads = {
        "age": ResidualRetrievalHead(
            input_dim=16, hidden_dim=8, label_min=18.0, label_max=80.0
        ),
        "height": ResidualRetrievalHead(
            input_dim=16, hidden_dim=8, label_min=140.0, label_max=210.0
        ),
    }
    model = RetroProfile(encoder=enc, pooling=pool, heads=heads, projector=proj)

    out = model(fake_bag_batch)
    assert set(out) == {"age", "height"}
    assert out["age"].shape == (3,)
    assert out["height"].shape == (3,)
    # Without a datastore these must lie in the configured label ranges.
    assert torch.all(out["age"] >= 18.0) and torch.all(out["age"] <= 80.0)
    assert torch.all(out["height"] >= 140.0) and torch.all(out["height"] <= 210.0)


def test_embed_speaker_returns_projected_embedding(fake_bag_batch):
    enc = TinyRandomEncoder(output_dim=32)
    pool = MILAttentionPooling(
        input_dim=32, hidden_dim=16, num_heads=2, target_names=["age", "height"]
    )
    proj = MLPHead(input_dim=32, hidden_dim=32, output_dim=16, l2_normalize=True)
    heads = {
        "age": ResidualRetrievalHead(input_dim=16, hidden_dim=8),
        "height": ResidualRetrievalHead(input_dim=16, hidden_dim=8),
    }
    model = RetroProfile(encoder=enc, pooling=pool, heads=heads, projector=proj)
    emb = model.embed_speaker(fake_bag_batch)
    assert emb.shape == (3, 16)
    # L2-normalized projector should give unit norms.
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_forward_with_components_includes_anchor_and_delta(fake_bag_batch):
    enc = TinyRandomEncoder(output_dim=32)
    pool = MILAttentionPooling(
        input_dim=32, hidden_dim=16, num_heads=1, target_names=["age", "height"]
    )
    proj = MLPHead(input_dim=32, hidden_dim=32, output_dim=16, l2_normalize=True)
    heads = {
        "age": ResidualRetrievalHead(input_dim=16, hidden_dim=8),
        "height": ResidualRetrievalHead(input_dim=16, hidden_dim=8),
    }
    model = RetroProfile(encoder=enc, pooling=pool, heads=heads, projector=proj)
    comps = model.forward_with_components(fake_bag_batch)
    for t in ("age", "height"):
        assert "pred" in comps[t]
        assert "embedding" in comps[t]
        assert comps[t]["embedding"].shape == (3, 16)
