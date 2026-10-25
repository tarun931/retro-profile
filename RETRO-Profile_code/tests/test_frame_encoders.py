"""Tests for the frame-level encoders used by the Gupta 2022 baseline.

The base logic (frame-mask downsampling, freeze/no_grad, the extract_frames
contract) is tested with a tiny stub HF model -- fast and offline. The real
wav2vec2/WavLM wrappers download ~360 MB, so that check is opt-in via the
RUN_HF_ENCODER_TEST=1 env var (run it on Colab, skip it locally).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from retro_profile.models.encoders.frame_encoders import _HFFrameEncoder


class _StubHF(nn.Module):
    """Minimal stand-in for an HF SSL model: returns (N, T', D) last_hidden_state."""

    def __init__(self, hidden=8, tprime=5):
        super().__init__()
        self.tprime = tprime
        self.proj = nn.Linear(1, hidden)          # a real param, so freezing matters

    def forward(self, input_values, attention_mask=None):
        n = input_values.shape[0]
        x = input_values.mean(dim=1).view(n, 1, 1).expand(n, self.tprime, 1)
        return type("Out", (), {"last_hidden_state": self.proj(x)})()   # (N, tprime, hidden)


# ---------- frame-mask downsampling ---------------------------------------


def test_downsample_mask_passthrough_when_same_length():
    mask = torch.tensor([[True, True, False, False]])
    out = _HFFrameEncoder._downsample_mask(mask, 4)
    assert out.dtype == torch.bool and out.tolist() == [[True, True, False, False]]


def test_downsample_mask_all_true_and_all_false_are_preserved():
    all_true = torch.ones(2, 16, dtype=torch.bool)
    all_false = torch.zeros(2, 16, dtype=torch.bool)
    assert _HFFrameEncoder._downsample_mask(all_true, 4).all()
    assert not _HFFrameEncoder._downsample_mask(all_false, 4).any()


def test_downsample_mask_changes_length():
    mask = torch.ones(3, 16000, dtype=torch.bool)
    out = _HFFrameEncoder._downsample_mask(mask, 50)
    assert out.shape == (3, 50) and out.dtype == torch.bool


# ---------- extract_frames contract (stub model) --------------------------


def test_extract_frames_shapes_and_mask():
    enc = _HFFrameEncoder(model=_StubHF(hidden=8, tprime=5), output_dim=8, freeze=True)
    wav = torch.randn(4, 1600)
    sample_mask = torch.ones(4, 1600, dtype=torch.bool)
    feats, frame_mask = enc.extract_frames(wav, sample_mask)
    assert feats.shape == (4, 5, 8)
    assert frame_mask.shape == (4, 5) and frame_mask.dtype == torch.bool
    assert enc.output_dim == 8


def test_frozen_encoder_produces_no_grad_and_disables_params():
    enc = _HFFrameEncoder(model=_StubHF(), output_dim=8, freeze=True)
    assert all(not p.requires_grad for p in enc.parameters())
    feats, _ = enc.extract_frames(torch.randn(2, 800), torch.ones(2, 800, dtype=torch.bool))
    assert not feats.requires_grad                       # forward ran under no_grad


def test_trainable_encoder_allows_gradients():
    enc = _HFFrameEncoder(model=_StubHF(), output_dim=8, freeze=False)
    assert any(p.requires_grad for p in enc.parameters())
    feats, _ = enc.extract_frames(torch.randn(2, 800), torch.ones(2, 800, dtype=torch.bool))
    assert feats.requires_grad                           # gradients can flow


def test_extract_frames_without_mask_returns_all_true():
    enc = _HFFrameEncoder(model=_StubHF(tprime=6), output_dim=8, freeze=True)
    feats, frame_mask = enc.extract_frames(torch.randn(2, 800), None)
    assert frame_mask.shape == (2, 6) and frame_mask.all()


# ---------- real HF encoders (opt-in; run on Colab) -----------------------


@pytest.mark.skipif(
    os.environ.get("RUN_HF_ENCODER_TEST") != "1",
    reason="set RUN_HF_ENCODER_TEST=1 to download and test the real HF encoders",
)
def test_real_encoders_extract_frames():
    from retro_profile.models.encoders.frame_encoders import (
        Wav2Vec2FrameEncoder,
        WavLMFrameEncoder,
    )

    wav = torch.randn(2, 16000)                          # 1 s @ 16 kHz
    mask = torch.ones(2, 16000, dtype=torch.bool)
    for enc in (WavLMFrameEncoder(freeze=True),
                Wav2Vec2FrameEncoder(freeze=False, gradient_checkpointing=False)):
        feats, fmask = enc.extract_frames(wav, mask)
        assert enc.output_dim == 768
        assert feats.shape[0] == 2 and feats.shape[-1] == 768
        assert fmask.shape == feats.shape[:2]
