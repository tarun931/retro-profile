"""Which SSL checkpoints may be given an attention mask.

`facebook/wav2vec2-base` has `feat_extract_norm == "group"`: it was pretrained
WITHOUT an attention mask, and passing one degrades its representations. WavLM
Base+ has "layer" and expects one. `frame_encoders.py` has always applied this
rule; `WavLMEncoder` did not, so swapping its checkpoint to wav2vec2 would have
silently fed the model an input it never saw in pretraining — invalidating the
encoder-controlled comparison these tests exist to enable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from retro_profile.models.encoders.wavlm import Wav2Vec2Encoder, WavLMEncoder


class _StubHF(nn.Module):
    """Minimal stand-in for an HF SSL model; records the mask it was given."""

    def __init__(self, feat_extract_norm: str, hidden: int = 8, layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers,
                                      feat_extract_norm=feat_extract_norm)
        self.proj = nn.Linear(1, hidden)
        self.last_attention_mask = "unset"

    def forward(self, input_values, attention_mask=None):
        self.last_attention_mask = attention_mask
        n, t = input_values.shape
        tprime = max(1, t // 4)
        h = self.proj(input_values[:, :tprime].unsqueeze(-1))
        return SimpleNamespace(hidden_states=(h, h, h), last_hidden_state=h)


def _encoder(cls, feat_extract_norm, monkeypatch):
    stub = _StubHF(feat_extract_norm)
    monkeypatch.setattr("retro_profile.models.encoders.wavlm.AutoModel",
                        SimpleNamespace(from_pretrained=lambda *a, **k: stub))
    enc = cls(freeze=False)
    return enc, stub


def _run(enc):
    b, k, t = 2, 3, 16
    return enc(torch.randn(b, k, t), torch.ones(b, k, t, dtype=torch.bool),
               torch.ones(b, k, dtype=torch.bool))


def test_layer_norm_model_receives_a_mask(monkeypatch):
    """WavLM Base+ was pretrained with masks and must get one."""
    enc, stub = _encoder(WavLMEncoder, "layer", monkeypatch)
    assert enc.use_attention_mask is True
    _run(enc)
    assert stub.last_attention_mask is not None


def test_group_norm_model_receives_none(monkeypatch):
    """wav2vec2-base was not, and must not."""
    enc, stub = _encoder(WavLMEncoder, "group", monkeypatch)
    assert enc.use_attention_mask is False
    _run(enc)
    assert stub.last_attention_mask is None


def test_wav2vec2_encoder_defaults_to_the_base_checkpoint(monkeypatch):
    enc, _ = _encoder(Wav2Vec2Encoder, "group", monkeypatch)
    assert enc.pretrained_name == "facebook/wav2vec2-base"
    assert enc.use_attention_mask is False


def test_output_shape_is_unaffected_by_the_mask_rule(monkeypatch):
    """Padded frames are still excluded via the downsampled frame mask."""
    for norm in ("layer", "group"):
        enc, _ = _encoder(WavLMEncoder, norm, monkeypatch)
        out = _run(enc)
        assert out.shape == (2, 3, 8)
        assert torch.isfinite(out).all()


def test_padded_utterances_are_zeroed_for_both_norms(monkeypatch):
    for norm in ("layer", "group"):
        enc, _ = _encoder(WavLMEncoder, norm, monkeypatch)
        b, k, t = 2, 3, 16
        utt = torch.tensor([[True, True, False], [True, False, False]])
        out = enc(torch.randn(b, k, t), torch.ones(b, k, t, dtype=torch.bool), utt)
        assert torch.count_nonzero(out[0, 2]) == 0
        assert torch.count_nonzero(out[1, 1]) == 0


@pytest.mark.parametrize("agg", ["weighted", "mean", "last"])
def test_layer_aggregation_modes_all_run(monkeypatch, agg):
    stub = _StubHF("group")
    monkeypatch.setattr("retro_profile.models.encoders.wavlm.AutoModel",
                        SimpleNamespace(from_pretrained=lambda *a, **k: stub))
    enc = Wav2Vec2Encoder(freeze=False, layer_aggregation=agg)
    assert _run(enc).shape == (2, 3, 8)
