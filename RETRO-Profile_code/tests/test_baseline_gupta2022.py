"""Smoke tests for the Gupta 2022 bi-encoder transformer mixture baseline.

Uses a tiny stub frame encoder (no wav2vec2 download) so it runs in ms on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.models.baselines.gupta2022 import Gupta2022Baseline
from retro_profile.models.encoders.frame_encoders import _HFFrameEncoder


class _StubFrameEncoder(nn.Module):
    """Frame encoder stub: waveform -> (N, tprime, output_dim) + realistic mask."""

    def __init__(self, output_dim=16, tprime=6):
        super().__init__()
        self.output_dim = output_dim
        self.tprime = tprime
        self.proj = nn.Linear(1, output_dim)

    def extract_frames(self, waveforms, sample_mask):
        x = waveforms[:, : self.tprime].unsqueeze(-1)              # (N, tprime, 1)
        feats = self.proj(x)                                      # (N, tprime, D)
        frame_mask = _HFFrameEncoder._downsample_mask(sample_mask, self.tprime)
        return feats, frame_mask


LABEL_STATS = {"age": {"mean": 40.0, "std": 12.0},
               "height": {"mean": 175.0, "std": 9.0}}


def _build_model():
    return Gupta2022Baseline(
        encoder=_StubFrameEncoder(output_dim=16, tprime=6),
        feature_dim=16, num_layers=1, nhead=2, dim_feedforward=32,
        dropout=0.0, fc_dim=8, label_stats=LABEL_STATS,
    )


def test_forward_returns_age_height_gender(fake_bag_batch):
    model = _build_model().eval()
    out = model(fake_bag_batch)
    assert set(out) == {"age", "height", "gender"}
    for k in ("age", "height", "gender"):
        assert out[k].shape == (3,)                              # B=3 speakers
        assert torch.isfinite(out[k]).all()


def test_gender_is_a_finite_logit(fake_bag_batch):
    """Model returns the gender LOGIT (for BCE-with-logits); sigmoid maps it to
    a valid [0,1] gate probability."""
    out = _build_model().eval()(fake_bag_batch)
    assert torch.isfinite(out["gender"]).all()
    gate = torch.sigmoid(out["gender"])
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


def test_predictions_are_in_real_units(fake_bag_batch):
    """With label stats set, denormalized preds should sit in a plausible real
    range (there is no clamp, so we only assert a loose sanity band)."""
    out = _build_model().eval()(fake_bag_batch)
    assert torch.all(out["age"] > 0) and torch.all(out["age"] < 120)
    assert torch.all(out["height"] > 100) and torch.all(out["height"] < 250)


def test_per_speaker_aggregation_is_mean_over_bag():
    """A speaker whose single utterance is repeated K times must give the same
    prediction as the single utterance (aggregation = masked mean; eval() so no
    dropout)."""
    torch.manual_seed(0)
    model = _build_model().eval()
    wav = torch.randn(1600)
    one = collate_bags([SpeakerBag([wav], 16000, "s", ["u0"],
                                   {"age": 30.0, "height": 170.0})], targets=("age", "height"))
    three = collate_bags([SpeakerBag([wav, wav, wav], 16000, "s", ["u0", "u1", "u2"],
                                     {"age": 30.0, "height": 170.0})], targets=("age", "height"))
    o1, o3 = model(one), model(three)
    assert torch.allclose(o1["age"], o3["age"], atol=1e-4)
    assert torch.allclose(o1["height"], o3["height"], atol=1e-4)


def test_embed_speaker_not_implemented(fake_bag_batch):
    """Gupta is not a retrieval method -> cannot build a kNN datastore."""
    with pytest.raises(NotImplementedError):
        _build_model().embed_speaker(fake_bag_batch)


# --- regression: transformer dropout is separate from pooled/FC dropout ------
# Gupta's own code runs PyTorch's default 0.1 inside the transformer encoders
# and reserves 0.5 for the pooled vectors. Passing one value into both put 0.5
# across all 12 encoder layers (6 per branch) and stopped the model learning --
# its best checkpoint was epoch 3 of 40. Do not re-merge these two arguments.

def _transformer_dropout_rates(model: Gupta2022Baseline) -> set[float]:
    """Every dropout probability living inside the two transformer stacks."""
    rates = set()
    for stack in (model.transformer_M, model.transformer_F):
        for m in stack.modules():
            if isinstance(m, nn.Dropout):
                rates.add(float(m.p))
    return rates


def test_transformer_dropout_is_independent_of_pooled_dropout():
    model = Gupta2022Baseline(
        encoder=_StubFrameEncoder(output_dim=16, tprime=6),
        feature_dim=16, num_layers=2, nhead=2, dim_feedforward=32,
        dropout=0.5, transformer_dropout=0.1, fc_dim=8, label_stats=LABEL_STATS,
    )
    assert _transformer_dropout_rates(model) == {0.1}, "0.5 leaked into the encoders"
    assert float(model.dropout.p) == 0.5, "pooled/FC dropout should stay at Gupta's 0.5"


def test_transformer_dropout_defaults_to_pytorch_default():
    """Default must be 0.1 even when a caller passes only `dropout`."""
    model = Gupta2022Baseline(
        encoder=_StubFrameEncoder(output_dim=16, tprime=6),
        feature_dim=16, num_layers=2, nhead=2, dim_feedforward=32,
        dropout=0.5, fc_dim=8, label_stats=LABEL_STATS,
    )
    assert _transformer_dropout_rates(model) == {0.1}
    assert float(model.dropout.p) == 0.5
