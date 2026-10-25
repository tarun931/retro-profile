"""Phase 7 checks: the eval path works for the Gupta baseline.

Covers (1) the evaluator running on a Gupta model with no datastore and no
forward_with_components, and (2) the LightningModule-agnostic weight load used by
evaluate.py -- including restoration of the label-stat buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.evaluation import evaluate_with_explanations
from retro_profile.models.baselines.gupta2022 import Gupta2022Baseline
from retro_profile.models.encoders.frame_encoders import _HFFrameEncoder

LABEL_STATS = {"age": {"mean": 40.0, "std": 12.0}, "height": {"mean": 175.0, "std": 9.0}}


class _StubFrameEncoder(nn.Module):
    def __init__(self, output_dim=16, tprime=6):
        super().__init__()
        self.output_dim = output_dim
        self.tprime = tprime
        self.proj = nn.Linear(1, output_dim)

    def extract_frames(self, waveforms, sample_mask):
        x = waveforms[:, : self.tprime].unsqueeze(-1)
        return self.proj(x), _HFFrameEncoder._downsample_mask(sample_mask, self.tprime)


def _model(label_stats=LABEL_STATS):
    return Gupta2022Baseline(
        encoder=_StubFrameEncoder(16, 6), feature_dim=16, num_layers=1, nhead=2,
        dim_feedforward=32, dropout=0.0, fc_dim=8, label_stats=label_stats,
    )


def _bag_batch(B=3):
    bags = [
        SpeakerBag([torch.randn(1600), torch.randn(1200)], 16000, f"s{i}", ["u0", "u1"],
                   {"age": 25.0 + 5 * i, "height": 168.0 + 4 * i})
        for i in range(B)
    ]
    return collate_bags(bags, targets=("age", "height"))


def test_evaluator_runs_on_gupta_model():
    """No datastore + no forward_with_components -> evaluator uses model(batch)."""
    model = _model().eval()
    batch = _bag_batch()
    df, metrics = evaluate_with_explanations(
        model=model, dataloader=[batch], target_names=["age", "height"],
        datastore=None, device="cpu",
    )
    assert len(df) == 3 * 2                                # 3 speakers x 2 targets
    assert set(df["target"]) == {"age", "height"}
    assert "height/mae" in metrics and "age/mae" in metrics
    assert metrics["height/mae"] > 1.0                    # real cm, not z-scored


def test_label_stat_buffers_survive_checkpoint_roundtrip():
    """evaluate.py loads model weights directly; label-stat buffers must restore
    so denormalization matches the trained model."""
    trained = _model(LABEL_STATS)
    # simulate a Lightning checkpoint state_dict (keys prefixed with 'model.')
    state = {f"model.{k}": v for k, v in trained.state_dict().items()}

    fresh = _model(label_stats=None)                      # identity stats (mean 0, std 1)
    assert float(fresh._age_mean) == 0.0
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    fresh.load_state_dict(model_state, strict=False)
    assert float(fresh._age_mean) == 40.0                 # restored from checkpoint
    assert float(fresh._height_std) == 9.0

    # and predictions now match the trained model exactly
    trained.eval(); fresh.eval()
    batch = _bag_batch()
    with torch.no_grad():
        assert torch.allclose(trained(batch)["age"], fresh(batch)["age"], atol=1e-5)
