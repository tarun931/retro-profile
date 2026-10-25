"""Single-target (age-only) RETRO-Profile must build, train and select.

`configs/experiment/retro_age_only_budget.yaml` drops height entirely to test
whether OrdCon does better when the shared projector (CLAUDE.md §5) only has to
order the space along ONE axis instead of two partly-conflicting ones.

Everything in this codebase has been exercised with exactly two regression
targets. These tests pin the places a one-target model could quietly break:
`RetroProfile` derives `target_names` from `heads.keys()` and averages pooled
embeddings across them, `MultitaskLoss` builds an OrdCon term per target, and
`RegressionMetrics.combined_mae` sums over whatever targets are present.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.losses.multitask import MultitaskLoss
from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.heads.residual import ResidualRetrievalHead
from retro_profile.models.heads.mlp import MLPHead
from retro_profile.models.pooling import MILAttentionPooling
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.training.metrics import RegressionMetrics


class _TinyEncoder(BaseEncoder):
    def __init__(self, output_dim: int = 32):
        super().__init__(output_dim=output_dim)
        self.proj = nn.Linear(1, output_dim)

    def forward(self, waveforms, sample_mask, utt_mask):
        m = sample_mask.to(waveforms.dtype)
        avg = (waveforms * m).sum(-1, keepdim=True) / m.sum(-1, keepdim=True).clamp_min(1.0)
        return self.proj(avg) * utt_mask.unsqueeze(-1).to(waveforms.dtype)


def _age_only_model() -> RetroProfile:
    return RetroProfile(
        encoder=_TinyEncoder(32),
        pooling=MILAttentionPooling(
            input_dim=32, hidden_dim=16, num_heads=2, target_names=["age"]
        ),
        projector=MLPHead(input_dim=32, hidden_dim=32, output_dim=16, l2_normalize=True),
        heads={
            "age": ResidualRetrievalHead(
                input_dim=16, hidden_dim=8, label_min=18.0, label_max=80.0,
                use_retrieval=False,
            )
        },
    )


def test_age_only_model_builds_and_runs(fake_bag_batch):
    model = _age_only_model()
    assert model.target_names == ["age"]

    out = model(fake_bag_batch)
    preds = out["predictions"] if "predictions" in out else out
    assert "age" in preds
    assert "height" not in preds
    assert preds["age"].shape[0] == fake_bag_batch.batch_size
    assert torch.isfinite(preds["age"]).all()


def test_age_only_predictions_are_in_real_units(fake_bag_batch):
    """CLAUDE.md §1: heads emit real label units and clamp to [min, max].

    §2: the direct MLP's final layer is zero-weight with a midpoint bias, so a
    freshly built head must sit at the midpoint, not pinned at label_min.
    """
    model = _age_only_model()
    out = model(fake_bag_batch)
    preds = (out["predictions"] if "predictions" in out else out)["age"]
    assert (preds >= 18.0).all() and (preds <= 80.0).all()
    assert torch.allclose(preds, torch.full_like(preds, 49.0), atol=1e-4)


def test_ordcon_survives_a_single_target(fake_bag_batch):
    """One target means one OrdCon axis -- the loss must still produce a gradient."""
    model = _age_only_model()
    loss_fn = MultitaskLoss(
        target_names=["age"], embedding_dim=16, ordcon_weight=0.3,
    )
    out = model(fake_bag_batch)
    preds = out["predictions"] if "predictions" in out else out
    embeddings = out.get("embeddings")

    labels = {"age": torch.linspace(20.0, 60.0, fake_bag_batch.batch_size)}
    mask = {"age": torch.ones_like(labels["age"], dtype=torch.bool)}

    components = loss_fn(preds, labels, mask, embeddings)
    total = components["total"]
    assert torch.isfinite(total)
    assert "ordcon/age" in components or "ordcon/total" in components
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert any(g.abs().sum() > 0 for g in grads)


def test_combined_mae_reduces_to_age_alone():
    """Selection must not silently become 'no targets present' for a 1-target run."""
    m = RegressionMetrics(["age"])
    labels = {"age": torch.tensor([30.0, 40.0])}
    preds = {"age": torch.tensor([32.0, 40.0])}
    m.update(preds, labels, {"age": torch.ones(2, dtype=torch.bool)})
    out = m.compute()
    assert "combined_mae" in out
    assert abs(out["combined_mae"] - 1.0 / 5.91) < 1e-6
