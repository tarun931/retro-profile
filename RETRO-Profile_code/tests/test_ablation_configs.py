"""Verify each ablation config composes into the model it claims to be.

Cheap insurance: a typo in one of these YAMLs would otherwise only surface after
an hour of GPU time, and could silently produce a mislabelled ablation row.
"""

from __future__ import annotations

import os

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "configs")
)

# experiment -> (pooling class, ordcon_weight, retrieval enabled)
EXPECTED = {
    "ablation_1_wavlm_mlp": ("MeanPooling", 0.0, False),
    "ablation_2_mil": ("MILAttentionPooling", 0.0, False),
    "ablation_3_ordcon": ("MILAttentionPooling", 0.3, False),
    "ablation_4_full": ("MILAttentionPooling", 0.3, True),
}


def _cfg(experiment: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="default", overrides=[f"experiment={experiment}"])


@pytest.mark.parametrize("experiment,expected", EXPECTED.items())
def test_ablation_config_composes_as_intended(experiment, expected):
    pooling_cls, ordcon_weight, retrieval = expected
    cfg = _cfg(experiment)

    assert cfg.model.pooling._target_.endswith(pooling_cls)
    assert float(cfg.training.loss.ordcon_weight) == ordcon_weight
    for target in ("age", "height"):
        assert bool(cfg.model.heads[target].use_retrieval) is retrieval


@pytest.mark.parametrize("experiment", list(EXPECTED))
def test_all_rows_share_the_same_data_budget(experiment):
    """Only the architecture may vary between rows -- never the data budget."""
    cfg = _cfg(experiment)
    assert int(cfg.data.max_audio_seconds) == 4
    assert int(cfg.data.batch_size) == 16
    assert int(cfg.data.bag_size) == 8
    assert bool(cfg.data.augmentation.enabled) is False


def test_only_the_full_row_uses_the_two_phase_curriculum():
    for experiment in ("ablation_1_wavlm_mlp", "ablation_2_mil", "ablation_3_ordcon"):
        cfg = _cfg(experiment)
        assert not bool(cfg.training.get("two_phase", {}).get("enabled", False))

    full = _cfg("ablation_4_full")
    assert bool(full.training.two_phase.enabled)
    # phase-2 checkpoint selection, so this row is comparable with rows 1-3
    assert bool(full.training.two_phase.reset_checkpoint_best)


def test_original_two_phase_experiment_keeps_legacy_selection():
    """The earlier experiment must stay reproducible: checkpoint reset stays off."""
    cfg = _cfg("retro_two_phase")
    assert bool(cfg.training.two_phase.enabled)
    assert not bool(cfg.training.two_phase.reset_checkpoint_best)
