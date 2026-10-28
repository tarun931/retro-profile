"""Model registry for non-Hydra entrypoints (build by short name)."""

from __future__ import annotations

from retro_profile.models.baselines import (
    Gupta2022Baseline,
    Jaid2023Baseline,
)
from retro_profile.models.retro_profile import RetroProfile
from retro_profile.utils.registry import Registry

models = Registry("models")
models._registry["retro_profile"] = RetroProfile
models._registry["baseline_gupta2022"] = Gupta2022Baseline
models._registry["baseline_jaid2023"] = Jaid2023Baseline


def build_model(name: str, **kwargs):
    return models.build(name, **kwargs)
