"""Model layer: encoders, pooling, heads, baselines, and the full RETRO-Profile."""

from retro_profile.models.base import BaseProfileModel
from retro_profile.models.baselines import (
    Gupta2022Baseline,
    Jaid2023Baseline,
)
from retro_profile.models.encoders import (
    BaseEncoder,
    Wav2Vec2Encoder,
    WavLMEncoder,
)
from retro_profile.models.heads import MLPHead, ResidualRetrievalHead, RetrievalHead
from retro_profile.models.pooling import AttentionPooling, MeanPooling, MILAttentionPooling
from retro_profile.models.registry import build_model, models
from retro_profile.models.retro_profile import RetroProfile

__all__ = [
    # Base
    "BaseProfileModel",
    # Encoders
    "BaseEncoder",
    "WavLMEncoder",
    "Wav2Vec2Encoder",
    # Pooling
    "MeanPooling",
    "AttentionPooling",
    "MILAttentionPooling",
    # Heads
    "MLPHead",
    "ResidualRetrievalHead",
    "RetrievalHead",
    # Baselines
    "Gupta2022Baseline",
    "Jaid2023Baseline",
    # Main
    "RetroProfile",
    # Registry
    "models",
    "build_model",
]
