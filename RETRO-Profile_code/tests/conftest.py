"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def fake_bag_batch():
    """A small synthetic batch of 3 speakers x ≤4 utterances x ≤2400 samples."""
    bags = [
        SpeakerBag(
            waveforms=[torch.randn(1600), torch.randn(1200), torch.randn(2400)],
            sample_rate=16_000,
            speaker_id="spk_001",
            utt_ids=["u0", "u1", "u2"],
            labels={"age": 28.0, "height": 178.5},
        ),
        SpeakerBag(
            waveforms=[torch.randn(2000), torch.randn(800)],
            sample_rate=16_000,
            speaker_id="spk_002",
            utt_ids=["u0", "u1"],
            labels={"age": 45.0, "height": 165.0},
        ),
        SpeakerBag(
            waveforms=[torch.randn(1800), torch.randn(2200), torch.randn(1500), torch.randn(1100)],
            sample_rate=16_000,
            speaker_id="spk_003",
            utt_ids=["u0", "u1", "u2", "u3"],
            labels={"age": 60.0, "height": 190.0},
        ),
    ]
    return collate_bags(bags, targets=("age", "height"))
