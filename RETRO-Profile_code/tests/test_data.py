"""Tests for the data layer."""

from __future__ import annotations

import torch

from retro_profile.data.collate import collate_bags
from retro_profile.data.base import SpeakerBag


def test_collate_bag_shapes(fake_bag_batch):
    b = fake_bag_batch
    assert b.batch_size == 3
    assert b.waveforms.shape[0] == 3
    assert b.waveforms.shape[1] == 4   # K_max == 4 (third bag has 4 utts)
    assert b.waveforms.shape[2] == 2400  # T_max
    assert b.utt_mask.shape == (3, 4)
    assert b.sample_mask.shape == (3, 4, 2400)


def test_collate_masks_align_with_padding(fake_bag_batch):
    b = fake_bag_batch
    # Speaker 0 has 3 real utts, last column should be padded.
    assert b.utt_mask[0].tolist() == [True, True, True, False]
    # Speaker 1 has 2 real utts.
    assert b.utt_mask[1].tolist() == [True, True, False, False]
    # Speaker 2 has 4 real utts.
    assert b.utt_mask[2].tolist() == [True, True, True, True]


def test_collate_labels_present(fake_bag_batch):
    b = fake_bag_batch
    assert torch.allclose(b.labels["age"], torch.tensor([28.0, 45.0, 60.0]))
    assert torch.allclose(b.labels["height"], torch.tensor([178.5, 165.0, 190.0]))
    assert b.label_mask["age"].all()
    assert b.label_mask["height"].all()


def test_missing_label_becomes_nan_and_mask_zero():
    bags = [
        SpeakerBag(
            waveforms=[torch.randn(800)],
            sample_rate=16_000,
            speaker_id="x",
            utt_ids=["u0"],
            labels={"height": 178.0},  # age missing on purpose
        ),
    ]
    b = collate_bags(bags, targets=("age", "height"))
    assert torch.isnan(b.labels["age"]).all()
    assert b.label_mask["age"].tolist() == [False]
    assert b.label_mask["height"].tolist() == [True]


def test_pad_zeroed_audio(fake_bag_batch):
    """Padded sample positions must be exactly zero."""
    b = fake_bag_batch
    # Speaker 0, utterance 0 is 1600 samples; positions 1600.. should be 0.
    pad_region = b.waveforms[0, 0, 1600:]
    assert torch.all(pad_region == 0)
