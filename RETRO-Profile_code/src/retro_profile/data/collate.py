"""Bag-aware collation for variable-length speaker bags.

A batch is a list of ``SpeakerBag`` objects with potentially different bag sizes
and different per-utterance lengths. We pack them into a single tensor of shape
``(B, K_max, T_max)`` plus boolean masks so that downstream layers (encoders,
MIL pooling) can ignore padding cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from retro_profile.data.base import SpeakerBag, SpeakerExample


@dataclass(slots=True)
class BagBatch:
    """A batched collection of speaker bags.

    Attributes:
        waveforms:    (B, K_max, T_max) float tensor, zero-padded.
        utt_mask:     (B, K_max) bool, True where an utterance exists.
        sample_mask:  (B, K_max, T_max) bool, True for non-padded audio samples.
        labels:       dict[str, FloatTensor (B,)] — speaker-level labels per target.
        label_mask:   dict[str, BoolTensor (B,)] — True where the label is present.
        speaker_ids:  list[str] of length B.
        utt_ids:      list[list[str]] of length B; inner lists may differ in length.
    """

    waveforms: torch.Tensor
    utt_mask: torch.Tensor
    sample_mask: torch.Tensor
    labels: dict[str, torch.Tensor]
    label_mask: dict[str, torch.Tensor]
    speaker_ids: list[str]
    utt_ids: list[list[str]]

    def to(self, device: torch.device | str) -> "BagBatch":
        return BagBatch(
            waveforms=self.waveforms.to(device, non_blocking=True),
            utt_mask=self.utt_mask.to(device, non_blocking=True),
            sample_mask=self.sample_mask.to(device, non_blocking=True),
            labels={k: v.to(device, non_blocking=True) for k, v in self.labels.items()},
            label_mask={k: v.to(device, non_blocking=True) for k, v in self.label_mask.items()},
            speaker_ids=self.speaker_ids,
            utt_ids=self.utt_ids,
        )

    @property
    def batch_size(self) -> int:
        return self.waveforms.shape[0]


def collate_bags(
    bags: Sequence[SpeakerBag],
    targets: Sequence[str] = ("age", "height"),
) -> BagBatch:
    """Collate a list of ``SpeakerBag`` into a ``BagBatch``."""
    if not bags:
        raise ValueError("Cannot collate an empty list of bags.")

    B = len(bags)
    K_max = max(len(b.waveforms) for b in bags)
    T_max = max(w.shape[-1] for b in bags for w in b.waveforms)

    waveforms = torch.zeros((B, K_max, T_max), dtype=torch.float32)
    utt_mask = torch.zeros((B, K_max), dtype=torch.bool)
    sample_mask = torch.zeros((B, K_max, T_max), dtype=torch.bool)

    for i, bag in enumerate(bags):
        for k, w in enumerate(bag.waveforms):
            t = w.shape[-1]
            waveforms[i, k, :t] = w
            sample_mask[i, k, :t] = True
            utt_mask[i, k] = True

    # Speaker-level labels
    labels: dict[str, torch.Tensor] = {}
    label_mask: dict[str, torch.Tensor] = {}
    for t in targets:
        vals = torch.full((B,), float("nan"), dtype=torch.float32)
        present = torch.zeros((B,), dtype=torch.bool)
        for i, bag in enumerate(bags):
            v = bag.labels.get(t)
            if v is not None:
                vals[i] = float(v)
                present[i] = True
        labels[t] = vals
        label_mask[t] = present

    return BagBatch(
        waveforms=waveforms,
        utt_mask=utt_mask,
        sample_mask=sample_mask,
        labels=labels,
        label_mask=label_mask,
        speaker_ids=[b.speaker_id for b in bags],
        utt_ids=[list(b.utt_ids) for b in bags],
    )


def collate_examples(
    examples: Sequence[SpeakerExample],
    targets: Sequence[str] = ("age", "height"),
) -> BagBatch:
    """Treat a list of single utterances as a batch of bag-size-1 bags.

    This lets the same model code run for utterance-level inference (e.g. when
    building the kNN datastore from the training set utterance-by-utterance and
    averaging per speaker afterwards).
    """
    bags = [
        SpeakerBag(
            waveforms=[ex.waveform],
            sample_rate=ex.sample_rate,
            speaker_id=ex.speaker_id,
            utt_ids=[ex.utt_id],
            labels=ex.labels,
        )
        for ex in examples
    ]
    return collate_bags(bags, targets=targets)


def make_collate_fn(targets: Sequence[str], mode: str = "bag"):
    """Factory that returns a callable suitable for ``DataLoader(collate_fn=...)``."""
    targets = tuple(targets)
    if mode == "bag":
        return lambda batch: collate_bags(batch, targets=targets)
    if mode == "utterance":
        return lambda batch: collate_examples(batch, targets=targets)
    raise ValueError(f"Unknown collate mode: {mode!r}")
