"""Per-speaker clip sampling for utterance-level training on corpora with many clips per speaker.

Gupta 2022 and Jaid 2023 train per clip, one pass over every clip per epoch. On TIMIT (10 clips
per speaker) that matches a bag model's 8 clips per speaker; on HeightCeleb (median ~108) it is
~13x more audio than RETRO sees. This sampler draws up to ``n`` fresh random clips per speaker
each epoch, so both kinds of model see the same audio per speaker per epoch (agreed 2026-10-12).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pandas as pd
import torch
from torch.utils.data import Sampler


class PerSpeakerUtteranceSampler(Sampler[int]):
    """Each epoch: up to ``n`` random row indices per speaker, all speakers shuffled together.

    Uses torch's global RNG, so ``seed_everything`` makes the draw reproducible."""

    def __init__(self, speaker_ids: Sequence[str], n: int) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        rows = pd.Series(range(len(speaker_ids))).groupby(pd.Series(list(speaker_ids)).astype(str))
        self._groups = [torch.as_tensor(g.to_numpy()) for _, g in rows]

    def __len__(self) -> int:
        return sum(min(self.n, len(g)) for g in self._groups)

    def __iter__(self) -> Iterator[int]:
        picks = torch.cat([g[torch.randperm(len(g))[: self.n]] for g in self._groups])
        return iter(picks[torch.randperm(len(picks))].tolist())
