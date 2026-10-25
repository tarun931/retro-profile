"""Baseline: WavLM + regression MLP (RQ1 baseline #1).

Proposal reference:
    RETRO-Profile proposal, Section 4.5 — RQ1 compares RETRO-Profile against
    "WavLM coupled with a regression MLP". This is the plain end-to-end
    regression baseline: no MIL pooling, no retrieval, no ordinal contrastive
    loss — just the frozen SSL front-end and a per-target MLP.

This is a STUB — only the class skeleton and config-compatible signature are
provided. Implement the real architecture inside ``forward`` (and optionally
``embed_speaker``). The training loop, datasets, and metrics work unchanged
once those are filled in.

Sketch of the intended method:
    * WavLM Base+ frozen, weighted layer aggregation
    * mean-pool over time, mean-pool over utterances (per speaker)
    * one MLP regression head per target (age, height)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.data.collate import BagBatch
from retro_profile.models.base import BaseProfileModel


class WavLMMLPBaseline(BaseProfileModel):
    """STUB: frozen WavLM + per-target regression MLP (RQ1 baseline #1)."""

    def __init__(
        self,
        encoder: nn.Module | None = None,
        target_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(target_names=target_names or ["age", "height"])
        self.encoder = encoder
        self._kwargs = kwargs
        # TODO(researcher): allocate the per-target regression heads here.

    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        # TODO(researcher): implement the WavLM + MLP forward pass.
        raise NotImplementedError(
            "WavLMMLPBaseline.forward() is a stub. Implement the frozen-WavLM "
            "plus per-target regression MLP pipeline here."
        )

    def embed_speaker(self, batch: BagBatch) -> torch.Tensor:
        # Optional — only needed if you want to plug this baseline into the
        # kNN datastore for an apples-to-apples retrieval comparison.
        raise NotImplementedError
