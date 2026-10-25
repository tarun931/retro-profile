"""Baseline: AnatomyNet — physics-inspired multi-task speech regressor.

This is a SUPPLEMENTARY baseline, NOT one of the proposal's three RQ1
comparison systems (those are wavlm_mlp, gupta2022, jaid2023). It is kept
around for the redundancy comparison: an earlier internal proposal whose
redundancy issue motivated RETRO-Profile. Retained as a STUB so it can be
re-trained and the redundancy claim confirmed with apples-to-apples numbers.

STUB. Implement inside ``forward``.
"""

from __future__ import annotations

import torch

from retro_profile.data.collate import BagBatch
from retro_profile.models.base import BaseProfileModel


class AnatomyNetBaseline(BaseProfileModel):
    """STUB: physics-inspired multitask baseline (supplementary, non-RQ1)."""

    def __init__(
        self,
        target_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(target_names=target_names or ["age", "height"])
        self._kwargs = kwargs

    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "AnatomyNetBaseline.forward() is a stub. Implement the original "
            "physics-inspired pipeline here."
        )
