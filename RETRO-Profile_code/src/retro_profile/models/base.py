"""Abstract base for any speaker-trait regression model.

Defines a uniform interface so the trainer / evaluator / datastore-builder can
treat RetroProfile and the baseline stubs interchangeably.

Required methods
----------------
forward(batch)         -> dict {target_name: (B,) prediction tensor}
embed_speaker(batch)   -> (B, D) speaker embedding (used to build the kNN
                          datastore; baselines that don't support kNN can raise
                          NotImplementedError)
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn

from retro_profile.data.collate import BagBatch


class BaseProfileModel(nn.Module, abc.ABC):
    """Common contract for all speaker-trait regression models."""

    target_names: list[str]

    def __init__(self, target_names: list[str]) -> None:
        super().__init__()
        self.target_names = list(target_names)

    @abc.abstractmethod
    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        """Return one prediction per target, each of shape (B,)."""

    def embed_speaker(self, batch: BagBatch) -> torch.Tensor:
        """Return a (B, D) speaker embedding suitable for the FAISS datastore.

        Default: not supported; baselines that want kNN must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement embed_speaker(); "
            f"it cannot be used to build a retrieval datastore."
        )
