"""Base class for self-supervised speech encoders.

All encoders consume a packed bag tensor of shape ``(B, K, T)`` and return
*per-utterance* embeddings of shape ``(B, K, D)``. They are responsible for:

  * resampling / padding internally if needed (we always feed 16 kHz);
  * mean-pooling the time axis of the SSL feature sequence (encoders return one
    embedding per utterance, not per frame — frame-level pooling for trait
    regression has not shown wins beyond utterance-level pooling, and saves
    a lot of memory in the bag setting);
  * masking out padded utterances using ``utt_mask``.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, abc.ABC):
    """Abstract base for SSL speech encoders."""

    output_dim: int  # D, the per-utterance embedding dimension

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    @abc.abstractmethod
    def forward(
        self,
        waveforms: torch.Tensor,         # (B, K, T)
        sample_mask: torch.Tensor,       # (B, K, T) bool — True for real samples
        utt_mask: torch.Tensor,          # (B, K)    bool — True for real utterances
    ) -> torch.Tensor:
        """Return per-utterance embeddings of shape (B, K, D)."""

    @staticmethod
    def _masked_mean_time(
        features: torch.Tensor,          # (BK, T', D)
        attention_mask: torch.Tensor,    # (BK, T') bool/long
    ) -> torch.Tensor:
        """Mean-pool a frame sequence along the time axis, respecting the mask."""
        m = attention_mask.to(features.dtype).unsqueeze(-1)  # (BK, T', 1)
        summed = (features * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        return summed / denom
