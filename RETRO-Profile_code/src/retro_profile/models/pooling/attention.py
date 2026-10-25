"""Standard (single-head) attention pooling.

Simpler than MIL pooling — there is no per-trait specialization, no gated
modulation. Useful as an ablation point: "does the gain come from attention
itself, or from the MIL-specific design?"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Single-head additive-attention pooling over a bag of utterance embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        utt_emb: torch.Tensor,        # (B, K, D)
        utt_mask: torch.Tensor,       # (B, K)
        target: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attn(utt_emb).squeeze(-1)              # (B, K)
        scores = scores.masked_fill(~utt_mask, -math.inf)
        weights = F.softmax(scores, dim=1)                    # (B, K)
        speaker_emb = torch.einsum("bk,bkd->bd", weights, utt_emb)
        return speaker_emb, weights
