"""Mean pooling — the baseline against which MIL is compared."""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """Bag-aware mean pooling. Returns one embedding per speaker."""

    def __init__(self, input_dim: int, **_: object) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(
        self,
        utt_emb: torch.Tensor,        # (B, K, D)
        utt_mask: torch.Tensor,       # (B, K)
        target: str | None = None,    # ignored — same pooling for every target
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = utt_mask.to(utt_emb.dtype).unsqueeze(-1)
        summed = (utt_emb * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        speaker_emb = summed / denom
        # Uniform attention weights for downstream introspection.
        attn = m.squeeze(-1) / m.squeeze(-1).sum(dim=1, keepdim=True).clamp_min(1e-6)
        return speaker_emb, attn
