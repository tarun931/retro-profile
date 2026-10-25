"""Standard regression losses with NaN-aware masking.

Why mask? HeightCeleb has only height labels, NISP has both, TIMIT has both —
so when training cross-corpus or fine-tuning on HeightCeleb, the age column is
NaN for many rows. We compute the loss only where labels are present.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedRegressionLoss(nn.Module):
    """MSE / L1 / Huber loss that ignores NaN labels.

    Args:
        kind: "mse" | "l1" | "smooth_l1" (Huber).
        delta: Beta parameter for smooth_l1.
    """

    def __init__(self, kind: str = "smooth_l1", delta: float = 1.0) -> None:
        super().__init__()
        self.kind = kind
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,   # bool or float; True/1.0 = include
    ) -> torch.Tensor:
        if mask is None:
            mask = ~torch.isnan(target)
        else:
            mask = mask & ~torch.isnan(target)
        m = mask.to(pred.dtype)

        # Replace NaNs with zeros so the elementwise loss is finite; we mask the
        # contribution out anyway.
        target_safe = torch.where(mask, target, torch.zeros_like(target))

        if self.kind == "mse":
            elem = F.mse_loss(pred, target_safe, reduction="none")
        elif self.kind == "l1":
            elem = F.l1_loss(pred, target_safe, reduction="none")
        elif self.kind == "smooth_l1":
            elem = F.smooth_l1_loss(pred, target_safe, beta=self.delta, reduction="none")
        else:
            raise ValueError(f"Unknown regression loss: {self.kind!r}")

        denom = m.sum().clamp_min(1.0)
        return (elem * m).sum() / denom
