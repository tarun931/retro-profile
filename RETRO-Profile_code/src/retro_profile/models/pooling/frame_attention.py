"""Frame-level attention pooling -- the first level of hierarchical attention.

Every encoder reduces a clip to one vector by a masked MEAN over its frames, so
each frame counts equally however much it says about the speaker. This module
replaces that mean with a learned weighting of the frames (Ilse-style gated
attention, the same form the clip-level MIL pooling uses). With it switched on
the model attends twice: over frames within each clip (here, shared by every
target) and then over clips within the bag (``MILAttentionPooling``, per target).

The scoring layer is zero-initialised, so at initialisation every real frame
gets the same weight and the output is exactly the masked mean. The model
therefore starts as the plain-mean model and only departs from it where the
data pays for it -- the same principle as the residual head's zero init.

A clip with no real frames (a padded slot in the bag) returns zeros, never NaN:
softmax over a row of -inf is NaN, and NaN * 0 is still NaN, so masking the
output afterwards would not have contained it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrameAttentionPooling(nn.Module):
    """Gated attention over the frames of each clip; returns one vector per clip.

    Args:
        input_dim:  D, the frame feature size.
        hidden_dim: hidden size of the attention MLP.
        dropout:    dropout on the frames fed to the scorer (not on the output).
        gated:      Ilse-style tanh * sigmoid gate (as in MILAttentionPooling).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.V = nn.Linear(input_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim) if gated else None
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.w.weight)          # uniform weights at init = masked mean
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        frames: torch.Tensor,        # (N, T', D)
        frame_mask: torch.Tensor,    # (N, T')  True / 1 for real frames
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pooled (N, D), weights (N, T')); weights sum to 1 over real frames."""
        mask = frame_mask.bool()
        h = self.dropout(frames)
        a = torch.tanh(self.V(h))
        if self.U is not None:
            a = a * torch.sigmoid(self.U(h))
        # Scores in float32: under fp16 autocast the masked fill and the softmax
        # would otherwise run at half precision.
        scores = self.w(a).squeeze(-1).float()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * mask
        # Rows with no real frame are all zero here; renormalising leaves them at 0.
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("nt,ntd->nd", weights.to(frames.dtype), frames)
        return pooled, weights
