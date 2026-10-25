"""Generic MLP regression / projection head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    """Multi-layer perceptron with GELU / dropout / optional L2-normalized output.

    Used both for:
      * direct regression (output_dim=1) when no retrieval is enabled
      * the embedding projector applied before OrdCon and FAISS indexing
        (l2_normalize=True so cosine retrieval is well-defined)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_hidden: int = 1,
        dropout: float = 0.1,
        l2_normalize: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}.get(activation, nn.GELU)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), act(), nn.Dropout(dropout)]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.l2_normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out
