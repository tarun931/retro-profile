"""Optimizer & scheduler factories."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler


def build_optimizer(
    parameters,
    name: str = "adamw",
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    **kwargs: Any,
) -> Optimizer:
    if name.lower() == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    if name.lower() == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    if name.lower() == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def linear_warmup_cosine(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> _LRScheduler:
    """Linear warmup followed by cosine decay to ``min_lr_ratio * lr``."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_scheduler(
    optimizer: Optimizer,
    name: str | None = "warmup_cosine",
    warmup_steps: int = 500,
    total_steps: int = 50_000,
    min_lr_ratio: float = 0.05,
    **kwargs: Any,
) -> _LRScheduler | None:
    if name is None or name.lower() == "none":
        return None
    if name.lower() == "warmup_cosine":
        return linear_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio)
    if name.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")
