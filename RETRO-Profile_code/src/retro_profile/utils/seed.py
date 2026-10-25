"""Reproducibility seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Seed value.
        deterministic: If True, also force cuDNN to deterministic mode. Slower but
            reproducible bit-for-bit on the same hardware.

    Returns:
        The seed that was set, for logging convenience.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS_WORKSPACE_CONFIG must be set for use_deterministic_algorithms on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)

    return seed
