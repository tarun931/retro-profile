"""Search-time configuration for retrieval heads.

Kept as a separate dataclass so it can be instantiated from Hydra (`_target_`)
and passed around without coupling to the FAISS implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Hyperparameters for kNN search at inference time.

    Attributes:
        k:           Number of neighbours retrieved per query.
        metric:      "cosine" or "l2". Must match the datastore's metric.
        weighting:   How to weight the retrieved labels. Either
                     "softmax_distance" or "uniform".
        temperature: Softmax temperature when weighting='softmax_distance'.
                     Smaller values place more mass on the closest neighbour.
    """

    k: int = 20
    metric: str = "cosine"
    weighting: str = "softmax_distance"
    temperature: float = 0.1
