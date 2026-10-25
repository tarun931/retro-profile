"""Pure kNN retrieval head — no learned residual.

Useful as an ablation: how much of RETRO-Profile's gain comes from the residual
correction vs. from retrieval interpolation alone?
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.retrieval.datastore import RetrievalDatastore


class RetrievalHead(nn.Module):
    """Predict ŷ = kNN-weighted mean of training labels. No learned parameters."""

    def __init__(
        self,
        input_dim: int,
        label_min: float = 0.0,
        label_max: float = 250.0,
        knn_k: int = 20,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.label_min = label_min
        self.label_max = label_max
        self.knn_k = knn_k
        self._datastore: RetrievalDatastore | None = None
        self._target_name: str | None = None
        # Required so PyTorch sees this as a leaf module with state_dict; we add
        # a single 1-element no-op buffer to keep PL happy on save/load.
        self.register_buffer("_noop", torch.zeros(1), persistent=False)

    def set_datastore(self, datastore: RetrievalDatastore, target_name: str) -> None:
        self._datastore = datastore
        self._target_name = target_name

    @property
    def has_datastore(self) -> bool:
        return self._datastore is not None and self._datastore.is_built

    def forward(
        self,
        speaker_emb: torch.Tensor,
        retrieval_emb: torch.Tensor | None = None,
        speaker_ids: list[str] | None = None,
    ) -> torch.Tensor:
        if not self.has_datastore:
            # No datastore yet -> return mid-range as a safe default. Train won't
            # learn anything from this; users should pretrain a different head
            # first, then build the datastore, then swap to RetrievalHead.
            mid = 0.5 * (self.label_min + self.label_max)
            return torch.full((speaker_emb.shape[0],), mid, device=speaker_emb.device)
        assert self._datastore is not None and self._target_name is not None
        query = retrieval_emb if retrieval_emb is not None else speaker_emb
        pred = self._datastore.knn_anchor(
            query, target=self._target_name, k=self.knn_k,
            exclude_speaker_ids=speaker_ids,
        )
        return pred.clamp(self.label_min, self.label_max)
