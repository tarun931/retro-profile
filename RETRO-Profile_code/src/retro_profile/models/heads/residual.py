"""Residual retrieval head — Component 3 of RETRO-Profile.

This is the key novelty for the *prediction* side of the model. Instead of
regressing height directly from the speaker embedding (which forces an MLP to
extrapolate over the whole 140-210 cm range), we:

    1.  Compute an *anchor* by retrieving the K nearest training speakers from
        a FAISS datastore and taking a similarity-weighted mean of their labels.
    2.  Predict only the *residual* delta on top of that anchor with a small MLP.

The residual is bounded (we tanh-squash it to ±delta_clip) so the model cannot
go far from the retrieval interpolation. This converts an open-ended regression
problem into bounded interpolation between real observed humans, and makes
predictions forensic-defensible: every prediction can be explained by listing
the K retrieved neighbours.

At training time the datastore is empty, so the head falls back to plain
MLP regression. After training, ``scripts/build_datastore.py`` populates the
datastore from the train set, after which ``use_retrieval=True`` activates the
anchor + residual path automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.retrieval.datastore import RetrievalDatastore


class ResidualRetrievalHead(nn.Module):
    """Predict ŷ = anchor(z) + Δ(z), where:

      * anchor(z) is a kNN-weighted mean of training labels for the target,
      * Δ(z) is a small MLP residual, tanh-clipped to ±delta_clip.

    Args:
        input_dim:    Speaker-embedding dim D fed into the head.
        hidden_dim:   Hidden size of the residual MLP.
        output_dim:   Always 1 for scalar regression. Kept for API symmetry.
        label_min:    Hard lower clip applied at the very end (e.g. 140 cm).
        label_max:    Hard upper clip applied at the very end (e.g. 210 cm).
        delta_clip:   Tanh saturation for the residual. A value of 5 means the
                      residual can move the anchor by at most ~5 of the label's
                      original units (cm or years). Tune per-target.
        use_retrieval: If False, the head degrades to pure MLP regression; useful
                       for the ``no_knn`` ablation and for the early epochs of
                       training when no datastore exists yet.
        knn_k:        Default K used at inference if the datastore doesn't override.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 1,
        label_min: float = 0.0,
        label_max: float = 250.0,
        delta_clip: float = 5.0,
        use_retrieval: bool = True,
        knn_k: int = 20,
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError("ResidualRetrievalHead only supports scalar regression.")
        self.input_dim = input_dim
        self.label_min = label_min
        self.label_max = label_max
        self.delta_clip = delta_clip
        self.use_retrieval = use_retrieval
        self.knn_k = knn_k

        # Direct prediction branch: used when no datastore is attached.
        # We initialize the last layer's bias to the midpoint of the label range
        # so that, at random init, predictions start near the centre of the valid
        # range (e.g. ~50 yrs for age, ~175 cm for height) instead of getting
        # clamped to label_min by the final clamp().
        self.direct = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        midpoint = 0.5 * (label_min + label_max)
        nn.init.zeros_(self.direct[-1].weight)
        nn.init.constant_(self.direct[-1].bias, midpoint)

        # Residual branch: predicts Δ on top of the retrieval anchor.
        # Initialize the last layer to zeros so that, at random init, the residual
        # is ~0 and the prediction equals the kNN anchor — a strong prior.
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        # The datastore is set externally via ``set_datastore`` after training.
        self._datastore: RetrievalDatastore | None = None
        self._target_name: str | None = None

    # ---- datastore wiring --------------------------------------------------
    def set_datastore(self, datastore: RetrievalDatastore, target_name: str) -> None:
        """Attach a populated datastore for this head's target."""
        self._datastore = datastore
        self._target_name = target_name

    @property
    def has_datastore(self) -> bool:
        return self.use_retrieval and self._datastore is not None and self._datastore.is_built

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        speaker_emb: torch.Tensor,            # (B, D) — already projected (l2-norm if cosine)
        return_components: bool = False,
        retrieval_emb: torch.Tensor | None = None,   # (B, D) query used for kNN search
        speaker_ids: list[str] | None = None,        # for leave-one-out retrieval
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if not self.has_datastore:
            pred = self.direct(speaker_emb).squeeze(-1)
            pred = pred.clamp(self.label_min, self.label_max)
            if return_components:
                return {"pred": pred, "anchor": None, "delta": pred}
            return pred

        # Retrieval anchor: similarity-weighted mean of K nearest speakers' labels.
        # The kNN *query* must live in the same space the datastore was built in
        # (``model.embed_speaker``); the residual MLP still reads the per-target
        # embedding. When no explicit query is given, fall back to speaker_emb.
        assert self._datastore is not None and self._target_name is not None
        query = retrieval_emb if retrieval_emb is not None else speaker_emb
        anchor = self._datastore.knn_anchor(
            query, target=self._target_name, k=self.knn_k,
            exclude_speaker_ids=speaker_ids,
        )
        # Residual MLP, tanh-clipped to ±delta_clip.
        delta = torch.tanh(self.residual(speaker_emb).squeeze(-1)) * self.delta_clip
        pred = (anchor + delta).clamp(self.label_min, self.label_max)
        if return_components:
            return {"pred": pred, "anchor": anchor, "delta": delta}
        return pred
