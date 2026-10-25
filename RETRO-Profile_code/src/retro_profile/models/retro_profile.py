"""RETRO-Profile — the full architecture.

Wires together:

    waveform bag (B, K, T)
        --[encoder]-->         per-utterance embeddings (B, K, D_enc)
        --[MIL pooling]-->     speaker embedding   (B, D_enc) [per target]
        --[projector]-->       projected embedding (B, D_proj) [shared]
            ↳ used for OrdCon loss & FAISS indexing
        --[residual head]-->   prediction (B,) per target
            internally: anchor = kNN over datastore, delta = small MLP

The projector is shared across targets so that the OrdCon-organized embedding
space is *one* space the datastore indexes — different targets share the same
neighbours, only the labels they read out differ. This is what makes the
forensic-style explanation coherent ("the same retrieved speakers explain both
the predicted age and the predicted height").

Training-time vs. inference-time
--------------------------------
At training time the datastore is empty, so the residual heads degrade to plain
MLP regression. We still train the projector with OrdCon so that, when the
datastore is later populated, kNN over the projected space is meaningful. After
training, ``scripts/build_datastore.py`` populates the datastore and attaches
it to all retrieval-aware heads.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from retro_profile.data.collate import BagBatch
from retro_profile.models.base import BaseProfileModel
from retro_profile.models.heads.mlp import MLPHead


class RetroProfile(BaseProfileModel):
    """The full RETRO-Profile model (encoder + MIL + projector + retrieval heads)."""

    def __init__(
        self,
        encoder: nn.Module,                          # BaseEncoder
        pooling: nn.Module,                          # MILAttentionPooling (or simpler)
        heads: dict[str, nn.Module],                 # {target: ResidualRetrievalHead | MLPHead}
        projector: nn.Module | None = None,          # MLPHead with l2_normalize=True
        retrieval: Any = None,                       # SearchConfig (informational)
        **_: Any,
    ) -> None:
        target_names = list(heads.keys())
        super().__init__(target_names=target_names)

        self.encoder = encoder
        self.pooling = pooling
        self.heads = nn.ModuleDict(heads)

        # If no projector is provided, fall back to identity (still 2D so OrdCon
        # gets a stable space, but no extra parameters).
        if projector is None:
            projector = MLPHead(
                input_dim=getattr(pooling, "output_dim", encoder.output_dim),
                hidden_dim=getattr(pooling, "output_dim", encoder.output_dim),
                output_dim=getattr(pooling, "output_dim", encoder.output_dim),
                num_hidden=1,
                l2_normalize=True,
            )
        self.projector = projector

        self.retrieval_cfg = retrieval

    # ---- inference paths ---------------------------------------------------
    def _encode_and_pool(
        self, batch: BagBatch, target: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch and pool per the given target.

        Returns:
            speaker_emb: (B, D_enc) pre-projector
            attn_weights: (B, K)
        """
        utt_emb = self.encoder(batch.waveforms, batch.sample_mask, batch.utt_mask)
        out = self.pooling(utt_emb, batch.utt_mask, target=target)
        speaker_emb, attn = out
        return speaker_emb, attn

    def _project(self, speaker_emb: torch.Tensor) -> torch.Tensor:
        return self.projector(speaker_emb)

    # ---- BaseProfileModel API ---------------------------------------------
    def _pool_and_project(
        self, batch: BagBatch
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        """Encode once, pool per target, project, and build the retrieval query.

        Returns (projected_per_target, attention_per_target, retrieval_emb) where
        ``retrieval_emb`` is identical to what ``embed_speaker`` produces — the
        vector the FAISS datastore is built from — so kNN queries and datastore
        entries live in the *same* space.
        """
        utt_emb = self.encoder(batch.waveforms, batch.sample_mask, batch.utt_mask)

        pooled: dict[str, torch.Tensor] = {}
        attn: dict[str, torch.Tensor] = {}
        for t in self.heads:
            pooled[t], attn[t] = self.pooling(utt_emb, batch.utt_mask, target=t)

        projected = {t: self._project(p) for t, p in pooled.items()}
        # Shared retrieval query: average the per-target pooled vectors, then
        # project (exactly the embed_speaker recipe used to build the datastore).
        avg_pooled = torch.stack([pooled[t] for t in self.target_names], dim=0).mean(dim=0)
        retrieval_emb = self._project(avg_pooled)
        return projected, attn, retrieval_emb

    def forward(self, batch: BagBatch) -> dict[str, torch.Tensor]:
        """Predict every target. Returns {target: (B,)}."""
        projected, _, retrieval_emb = self._pool_and_project(batch)

        preds: dict[str, torch.Tensor] = {}
        for t, head in self.heads.items():
            # Retrieval heads take the per-target embedding for the residual MLP
            # plus the shared retrieval query (and speaker ids for leave-one-out).
            if hasattr(head, "has_datastore"):
                preds[t] = head(projected[t], retrieval_emb=retrieval_emb,
                                speaker_ids=batch.speaker_ids)
            else:
                preds[t] = head(projected[t]).squeeze(-1)
        return preds

    def forward_with_components(
        self, batch: BagBatch
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Like ``forward`` but also returns embeddings + attention + components.

        Returns a dict keyed by target with sub-dicts:
            {pred, anchor, delta, embedding, retrieval_embedding, attention}
        ``embedding`` is this target's projected vector (what its head reads);
        ``retrieval_embedding`` is the shared vector the datastore indexes, and
        is what the LightningModule applies OrdCon to.
        """
        projected, attn, retrieval_emb = self._pool_and_project(batch)
        out: dict[str, dict[str, torch.Tensor]] = {}
        for t, head in self.heads.items():
            if hasattr(head, "set_datastore"):
                comp = head(projected[t], return_components=True,
                            retrieval_emb=retrieval_emb, speaker_ids=batch.speaker_ids)
            else:
                comp = {"pred": head(projected[t]).squeeze(-1), "anchor": None, "delta": None}
            comp["embedding"] = projected[t]
            # The vector the FAISS datastore indexes and queries -- the space
            # OrdCon is meant to organise (module docstring). It is identical
            # for every target. With one target, or with pooling that ignores
            # the target, it equals `embedding`; with per-target MIL stacks it
            # does not, because the projector's weights are shared but its
            # inputs differ.
            comp["retrieval_embedding"] = retrieval_emb
            a = attn[t]
            comp["attention"] = a if isinstance(a, torch.Tensor) else a.get(t)
            out[t] = comp
        return out

    def embed_speaker(self, batch: BagBatch) -> torch.Tensor:
        """Return the projector-output embedding used for FAISS indexing.

        We average the per-target pooled embeddings so the *single* index is
        usable for every retrieval head. (Different targets index the same
        neighbours with different label columns.)
        """
        # Single source of truth: the same vector `forward` uses as its kNN query,
        # so the datastore is always built in the space we later search.
        _, _, retrieval_emb = self._pool_and_project(batch)
        return retrieval_emb
