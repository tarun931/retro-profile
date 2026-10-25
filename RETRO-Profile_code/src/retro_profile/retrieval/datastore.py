"""FAISS-backed retrieval datastore for the residual / pure-retrieval heads.

The datastore stores **one entry per training speaker**, keyed by an L2-normalized
embedding vector and carrying that speaker's labels (age, height). At inference,
we query with the test speaker's embedding and weight the retrieved labels by
softmax-of-similarity to form an anchor.

Why per-speaker, not per-utterance?
    *  Per-utterance datastores would have to be averaged anyway, and would
       leak intra-speaker correlation into the kNN distance — hurting cross-
       corpus generalization. Aggregating to speaker level once, up-front, is
       both cleaner and faster (smaller index, fewer queries).

Persistence
    *  ``save(dir)`` writes the FAISS index + a parquet sidecar of labels.
    *  ``load(dir)`` recovers both. Datastores are independent of model weights;
       you can re-build one without re-training, and the trained heads will pick
       up the new neighbours automatically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@dataclass
class DatastoreEntry:
    """In-memory record before the FAISS index is built."""
    speaker_id: str
    embedding: np.ndarray            # (D,) float32, expected to be L2-normalized
    labels: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalDatastore:
    """Speaker-keyed FAISS index + label table.

    Args:
        embedding_dim:  D of stored vectors.
        metric:         "cosine" (recommended; uses inner product on L2-normalized
                        vectors) or "l2".
        temperature:    Softmax temperature applied to similarities when computing
                        the kNN-weighted label anchor.
        weighting:      "softmax_distance" (default) or "uniform".
    """

    def __init__(
        self,
        embedding_dim: int,
        metric: str = "cosine",
        temperature: float = 0.1,
        weighting: str = "softmax_distance",
    ) -> None:
        if not _HAS_FAISS:
            raise ImportError(
                "faiss is required for RetrievalDatastore. "
                "Install with `pip install faiss-cpu` or `faiss-gpu`."
            )
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.temperature = temperature
        self.weighting = weighting

        self._entries: list[DatastoreEntry] = []
        self._index: faiss.Index | None = None
        self._labels_df: pd.DataFrame | None = None

    # ---- construction ------------------------------------------------------
    def add(self, entry: DatastoreEntry) -> None:
        if self._index is not None:
            raise RuntimeError("Cannot add entries after build(); rebuild instead.")
        if entry.embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Embedding dim mismatch: expected ({self.embedding_dim},) "
                f"got {entry.embedding.shape}"
            )
        self._entries.append(entry)

    def build(self) -> None:
        """Construct the FAISS index from accumulated entries."""
        if not self._entries:
            raise RuntimeError("Datastore is empty; call add() before build().")

        vecs = np.stack([e.embedding for e in self._entries]).astype(np.float32)
        if self.metric == "cosine":
            # Cosine via inner product on L2-normalized vectors. We re-normalize
            # defensively in case callers forgot.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
            vecs = vecs / norms
            index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.metric == "l2":
            index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")
        index.add(vecs)
        self._index = index

        # Sidecar table for fast label lookup by row.
        self._labels_df = pd.DataFrame(
            [{"speaker_id": e.speaker_id, **e.labels} for e in self._entries]
        )

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._labels_df is not None

    def __len__(self) -> int:
        return self._index.ntotal if self._index is not None else len(self._entries)

    # ---- search ------------------------------------------------------------
    def search(
        self, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (similarities, indices) for each query row.

        Similarities are cosine in [-1, 1] when metric='cosine', or *negated*
        squared L2 when metric='l2' (so larger = closer in both cases).
        """
        if not self.is_built:
            raise RuntimeError("Datastore not built; call build() first.")
        device = queries.device
        q = queries.detach().to("cpu", dtype=torch.float32).numpy()
        if self.metric == "cosine":
            norms = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
            q = q / norms
        D, I = self._index.search(q, k)                 # noqa: E741
        if self.metric == "l2":
            D = -D                                      # convert to similarity
        return torch.from_numpy(D).to(device), torch.from_numpy(I).to(device)

    def _exclusion_mask(
        self, idx: torch.Tensor, exclude_speaker_ids: Sequence[str]
    ) -> torch.Tensor:
        """(B, k) bool mask: True where the retrieved row IS the query speaker."""
        if len(exclude_speaker_ids) != idx.shape[0]:
            raise ValueError(
                f"exclude_speaker_ids has {len(exclude_speaker_ids)} entries but "
                f"there are {idx.shape[0]} queries."
            )
        stored = self._labels_df["speaker_id"].to_numpy().astype(str)
        retrieved = stored[idx.detach().cpu().numpy().clip(min=0)]      # (B, k) str
        query = np.asarray([str(s) for s in exclude_speaker_ids])[:, None]
        return torch.from_numpy(retrieved == query).to(idx.device)

    def _search_excluding_self(
        self, queries: torch.Tensor, k: int, exclude_speaker_ids: Sequence[str] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k search, optionally dropping the query's own datastore entry.

        Leave-one-out retrieval. Without it, a *training* speaker retrieves itself
        at similarity 1.0, the softmax puts nearly all mass on its own label, and
        the residual head learns nothing (the anchor is already the ground truth).
        Test speakers are not in the datastore, so this only affects training.
        """
        if not exclude_speaker_ids:
            return self.search(queries, k)

        # Over-fetch by one so k real neighbours survive the self-match, then
        # re-select the best k after masking.
        n = len(self)
        k_search = min(k + 1, n)
        sims, idx = self.search(queries, k_search)
        sims = sims.masked_fill(self._exclusion_mask(idx, exclude_speaker_ids), float("-inf"))
        k_keep = min(k, k_search)
        sims, order = torch.topk(sims, k_keep, dim=1)
        return sims, torch.gather(idx, 1, order)

    def knn_anchor(
        self,
        queries: torch.Tensor,
        target: str,
        k: int,
        exclude_speaker_ids: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return the kNN-weighted label anchor for each query.

        ``exclude_speaker_ids`` enables leave-one-out retrieval (pass the query
        speakers' ids so a speaker never retrieves itself).

        Weights are normalised over VALID neighbours only. A neighbour is invalid
        if its label for ``target`` is missing, it was excluded above, or FAISS
        returned no result for that slot (index -1, which happens when ``k``
        exceeds the datastore size). A query whose every neighbour is invalid
        gets the datastore's mean ``target`` label -- the training-set mean,
        since the datastore indexes training speakers. Without this, a softmax
        over a row of -inf returns NaN. Raises if the datastore holds no label
        for ``target`` at all: it cannot supply that anchor, and a silent value
        would hide the misconfiguration.
        """
        if not self.is_built:
            raise RuntimeError("Datastore not built; call build() first.")
        if target not in self._labels_df.columns:
            raise KeyError(f"Target {target!r} not in datastore labels: "
                           f"{list(self._labels_df.columns)}")

        sims, idx = self._search_excluding_self(queries, k, exclude_speaker_ids)
        device = queries.device

        labels = torch.tensor(
            self._labels_df[target].to_numpy(dtype=np.float32), device=device
        )
        known = labels[~torch.isnan(labels)]
        if known.numel() == 0:
            raise ValueError(
                f"Datastore holds no {target!r} labels, so it cannot supply a "
                f"{target!r} anchor. Build it from speakers that have this label."
            )
        fallback = known.mean()

        # FAISS marks an empty slot with index -1. Indexing with -1 would read
        # the LAST label, so clamp for the gather and mark the slot invalid.
        gathered = labels[idx.clamp_min(0)]                 # (B, k)

        # Invalid = no result, missing label, or dropped by the exclusion above.
        invalid = (idx < 0) | torch.isnan(gathered) | ~torch.isfinite(sims)
        has_valid = (~invalid).any(dim=1)                   # (B,)

        if self.weighting == "uniform":
            valid = ~invalid
            gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
            denom = valid.sum(dim=1).clamp_min(1).to(gathered.dtype)
            anchor = gathered.sum(dim=1) / denom
        else:
            # softmax_distance — temperature-softmaxed similarity.
            masked_sims = sims.masked_fill(invalid, float("-inf"))
            # A row with no valid neighbour would softmax all -inf into NaN.
            # Give it finite logits; its value is replaced by the fallback below.
            masked_sims = torch.where(
                has_valid.unsqueeze(1), masked_sims, torch.zeros_like(masked_sims)
            )
            weights = torch.softmax(masked_sims / max(self.temperature, 1e-6), dim=1)
            gathered = torch.where(invalid, torch.zeros_like(gathered), gathered)
            anchor = (weights * gathered).sum(dim=1)

        return torch.where(has_valid, anchor, fallback.to(anchor.dtype).expand_as(anchor))

    def neighbours(
        self,
        queries: torch.Tensor,
        k: int,
        exclude_speaker_ids: Sequence[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Return the k retrieved neighbours per query as a list of dicts.

        Used by the evaluator to produce forensic-style explanations:
            "predicted height = 178 cm because retrieved {sp1: 175, sp2: 182, ...}"
        """
        sims, idx = self._search_excluding_self(queries, k, exclude_speaker_ids)
        out: list[list[dict[str, Any]]] = []
        for i in range(idx.shape[0]):
            row = []
            for j in range(idx.shape[1]):
                meta = self._labels_df.iloc[int(idx[i, j].item())].to_dict()
                meta["similarity"] = float(sims[i, j].item())
                row.append(meta)
            out.append(row)
        return out

    # ---- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        if not self.is_built:
            raise RuntimeError("Cannot save an unbuilt datastore.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        self._labels_df.to_parquet(directory / "labels.parquet")
        meta = {
            "embedding_dim": self.embedding_dim,
            "metric": self.metric,
            "temperature": self.temperature,
            "weighting": self.weighting,
        }
        pd.Series(meta).to_json(directory / "meta.json")

    @classmethod
    def load(cls, directory: str | Path) -> "RetrievalDatastore":
        directory = Path(directory)
        meta = pd.read_json(directory / "meta.json", typ="series").to_dict()
        ds = cls(
            embedding_dim=int(meta["embedding_dim"]),
            metric=str(meta["metric"]),
            temperature=float(meta["temperature"]),
            weighting=str(meta["weighting"]),
        )
        ds._index = faiss.read_index(str(directory / "index.faiss"))
        ds._labels_df = pd.read_parquet(directory / "labels.parquet")
        return ds
