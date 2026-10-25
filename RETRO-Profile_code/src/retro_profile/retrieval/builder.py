"""Build a :class:`RetrievalDatastore` from a trained RETRO-Profile model.

Workflow (used by ``scripts/build_datastore.py``):

  1. Load a trained Lightning checkpoint.
  2. Iterate over the *train* split utterance-by-utterance.
  3. Encode + project each utterance, then average per speaker into a single
     vector. (Per-speaker averaging is intentional — see datastore.py docstring.)
  4. Add (speaker_id, vector, labels) entries to the datastore and ``build()``.
  5. Save to disk.

The model used here is whatever owns ``embed_speaker(...)`` — the abstract
contract that RetroProfile and any baselines that want kNN can implement.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from retro_profile.retrieval.datastore import DatastoreEntry, RetrievalDatastore


@torch.no_grad()
def build_datastore(
    model: torch.nn.Module,                 # must implement embed_speaker(BagBatch) -> (B, D)
    dataloader: DataLoader,                 # utterance-mode loader (each item = 1 utterance)
    targets: Iterable[str] = ("age", "height"),
    embedding_dim: int | None = None,
    metric: str = "cosine",
    temperature: float = 0.1,
    weighting: str = "softmax_distance",
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
) -> RetrievalDatastore:
    """Encode all utterances, average per speaker, build a datastore."""
    model = model.to(device).eval()
    targets = list(targets)

    speaker_vecs: dict[str, list[np.ndarray]] = defaultdict(list)
    speaker_labels: dict[str, dict[str, float]] = {}

    for batch in dataloader:
        batch = batch.to(device)
        emb = model.embed_speaker(batch)                       # (B, D)
        emb_np = emb.detach().cpu().numpy().astype(np.float32)

        for i, sid in enumerate(batch.speaker_ids):
            speaker_vecs[sid].append(emb_np[i])
            if sid not in speaker_labels:
                speaker_labels[sid] = {
                    t: float(batch.labels[t][i].item())
                    for t in targets
                    if bool(batch.label_mask[t][i].item())
                }

    if embedding_dim is None:
        any_vec = next(iter(speaker_vecs.values()))[0]
        embedding_dim = int(any_vec.shape[0])

    ds = RetrievalDatastore(
        embedding_dim=embedding_dim,
        metric=metric,
        temperature=temperature,
        weighting=weighting,
    )
    for sid, vecs in speaker_vecs.items():
        avg = np.mean(np.stack(vecs), axis=0)
        if metric == "cosine":
            n = np.linalg.norm(avg) + 1e-12
            avg = avg / n
        ds.add(DatastoreEntry(
            speaker_id=sid,
            embedding=avg.astype(np.float32),
            labels=speaker_labels.get(sid, {}),
        ))
    ds.build()
    return ds


def attach_datastore(model: torch.nn.Module, datastore: RetrievalDatastore) -> None:
    """Wire a built datastore into every retrieval-aware head on the model.

    Conventions:
      * the model exposes ``self.heads`` as a ``ModuleDict`` keyed by target name;
      * each value is a head that exposes ``set_datastore(datastore, target_name)``
        if it supports retrieval (otherwise the call is skipped).
    """
    heads = getattr(model, "heads", None)
    if heads is None:
        raise AttributeError(f"Model {type(model).__name__} has no .heads ModuleDict.")
    for name, head in heads.items():
        if hasattr(head, "set_datastore"):
            head.set_datastore(datastore, target_name=name)


def save_built(datastore: RetrievalDatastore, directory: str | Path) -> None:
    datastore.save(directory)


def load_built(directory: str | Path) -> RetrievalDatastore:
    return RetrievalDatastore.load(directory)
