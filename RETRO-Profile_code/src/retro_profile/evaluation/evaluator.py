"""Evaluator that produces per-speaker predictions plus forensic explanations.

Output: a pandas DataFrame with columns
    speaker_id, target, prediction, anchor, delta, ground_truth, error,
    neighbour_<i>_speaker, neighbour_<i>_label, neighbour_<i>_similarity

This is what the thesis chapter on forensic defensibility cites: every
prediction comes with the K retrieved training speakers that anchored it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
import torch
from torch.utils.data import DataLoader

from retro_profile.retrieval.datastore import RetrievalDatastore
from retro_profile.training.metrics import RegressionMetrics


def _attention_stats(
    comps: dict | None, target: str, i: int, utt_mask: torch.Tensor
) -> dict[str, float]:
    """Summarise how *selective* the MIL attention was for one speaker.

    ``attn_entropy`` is Shannon entropy normalised by log(K), so:
        1.0 -> perfectly uniform  (attention collapsed; MIL == mean pooling)
        0.0 -> all weight on one clip (maximally selective)
    Together with ``attn_max`` (uniform would be 1/K) this is the diagnostic for
    whether MIL actually learned to discriminate between utterances.
    """
    empty = {"n_clips": float("nan"), "attn_max": float("nan"),
             "attn_entropy": float("nan"), "n_eff": float("nan")}
    if comps is None or comps.get(target, {}).get("attention") is None:
        return empty
    attn = comps[target]["attention"]
    if not isinstance(attn, torch.Tensor) or attn.dim() != 2:
        return empty

    weights = attn[i][utt_mask.bool()].detach().float().cpu()
    k = int(weights.numel())
    if k == 0:
        return empty
    total = weights.sum().clamp_min(1e-12)
    p = (weights / total).clamp_min(1e-12)
    entropy = 1.0 if k == 1 else float(-(p * p.log()).sum() / math.log(k))
    # n_eff (participation ratio, 1 / sum p^2): the *effective* number of clips
    # being averaged. Uniform weights over K clips give n_eff = K; all weight on
    # one clip gives 1. This is the quantity §3.5 turns on -- if MIL selectivity
    # costs more in lost sqrt(K) variance reduction than it gains in relevance,
    # n_eff < K is the mechanism.
    n_eff = float(1.0 / (p * p).sum().clamp_min(1e-12))
    return {"n_clips": float(k), "attn_max": float(p.max()),
            "attn_entropy": entropy, "n_eff": n_eff}


def _attention_rows(
    comps: dict | None, target: str, i: int, speaker_id: str,
    utt_mask: torch.Tensor, utt_ids: list[str],
) -> list[dict]:
    """One row per real utterance: its MIL attention weight, normalised."""
    if comps is None or comps.get(target, {}).get("attention") is None:
        return []
    attn = comps[target]["attention"]
    if not isinstance(attn, torch.Tensor) or attn.dim() != 2:
        return []
    mask = utt_mask.bool()
    weights = attn[i][mask].detach().float().cpu()
    if weights.numel() == 0:
        return []
    p = weights / weights.sum().clamp_min(1e-12)
    kept = [u for u, m in zip(utt_ids, mask.tolist(), strict=False) if m]
    order = p.argsort(descending=True).tolist()
    rank = {idx: r for r, idx in enumerate(order)}
    return [
        {
            "speaker_id": speaker_id,
            "target": target,
            "utt_id": kept[j] if j < len(kept) else f"utt_{j}",
            "weight": float(p[j]),
            "rank": rank[j],
            "n_clips": int(p.numel()),
        }
        for j in range(p.numel())
    ]


def _frame_attention_rows(model: torch.nn.Module, batch) -> list[dict]:
    """One row per real clip: how the frame-level attention spread over its frames.

    Empty unless the encoder pools frames with attention (``frame_pooling:
    attention``). ``eff_share`` is n_eff / n_frames: 1.0 is a plain mean over the
    clip's frames; small values mean a few frames summarise the clip. ``weights``
    holds the raw per-frame weights (float16) for later analysis.
    """
    enc = getattr(model, "encoder", None)
    fa = getattr(enc, "last_frame_attention", None)
    if fa is None:
        return []
    enc.last_frame_attention = None              # never reuse a stale batch
    weights, mask = fa[0].detach().float().cpu(), fa[1].detach().bool().cpu()
    k_max = batch.utt_mask.shape[1]
    rows: list[dict] = []
    for b, sid in enumerate(batch.speaker_ids):
        for k, uid in enumerate(batch.utt_ids[b]):
            if not bool(batch.utt_mask[b, k]):
                continue
            idx = b * k_max + k                  # the encoder flattens (B, K) -> B*K
            w = weights[idx][mask[idx]]
            n = int(w.numel())
            if n == 0:
                continue
            p = (w / w.sum().clamp_min(1e-12)).clamp_min(1e-12)
            n_eff = float(1.0 / (p * p).sum())
            rows.append({
                "speaker_id": sid,
                "utt_id": uid,
                "n_frames": n,
                "n_eff": n_eff,
                "eff_share": n_eff / n,
                "max_weight": float(p.max()),
                "entropy": 1.0 if n == 1 else float(-(p * p.log()).sum() / math.log(n)),
                "weights": w.numpy().astype("float16"),
            })
    return rows


@torch.no_grad()
def evaluate_with_explanations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_names: Iterable[str],
    datastore: RetrievalDatastore | None = None,
    k_neighbours: int = 5,
    device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    return_attention: bool = False,
    frame_attention: list[dict] | None = None,
    embeddings: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]] | tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Run inference and return predictions plus aggregate metrics.

    With ``return_attention=True`` a third frame is returned holding one row per
    (speaker, target, utterance) with that utterance's MIL attention weight --
    the raw material for asking *what* the attention is selecting on, rather
    than only how concentrated it is.

    Pass a list as ``frame_attention`` to collect one row per clip describing the
    frame-level attention (see ``_frame_attention_rows``); it stays empty for
    models that average frames.

    Pass a list as ``embeddings`` to collect, per row, the retrieval embedding --
    the vector the kNN datastore is queried with -- so kNN settings (k,
    weighting) can be swept offline without re-running the encoder.
    """
    target_names = list(target_names)
    model = model.to(device).eval()

    metrics = RegressionMetrics(target_names)
    rows: list[dict] = []
    attn_rows: list[dict] = []

    for batch in dataloader:
        batch = batch.to(device)

        # Run with components if available so we can record anchor/delta.
        if hasattr(model, "forward_with_components"):
            comps = model.forward_with_components(batch)
            preds = {t: c["pred"] for t, c in comps.items()}
        else:
            comps = None
            preds = model(batch)

        # Optional: retrieve neighbours from the datastore for an explanation.
        nbrs = None
        if datastore is not None and datastore.is_built and hasattr(model, "embed_speaker"):
            emb = model.embed_speaker(batch)
            nbrs = datastore.neighbours(emb, k=k_neighbours)

        if frame_attention is not None:
            frame_attention.extend(_frame_attention_rows(model, batch))
        if embeddings is not None and comps is not None:
            # Identical for every target (one shared retrieval query).
            q = next(iter(comps.values())).get("retrieval_embedding")
            if q is not None:
                q = q.detach().float().cpu().numpy()
                for i, sid in enumerate(batch.speaker_ids):
                    uids = batch.utt_ids[i]
                    embeddings.append({"speaker_id": sid,
                                       "utt_id": uids[0] if len(uids) == 1 else "",
                                       "embedding": q[i]})

        metrics.update(preds, batch.labels, batch.label_mask)

        for i, sid in enumerate(batch.speaker_ids):
            for t in target_names:
                if t not in preds:
                    continue
                gt = float(batch.labels[t][i].item()) if bool(batch.label_mask[t][i].item()) else float("nan")
                pred = float(preds[t][i].item())
                anchor = (
                    float(comps[t]["anchor"][i].item())
                    if comps is not None and comps[t].get("anchor") is not None
                    else float("nan")
                )
                delta = (
                    float(comps[t]["delta"][i].item())
                    if comps is not None and comps[t].get("delta") is not None
                    else float("nan")
                )
                # In utterance mode each bag holds exactly one clip, so utt_id
                # identifies the row; in bag mode it is empty (the row is the
                # speaker, not any single clip).
                uids = batch.utt_ids[i]
                row: dict = {
                    "speaker_id": sid,
                    "utt_id": uids[0] if len(uids) == 1 else "",
                    "target": t,
                    "prediction": pred,
                    "anchor": anchor,
                    "delta": delta,
                    "ground_truth": gt,
                    "error": pred - gt if not torch.isnan(torch.tensor(gt)) else float("nan"),
                }
                row.update(_attention_stats(comps, t, i, batch.utt_mask[i]))
                if return_attention:
                    attn_rows.extend(
                        _attention_rows(comps, t, i, sid, batch.utt_mask[i], uids)
                    )
                if nbrs is not None:
                    for j, n in enumerate(nbrs[i]):
                        row[f"neighbour_{j}_speaker"] = n.get("speaker_id")
                        row[f"neighbour_{j}_label"] = n.get(t)
                        row[f"neighbour_{j}_similarity"] = n.get("similarity")
                rows.append(row)

    df = pd.DataFrame(rows)
    if return_attention:
        return df, metrics.compute(), pd.DataFrame(attn_rows)
    return df, metrics.compute()
