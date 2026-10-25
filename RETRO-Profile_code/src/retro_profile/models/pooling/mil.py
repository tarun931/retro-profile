"""MIL attention pooling — Component 1 of RETRO-Profile.

This is the bag-of-utterances novelty. We adapt the gated-attention MIL pooling
of Ilse et al. 2018 (https://arxiv.org/abs/1802.04712) to the speech setting,
extending it with **per-target attention heads** so that the pooling can be
*specialized for each trait*: vowel-rich utterances get more weight for height
(VTL signal), sustained voicing gets more weight for age (jitter / shimmer).

We borrow the MIL-for-speaker-attributes idea from RL-MIL (Salkhordeh-Ziabari
2024) but use a fully differentiable (non-RL) head, which is cheaper and
empirically as strong on bag sizes < 16.

Inputs / outputs
================
Inputs
    utt_emb     (B, K, D)  per-utterance encoder embeddings
    utt_mask    (B, K)      bool, True for real utterances
    target      str | None  if given, return the pooling for that head only;
                            otherwise return a dict keyed by target name
Outputs
    speaker_emb (B, D)     pooled speaker embedding for the target
    attn_weights (B, K)    attention distribution (sums to 1 over real utts)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GatedAttentionHead(nn.Module):
    """A single MIL gated-attention head (Ilse et al. 2018).

    score_k = w^T * (tanh(V * h_k) ⊙ sigmoid(U * h_k))

    The gate sigmoid(U h_k) modulates which dimensions of the tanh score matter
    for *this* utterance, allowing the head to attend differently to vowel-rich
    vs. plosive-rich utterances within the same bag.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, gated: bool) -> None:
        super().__init__()
        self.gated = gated
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        if gated:
            self.U = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, utt_emb: torch.Tensor, utt_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dropout(utt_emb)                                # (B, K, D)
        a = torch.tanh(self.V(h))                                # (B, K, H)
        if self.gated:
            a = a * torch.sigmoid(self.U(h))
        scores = self.w(a).squeeze(-1)                           # (B, K)
        scores = scores.masked_fill(~utt_mask, -math.inf)
        weights = F.softmax(scores, dim=1)                        # (B, K)
        pooled = torch.einsum("bk,bkd->bd", weights, utt_emb)     # (B, D)
        return pooled, weights


class MILAttentionPooling(nn.Module):
    """Multi-head gated MIL pooling with optional per-target heads.

    Args:
        input_dim:     D, the encoder embedding dim.
        hidden_dim:    Hidden size inside each attention head.
        num_heads:     Number of independent heads. Their pooled outputs are
                       *averaged* (not concatenated) so the speaker embedding
                       stays D-dim — required for the residual head and FAISS
                       index to remain compatible.
        dropout:       Dropout inside the attention MLP.
        gated:         If True, use Ilse-style gated attention (recommended).
        target_names:  Optional list of trait names ("age", "height"). If given,
                       a *separate* set of ``num_heads`` heads is allocated per
                       target, and ``forward(target=...)`` returns the per-target
                       pooled embedding. If None, a single shared set of heads
                       is used regardless of target.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        gated: bool = True,
        target_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_heads = num_heads
        self.target_names = target_names

        def _make_heads() -> nn.ModuleList:
            return nn.ModuleList(
                _GatedAttentionHead(input_dim, hidden_dim, dropout, gated)
                for _ in range(num_heads)
            )

        if target_names is None:
            self.heads = _make_heads()
            self.target_heads: nn.ModuleDict | None = None
        else:
            self.heads = None  # type: ignore[assignment]
            self.target_heads = nn.ModuleDict({t: _make_heads() for t in target_names})

    # ---- internals ---------------------------------------------------------
    def _pool_with_heads(
        self,
        heads: nn.ModuleList,
        utt_emb: torch.Tensor,
        utt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_list, weight_list = [], []
        for h in heads:
            p, w = h(utt_emb, utt_mask)
            pooled_list.append(p)
            weight_list.append(w)
        # Average across heads -> (B, D); average attention -> (B, K)
        pooled = torch.stack(pooled_list, dim=0).mean(dim=0)
        weights = torch.stack(weight_list, dim=0).mean(dim=0)
        return pooled, weights

    # ---- forward -----------------------------------------------------------
    def forward(
        self,
        utt_emb: torch.Tensor,
        utt_mask: torch.Tensor,
        target: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.target_heads is None:
            assert self.heads is not None
            return self._pool_with_heads(self.heads, utt_emb, utt_mask)

        if target is not None:
            if target not in self.target_heads:
                raise KeyError(f"No MIL head for target {target!r}; have {list(self.target_heads)}")
            return self._pool_with_heads(self.target_heads[target], utt_emb, utt_mask)

        # Return a dict over all targets (used during training when we compute
        # multiple regression losses in a single forward pass).
        pooled, weights = {}, {}
        for t, heads in self.target_heads.items():
            p, w = self._pool_with_heads(heads, utt_emb, utt_mask)
            pooled[t] = p
            weights[t] = w
        return pooled, weights
