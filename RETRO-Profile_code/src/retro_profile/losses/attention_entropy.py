"""Penalise MIL attention that concentrates on too few utterances.

Motivation (SESSION_HANDOFF.md §3.8e, E16/E18). Height attention concentrates
hard -- an effective bag size of 2.6-4.2 out of 8 -- and on height, MIL pooling
LOSES to plain averaging (5.41 +/- 0.16 against 5.09 +/- 0.09). The calibration
diagnostic explains why: the concentrated model produces predictions almost as
spread out as real heights (std ratio 0.97) while its within-gender correlation
is only 0.36, so most of that spread is noise. Concentrating forfeits the
sqrt(K) variance reduction of averaging and buys nothing in return.

This term interpolates continuously between the two, which is a strictly better
experiment than comparing them as a binary. At weight 0 the pooling is ordinary
MIL attention; as the weight grows the attention is driven toward uniform, i.e.
toward mean pooling. Sweeping the weight turns "MIL versus averaging" into a
curve, and an interior optimum -- if one exists -- would be evidence that some
selectivity helps while the current amount is too much.

It also removes a confound in the existing evidence. Today "averaging" is
measured by `evaluate.py --unit utterance`, which runs the model on single
clips; a bag-trained model has never seen an isolated clip, so that comparison
mixes pooling with an out-of-distribution input. Varying this weight changes
ONLY the pooling, with bag-mode evaluation on both sides.

The penalty is the KL divergence from the attention distribution to uniform,
normalised by log K so the weight means the same thing at any bag size:

    KL(a || u) / log K  =  1 - H(a) / log K

which is 0 for uniform attention and 1 when all weight sits on one instance.
Since n_eff = exp(H(a)), driving this to 0 drives n_eff to K.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-12


class AttentionEntropyPenalty(nn.Module):
    """Normalised KL-to-uniform over per-target MIL attention weights.

    Args:
        weight: multiplier on the penalty. 0 disables it entirely (the default,
            so existing configs are unaffected).
        per_target_weights: optional per-target overrides, e.g. penalise height
            but leave age alone -- age attention is already near-uniform
            (n_eff 7.3-7.5 of 8), so a penalty there has almost nothing to act on.
    """

    def __init__(
        self,
        weight: float = 0.0,
        per_target_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.per_target_weights = dict(per_target_weights or {})

    @staticmethod
    def _normalised_kl(weights: torch.Tensor, utt_mask: torch.Tensor) -> torch.Tensor:
        """1 - H(a)/log K per bag, averaged over the batch. (B, K) -> scalar."""
        mask = utt_mask.to(weights.dtype)
        w = (weights * mask).clamp_min(0.0)
        # Renormalise defensively: the pooling already applies a masked softmax,
        # but a resumed or half-trained model can emit weights that do not sum
        # to 1 over the valid slots.
        w = w / w.sum(dim=1, keepdim=True).clamp_min(_EPS)

        # x*log(x) -> 0 as x -> 0, so the clamp only guards the log.
        ent = -(w * torch.log(w.clamp_min(_EPS)) * mask).sum(dim=1)          # (B,)

        # log of the number of VALID instances, not the padded width -- a bag of
        # 3 clips is already uniform at n_eff 3 and must not be penalised for it.
        k_valid = mask.sum(dim=1)                                            # (B,)
        log_k = torch.log(k_valid.clamp_min(1.0))

        # Bags with a single valid clip have log K = 0 and no possible spread;
        # they contribute nothing rather than dividing by zero.
        usable = k_valid > 1
        if not bool(usable.any()):
            return weights.new_zeros(())
        return (1.0 - ent[usable] / log_k[usable].clamp_min(_EPS)).mean()

    def forward(
        self,
        attention: dict[str, torch.Tensor] | None,
        utt_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Return {"attn_kl/<target>": ..., "attn_kl/total": weighted sum}."""
        out: dict[str, torch.Tensor] = {}
        if not attention or utt_mask is None:
            return out

        total: torch.Tensor | None = None
        for t, w in attention.items():
            if w is None or w.dim() != 2:
                continue
            kl = self._normalised_kl(w, utt_mask)
            out[f"attn_kl/{t}"] = kl.detach()
            lam = float(self.per_target_weights.get(t, self.weight))
            if lam == 0.0:
                continue
            term = lam * kl
            total = term if total is None else total + term

        if total is not None:
            out["attn_kl/total"] = total
        return out
