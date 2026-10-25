"""Multitask loss combining masked regression + OrdCon.

Provides a single ``forward(...)`` that the LightningModule can call, returning
a scalar ``total`` for backprop and a dict of named components for logging.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from retro_profile.losses.attention_entropy import AttentionEntropyPenalty
from retro_profile.losses.ordcon import OrdConLoss
from retro_profile.losses.regression import MaskedRegressionLoss


class MultitaskLoss(nn.Module):
    """Weighted sum of regression losses + OrdCon."""

    def __init__(
        self,
        target_names: list[str],
        embedding_dim: int,
        regression_kind: str = "smooth_l1",
        regression_weight: float = 1.0,
        ordcon_weight: float = 0.5,
        per_target_weights: dict[str, float] | None = None,
        ordcon_kwargs: dict | None = None,
        attention_entropy_weight: float = 0.0,
        attention_entropy_per_target: dict[str, float] | None = None,
        label_std: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.target_names = list(target_names)
        self.regression_weight = regression_weight
        self.ordcon_weight = ordcon_weight
        self.per_target_weights = per_target_weights or {t: 1.0 for t in target_names}

        self.regression = MaskedRegressionLoss(kind=regression_kind)
        # Penalises MIL attention that concentrates on few utterances. Weight 0
        # (the default) makes this inert, so existing configs are unaffected.
        self.attention_entropy = AttentionEntropyPenalty(
            weight=attention_entropy_weight,
            per_target_weights=attention_entropy_per_target,
        )
        # `label_std` lets OrdCon express `magnitude_scale` in real label units
        # when the labels it receives have been z-scored (see losses/ordcon.py).
        self.ordcon = OrdConLoss(
            embedding_dim=embedding_dim,
            target_names=target_names,
            label_std=label_std,
            **(ordcon_kwargs or {}),
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],          # {t: (B,)}
        labels: dict[str, torch.Tensor],               # {t: (B,)}
        label_mask: dict[str, torch.Tensor],           # {t: (B,) bool}
        embeddings: torch.Tensor | None,               # (B, D) or None to skip OrdCon
        attention: dict[str, torch.Tensor] | None = None,   # {t: (B, K)}
        utt_mask: torch.Tensor | None = None,               # (B, K) bool
    ) -> dict[str, torch.Tensor]:
        components: dict[str, torch.Tensor] = {}

        # Regression term
        reg_total = predictions[self.target_names[0]].new_zeros(())
        for t in self.target_names:
            if t not in predictions:
                continue
            l = self.regression(predictions[t], labels[t], mask=label_mask[t])
            w = float(self.per_target_weights.get(t, 1.0))
            components[f"reg/{t}"] = l
            reg_total = reg_total + w * l
        components["reg/total"] = reg_total

        # OrdCon term
        if embeddings is not None and self.ordcon_weight > 0:
            ord_components = self.ordcon(embeddings, labels, label_mask)
            components.update(ord_components)
            ord_total = ord_components["ordcon/total"]
        else:
            ord_total = reg_total.new_zeros(())
            components["ordcon/total"] = ord_total

        # Attention-concentration term. Optional and off by default; see
        # losses/attention_entropy.py for why it exists.
        attn_components = self.attention_entropy(attention, utt_mask)
        components.update(attn_components)
        attn_total = attn_components.get("attn_kl/total", reg_total.new_zeros(()))

        total = (
            self.regression_weight * reg_total
            + self.ordcon_weight * ord_total
            + attn_total          # already scaled by its own weight(s)
        )
        components["total"] = total
        return components
