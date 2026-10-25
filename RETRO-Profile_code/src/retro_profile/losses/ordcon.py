"""Order-enhanced Contrastive Loss (OrdCon) — Component 2 of RETRO-Profile.

Adapted from Wang et al. 2025 (IEEE TIFS) "Order-Enhanced Contrastive Learning
for Face Age Estimation" to scalar speech-trait regression.

The key idea: kNN retrieval only works if the embedding space is *organized
along the target axis*. A pure regression loss on an MLP head does not give
us that — the embedding can encode anything (accent, channel, content) as long
as the head can decode height out of it. OrdCon explicitly forces:

    sign(target_i - target_j)  ==  sign( <z_i - z_anchor, direction>  -  <z_j - z_anchor, direction> )

i.e. the *direction* between two embeddings should align with the sign of
their target difference. Pairs with larger label gaps should be further apart
in the projected space.

We implement this with a margin-ranking formulation that's stable under
mini-batching and works with any number of targets:

    For every (i, j) pair within a batch:
       Δy = y_i - y_j
       Δz = ||z_i - z_j||
       L  =  ReLU( margin - sign(Δy) * (z_i - z_j) · u )                       (direction)
           + ReLU( |Δy / scale| - Δz * gain )                                   (magnitude alignment)

where ``u`` is a *learnable unit direction* per target, and ``scale``/``gain``
are constants that make the magnitude term scale-invariant across targets.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdConLoss(nn.Module):
    """Order-enhanced contrastive loss for scalar regression embeddings.

    Args:
        embedding_dim:   D of the projected embedding space. A learnable unit
                         direction of dim D is allocated per target.
        target_names:    Sequence of target names (one direction vector per target).
        margin:          Hinge margin for the direction-alignment term.
        magnitude_gain:  How tightly the embedding distance must track the
                         absolute label gap. Larger -> stricter geometry.
        magnitude_scale: Label gap that should map to an embedding distance of
                         ``1 / magnitude_gain``. A float applies to every target;
                         a ``{target: scale}`` mapping sets each one separately.
                         Its UNITS are set by ``magnitude_units`` -- read that
                         before choosing a value.
        magnitude_units: ``"loss"`` (default) -- ``magnitude_scale`` is in the
                         units the labels arrive in. The LightningModule z-scores
                         labels before the loss, so this means STANDARD
                         DEVIATIONS. This is the behaviour of every run up to
                         2026-10-10, and the default so they reproduce exactly.
                         ``"label"`` -- ``magnitude_scale`` is in real label
                         units (years, cm); gaps are converted back using
                         ``label_std``.
        label_std:       ``{target: std}`` the labels were divided by before
                         reaching this loss, or None if they arrive in real
                         units. Only consulted when ``magnitude_units="label"``.

    Choosing ``magnitude_scale`` (measured on TIMIT train, 2026-10-10). The
    projector L2-normalises, so no two embeddings are more than 2.0 apart, and
    the magnitude hinge can only ever be satisfied for label gaps up to
    ``2 * magnitude_scale``. Hence:

      * ``5`` in ``"loss"`` units = 38 yr / 49 cm. Satisfiable for every pair, so
        the term is mostly inactive -- the historical setting.
      * ``5`` in ``"label"`` units = 5 yr / 5 cm. UNSATISFIABLE for 26% of age
        pairs and 53% of height pairs: a constant push that can never be met.
        This is what the old docstring claimed and it is not a usable setting.
      * about half the train label range (~27 yr / ~27 cm on TIMIT) maps the
        full range onto the diameter of the unit sphere: satisfiable
        everywhere, active wherever two speakers sit closer than their label
        gap warrants.
    """

    def __init__(
        self,
        embedding_dim: int,
        target_names: list[str],
        margin: float = 0.1,
        magnitude_gain: float = 1.0,
        magnitude_scale: float | Mapping[str, float] = 5.0,
        magnitude_units: str = "loss",
        label_std: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.target_names = list(target_names)
        self.margin = margin
        self.magnitude_gain = magnitude_gain

        if magnitude_units not in ("loss", "label"):
            raise ValueError(
                f"magnitude_units must be 'loss' or 'label', got {magnitude_units!r}"
            )
        self.magnitude_units = magnitude_units

        if isinstance(magnitude_scale, (int, float)):
            scales = {t: float(magnitude_scale) for t in self.target_names}
        else:
            missing = [t for t in self.target_names if t not in magnitude_scale]
            if missing:
                raise ValueError(f"magnitude_scale has no entry for targets {missing}")
            scales = {t: float(magnitude_scale[t]) for t in self.target_names}
        bad = {t: v for t, v in scales.items() if not v > 0}
        if bad:
            raise ValueError(f"magnitude_scale must be positive, got {bad}")
        self.magnitude_scale = scales

        # Labels reach this loss z-scored when the LightningModule normalises
        # targets. In "label" units the gap must be converted back to years/cm
        # before dividing by the scale; in "loss" units it is used as received.
        self.label_std = {t: float(v) for t, v in (label_std or {}).items()}
        self._gap_factor = {
            t: (self.label_std.get(t, 1.0) if magnitude_units == "label" else 1.0) / scales[t]
            for t in self.target_names
        }

        # One learnable unit-direction per target, initialized randomly.
        self.directions = nn.ParameterDict(
            {t: nn.Parameter(torch.randn(embedding_dim) * 0.01) for t in target_names}
        )

    def forward(
        self,
        embeddings: torch.Tensor,             # (B, D)
        labels: dict[str, torch.Tensor],      # {target: (B,)}
        label_mask: dict[str, torch.Tensor],  # {target: (B,) bool}
    ) -> dict[str, torch.Tensor]:
        if embeddings.dim() != 2:
            raise ValueError(f"Expected (B, D), got {tuple(embeddings.shape)}")

        out: dict[str, torch.Tensor] = {}
        total = embeddings.new_zeros(())
        any_present = False

        for t in self.target_names:
            if t not in labels:
                continue
            y = labels[t]
            m = label_mask[t]
            valid = m.nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() < 2:
                out[f"ordcon/{t}"] = embeddings.new_zeros(())
                out[f"ordcon/{t}/direction"] = embeddings.new_zeros(())
                out[f"ordcon/{t}/magnitude"] = embeddings.new_zeros(())
                continue
            any_present = True

            z = embeddings[valid]                              # (n, D)
            y_v = y[valid]                                     # (n,)
            n = z.shape[0]

            # Pairwise label and embedding differences.
            dy = y_v.unsqueeze(0) - y_v.unsqueeze(1)           # (n, n)
            dz = z.unsqueeze(0) - z.unsqueeze(1)               # (n, n, D)

            # Direction alignment: sign(dy) * <dz, u> should be >= margin.
            u = F.normalize(self.directions[t], p=2, dim=0)    # (D,)
            proj = torch.einsum("ijd,d->ij", dz, u)            # (n, n)
            sign = torch.sign(dy)                              # (n, n) in {-1, 0, +1}

            non_zero = sign != 0
            direction_term = F.relu(self.margin - sign * proj)
            direction_term = direction_term * non_zero.to(direction_term.dtype)
            n_pairs = non_zero.sum().clamp_min(1).to(direction_term.dtype)
            direction_loss = direction_term.sum() / n_pairs

            # Magnitude alignment: ||dz|| should be at least |dy|/scale * gain.
            dist = dz.norm(p=2, dim=-1)                        # (n, n)
            target_dist = dy.abs() * self._gap_factor[t]
            magnitude_term = F.relu(target_dist - dist * self.magnitude_gain)
            # Exclude diagonal so self-pairs don't dominate.
            eye = torch.eye(n, device=z.device, dtype=torch.bool)
            magnitude_term = magnitude_term.masked_fill(eye, 0.0)
            magnitude_loss = magnitude_term.sum() / max(n * (n - 1), 1)

            loss_t = direction_loss + magnitude_loss
            out[f"ordcon/{t}"] = loss_t
            # Logged separately: a magnitude term stuck near zero means it is
            # inactive, which a combined number cannot reveal.
            out[f"ordcon/{t}/direction"] = direction_loss
            out[f"ordcon/{t}/magnitude"] = magnitude_loss
            total = total + loss_t

        out["ordcon/total"] = total if any_present else embeddings.new_zeros(())
        return out
