"""Uncertainty-based multi-task loss weighting (Kendall et al. 2018).

Used by the Gupta 2022 baseline (bi-encoder transformer mixture), which balances
age, height, and gender losses with a *learnable* log-variance per task instead
of hand-tuned weights. Given a raw, already-reduced (NaN-aware) scalar loss
``L_t`` per task, the weighted contribution follows Gupta's exact form:

    exp(-s_t) * L_t + s_t          where s_t = log(sigma_t^2) is learnable.

The ``+ s_t`` term is the regularizer that stops the optimizer from trivially
driving every weight to zero. At init (s_t = 0) the weight is exp(0) = 1 and the
regularizer is 0, so the total is just the sum of the raw task losses.

This module only *weights* task losses; the per-task losses themselves (masked
MSE for age/height, BCE for gender) are computed by the caller so NaN masking
lives in one place.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyLoss(nn.Module):
    """Learnable-uncertainty weighting over a set of named task losses.

    Args:
        task_names: names of the tasks to weight (e.g. ["age", "height", "gender"]).
                    One learnable log-variance parameter is allocated per task.
    """

    def __init__(self, task_names: list[str]) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("UncertaintyLoss needs at least one task name.")
        self.task_names = list(task_names)
        # One learnable log-variance per task, initialized to 0 (weight = 1).
        self.log_vars = nn.ParameterDict(
            {t: nn.Parameter(torch.zeros(())) for t in self.task_names}
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Combine raw per-task scalar losses into one weighted total.

        Args:
            losses: {task_name: scalar_loss_tensor}. Tasks absent from this dict
                    (e.g. gender missing for a corpus) are simply skipped.

        Returns:
            A dict with per-task weighted contributions (``uncertainty/<t>``),
            the current log-variances (``log_var/<t>``, detached for logging),
            and the scalar ``total`` to backprop.
        """
        out: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for t in self.task_names:
            if t not in losses:
                continue
            s = self.log_vars[t]
            weighted = torch.exp(-s) * losses[t] + s
            out[f"uncertainty/{t}"] = weighted
            out[f"log_var/{t}"] = s.detach()
            total = weighted if total is None else total + weighted

        if total is None:
            # No task losses supplied; return a zero on the params' device.
            any_param = next(iter(self.log_vars.values()))
            total = torch.zeros((), device=any_param.device)
        out["total"] = total
        return out
