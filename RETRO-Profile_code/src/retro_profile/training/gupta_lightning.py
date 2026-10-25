"""LightningModule for the multi-task RQ1 baselines (Gupta 2022, Jaid 2023).

Reuses RETRO-Profile's training harness (normalization, real-unit metrics, the
step/epoch hooks) but swaps OrdCon out for a baseline's own recipe:

  * task losses: masked regression for age & height (z-scored) + BCE-with-logits
    for gender;
  * combined either by learnable-uncertainty weighting (Kendall) or a plain sum;
  * discriminative learning rates: a low LR for a (fine-tuned) encoder and a
    higher LR for newly-initialised layers -- standard for stable fine-tuning.

Gender is supervised only as an auxiliary task; it is never reported (metrics
cover age & height, in real units, per speaker).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from retro_profile.data.collate import BagBatch
from retro_profile.losses.regression import MaskedRegressionLoss
from retro_profile.losses.uncertainty import UncertaintyLoss
from retro_profile.training.lightning_module import RetroProfileLitModule
from retro_profile.training.metrics import RegressionMetrics
from retro_profile.training.optim import build_scheduler


class MultiTaskProfilingLitModule(RetroProfileLitModule):
    """Trainer for the multi-task baselines (age + height regression + gender).

    Shared by both RQ1 baselines, which differ only in how the task losses are
    combined and which regression loss they use:

      * **Gupta 2022** -- MSE + learnable-uncertainty weighting
        (``regression_kind='mse'``, ``loss_combination='uncertainty'``),
      * **Jaid 2023**  -- MAE + a plain sum
        (``regression_kind='l1'``,  ``loss_combination='sum'``).

    Gender is supervised only as an auxiliary task (and, for Gupta, to make the
    mixture gate gender-meaningful); it is never reported.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_names: list[str],
        gender_name: str = "gender",
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-2,
        scheduler: dict[str, Any] | None = None,
        label_stats: dict[str, dict[str, float]] | None = None,
        normalize_targets: bool = True,
        regression_kind: str = "mse",
        loss_combination: str = "uncertainty",
        combined_normalisers: dict[str, float] | None = None,
        mixup_alpha: float | None = None,
        **_: Any,
    ) -> None:
        # Skip RetroProfileLitModule.__init__ (which builds MultitaskLoss/OrdCon);
        # set up the LightningModule ourselves with the baseline's loss instead.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(ignore=["model"])
        if loss_combination not in {"uncertainty", "sum"}:
            raise ValueError(
                f"loss_combination must be 'uncertainty' or 'sum', got {loss_combination!r}"
            )

        self.model = model
        # Mixup on training batches (training/mixup.py; Gupta / Truong); None = off, exactly as before.
        self.mixup_alpha = mixup_alpha
        self.target_names = list(target_names)             # regression targets (age, height)
        self.gender_name = gender_name
        self.normalize_targets = normalize_targets
        self.label_stats = label_stats or {}
        # Push train-set stats into the model so it denormalizes to real units.
        if self.label_stats and hasattr(self.model, "set_label_stats"):
            self.model.set_label_stats(self.label_stats)

        self.loss_combination = loss_combination
        self.reg_loss = MaskedRegressionLoss(kind=regression_kind)
        self.uncertainty_loss = (
            UncertaintyLoss(self.target_names + [gender_name])
            if loss_combination == "uncertainty" else None
        )

        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        self.scheduler_cfg = scheduler or {
            "name": "warmup_cosine", "warmup_steps": 500, "total_steps": 30_000,
        }

        # `combined_normalisers` parametrises `val/combined_mae`, the
        # checkpoint-selection metric (see RegressionMetrics.compute).
        self.train_metrics = RegressionMetrics(self.target_names, combined_normalisers)
        self.val_metrics = RegressionMetrics(self.target_names, combined_normalisers)
        self.test_metrics = RegressionMetrics(self.target_names, combined_normalisers)

    # ---- gender loss -------------------------------------------------------
    def _gender_loss(self, logit: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """Masked BCE-with-logits on the predicted gender score.

        The model returns a gender *logit* (not a probability); BCE-with-logits
        is the AMP/fp16-safe, numerically-stable choice (plain binary_cross_entropy
        is unsafe to autocast).
        """
        m = (mask & ~torch.isnan(target)).to(logit.dtype)
        target_safe = torch.where(m.bool(), target, torch.zeros_like(target))
        elem = F.binary_cross_entropy_with_logits(logit, target_safe, reduction="none")
        return (elem * m).sum() / m.sum().clamp_min(1.0)

    # ---- core step (overrides base) ---------------------------------------
    def _shared_step(self, batch: BagBatch, stage: str):
        preds = self.model(batch)                          # {age, height, gender}

        # Regression task losses in z-score space (age/height).
        norm_preds = self._normalize({t: preds[t] for t in self.target_names})
        norm_labels = self._normalize(batch.labels)
        task_losses: dict[str, torch.Tensor] = {}
        for t in self.target_names:
            task_losses[t] = self.reg_loss(norm_preds[t], norm_labels[t], mask=batch.label_mask[t])

        # Gender task loss (BCE), if the label is present.
        g = self.gender_name
        if g in preds and g in batch.labels:
            task_losses[g] = self._gender_loss(preds[g], batch.labels[g], batch.label_mask[g])

        if self.uncertainty_loss is not None:
            weighted = self.uncertainty_loss(task_losses)  # learnable-uncertainty weighting
            total = weighted["total"]
        else:                                              # plain sum (Jaid 2023)
            weighted = {}
            total = torch.stack(list(task_losses.values())).sum()

        loss_components: dict[str, torch.Tensor] = {"total": total}
        for t in self.target_names:
            loss_components[f"reg/{t}"] = task_losses[t]
        if g in task_losses:
            loss_components[f"bce/{g}"] = task_losses[g]
        for k, v in weighted.items():
            if k.startswith("log_var/"):
                loss_components[k] = v

        # Metrics on real-unit age/height only (gender excluded from reporting).
        metrics = {"train": self.train_metrics, "val": self.val_metrics,
                   "test": self.test_metrics}[stage]
        metrics.update({t: preds[t] for t in self.target_names}, batch.labels, batch.label_mask)
        return loss_components, preds

    # ---- optimizer (discriminative LR) ------------------------------------
    def configure_optimizers(self):
        enc_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc_params if name.startswith("model.encoder") else head_params).append(p)

        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": self.head_lr})
        if enc_params:                                      # empty when encoder is frozen (Config B)
            groups.append({"params": enc_params, "lr": self.encoder_lr})

        opt = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        sched = build_scheduler(opt, **self.scheduler_cfg)
        if sched is None:
            return opt
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# Backwards-compatible name: this module started life as the Gupta-only trainer.
# Existing configs and checkpoints refer to it by that name.
Gupta2022LitModule = MultiTaskProfilingLitModule
