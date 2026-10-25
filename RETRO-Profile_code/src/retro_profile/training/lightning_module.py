"""PyTorch Lightning module for trait regression.

Generic over the underlying model — works with RetroProfile *and* any baseline
that conforms to ``BaseProfileModel`` (i.e. exposes ``forward(batch) -> dict``).
The Lightning module also carries the loss, the metrics, and the per-target
label normalization stats so we report errors in *original units* (years, cm)
even when the heads are trained on z-scored targets.
"""

from __future__ import annotations

import math
from typing import Any

import pytorch_lightning as pl
import torch

from retro_profile.data.collate import BagBatch
from retro_profile.losses.multitask import MultitaskLoss
from retro_profile.training.metrics import RegressionMetrics
from retro_profile.training.mixup import mixup_batch
from retro_profile.training.optim import build_optimizer, build_scheduler


class RetroProfileLitModule(pl.LightningModule):
    """LightningModule wrapping any BaseProfileModel."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_names: list[str],
        embedding_dim: int,
        regression_kind: str = "smooth_l1",
        regression_weight: float = 1.0,
        ordcon_weight: float = 0.5,
        ordcon_kwargs: dict[str, Any] | None = None,
        attention_entropy_weight: float = 0.0,
        attention_entropy_per_target: dict[str, float] | None = None,
        optimizer: dict[str, Any] | None = None,
        scheduler: dict[str, Any] | None = None,
        label_stats: dict[str, dict[str, float]] | None = None,
        normalize_targets: bool = True,
        encoder_lr: float | None = None,
        combined_normalisers: dict[str, float] | None = None,
        ordcon_embedding: str = "retrieval",
        frame_attention_lr: float | None = None,
        mixup_alpha: float | None = None,
    ) -> None:
        super().__init__()
        # Hyperparam logging — exclude the model object itself.
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        # Mixup on training batches (training/mixup.py); None = off, the step is exactly as before.
        self.mixup_alpha = mixup_alpha
        self.target_names = list(target_names)
        self.normalize_targets = normalize_targets
        self.label_stats = label_stats or {}

        # Which embedding OrdCon organises. "retrieval" is the shared vector the
        # datastore indexes -- the design in models/retro_profile.py. Before
        # 2026-10-10 the loss silently took the FIRST target's projected vector
        # instead ("first_target"); kept only so old runs can be reproduced.
        if ordcon_embedding not in ("retrieval", "first_target"):
            raise ValueError(
                f"ordcon_embedding must be 'retrieval' or 'first_target', "
                f"got {ordcon_embedding!r}"
            )
        self.ordcon_embedding = ordcon_embedding

        # The std each label is divided by in `_normalize`, so OrdCon can state
        # its magnitude scale in years/cm. Must mirror `_normalize` exactly:
        # nothing when targets are not normalised.
        label_std = (
            {t: float(s["std"]) for t, s in self.label_stats.items()}
            if self.normalize_targets else None
        )

        self.loss_fn = MultitaskLoss(
            target_names=self.target_names,
            embedding_dim=embedding_dim,
            regression_kind=regression_kind,
            regression_weight=regression_weight,
            ordcon_weight=ordcon_weight,
            ordcon_kwargs=ordcon_kwargs,
            attention_entropy_weight=attention_entropy_weight,
            attention_entropy_per_target=attention_entropy_per_target,
            label_std=label_std,
        )

        self.optimizer_cfg = optimizer or {"name": "adamw", "lr": 3e-4, "weight_decay": 1e-2}
        # Discriminative LR for fine-tuning. None (the default) keeps the single
        # param group used by every frozen-encoder run. See configure_optimizers.
        self.encoder_lr = encoder_lr
        # Own LR for the frame-level (temporal) attention, when set. At the head LR (1e-4) its
        # zero-initialised scorer never left the plain mean in 60 epochs (E26, eff_share 1.000).
        self.frame_attention_lr = frame_attention_lr
        self.scheduler_cfg = scheduler or {
            "name": "warmup_cosine",
            "warmup_steps": 500,
            "total_steps": 50_000,
        }

        # `combined_normalisers` parametrises `val/combined_mae`, the
        # checkpoint-selection metric (see RegressionMetrics.compute).
        self.train_metrics = RegressionMetrics(self.target_names, combined_normalisers)
        self.val_metrics = RegressionMetrics(self.target_names, combined_normalisers)
        self.test_metrics = RegressionMetrics(self.target_names, combined_normalisers)

    # ---- normalization -----------------------------------------------------
    #
    # The model heads always produce predictions in **real label units**
    # (e.g. years for age, cm for height). The clamp at (label_min, label_max)
    # inside each head only makes sense in real units, and the kNN anchor read
    # from the datastore is also in real units (because the labels stored in
    # the datastore are real). So predictions and `batch.labels` are already
    # on the same scale.
    #
    # To make the optimizer's job easier we still z-score *both sides* when
    # computing the loss (so age/cm trade off cleanly and the smooth-L1 beta is
    # in dimensionless std-units). The normalization is purely loss-side; the
    # head and the metrics both operate in real label units.
    def _normalize(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Z-score a dict of per-target tensors against the train-set stats."""
        if not self.normalize_targets or not self.label_stats:
            return values
        out = {}
        for t, v in values.items():
            stats = self.label_stats.get(t)
            if stats is None:
                out[t] = v
                continue
            out[t] = (v - stats["mean"]) / stats["std"]
        return out

    def _ordcon_input(self, components: dict[str, dict[str, Any]]) -> torch.Tensor:
        """The embedding OrdCon is applied to.

        NOT "any target's embedding". The projector's weights are shared, but
        with per-target MIL stacks its inputs differ, so `components[t]
        ["embedding"]` is a different vector for each target. Taking the first
        one -- the behaviour until 2026-10-10 -- meant both the age AND height
        ordering terms shaped the AGE-pooled vector, while FAISS searched a
        third vector that OrdCon never touched directly.
        """
        first = next(iter(components.values()))
        if self.ordcon_embedding == "retrieval" and first.get("retrieval_embedding") is not None:
            return first["retrieval_embedding"]
        return first["embedding"]

    # ---- core step ---------------------------------------------------------
    def _shared_step(
        self, batch: BagBatch, stage: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        components = self.model.forward_with_components(batch) \
            if hasattr(self.model, "forward_with_components") else None

        if components is not None:
            preds = {t: c["pred"] for t, c in components.items()}
            embeddings = self._ordcon_input(components)
            # Per-target MIL weights (B, K), for the attention-concentration
            # penalty. Absent for poolings that have no weights (e.g. mean).
            attention = {
                t: c["attention"] for t, c in components.items()
                if isinstance(c.get("attention"), torch.Tensor)
            } or None
        else:
            preds = self.model(batch)
            embeddings = None
            attention = None

        # Predictions and labels are both in REAL label units (years, cm).
        # We z-score both sides for the loss only — this is purely a scale
        # adjustment so that targets with very different ranges (e.g. age in
        # years vs height in cm) contribute comparably to the total loss.
        # Metrics are reported in real units below.
        loss_preds  = self._normalize(preds)
        loss_labels = self._normalize(batch.labels)

        loss_components = self.loss_fn(
            predictions=loss_preds,
            labels=loss_labels,
            label_mask=batch.label_mask,
            embeddings=embeddings,
            attention=attention,
            utt_mask=getattr(batch, "utt_mask", None),
        )

        # Metric updates use real-unit predictions and labels directly.
        if stage == "train":
            self.train_metrics.update(preds, batch.labels, batch.label_mask)
        elif stage == "val":
            self.val_metrics.update(preds, batch.labels, batch.label_mask)
        elif stage == "test":
            self.test_metrics.update(preds, batch.labels, batch.label_mask)

        return loss_components, preds

    # ---- Lightning hooks ---------------------------------------------------
    def training_step(self, batch: BagBatch, batch_idx: int) -> torch.Tensor:
        if self.mixup_alpha:                     # off by default: no random number drawn, batch untouched
            batch = mixup_batch(batch, self.mixup_alpha)
        loss_components, _ = self._shared_step(batch, "train")
        # Log scalar components.
        for k, v in loss_components.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, batch_size=batch.batch_size)
        return loss_components["total"]

    def validation_step(self, batch: BagBatch, batch_idx: int) -> torch.Tensor:
        loss_components, _ = self._shared_step(batch, "val")
        for k, v in loss_components.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss_components["total"]

    def test_step(self, batch: BagBatch, batch_idx: int) -> torch.Tensor:
        loss_components, _ = self._shared_step(batch, "test")
        for k, v in loss_components.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        return loss_components["total"]

    def on_train_epoch_end(self) -> None:
        for k, v in self.train_metrics.compute().items():
            self.log(f"train/{k}", v, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        for k, v in self.val_metrics.compute().items():
            self.log(f"val/{k}", v, on_epoch=True, prog_bar=k.endswith("/mae"))
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        for k, v in self.test_metrics.compute().items():
            self.log(f"test/{k}", v, on_epoch=True)
        self.test_metrics.reset()

    # ---- Optimizer ---------------------------------------------------------
    # Only the *pretrained backbone* belongs in the slow group. WavLMEncoder also
    # owns `layer_weights`, which is freshly initialised to zeros and must train
    # at the head LR like any other new parameter -- hence matching on
    # "model.encoder.model." (the HF module) rather than "model.encoder".
    _BACKBONE_PREFIX = "model.encoder.model."
    _FRAME_ATTENTION_PREFIX = "model.encoder.frame_pool."

    def _param_groups(self) -> list[dict[str, Any]]:
        """Two LR groups when fine-tuning: slow backbone, fast new layers -- plus, when
        ``frame_attention_lr`` is set, a third for the frame-level attention.

        A single LR across a 94.4 M pretrained encoder and randomly-initialised
        heads fails both ways -- high enough to train the heads destroys the
        pretrained features, low enough to protect them leaves the heads unlearnt.
        """
        base_lr = self.optimizer_cfg.get("lr", 3e-4)
        backbone, frame, new = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(self._BACKBONE_PREFIX):
                backbone.append(p)
            elif self.frame_attention_lr is not None and name.startswith(self._FRAME_ATTENTION_PREFIX):
                frame.append(p)
            else:
                new.append(p)

        groups: list[dict[str, Any]] = []
        if new:
            groups.append({"params": new, "lr": base_lr})
        if backbone:                    # empty whenever the encoder is frozen
            groups.append({"params": backbone, "lr": self.encoder_lr})
        if frame:
            groups.append({"params": frame, "lr": self.frame_attention_lr})
        return groups

    def configure_optimizers(self):
        if self.encoder_lr is None and self.frame_attention_lr is None:
            params = [p for p in self.parameters() if p.requires_grad]
        else:
            params = self._param_groups()
        opt = build_optimizer(params, **self.optimizer_cfg)
        sched = build_scheduler(opt, **self.scheduler_cfg)
        if sched is None:
            return opt
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
