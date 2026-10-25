"""Training entrypoint.

Usage:
    python scripts/train.py                                  # default = retro_profile + timit
    python scripts/train.py experiment=retro_profile_full
    python scripts/train.py experiment=no_mil data=nisp
    python scripts/train.py model=baseline_wavlm_mlp data=timit
    python scripts/train.py experiment=retro_profile_full resume_from=auto   # resume after a disconnect

Hydra composes:
    configs/default.yaml
        + configs/model/<model>.yaml
        + configs/data/<data>.yaml
        + configs/training/<training>.yaml
        + configs/experiment/<experiment>.yaml  (optional, applied last)
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from retro_profile.training import (
    LogTrainableParamCount,
    PerEpochDiagnostics,
    PeriodicDatastoreRebuild,
    RetroProfileLitModule,
    TwoPhaseRetrievalCurriculum,
)
from retro_profile.utils import get_logger, instantiate, seed_everything, setup_logging


def embedding_dim_of(model) -> int:
    """Width of the space OrdCon operates on -- RETRO-Profile only.

    Tries the shared projector, then an encoder, then a generic ``feature_dim``.
    Baselines that have none of these never reach here: they use a different
    LightningModule and have no OrdCon term.
    """
    for owner, attr in (
        (getattr(model, "projector", None), "output_dim"),
        (getattr(model, "encoder", None), "output_dim"),
        (model, "feature_dim"),
        (model, "output_dim"),
    ):
        if owner is not None and hasattr(owner, attr):
            return int(getattr(owner, attr))
    raise AttributeError(
        f"Cannot infer an embedding dim for {type(model).__name__}: it exposes "
        "no projector.output_dim, encoder.output_dim, feature_dim or output_dim."
    )


def apply_sharing_strategy(env: dict[str, str] | None = None) -> str | None:
    """Switch how DataLoader workers hand tensors to the main process, when TORCH_SHARING_STRATEGY is set.

    The default ("file_descriptor") passes one open file per shared tensor. Joint Jaid (job 60955713,
    13 Oct 2026: batch 64, 112 fast epochs) exhausted the open-files limit and died with "received 0
    items of ancdata" / "Pin memory thread exited unexpectedly". TORCH_SHARING_STRATEGY=file_system
    uses named shared-memory files instead. Unset = PyTorch's default, unchanged; results are identical.
    """
    strategy = (os.environ if env is None else env).get("TORCH_SHARING_STRATEGY")
    if strategy:
        torch.multiprocessing.set_sharing_strategy(strategy)
    return strategy


def _build_loggers(cfg: DictConfig, output_dir: Path) -> list:
    loggers = []
    log_cfg = cfg.get("logger", {})
    if log_cfg.get("tensorboard", {}).get("enabled", True):
        loggers.append(TensorBoardLogger(save_dir=str(output_dir), name="tb"))
    if log_cfg.get("wandb", {}).get("enabled", False):
        loggers.append(
            WandbLogger(
                project=log_cfg["wandb"].get("project", "retro-profile"),
                entity=log_cfg["wandb"].get("entity"),
                name=cfg.run_name,
                save_dir=str(output_dir),
            )
        )
    return loggers


def _build_callbacks(cfg: DictConfig, output_dir: Path) -> list:
    callbacks: list = [
        LogTrainableParamCount(),
        LearningRateMonitor(logging_interval="step"),
    ]
    # Per-epoch validation diagnostics (accuracy + attention + anchor/residual
    # split). On by default: it costs seconds per epoch and is the only record
    # of *why* a curve looks the way it does. Never runs on the test split.
    diag_cfg = cfg.training.get("diagnostics", {}) or {}
    if diag_cfg.get("enabled", True):
        callbacks.append(
            PerEpochDiagnostics(
                output_path=output_dir / "diagnostics.csv",
                every_n_epochs=int(diag_cfg.get("every_n_epochs", 1)),
            )
        )

    es_cfg = cfg.training.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/total"),
                mode=es_cfg.get("mode", "min"),
                patience=int(es_cfg.get("patience", 10)),
            )
        )
    ck_cfg = cfg.training.get("checkpoint", {})
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            monitor=ck_cfg.get("monitor", "val/total"),
            mode=ck_cfg.get("mode", "min"),
            save_top_k=int(ck_cfg.get("save_top_k", 3)),
            save_last=bool(ck_cfg.get("save_last", True)),
            filename="epoch={epoch}-step={step}-val={" + ck_cfg.get("monitor", "val/total") + ":.4f}",
            auto_insert_metric_name=False,
        )
    )
    # Optional extra copies: the best epoch for EACH trait on its own validation
    # MAE, alongside (never instead of) the combined checkpoint above. Age keeps
    # improving late while height peaks early, so one combined checkpoint is
    # saved at a compromise point; these let each trait be read at its own best.
    # Not rule R5: nothing early-stops on them and each copy is only ever scored
    # on its own trait. Appended AFTER the main checkpoint, because Lightning's
    # trainer.checkpoint_callback (best_checkpoint.txt, the final test pass) is
    # the FIRST ModelCheckpoint in the list.
    per_target = list(ck_cfg.get("per_target", []) or [])
    if per_target and bool((cfg.training.get("two_phase", {}) or {}).get("enabled", False)):
        raise ValueError(
            "training.checkpoint.per_target cannot be combined with two_phase: the phase "
            "switch resets every ModelCheckpoint and would leave orphaned phase-1 copies "
            "next to the phase-2 ones. Measure kNN with evaluate.py --force-retrieval instead."
        )
    for t in per_target:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(output_dir / "checkpoints"),
                monitor=f"val/{t}/mae",
                mode="min",
                save_top_k=1,
                save_last=False,
                filename=f"best_{t}-epoch={{epoch}}-val_{t}={{val/{t}/mae:.4f}}",
                auto_insert_metric_name=False,
            )
        )
    # Two-phase curriculum for Novelty 3: phase 1 trains the direct head while
    # MIL+OrdCon organize the space, then the datastore is attached so the
    # residual head actually trains. Takes precedence over a plain rebuild.
    two_phase = cfg.training.get("two_phase", {}) or {}
    if bool(two_phase.get("enabled", False)):
        callbacks.append(
            TwoPhaseRetrievalCurriculum(
                start_epoch=int(two_phase.get("start_epoch", 20)),
                rebuild_every_n_epochs=int(two_phase.get("rebuild_every_n_epochs", 0)),
                freeze_embedding=bool(two_phase.get("freeze_embedding", True)),
                reset_early_stopping=bool(two_phase.get("reset_early_stopping", True)),
                restore_best_phase1=bool(two_phase.get("restore_best_phase1", True)),
                reset_checkpoint_best=bool(two_phase.get("reset_checkpoint_best", False)),
            )
        )
    else:
        ds_rebuild = cfg.training.get("datastore_rebuild", {})
        if int(ds_rebuild.get("every_n_epochs", 0)) > 0:
            callbacks.append(
                PeriodicDatastoreRebuild(
                    every_n_epochs=int(ds_rebuild["every_n_epochs"]),
                    start_epoch=int(ds_rebuild.get("start_epoch", 5)),
                )
            )
    return callbacks


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> float:
    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=output_dir, level="INFO")
    log = get_logger("train")

    log.info(f"Run name: {cfg.run_name}")
    log.info(f"Output dir: {output_dir}")
    log.info("Resolved config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(int(cfg.seed))
    strategy = apply_sharing_strategy()
    if strategy:
        log.info(f"DataLoader sharing strategy: {torch.multiprocessing.get_sharing_strategy()} "
                 "(TORCH_SHARING_STRATEGY)")

    # ---- DataModule -------------------------------------------------------
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")
    log.info(
        f"Datamodule: {type(datamodule).__name__}; "
        f"train speakers={datamodule.train_set.num_speakers}, "
        f"val speakers={datamodule.val_set.num_speakers}"
    )
    log.info(f"Label stats: {datamodule.label_stats}")

    # ---- Model ------------------------------------------------------------
    model = instantiate(cfg.model)
    log.info(f"Model: {type(model).__name__}")

    # ---- Trainer steps estimate (for warmup_cosine total_steps) -----------
    steps_per_epoch = max(1, len(datamodule.train_dataloader()))
    total_steps = int(cfg.training.max_epochs * steps_per_epoch
                      / max(1, cfg.training.get("accumulate_grad_batches", 1)))
    sched_cfg = OmegaConf.to_container(cfg.training.scheduler, resolve=True)
    sched_cfg["total_steps"] = total_steps

    # ---- Lightning module (config-selectable) -----------------------------
    # Default is RETRO-Profile; the multi-task RQ1 baselines request their own
    # LightningModule via `training.lit_module` ("multitask", or the legacy
    # alias "gupta"). Their hyperparameters live under `training.baseline`
    # (legacy key: `training.gupta`).
    lit_kind = cfg.training.get("lit_module", "retro")
    # Mixup (training/mixup.py, 2026-10-18): only when a config sets training.mixup.enabled; every
    # earlier config lacks the key, so it stays off and those runs train exactly as before.
    mixup_cfg = cfg.training.get("mixup", None) or {}
    mixup_alpha = float(mixup_cfg.get("alpha", 1.0)) if mixup_cfg.get("enabled", False) else None

    # Normalisers for `val/combined_mae`, the checkpoint-selection metric
    # (SESSION_HANDOFF.md §7.2). None falls back to the TIMIT chance MAEs in
    # RegressionMetrics; override per corpus via `training.combined_normalisers`.
    combined_normalisers = (
        OmegaConf.to_container(cfg.training.combined_normalisers, resolve=True)
        if cfg.training.get("combined_normalisers") is not None
        else None
    )

    if lit_kind in ("multitask", "gupta"):
        from retro_profile.training import MultiTaskProfilingLitModule

        bcfg = OmegaConf.to_container(
            cfg.training.get("baseline", None) or cfg.training.get("gupta", {}) or {},
            resolve=True,
        )
        lit = MultiTaskProfilingLitModule(
            model=model,
            target_names=list(model.target_names),
            gender_name=bcfg.get("gender_name", "gender"),
            encoder_lr=float(bcfg.get("encoder_lr", 1e-5)),
            head_lr=float(bcfg.get("head_lr", 1e-4)),
            weight_decay=float(bcfg.get("weight_decay", 1e-2)),
            regression_kind=bcfg.get("regression_kind", "mse"),
            loss_combination=bcfg.get("loss_combination", "uncertainty"),
            scheduler=sched_cfg,
            label_stats=datamodule.label_stats,
            normalize_targets=bool(cfg.data.get("normalize_targets", True)),
            combined_normalisers=combined_normalisers,
            mixup_alpha=mixup_alpha,
        )
    else:
        loss_cfg = OmegaConf.to_container(cfg.training.loss, resolve=True)
        lit = RetroProfileLitModule(
            model=model,
            target_names=list(model.target_names),
            # Only RETRO needs this (it sizes OrdCon's learnable per-target axis);
            # the baselines have no such space, and some have neither a projector
            # nor an `encoder` attribute at all.
            embedding_dim=embedding_dim_of(model),
            regression_kind=loss_cfg.get("regression_kind", "smooth_l1"),
            regression_weight=float(loss_cfg.get("regression_weight", 1.0)),
            ordcon_weight=float(loss_cfg.get("ordcon_weight", 0.0)),
            ordcon_kwargs=loss_cfg.get("ordcon_kwargs"),
            # Which embedding OrdCon organises; see RetroProfileLitModule.
            ordcon_embedding=loss_cfg.get("ordcon_embedding", "retrieval"),
            # Penalise MIL attention that concentrates on few utterances
            # (losses/attention_entropy.py). 0.0 keeps every existing run
            # bit-identical.
            attention_entropy_weight=float(loss_cfg.get("attention_entropy_weight", 0.0)),
            attention_entropy_per_target=loss_cfg.get("attention_entropy_per_target"),
            optimizer=OmegaConf.to_container(cfg.training.optimizer, resolve=True),
            scheduler=sched_cfg,
            label_stats=datamodule.label_stats,
            normalize_targets=bool(cfg.data.get("normalize_targets", True)),
            # Set only when fine-tuning the encoder; None keeps the single
            # param group every frozen-encoder run has used.
            encoder_lr=(
                float(cfg.training.encoder_lr)
                if cfg.training.get("encoder_lr") is not None
                else None
            ),
            combined_normalisers=combined_normalisers,
            # Own LR for frame-level (temporal) attention; None = trains at the head LR, as before.
            frame_attention_lr=(
                float(cfg.training.frame_attention_lr)
                if cfg.training.get("frame_attention_lr") is not None
                else None
            ),
            mixup_alpha=mixup_alpha,
        )

    # ---- Trainer ----------------------------------------------------------
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=int(cfg.training.max_epochs),
        gradient_clip_val=float(cfg.training.get("gradient_clip_val", 0.0)) or None,
        accumulate_grad_batches=int(cfg.training.get("accumulate_grad_batches", 1)),
        precision=cfg.get("precision", "32"),
        accelerator=("gpu" if (cfg.device == "auto" and torch.cuda.is_available())
                     else (cfg.device if cfg.device != "auto" else "cpu")),
        devices="auto",
        callbacks=_build_callbacks(cfg, output_dir),
        logger=_build_loggers(cfg, output_dir),
        log_every_n_steps=10,
        val_check_interval=cfg.training.get("val_check_interval", 1.0),
    )

    # ---- Resume handling --------------------------------------------------
    # resume_from: null -> fresh; "auto"/"last" -> <output_dir>/checkpoints/last.ckpt
    # if present (else fresh); otherwise an explicit path. A missing path warns and
    # starts fresh so the SAME command works on a first run and on a resume.
    resume_ckpt: str | None = None
    resume_cfg = cfg.get("resume_from")
    if resume_cfg:
        if str(resume_cfg).lower() in {"auto", "last"}:
            cand = output_dir / "checkpoints" / "last.ckpt"
            if cand.exists():
                resume_ckpt = str(cand)
            else:
                log.warning("resume_from=auto but no last.ckpt found; starting fresh.")
        elif Path(resume_cfg).exists():
            resume_ckpt = str(resume_cfg)
        else:
            log.warning(f"resume_from={resume_cfg!r} not found; starting fresh.")
    if resume_ckpt:
        log.info(f"Resuming training from checkpoint: {resume_ckpt}")

    trainer.fit(lit, datamodule=datamodule, ckpt_path=resume_ckpt)

    log.info("Training complete.")
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint: {best_path}")
    # The one source of truth for which checkpoint to evaluate or keep. It is
    # phase-aware, unlike a lowest-val= scan of the folder, which picks an
    # orphaned phase-1 file in a two-phase run. Read by scripts/select_checkpoint.py.
    if best_path:
        (output_dir / "best_checkpoint.txt").write_text(f"{best_path}\n")

    # Final test pass on the held-out test split if it's available.
    try:
        datamodule.setup("test")
        results = trainer.test(lit, datamodule=datamodule, ckpt_path="best")
        log.info(f"Test results: {results}")
    except Exception as e:
        log.warning(f"Skipping final test: {e}")

    # Return the monitored metric so Hydra sweeps can use it.
    monitor_key = cfg.training.checkpoint.get("monitor", "val/total")
    metric = trainer.callback_metrics.get(monitor_key)
    return float(metric) if metric is not None else float("nan")


if __name__ == "__main__":
    main()
