"""Custom Lightning callbacks specific to RETRO-Profile."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from retro_profile.retrieval.builder import attach_datastore, build_datastore


def _build_and_attach(trainer: pl.Trainer, pl_module: pl.LightningModule,
                      loader_key: str, split: str):
    """Build a datastore from the train split and attach it to the model's heads.

    Restores training mode afterwards -- ``build_datastore`` puts the model in
    eval() and we are being called mid-training.
    """
    dm = trainer.datamodule
    if dm is None or not hasattr(dm, loader_key):
        return None
    was_training = pl_module.model.training
    loader = getattr(dm, loader_key)(split)
    ds = build_datastore(
        model=pl_module.model,
        dataloader=loader,
        targets=pl_module.target_names,
        device=pl_module.device,
    )
    attach_datastore(pl_module.model, ds)
    if was_training:
        pl_module.model.train()
    return ds


class PeriodicDatastoreRebuild(pl.Callback):
    """Rebuild the FAISS datastore every N epochs.

    Useful late in training, once the projector has converged enough that
    embeddings are stable. Building during early epochs is wasteful — the
    embedding space hasn't organized yet, so the kNN anchors are noise.

    Args:
        every_n_epochs: Rebuild cadence; 0 disables.
        start_epoch:    Earliest epoch at which to start rebuilding.
        train_loader_key: Name on the trainer's datamodule used to fetch the
                          utterance-mode dataloader for building.
    """

    def __init__(
        self,
        every_n_epochs: int = 0,
        start_epoch: int = 5,
        train_loader_key: str = "utterance_loader",
    ) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.start_epoch = start_epoch
        self.train_loader_key = train_loader_key

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.every_n_epochs <= 0:
            return
        epoch = trainer.current_epoch
        if epoch < self.start_epoch or (epoch + 1) % self.every_n_epochs != 0:
            return
        ds = _build_and_attach(trainer, pl_module, self.train_loader_key, "train")
        if ds is not None:
            pl_module.print(
                f"[PeriodicDatastoreRebuild] Rebuilt datastore at epoch {epoch} (size={len(ds)})."
            )


class TwoPhaseRetrievalCurriculum(pl.Callback):
    """Two-phase training for Novelty 3 (the kNN residual head).

    Phase 1 (epochs < ``start_epoch``): the datastore is empty, so each head runs
    its plain ``direct`` MLP while MIL pooling + OrdCon organize the embedding
    space. The ``residual`` branch is allocated but receives NO gradient.

    Phase 2 (epochs >= ``start_epoch``): we build the datastore from the training
    split and attach it. Every head switches to ``anchor + residual``, so the
    residual branch finally trains -- it learns to correct the kNN interpolation.

    Why freeze the embedding path in phase 2 (``freeze_embedding=True``)?
    The datastore is a snapshot of the projected space. If the encoder/pooling/
    projector keep moving, that snapshot goes stale and the retrieved neighbours
    stop matching the query space. Freezing them makes the index valid for the
    whole of phase 2 and focuses phase 2 on what it is for: learning the residual.
    (Set it False and use ``rebuild_every_n_epochs`` to keep adapting instead.)

    Note: leave-one-out retrieval (see ``RetrievalDatastore.knn_anchor``) is what
    stops a training speaker from retrieving its own label here.
    """

    def __init__(
        self,
        start_epoch: int = 20,
        rebuild_every_n_epochs: int = 0,
        freeze_embedding: bool = True,
        reset_early_stopping: bool = True,
        restore_best_phase1: bool = True,
        reset_checkpoint_best: bool = False,
        split: str = "train",
        train_loader_key: str = "utterance_loader",
    ) -> None:
        super().__init__()
        self.start_epoch = start_epoch
        self.rebuild_every_n_epochs = rebuild_every_n_epochs
        self.freeze_embedding = freeze_embedding
        self.reset_early_stopping = reset_early_stopping
        self.restore_best_phase1 = restore_best_phase1
        self.reset_checkpoint_best = reset_checkpoint_best
        self.split = split
        self.train_loader_key = train_loader_key
        self.activated = False
        self._suspended_patience: dict[int, int] = {}

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _reset_checkpoint_tracking(trainer: pl.Trainer) -> int:
        """Make ModelCheckpoint start selecting the best *phase-2* checkpoint.

        Off by default (``reset_checkpoint_best=False``), which preserves the
        original behaviour: ModelCheckpoint keeps scoring across both phases, so
        "best" is usually a phase-1 model and the phase-2 model must be read from
        ``last.ckpt``.

        Turning it on clears the tracker at the transition, so "best" becomes the
        best phase-2 model by validation. That makes a two-phase run directly
        comparable with single-phase runs, which are all evaluated at their
        best-by-validation checkpoint. Must run AFTER ``restore_best_phase1``,
        which reads ``best_model_path``.
        """
        n = 0
        for cb in trainer.callbacks:
            if not isinstance(cb, ModelCheckpoint):
                continue
            worst = float("inf") if cb.mode == "min" else float("-inf")
            cb.best_k_models = {}
            cb.kth_best_model_path = ""
            cb.kth_value = torch.tensor(worst)
            cb.best_model_path = ""
            cb.best_model_score = None
            n += 1
        return n

    def _freeze_embedding_path(self, pl_module: pl.LightningModule) -> int:
        """Freeze encoder + pooling + projector so the indexed space is stable."""
        frozen = 0
        for name in ("encoder", "pooling", "projector"):
            module = getattr(pl_module.model, name, None)
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += p.numel()
        return frozen

    def _suspend_early_stopping(self, trainer: pl.Trainer) -> None:
        """Stop EarlyStopping from ending the run before phase 2 ever starts.

        Phase 1 converges and plateaus well before ``start_epoch``, so a normal
        patience budget will happily terminate training first -- and the residual
        head would never be trained at all. We park patience at a huge value and
        restore the real one at the transition.
        """
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping) and id(cb) not in self._suspended_patience:
                self._suspended_patience[id(cb)] = cb.patience
                cb.patience = 10**9

    def _restore_early_stopping(self, trainer: pl.Trainer, reset: bool) -> None:
        """Hand phase 2 back its real patience budget (optionally a fresh one)."""
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.patience = self._suspended_patience.pop(id(cb), cb.patience)
                if reset:
                    cb.wait_count = 0
                    cb.best_score = torch.tensor(
                        float("inf") if cb.mode == "min" else float("-inf")
                    )

    # ---- resume support ---------------------------------------------------
    def state_dict(self) -> dict:
        """Persist the phase flag so a resumed run does not re-run phase 1.

        Without this, resuming mid-phase-2 would restore the best *phase-1*
        weights and wipe the residual head that phase 2 had already trained.
        """
        return {"activated": self.activated}

    def load_state_dict(self, state_dict: dict) -> None:
        self.activated = bool(state_dict.get("activated", False))

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.activated:
            # Resumed inside phase 2: re-attach the datastore and re-freeze the
            # embedding path (neither survives a process restart), but do NOT
            # roll weights back to phase 1.
            ds = _build_and_attach(trainer, pl_module, self.train_loader_key, self.split)
            if ds is not None:
                if self.freeze_embedding:
                    self._freeze_embedding_path(pl_module)
                pl_module.print(
                    f"[TwoPhase] resumed in PHASE 2: datastore re-attached (size={len(ds)})."
                )
        else:
            self._suspend_early_stopping(trainer)

    @staticmethod
    def _restore_best_phase1_weights(trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Roll the model back to phase 1's best-by-validation checkpoint.

        Phase 1 keeps improving, peaks, then drifts. Freezing "wherever we happen
        to be at start_epoch" can therefore capture an already-degraded embedding.
        Restoring the best checkpoint makes phase 2 start from phase 1's best
        model and removes the sensitivity to exactly when we switch over.
        """
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                ckpt = torch.load(cb.best_model_path, map_location="cpu", weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                model_state = {k[len("model."):]: v for k, v in state.items()
                               if k.startswith("model.")}
                if model_state:
                    pl_module.model.load_state_dict(model_state, strict=False)
                    return cb.best_model_path
        return None

    # ---- hook -------------------------------------------------------------
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch < self.start_epoch:
            return

        if not self.activated:
            # Roll back to phase 1's best model BEFORE indexing it, so the
            # datastore is built from the best embedding rather than a drifted one.
            restored = None
            if self.restore_best_phase1:
                restored = self._restore_best_phase1_weights(trainer, pl_module)
            ds = _build_and_attach(trainer, pl_module, self.train_loader_key, self.split)
            if ds is None:
                pl_module.print(
                    "[TwoPhase] datamodule has no utterance loader; staying in phase 1."
                )
                return
            self.activated = True
            msg = f"[TwoPhase] PHASE 2 at epoch {epoch}: datastore attached (size={len(ds)})"
            if restored:
                msg += f", restored best phase-1 weights ({Path(restored).name})"
            if self.freeze_embedding:
                msg += f", froze {self._freeze_embedding_path(pl_module):,} embedding params"
            self._restore_early_stopping(trainer, reset=self.reset_early_stopping)
            msg += ", early-stopping restored"
            # After restore_best_phase1 (which reads best_model_path), optionally
            # let ModelCheckpoint start selecting the best PHASE-2 checkpoint.
            if self.reset_checkpoint_best and self._reset_checkpoint_tracking(trainer):
                msg += ", checkpoint selection reset to phase 2"
            pl_module.print(msg + ". Residual head now trains.")
            return

        # Already in phase 2: optionally refresh a stale index.
        if self.rebuild_every_n_epochs > 0 and (epoch - self.start_epoch) % self.rebuild_every_n_epochs == 0:
            ds = _build_and_attach(trainer, pl_module, self.train_loader_key, self.split)
            if ds is not None:
                pl_module.print(f"[TwoPhase] datastore refreshed at epoch {epoch} (size={len(ds)}).")


class LogTrainableParamCount(pl.Callback):
    """Log how many parameters are actually being trained (frozen-aware)."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        pl_module.print(
            f"[LogTrainableParamCount] total={total:,}, trainable={trainable:,} "
            f"({100.0 * trainable / max(total, 1):.2f}%)"
        )


class PerEpochDiagnostics(pl.Callback):
    """Append one row per (epoch, target) of *validation* diagnostics to a CSV.

    Lightning already logs val MAE/RMSE per epoch, but not the things that
    explain a curve: how selective the MIL attention is, what it is selecting
    on, and how much of a retrieval prediction is the kNN anchor rather than the
    learned residual. This records all of it, every epoch, so the run can be
    read afterwards instead of guessed at.

    **Validation only, deliberately.** Evaluating the test set each epoch would
    let the shape of the test curve influence which epoch we keep, and the test
    number would stop being an unbiased estimate. The dev split answers the same
    diagnostic questions at no such cost.

    Cost is ~70 speakers x bag_size clips per epoch -- seconds against a
    minute-long epoch.

    Columns:
        epoch, target, mae, rmse, pearson, n
        attn_entropy, attn_max, n_eff        -- how concentrated the attention is
        w_SA, w_SI, w_SX                     -- TIMIT sentence-type preference,
                                                relative to uniform (1.0 = mean pooling)
        anchor_share                         -- |anchor| / (|anchor| + |delta|);
                                                1.0 = the kNN anchor is the whole
                                                prediction and the residual is inert
        retrieval_active                     -- whether a datastore was attached
        frame_eff_share                      -- frame-level attention, mean n_eff /
                                                n_frames per clip (1.0 = plain mean;
                                                NaN without frame attention)
    """

    #: TIMIT utterance ids encode sentence type: SA (identical text for every
    #: speaker), SX (shared across groups), SI (unique per speaker).
    SENTENCE_TYPES = ("SA", "SI", "SX")

    def __init__(self, output_path: str | Path, every_n_epochs: int = 1) -> None:
        self.output_path = Path(output_path)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self._rows: list[dict] = []

    @staticmethod
    def _sentence_type(utt_id: str) -> str:
        u = str(utt_id).upper()
        for prefix in ("SA", "SX", "SI"):
            if u.startswith(prefix):
                return prefix
        return "other"

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if epoch % self.every_n_epochs:
            return

        loader = trainer.datamodule.val_dataloader() if trainer.datamodule else None
        if loader is None:
            return

        # Imported here: evaluation imports training.metrics, so a module-level
        # import would be circular.
        from retro_profile.evaluation.evaluator import evaluate_with_explanations

        model = pl_module.model
        was_training = model.training
        # `model.heads` is an nn.ModuleDict, which has no .get() -- an earlier
        # version used one and silently reported retrieval as never active,
        # which suppressed the phase-switch marker on every diagnostic plot.
        retrieval_active = False
        heads = getattr(model, "heads", None)
        if heads is not None:
            try:
                retrieval_active = any(
                    bool(getattr(h, "has_datastore", False)) for h in heads.values()
                )
            except (AttributeError, TypeError):
                retrieval_active = False

        frame_rows: list[dict] = []
        try:
            df, _, attn = evaluate_with_explanations(
                model=model,
                dataloader=loader,
                target_names=list(pl_module.target_names),
                datastore=None,          # the heads already hold their own datastore
                device=pl_module.device,
                return_attention=True,
                frame_attention=frame_rows,
            )
        except Exception as exc:                       # diagnostics must never kill a run
            print(f"[PerEpochDiagnostics] skipped epoch {epoch}: {exc}")
            return
        finally:
            model.train(was_training)

        if not attn.empty:
            attn = attn.copy()
            attn["type"] = attn["utt_id"].map(self._sentence_type)
            attn["uniform"] = 1.0 / attn["n_clips"]

        for target, g in df.groupby("target"):
            g = g.dropna(subset=["prediction", "ground_truth"])
            if g.empty:
                continue
            err = g["prediction"] - g["ground_truth"]
            row = {
                "epoch": epoch,
                "target": target,
                "mae": float(err.abs().mean()),
                "rmse": float((err**2).mean() ** 0.5),
                "pearson": float(g["prediction"].corr(g["ground_truth"])),
                "n": int(len(g)),
                "retrieval_active": retrieval_active,
            }
            for col in ("attn_entropy", "attn_max", "n_eff"):
                row[col] = float(g[col].mean()) if col in g else float("nan")
            # Frame-level attention (shared by every target, so the same value on
            # each target's row). 1.0 = still a plain mean over the clip's frames.
            row["frame_eff_share"] = (
                float(sum(r["eff_share"] for r in frame_rows) / len(frame_rows))
                if frame_rows else float("nan")
            )

            # How much of the prediction is the retrieved anchor vs the learned
            # residual. Near 1.0 means the residual has stopped contributing.
            if "anchor" in g and g["anchor"].notna().any():
                a, d = g["anchor"].abs(), g["delta"].abs()
                # NOTE: |anchor| is an absolute label (~30 yr, ~175 cm), so this
                # ratio is dominated by scale and always sits near 1.0. Read
                # mean_abs_delta and anchor_mae for the residual's real effect.
                row["anchor_share"] = float((a / (a + d).clip(lower=1e-9)).mean())
                row["mean_abs_delta"] = float(d.mean())
                row["anchor_mae"] = float(
                    (g["anchor"] - g["ground_truth"]).abs().mean()
                )
            else:
                row["anchor_share"] = float("nan")
                row["mean_abs_delta"] = float("nan")
                row["anchor_mae"] = float("nan")

            if not attn.empty:
                sub = attn[attn["target"] == target]
                for st in self.SENTENCE_TYPES:
                    s = sub[sub["type"] == st]
                    row[f"w_{st}"] = (
                        float((s["weight"] / s["uniform"]).mean()) if len(s) else float("nan")
                    )
            self._rows.append(row)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_csv(self.output_path, index=False)
