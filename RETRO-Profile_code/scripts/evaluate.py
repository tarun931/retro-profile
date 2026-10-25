"""Evaluate a trained RETRO-Profile model in-domain and cross-corpus.

Usage:
    python scripts/evaluate.py \
        --ckpt outputs/<run>/checkpoints/last.ckpt \
        --config-dir outputs/<run>/.hydra \
        --datastore outputs/<run>/datastore \
        --eval timit nisp heightceleb \
        --output outputs/<run>/eval

For each requested dataset, we run the trained model on the test split (or, with
``--split val``, on the validation speakers -- for choosing settings only) and
write a CSV of predictions + neighbour explanations, plus a JSON summary of
MAE / RMSE / Pearson per target, broken down overall / male / female.

``--unit`` selects the evaluation protocol:

    speaker    one prediction per speaker, pooled over a bag (our protocol).
    utterance  one prediction per clip -- what Gupta 2022, Jaid 2023 and most
               published TIMIT work report, so it is the only mode whose
               numbers are comparable to a paper. Per-speaker metrics are also
               derived from the *same* predictions by averaging within a
               speaker, so both protocols come from one inference pass.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from retro_profile.data.registry import build_datamodule
from retro_profile.evaluation import evaluate_with_explanations
from retro_profile.retrieval import RetrievalDatastore, attach_datastore
from retro_profile.utils import (
    get_logger,
    instantiate,
    load_config,
    seed_everything,
    setup_logging,
)

# Per-corpus DataModule defaults. Users override at the CLI as needed.
CORPUS_DEFAULTS: dict[str, dict] = {
    "timit": {"root": "data/timit"},
    "nisp": {"root": "data/nisp"},
    "heightceleb": {"root": "data/heightceleb"},
    # 60/20/20 split by co-appearance group (prepare_heightceleb.py --split-mode grouped)
    "heightceleb_split2": {"root": "data/heightceleb_split2"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--config-dir", required=True, type=Path)
    p.add_argument("--datastore", default=None, type=Path,
                   help="Optional path to a saved datastore directory.")
    p.add_argument("--eval", nargs="+", required=True,
                   choices=list(CORPUS_DEFAULTS),
                   help="Corpora to evaluate on.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--force-retrieval", action="store_true",
                   help="Switch kNN retrieval ON for a model trained with it OFF. The "
                        "residual branch was never trained and is zero-initialised, so "
                        "every prediction equals the kNN anchor: this measures how good "
                        "retrieval over the learned space is, not a trained RETRO model. "
                        "Requires --datastore.")
    p.add_argument("--k-neighbours", type=int, default=5,
                   help="Number of neighbours to record per prediction.")
    p.add_argument("--unit", choices=("speaker", "utterance"), default="speaker",
                   help="Evaluation unit. 'speaker' = one prediction per speaker "
                        "from a bag (our protocol). 'utterance' = one prediction "
                        "per clip (the protocol used by Gupta 2022 / Jaid 2023 and "
                        "most published TIMIT work); per-speaker numbers are then "
                        "also derived from the same predictions by averaging.")
    p.add_argument("--split", choices=("test", "val"), default="test",
                   help="Speakers to evaluate. 'val' (the 70 validation speakers) exists "
                        "for CHOOSING settings such as the kNN k; never report it as a "
                        "result. Recorded in __meta.eval.split.")
    p.add_argument("--save-embeddings", action="store_true",
                   help="Also write embeddings_<corpus>.npz: each row's retrieval "
                        "embedding (the kNN query), for offline kNN sweeps.")
    p.add_argument("--bag-size", type=int, default=None,
                   help="Clips per speaker in each --unit speaker bag. Default: the corpus config's "
                        "bag_size, DRAWN AT RANDOM when a speaker has more clips (TIMIT: 8 of 10). "
                        "Set it to at least every speaker's clip count (TIMIT: 10) to score on all "
                        "clips in a fixed order -- the same clips for every model. Recorded in "
                        "__meta.eval.bag_size (null = the corpus default).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_bag_size(data_cfg: dict, bag_size: int | None) -> dict:
    """The corpus data config with ``bag_size`` overridden when one is given (2026-10-15)."""
    if bag_size is None:
        return data_cfg
    if bag_size < 1:
        raise SystemExit(f"--bag-size must be >= 1, got {bag_size}")
    return {**data_cfg, "bag_size": int(bag_size)}


# --- metric helpers ---------------------------------------------------------
# Metrics are recomputed from the prediction DataFrame rather than taken from
# RegressionMetrics, so that the utterance-level and speaker-level numbers come
# from one inference pass and cannot drift apart.

def _metrics_by_target(df: pd.DataFrame) -> dict[str, float]:
    """Flat {target}/{mae,rmse,pearson} over the rows of ``df``."""
    out: dict[str, float] = {}
    for target, g in df.groupby("target"):
        g = g.dropna(subset=["prediction", "ground_truth"])
        if g.empty:
            continue
        err = g["prediction"] - g["ground_truth"]
        out[f"{target}/mae"] = float(err.abs().mean())
        out[f"{target}/rmse"] = float((err**2).mean() ** 0.5)
        out[f"{target}/pearson"] = float(g["prediction"].corr(g["ground_truth"]))
        out[f"{target}/n"] = int(len(g))
    return out


def _gender_blocks(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Overall + male + female metrics. Papers report M/F separately."""
    blocks = {"overall": _metrics_by_target(df)}
    if "gender" in df.columns:
        for code, name in (("M", "male"), ("F", "female")):
            sub = df[df["gender"] == code]
            if not sub.empty:
                blocks[name] = _metrics_by_target(sub)
    return blocks


def _to_speaker_level(df: pd.DataFrame) -> pd.DataFrame:
    """Average per-utterance predictions within each speaker (the bag protocol)."""
    agg: dict[str, str] = {"prediction": "mean", "ground_truth": "first"}
    if "gender" in df.columns:
        agg["gender"] = "first"
    return df.groupby(["speaker_id", "target"], as_index=False).agg(agg)


def _write_frame_attention(rows: list[dict], out_dir: Path, corpus: str) -> pd.DataFrame:
    """Per-clip frame-attention summary (CSV) and raw weights (NPZ keyed 'speaker|utt')."""
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "weights"} for r in rows])
    table.to_csv(out_dir / f"frame_attention_{corpus}.csv", index=False)
    np.savez_compressed(out_dir / f"frame_attention_{corpus}.npz",
                        **{f"{r['speaker_id']}|{r['utt_id']}": r["weights"] for r in rows})
    return table


def _write_embeddings(rows: list[dict], out_dir: Path, corpus: str) -> None:
    """Retrieval embeddings (the kNN queries), one per prediction row, as an NPZ."""
    np.savez(out_dir / f"embeddings_{corpus}.npz",
             speaker_id=np.array([r["speaker_id"] for r in rows]),
             utt_id=np.array([r["utt_id"] for r in rows]),
             embedding=np.stack([r["embedding"] for r in rows]).astype(np.float32))


def _force_retrieval(model) -> int:
    """Turn retrieval on in every head that supports it; return how many.

    For a model trained with ``use_retrieval: false``. Both head branches are
    always constructed, and the residual's last layer is zero-initialised and
    never received a gradient, so with a datastore attached the prediction is
    exactly the kNN anchor (clamped to the label range).
    """
    n = 0
    for head in getattr(model, "heads", {}).values():
        if hasattr(head, "use_retrieval"):
            head.use_retrieval = True
            n += 1
    return n


def _run_metadata(cfg, args, model, ckpt: dict, datastore) -> dict:
    """Provenance for this result: the budget and protocol that produced it.

    A number is not comparable to another number without these. Two of these
    fields have already invalidated runs in this project:

      * ``selection.monitor`` -- monitoring val/height/mae alone early-stops
        before age converges (SESSION_HANDOFF.md §6.9);
      * ``eval.unit`` -- per-speaker bags are ~sqrt(8) easier than the
        per-utterance protocol every published TIMIT paper reports.

    Written into summary.json so a result can never be read without its budget.
    """
    def g(obj, path, default=None):
        for key in path.split("."):
            if obj is None:
                return default
            obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        return default if obj is None else obj

    cfgd = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Best epoch is encoded in the checkpoint filename and/or its payload.
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    ordcon_w = g(cfgd, "training.loss.ordcon_weight")
    # How each clip is reduced over time. Only the WavLMEncoder family has the
    # option; configs from before 2026-10-11 lack the key and used the mean.
    enc_target = str(g(cfgd, "model.encoder._target_", ""))
    frame_pooling = (g(cfgd, "model.encoder.frame_pooling", "mean")
                     if enc_target.endswith(("WavLMEncoder", "Wav2Vec2Encoder", "HuBERTEncoder"))
                     else None)

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        rev = None

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_rev": rev,
        "run_name": g(cfgd, "run_name"),
        "seed": g(cfgd, "seed"),
        "model": {
            "name": g(cfgd, "model.name"),
            "encoder": g(cfgd, "model.encoder._target_"),
            "pretrained_name": g(cfgd, "model.encoder.pretrained_name"),
            "encoder_frozen": g(cfgd, "model.encoder.freeze"),
            "frame_pooling": frame_pooling,
            "trainable_params": trainable,
            "total_params": total,
        },
        "budget": {
            "max_audio_seconds": g(cfgd, "data.max_audio_seconds"),
            "batch_size": g(cfgd, "data.batch_size"),
            "bag_size": g(cfgd, "data.bag_size"),
            "accumulate_grad_batches": g(cfgd, "training.accumulate_grad_batches"),
            "effective_batch": (g(cfgd, "data.batch_size", 0) or 0)
                               * (g(cfgd, "training.accumulate_grad_batches", 1) or 1),
            "max_epochs": g(cfgd, "training.max_epochs"),
            "best_epoch": epoch,
            "precision": g(cfgd, "precision"),
            "augmentation": g(cfgd, "data.augmentation.enabled"),
            "train_unit": g(cfgd, "data.train_unit", "bag"),
            # utterance-level training: clips drawn per speaker per epoch (null = every clip)
            "utts_per_speaker": g(cfgd, "data.utts_per_speaker"),
            "lr": g(cfgd, "training.optimizer.lr"),
            "encoder_lr": g(cfgd, "training.encoder_lr") or g(cfgd, "training.gupta.encoder_lr"),
            # frame-level (temporal) attention's own LR; None = the head LR (every run before 13 Oct)
            "frame_attention_lr": g(cfgd, "training.frame_attention_lr"),
            # mixup on training batches (training/mixup.py); None = off (every run before 18 Oct)
            "mixup_alpha": (g(cfgd, "training.mixup.alpha", 1.0)
                            if g(cfgd, "training.mixup.enabled", False) else None),
        },
        # Configs written before the 2026-10-10 fix have no `ordcon_embedding`;
        # those runs applied OrdCon to the FIRST target's projected vector, not
        # to the vector the datastore indexes. Results from the two are not
        # comparable for two-target MIL models.
        "ordcon": (
            {
                "weight": ordcon_w,
                "embedding": g(cfgd, "training.loss.ordcon_embedding",
                               "first_target (pre-2026-10-10)"),
                "magnitude_scale": g(cfgd, "training.loss.ordcon_kwargs.magnitude_scale"),
                "magnitude_units": g(cfgd, "training.loss.ordcon_kwargs.magnitude_units",
                                     "loss"),
            }
            if ordcon_w
            else {"weight": ordcon_w}
        ),
        "selection": {
            "monitor": g(cfgd, "training.checkpoint.monitor"),
            "early_stopping": g(cfgd, "training.early_stopping.enabled"),
            "early_stopping_monitor": g(cfgd, "training.early_stopping.monitor"),
        },
        "eval": {
            "unit": args.unit,
            # Clips per speaker bag at evaluation; None = the corpus config's value, drawn at
            # random when a speaker has more clips (TIMIT: 8 of 10).
            "bag_size": getattr(args, "bag_size", None),
            # 'val' only ever chooses settings (e.g. kNN k); a result is 'test'.
            "split": getattr(args, "split", "test"),
            "checkpoint": Path(args.ckpt).name,
            "retrieval": datastore is not None,
            # True = retrieval switched on at TEST time for a model trained
            # without it: predictions are the kNN anchor alone, residual untrained.
            "retrieval_forced": bool(getattr(args, "force_retrieval", False)),
            "datastore_size": len(datastore) if datastore is not None else 0,
        },
        "env": {
            "torch": torch.__version__,
            "host": socket.gethostname(),
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }


def _attach_gender(df: pd.DataFrame, root: str | None) -> pd.DataFrame:
    """Merge the manifest's gender column in, for the M/F breakdown."""
    if not root:
        return df
    manifest = Path(root) / "manifests" / "test.csv"
    if not manifest.is_file():
        return df
    g = pd.read_csv(manifest, usecols=["speaker_id", "gender"])
    g = g.drop_duplicates("speaker_id")
    # Normalise 'M'/'male'/'Male' -> 'M' so the split is encoding-agnostic.
    g["gender"] = g["gender"].astype(str).str.upper().str[0]
    return df.merge(g, on="speaker_id", how="left")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=args.output, level="INFO")
    log = get_logger("evaluate")
    seed_everything(args.seed)

    cfg = load_config(args.config_dir / "config.yaml")
    model = instantiate(cfg.model)

    # Restore model weights directly from the checkpoint's state_dict. This is
    # LightningModule-agnostic (works whether the run used RetroProfileLitModule
    # or Gupta2022LitModule), and it restores any registered buffers such as the
    # Gupta baseline's label-stat buffers used to denormalize predictions.
    ckpt = torch.load(str(args.ckpt), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        log.warning(f"{len(missing)} missing keys when loading model weights (ok if buffers).")
    model.eval()

    # Attach the datastore to retrieval-aware heads, if provided.
    datastore = None
    if args.force_retrieval and (args.datastore is None or not args.datastore.exists()):
        raise SystemExit("--force-retrieval needs an existing --datastore")
    if args.datastore is not None and args.datastore.exists():
        datastore = RetrievalDatastore.load(args.datastore)
        if args.force_retrieval:
            log.info(f"Retrieval FORCED on in {_force_retrieval(model)} heads "
                     "(kNN anchor only; residual untrained)")
        attach_datastore(model, datastore)
        log.info(f"Loaded datastore from {args.datastore} ({len(datastore)} entries)")

    summary: dict[str, dict[str, float]] = {}

    for corpus in args.eval:
        log.info(f"\n=== Evaluating on {corpus} ===")
        # Pull defaults from the matching cfg/data file if it exists, else use CORPUS_DEFAULTS.
        ddir = Path(__file__).resolve().parent.parent / "configs" / "data" / f"{corpus}.yaml"
        if ddir.exists():
            data_cfg = OmegaConf.to_container(load_config(ddir), resolve=True)
            data_cfg.pop("_target_", None)
        else:
            data_cfg = {"name": corpus, **CORPUS_DEFAULTS[corpus]}
        data_cfg = apply_bag_size(data_cfg, getattr(args, "bag_size", None))

        dm = build_datamodule(corpus, **{k: v for k, v in data_cfg.items() if k != "name"})
        split = getattr(args, "split", "test")
        if args.unit == "utterance":
            # One clip per row: bags of size 1, so any in-model bag aggregation
            # is a no-op and each prediction stands alone.
            loader = dm.utterance_loader(split=split)
        elif split == "val":
            dm.setup("fit")                  # the validation set is built with "fit"
            loader = dm.val_dataloader()
        else:
            dm.setup("test")
            loader = dm.test_dataloader()

        frame_rows: list[dict] = []
        emb_rows: list[dict] | None = [] if getattr(args, "save_embeddings", False) else None
        df, _, attn_df = evaluate_with_explanations(
            model=model,
            dataloader=loader,
            target_names=model.target_names,
            datastore=datastore,
            k_neighbours=args.k_neighbours,
            return_attention=True,
            frame_attention=frame_rows,
            embeddings=emb_rows,
        )
        df = _attach_gender(df, data_cfg.get("root"))
        if emb_rows:
            _write_embeddings(emb_rows, args.output, corpus)
            log.info(f"Wrote embeddings_{corpus}.npz  ({len(emb_rows)} rows, split={split})")
        # Frame-level attention (hierarchical models only): did it stay a plain
        # mean over each clip's frames, or learn to pick frames?
        if frame_rows:
            fa = _write_frame_attention(frame_rows, args.output, corpus)
            log.info(f"Wrote frame_attention_{corpus}.csv/.npz  ({len(fa)} clips, "
                     f"mean eff_share {fa['eff_share'].mean():.3f}; 1.0 = plain mean)")

        out_csv = args.output / f"predictions_{corpus}.csv"
        df.to_csv(out_csv, index=False)
        # Per-utterance MIL weights. Empty for models without attention pooling
        # (the baselines), so only written when there is something to write.
        if not attn_df.empty:
            attn_csv = args.output / f"attention_{corpus}.csv"
            attn_df.to_csv(attn_csv, index=False)
            log.info(f"Wrote {attn_csv}  ({len(attn_df)} utterance weights)")
        log.info(f"Wrote {out_csv}  ({len(df)} rows, unit={args.unit})")

        # Primary block keeps the flat {target}/{metric} shape that
        # collect_results.py expects.
        blocks = {args.unit: _gender_blocks(df)}
        if args.unit == "utterance":
            # Same predictions, aggregated to speakers: the delta between these
            # two blocks *is* the bag-pooling effect, with nothing else varying.
            spk = _to_speaker_level(df)
            spk.to_csv(args.output / f"predictions_{corpus}_speaker.csv", index=False)
            blocks["speaker"] = _gender_blocks(spk)

        summary[corpus] = blocks[args.unit]["overall"]
        summary[f"{corpus}__unit"] = args.unit
        summary[f"{corpus}__breakdown"] = blocks

        for unit_name, block in blocks.items():
            for subset, m in block.items():
                pretty = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in m.items()
                )
                log.info(f"  {corpus} | {unit_name:9s} | {subset:7s} | {pretty}")

    summary["__meta"] = _run_metadata(cfg, args, model, ckpt, datastore)

    out_json = args.output / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
