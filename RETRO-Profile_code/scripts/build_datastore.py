"""Build a FAISS retrieval datastore from a trained model + dataset.

Usage:
    python scripts/build_datastore.py \
        --ckpt outputs/<run>/checkpoints/last.ckpt \
        --config-dir outputs/<run>/.hydra \
        --output outputs/<run>/datastore

The script restores the LightningModule from a checkpoint, takes the underlying
model, walks the training-split utterance loader, and builds a FAISS index of
per-speaker embeddings + labels. The result is saved to ``--output`` so the
``evaluate.py`` script (and any retrieval-aware head) can attach it later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from retro_profile.retrieval import build_datastore
from retro_profile.training import RetroProfileLitModule
from retro_profile.utils import (
    get_logger,
    instantiate,
    load_config,
    seed_everything,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path,
                   help="Path to a Lightning checkpoint (.ckpt)")
    p.add_argument("--config-dir", required=True, type=Path,
                   help="Path to the Hydra-saved config directory of the run "
                        "(typically <output_dir>/.hydra).")
    p.add_argument("--output", required=True, type=Path,
                   help="Directory in which to save index.faiss + labels.parquet")
    p.add_argument("--split", default="train", help="Data split to index (default: train)")
    p.add_argument("--metric", default="cosine", choices=["cosine", "l2"])
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--weighting", default="softmax_distance",
                   choices=["softmax_distance", "uniform"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir=args.output, level="INFO")
    log = get_logger("build_datastore")
    seed_everything(args.seed)

    cfg = load_config(args.config_dir / "config.yaml")
    log.info(f"Loaded config from {args.config_dir / 'config.yaml'}")

    # Build model + datamodule from the saved Hydra config.
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data)
    datamodule.setup("fit")

    # Load the Lightning checkpoint and copy weights into our fresh model.
    embedding_dim = int(getattr(getattr(model, "projector", model.encoder), "output_dim",
                                getattr(model.encoder, "output_dim", 256)))
    lit = RetroProfileLitModule.load_from_checkpoint(
        str(args.ckpt),
        model=model,
        target_names=list(model.target_names),
        embedding_dim=embedding_dim,
        label_stats=datamodule.label_stats,
        normalize_targets=bool(cfg.data.get("normalize_targets", True)),
        strict=False,
    )
    model = lit.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = datamodule.utterance_loader(args.split)
    log.info(f"Indexing {args.split!r} split — {len(loader.dataset)} utterances")

    ds = build_datastore(
        model=model,
        dataloader=loader,
        targets=model.target_names,
        metric=args.metric,
        temperature=args.temperature,
        weighting=args.weighting,
        device=device,
    )
    log.info(f"Built datastore with {len(ds)} speaker entries")
    ds.save(args.output)
    log.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
