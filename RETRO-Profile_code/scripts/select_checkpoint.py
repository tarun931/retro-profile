"""Choose the checkpoint to evaluate or keep -- the one TRAINING chose as best.

Three scripts used to re-derive "best" by taking the lowest val=... in the
checkpoints/ folder: the training slurm scripts, reevaluate.slurm and
prune_checkpoints.sh. That is wrong for the two-phase retrieval curriculum.
With `reset_checkpoint_best`, ModelCheckpoint forgets phase 1 at the switch but
leaves the phase-1 files on disk, and because retrieval makes validation worse,
a phase-1 file always has the lowest val=. On
retro_finetune_budget_fixed_timit_s42 (2026-10-10) training chose epoch 39 and
evaluation used epoch 16: a phase-1 model whose residual had never trained.

Order of preference:
  1. runs/<run>/best_checkpoint.txt, written by train.py from
     ModelCheckpoint.best_model_path -- phase-aware by construction.
  2. For runs that predate that file: the lowest val= in the filename, but for a
     two-phase run with reset_checkpoint_best ONLY among epochs >= start_epoch.
     A two-phase run with no phase-2 checkpoint is an error, never a silent
     fallback to phase 1.

Usage:  python scripts/select_checkpoint.py runs/<run>      # prints the path
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

_NAME = re.compile(r"epoch=(\d+)-step=\d+-val=([0-9.]+)\.ckpt$")


def phase2_start(run_dir: Path) -> int | None:
    """First phase-2 epoch if this run reset checkpoint tracking there, else None."""
    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    tp = (cfg.get("training") or {}).get("two_phase") or {}
    if tp.get("enabled") and tp.get("reset_checkpoint_best"):
        return int(tp.get("start_epoch", 0))
    return None


def select_checkpoint(run_dir: str | Path) -> Path:
    """Return the checkpoint training selected; see the module docstring."""
    run_dir = Path(run_dir)
    ck = run_dir / "checkpoints"

    best_file = run_dir / "best_checkpoint.txt"
    if best_file.exists():
        # Stored as an absolute cluster path. Resolve by name inside this run so
        # a copied or moved run still works.
        named = ck / Path(best_file.read_text().strip()).name
        if named.exists():
            return named

    cands = []
    for f in ck.glob("*.ckpt"):
        m = _NAME.search(f.name)
        if m:
            cands.append((float(m.group(2)), int(m.group(1)), f))
    if not cands:
        raise FileNotFoundError(f"no epoch=..-val=.. checkpoints in {ck}")

    start = phase2_start(run_dir)
    if start is not None:
        cands = [c for c in cands if c[1] >= start]
        if not cands:
            raise FileNotFoundError(
                f"{run_dir} is a two-phase run but has no checkpoint from epoch "
                f">= {start}; refusing to fall back to a phase-1 model"
            )
    return min(cands)[2]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/select_checkpoint.py runs/<run>")
    try:
        print(select_checkpoint(sys.argv[1]))
    except FileNotFoundError as e:
        sys.exit(f"select_checkpoint: {e}")


if __name__ == "__main__":
    main()
