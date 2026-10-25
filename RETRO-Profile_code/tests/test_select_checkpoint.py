"""Regression tests for scripts/select_checkpoint.py (2026-10-10).

The bug: the slurm scripts, reevaluate.slurm and the prune script each picked
"best" as the lowest val= in checkpoints/. In a two-phase run the checkpoint
tracker is reset at the switch but the phase-1 files stay on disk, and since
retrieval makes validation worse a phase-1 file always wins. On
retro_finetune_budget_fixed_timit_s42 training chose epoch 39 and evaluation
used epoch 16 -- a phase-1 model with an untrained residual. The prune script
would then have deleted epoch 39.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sel", ROOT / "scripts" / "select_checkpoint.py")
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

# The exact folder contents of the run that exposed the bug.
REAL_FILES = [
    "epoch=16-step=1666-val=1.2676.ckpt", "epoch=17-step=1764-val=1.3384.ckpt",
    "epoch=19-step=1960-val=1.3378.ckpt", "epoch=39-step=3920-val=1.3983.ckpt",
    "epoch=56-step=5586-val=1.4184.ckpt", "epoch=59-step=5880-val=1.4190.ckpt",
    "last.ckpt",
]
TWO_PHASE = {"enabled": True, "start_epoch": 30, "reset_checkpoint_best": True}


def _run(tmp_path, files, two_phase=None, name="run"):
    run = tmp_path / "runs" / name
    (run / "checkpoints").mkdir(parents=True)
    (run / ".hydra").mkdir()
    for f in files:
        (run / "checkpoints" / f).write_bytes(b"")
    cfg = {"training": {"two_phase": two_phase or {"enabled": False}}}
    (run / ".hydra" / "config.yaml").write_text(yaml.safe_dump(cfg))
    return run


def test_two_phase_run_with_orphaned_phase1_files_picks_the_phase2_best(tmp_path):
    run = _run(tmp_path, REAL_FILES, TWO_PHASE)
    assert sel.select_checkpoint(run).name == "epoch=39-step=3920-val=1.3983.ckpt"


def test_the_old_lowest_val_rule_would_have_picked_phase1(tmp_path):
    """Pins the failure itself, so the test above cannot pass vacuously."""
    vals = {f: float(m.group(1)) for f in REAL_FILES if (m := re.search(r"val=([0-9.]+)\.ckpt", f))}
    assert min(vals, key=vals.get).startswith("epoch=16-")


def test_best_checkpoint_file_written_by_training_wins(tmp_path):
    run = _run(tmp_path, REAL_FILES, TWO_PHASE)
    # Absolute cluster path, as train.py writes it; resolved by name locally.
    (run / "best_checkpoint.txt").write_text(
        "/data/x/runs/run/checkpoints/epoch=56-step=5586-val=1.4184.ckpt\n")
    assert sel.select_checkpoint(run).name == "epoch=56-step=5586-val=1.4184.ckpt"


def test_a_best_file_naming_a_deleted_checkpoint_falls_back_safely(tmp_path):
    run = _run(tmp_path, REAL_FILES, TWO_PHASE)
    (run / "best_checkpoint.txt").write_text("/somewhere/epoch=58-step=1-val=1.0.ckpt\n")
    assert sel.select_checkpoint(run).name == "epoch=39-step=3920-val=1.3983.ckpt"


def test_single_phase_runs_behave_exactly_as_before(tmp_path):
    files = ["epoch=16-step=1-val=1.3382.ckpt", "epoch=35-step=1-val=1.3578.ckpt",
             "epoch=59-step=1-val=1.3391.ckpt", "last.ckpt"]
    assert sel.select_checkpoint(_run(tmp_path, files)).name.startswith("epoch=16-")


def test_two_phase_without_reset_uses_every_epoch(tmp_path):
    run = _run(tmp_path, REAL_FILES, {**TWO_PHASE, "reset_checkpoint_best": False})
    assert sel.select_checkpoint(run).name.startswith("epoch=16-")


def test_two_phase_run_with_no_phase2_checkpoint_refuses(tmp_path):
    run = _run(tmp_path, REAL_FILES[:3], TWO_PHASE)
    with pytest.raises(FileNotFoundError, match="refusing to fall back"):
        sel.select_checkpoint(run)


def test_cli_prints_the_path_and_fails_loudly(tmp_path):
    run = _run(tmp_path, REAL_FILES, TWO_PHASE)
    ok = subprocess.run(["python", str(ROOT / "scripts" / "select_checkpoint.py"), str(run)],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout.strip().endswith("epoch=39-step=3920-val=1.3983.ckpt")
    empty = _run(tmp_path, [], name="empty")
    bad = subprocess.run(["python", str(ROOT / "scripts" / "select_checkpoint.py"), str(empty)],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and bad.stdout == ""


def _bash4() -> str | None:
    for b in ("bash", "/usr/local/bin/bash", "/opt/homebrew/bin/bash"):
        exe = shutil.which(b)
        if exe:
            out = subprocess.run([exe, "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True)
            if out.stdout.strip().isdigit() and int(out.stdout) >= 4:
                return exe
    return None


@pytest.mark.skipif(_bash4() is None, reason="prune_checkpoints.sh needs bash >= 4 (mapfile); the GPU cluster has it, macOS /bin/bash is 3.2")
def test_prune_dry_run_keeps_the_phase2_best(tmp_path):
    proj = tmp_path
    (proj / "scripts" / "cluster").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "select_checkpoint.py", proj / "scripts")
    shutil.copy(ROOT / "scripts" / "cluster" / "prune_checkpoints.sh", proj / "scripts" / "cluster")
    run = _run(proj, REAL_FILES, TWO_PHASE, name="retro_finetune_budget_fixed_timit_s42")
    (run / "checkpoints" / "best_age-epoch=17-val_age=0.5000.ckpt").write_bytes(b"")
    out = subprocess.run(
        [_bash4(), str(proj / "scripts" / "cluster" / "prune_checkpoints.sh"), "--dry"],
        capture_output=True, text=True,
        env={**os.environ, "PROJECT_DIR": str(proj),
             "RUNS": "runs/retro_finetune_budget_fixed_timit_s42/"},
    ).stdout
    assert "keep  epoch=39-step=3920-val=1.3983.ckpt" in out
    assert "would delete epoch=39" not in out
    assert "best_age" not in out          # per-trait copies are never candidates
