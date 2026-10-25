#!/bin/bash
# Reclaim checkpoint disk space.  🔵 RUN ON the GPU cluster (login node).
#
# WHY. `save_top_k: 3` was raised from 1 after the E11 selection mistake left a
# better epoch unrecoverable. That was the right call, but it triples storage per
# run, and <project> is the SUPERVISOR'S shared project (rule R1): it went from 162 GB
# to 356 GB of a 500 GB quota during one session, almost all of it ours.
# <scratch> is not an escape hatch -- it sits at 99% (rule R2).
#
# WHAT IT DOES. For each run it keeps `last.ckpt` plus the single best
# checkpoint, and deletes the rest. "Best" is read from the filename, which
# train.py writes as
#     epoch={epoch}-step={step}-val={monitor:.4f}.ckpt
# and every config now monitors val/combined_mae, where LOWER IS BETTER. Nothing
# else in a run directory is touched -- logs, diagnostics, eval outputs, hydra
# configs and attention reports all stay, so every recorded number remains
# reproducible from the artefacts.
#
# It never deletes the last remaining checkpoint of a run, and never touches a
# run whose checkpoints are already down to two or fewer. If the best cannot be
# identified it skips the run rather than guess.
#
# Run it with the `retro` conda env active: "best" comes from
# scripts/select_checkpoint.py (python + yaml).
#
# USAGE
#   bash scripts/cluster/prune_checkpoints.sh --dry     # show what would go
#   bash scripts/cluster/prune_checkpoints.sh           # do it
#   RUNS="runs/foo runs/bar" bash scripts/cluster/prune_checkpoints.sh --dry
#
# For runs that are fully superseded, delete the whole checkpoints/ directory by
# hand instead -- this script deliberately will not, because deciding a run is
# finished with is a judgement call and not a disk-space one.

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

cd "${PROJECT_DIR}"
RUNS="${RUNS:-$(ls -d runs/*/ 2>/dev/null || true)}"

freed=0
for run in ${RUNS}; do
  ck="${run%/}/checkpoints"
  [ -d "${ck}" ] || continue

  # Every .ckpt except last.ckpt is a candidate.
  # best_<trait>-*.ckpt are the per-trait copies (training.checkpoint.per_target):
  # one file each, always kept, never candidates for deletion.
  mapfile -t cands < <(find "${ck}" -maxdepth 1 -name '*.ckpt' ! -name 'last.ckpt' ! -name 'best_*' | sort)
  [ "${#cands[@]}" -le 1 ] && continue

  # Keep the checkpoint TRAINING chose (scripts/select_checkpoint.py). The old
  # lowest-val= scan kept an orphaned PHASE-1 file in two-phase runs and would
  # have deleted the real phase-2 best (2026-10-10). Needs the `retro` env
  # active for python + yaml.
  best=$(python scripts/select_checkpoint.py "${run%/}" 2>/dev/null) || best=""

  if [ -z "${best}" ]; then
    echo "  ${run%/}: could not identify a best checkpoint -- skipping"
    continue
  fi

  echo "${run%/}"
  echo "    keep  $(basename "${best}")  (+ last.ckpt)"
  for f in "${cands[@]}"; do
    [ "${f}" = "${best}" ] && continue
    sz=$(du -m "${f}" 2>/dev/null | cut -f1)
    freed=$(( freed + ${sz:-0} ))
    if [ "${DRY}" -eq 1 ]; then
      echo "    would delete $(basename "${f}")  (${sz} MB)"
    else
      echo "    deleting     $(basename "${f}")  (${sz} MB)"
      rm -f "${f}"
    fi
  done
done

printf '\n%s %d MB (%.1f GB)\n' \
  "$([ "${DRY}" -eq 1 ] && echo 'would free' || echo 'freed')" \
  "${freed}" "$(echo "${freed}" | awk '{print $1/1024}')"
[ "${DRY}" -eq 1 ] && echo "(dry run -- nothing was deleted)"
exit 0
