#!/bin/bash
# OrdCon lambda sweep -- MECHANISM analysis, no training.
#
# Runs scripts/analyse_ordcon.py section A (embedding ordering) over every run
# in the lambda sweep, including the lambda = 0.3 run that already exists.
#
# Why this is a separate job. The lambda sweep trains with retrieval OFF (it
# inherits retro_noknn_budget), so retro_finetune.slurm's WITH_RETRIEVAL=0 path
# never builds a datastore and never calls analyse_ordcon -- the runs would
# finish with no mechanism metrics at all. Section A needs only the checkpoint
# and the config, not a datastore, so this job is a few minutes per run.
#
# WHAT TO READ, in order:
#   distance_spearman  -- does DISTANCE in the space track LABEL DIFFERENCE?
#                         This is the property retrieval consumes, it is the
#                         weak one (0.22-0.44 in the existing runs), and it is
#                         the ONLY metric here that is comparable across the
#                         whole sweep -- see the lambda = 0 warning below.
#   ordering_accuracy  -- does the space order along the learned per-target
#                         axis? 0.5 is chance; existing runs reach 0.69-0.73.
#   axis_spearman      -- the same ordering, as a rank correlation.
#
# !! LAMBDA = 0 WARNING -- do not misread this row.
#
# OrdConLoss is CONSTRUCTED unconditionally (losses/multitask.py:45), so its
# learnable per-target `directions` exist in every checkpoint and this script
# will happily report numbers for them. But the loss is only COMPUTED when
# ordcon_weight > 0 (losses/multitask.py:74), so at lambda = 0 that axis never
# receives a gradient and stays at its RANDOM INIT.
#
# Therefore at lambda = 0 the axis metrics -- axis_pearson, axis_spearman and
# ordering_accuracy -- are measured along a random direction and will sit near
# chance BY CONSTRUCTION. That is not evidence the embedding space is
# unordered; it is only evidence that the untrained axis does not point at
# anything. Read distance_spearman for the lambda = 0 row: it is computed from
# raw pairwise embedding distances (analyse_ordcon.py, `dz`) and never touches
# the direction, so it is the one number comparable across all five lambdas.
#
# The axis metrics ARE comparable among lambda = 0.1 / 0.3 / 0.5 / 1.0, where
# the axis is trained in every case.
#
# Final MAE is NOT the metric for this experiment. OrdCon's claim is about the
# geometry of the space, and it has met that claim in every run while never
# improving MAE. If lambda = 0 matches lambda = 0.3 on these three numbers, then
# fine-tuned WavLM already orders the space and Novelty 2 adds nothing -- a
# clean negative worth stating, and the point of the sweep.
#
#   sbatch scripts/cluster/ordcon_analysis.slurm

#SBATCH --job-name=ordcon_an
#SBATCH --account=<project>
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
ENV_NAME="${ENV_NAME:-retro}"
SPLIT="${SPLIT:-val}"

# lambda = 0.3 is retro_noknn_budget, already trained -- included so the sweep
# is read as five points, not four plus a number from another table.
RUNS="${RUNS:-retro_ordcon_l000_timit_s42 retro_ordcon_l010_timit_s42 retro_noknn_budget_timit_s42 retro_ordcon_l050_timit_s42 retro_ordcon_l100_timit_s42}"

cd "${PROJECT_DIR}"
mkdir -p logs

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | ordcon analysis | split=${SPLIT} | $(hostname) ==="

for RUN in ${RUNS}; do
    RUN_DIR="runs/${RUN}"
    if [ ! -d "${RUN_DIR}" ]; then
        echo "!! missing ${RUN_DIR} -- skipped"
        continue
    fi

    # Select by the val= field in the filename, NOT `ls -t`. save_top_k is 3, so
    # "newest" means "whichever most recently entered the top three" -- that
    # mistake evaluated a sub-optimal checkpoint in 7 of 10 runs (bug §6.13).
    CKPT=$(for f in "${RUN_DIR}/checkpoints/"*.ckpt; do
        if [ "$(basename "${f}")" = "last.ckpt" ]; then continue; fi
        v=$(echo "${f}" | sed -n 's/.*val=\([0-9.]*\)\.ckpt$/\1/p')
        printf '%s\t%s\n' "${v:-999999}" "${f}"
      done | sort -g | head -1 | cut -f2)

    if [ -z "${CKPT}" ]; then
        echo "!! no checkpoint in ${RUN_DIR}/checkpoints -- skipped"
        continue
    fi

    echo "--- ${RUN}  ckpt=$(basename "${CKPT}")"
    python scripts/analyse_ordcon.py \
        --run "${RUN_DIR}" \
        --ckpt "${CKPT}" \
        --config-dir "${RUN_DIR}/.hydra" \
        --split "${SPLIT}" \
        --output "${RUN_DIR}/ordcon_report.json"
done

echo
echo "=== summary: lambda -> ordering metrics ==="
python - <<'PY'
import json, os
runs = os.environ.get("RUNS", "").split() or [
    "retro_ordcon_l000_timit_s42","retro_ordcon_l010_timit_s42",
    "retro_noknn_budget_timit_s42","retro_ordcon_l050_timit_s42",
    "retro_ordcon_l100_timit_s42"]
# Most specific key first: label() returns the first substring match, and
# "noknn_budget" is a substring of "noknn_budget_fixed".
lam = {"noknn_budget_fixed":"0.3 (fixed wiring)","l000":"0.0","l010":"0.1",
       "noknn_budget":"0.3 (existing)","l050":"0.5","l100":"1.0"}
def label(r):
    for k,v in lam.items():
        if k in r: return v
    return r
print(f"{'lambda':16s} {'target':7s} {'dist_spearman':>14s} {'ord_acc':>8s} {'axis_spear':>11s}")
print("-"*62)
for r in runs:
    p=f"runs/{r}/ordcon_report.json"
    if not os.path.exists(p):
        print(f"{label(r):16s} [no report]"); continue
    d=json.load(open(p)).get("ordering",{})
    for t in ("age","height"):
        o=d.get(t)
        if not o: continue
        note = "  <- axis untrained at lambda=0; read distance_spearman only" if "l000" in r else ""
        print(f"{label(r):16s} {t:7s} {o.get('distance_spearman',float('nan')):14.3f} "
              f"{o.get('ordering_accuracy',float('nan')):8.3f} {o.get('axis_spearman',float('nan')):11.3f}{note}")
PY

echo "=== $(date) | done ==="
