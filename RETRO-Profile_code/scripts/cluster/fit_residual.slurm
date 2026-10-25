#!/bin/bash
# Second stage for RETRO (2026-10-15): train ONLY each trait's residual correction on top of kNN.  🔵 SUBMIT ON the GPU cluster.
#
# For every run in RUNS: freeze training's chosen model, build the kNN datastore, choose k per trait on validation,
# train only the residual MLPs (best validation epoch kept; epoch 0 = kNN alone), and score validation + test
# three ways (kNN off / kNN alone / kNN + residual) on bags of BAG_SIZE clips. See scripts/fit_residual.py.
# Writes runs/<run>/stage2${SUFFIX}/. First checks that its kNN-off test predictions reproduce runs/<run>/${CHECK_SUB}.
#
#   sbatch --export=ALL,RUNS="runs/a runs/b" scripts/cluster/fit_residual.slurm
#
# Defaults: BAG_SIZE=10 (TIMIT: all clips), SUFFIX=_b10, CHECK_SUB=eval_spk (b10 runs score test on all 10 there).
# EXTRA_ARGS: extra fit_residual.py options, e.g. EXTRA_ARGS="--train-from-run --calibrated" for the cross-corpus Table 5
# runs (datastore and residual training from the run's own training corpora; k and residual epoch on calibrated validation).

#SBATCH --job-name=stage2
#SBATCH --account=<project>
#SBATCH --time=02:00:00
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
EVAL_CORPUS="${EVAL_CORPUS:-timit}"
BAG_SIZE="${BAG_SIZE:-10}"
SUFFIX="${SUFFIX:-_b10}"
CHECK_SUB="${CHECK_SUB:-eval_spk}"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

cd "${PROJECT_DIR}"
mkdir -p logs
if [ -z "${RUNS:-}" ]; then echo "RUNS is empty"; exit 1; fi

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | stage 2 (residual) | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for RUN_DIR in ${RUNS}; do
  RUN_DIR="${RUN_DIR%/}"
  echo "--- ${RUN_DIR}"
  python scripts/fit_residual.py --run "${RUN_DIR}" --corpus "${EVAL_CORPUS}" --bag-size "${BAG_SIZE}" \
      --suffix "${SUFFIX}" --check-sub "${CHECK_SUB}" --num-workers 8 "${EXTRA[@]}"
done
echo "=== $(date) | done ==="
