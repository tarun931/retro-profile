#!/bin/bash
# Table 3 (RQ4, 2026-10-16): neighbour quality of each run's retrieval space (scripts/neighbour_quality.py).  🔵 SUBMIT ON the GPU cluster.
#
# For every run in RUNS (each must already have its second stage, runs/<run>/stage2${SUFFIX}/): load the frozen model
# and the second stage's own lookup table, embed the test speakers on bags of BAG_SIZE clips, check the kNN-alone
# predictions match the second stage's, then write gap@k (k = 1/5/20), the distance-label Spearman and the kNN-only MAE
# to runs/<run>/stage2${SUFFIX}/neighbour_quality.json. No training.
#
#   sbatch --export=ALL,RUNS="runs/a runs/b" scripts/cluster/neighbour_quality.slurm

#SBATCH --job-name=nbrq
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
EVAL_CORPUS="${EVAL_CORPUS:-timit}"
BAG_SIZE="${BAG_SIZE:-10}"
SUFFIX="${SUFFIX:-_b10}"

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

echo "=== $(date) | neighbour quality | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for RUN_DIR in ${RUNS}; do
  RUN_DIR="${RUN_DIR%/}"
  echo "--- ${RUN_DIR}"
  python scripts/neighbour_quality.py --run "${RUN_DIR}" --corpus "${EVAL_CORPUS}" --bag-size "${BAG_SIZE}" \
      --suffix "${SUFFIX}" --num-workers 8
done
echo "=== $(date) | done ==="
