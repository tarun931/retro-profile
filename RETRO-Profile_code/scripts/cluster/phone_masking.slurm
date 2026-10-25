#!/bin/bash
# Phone-class masking on TIMIT test for the Table 1 models (scripts/phone_masking.py, 2026-10-18).  🔵 SUBMIT ON the GPU cluster.
#
# Gupta et al. 2022's Table 4 analysis for all four Table 1 models (4 x 5 seeds, their chosen checkpoints): each of the
# 7 TIMIT phone classes silenced in turn in the 1,680 test clips, plus the unmasked audio, which must reproduce the
# stored Table 1 predictions or the job stops. No training. Writes ${OUT}/predictions_<exp>_s<seed>.csv and
# ${OUT}/masked_fraction.csv. Needs TIMIT's .PHN files beside the test audio.
#
#   sbatch --export=ALL,OUT=runs/phone_masking_table1 scripts/cluster/phone_masking.slurm

#SBATCH --job-name=phonemask
#SBATCH --account=<project>
#SBATCH --time=04:00:00
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
OUT="${OUT:-runs/phone_masking_table1}"

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

echo "=== $(date) | phone masking | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# ONLY_EXP (optional): score only these experiments, e.g. ONLY_EXP=gupta2022_finetune_w2v2_lr3e6 (2026-10-19)
python scripts/phone_masking.py --out "${OUT}" --num-workers 8 ${ONLY_EXP:+--only-exp ${ONLY_EXP}}
echo "=== $(date) | done ==="
