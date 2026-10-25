#!/bin/bash
# Fun demo (2026-10-18), CPU version: one recording's age and height from each Table 1 model (scripts/predict_clip.py).  🔵 SUBMIT ON the GPU cluster.
#
# Same as predict_clip.slurm but on the CPU partition (comp): no GPU, so it does not count against the 4-GPU limit.
# One deployed copy per model (best validation score); about 3 min on CPU. Writes ${OUT} and ${OUT%.csv}_neighbours.csv.
# No training; not part of any result.
#
#   sbatch --export=ALL,AUDIO=fun/clip_001.wav,OUT=fun/predictions_clip_001.csv scripts/cluster/predict_clip_cpu.slurm

#SBATCH --job-name=predclip_cpu
#SBATCH --account=<project>
#SBATCH --time=00:30:00
#SBATCH --partition=comp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
ENV_NAME="${ENV_NAME:-retro}"

cd "${PROJECT_DIR}"
mkdir -p logs
if [ -z "${AUDIO:-}" ] || [ -z "${OUT:-}" ]; then echo "AUDIO and OUT are required"; exit 1; fi

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | predict one clip (CPU) | $(hostname) ==="
python scripts/predict_clip.py --audio "${AUDIO}" --out "${OUT}"
echo "=== $(date) | done ==="
