#!/bin/bash
# Gupta 2022 -- Config A, the faithful reproduction run.
#
# Target (paper Table 1, PER-UTTERANCE, M/F):
#     age    MAE 3.96 / 4.48    RMSE 5.54 / 6.49
#     height MAE 5.58 / 5.07    RMSE 7.30 / 6.43
#
#   sbatch scripts/cluster/gupta_finetune.slurm
#
# Configured for project <project>, user <user>. Nothing left to fill in --
# submit as-is. Only --partition/--gres are worth revisiting (see below).

#SBATCH --job-name=gupta_ft
#SBATCH --account=<project>               # project <project> -- Large language models for detecting misinformation
#SBATCH --time=03:00:00           # measured: ~31 s/epoch on an L40S, so 50 epochs
                                 # is ~30 min. 3 h is ample and schedules far sooner
                                 # than a 12 h request.
#SBATCH --partition=gpu              # confirmed via sinfo: A100 x32, L40S x80, A40 x12, T4 x8
# #SBATCH --qos=...                 # not needed: <project> uses the default QoS. Add only if a job is rejected.
#SBATCH --gres=gpu:L40S:1            # 80 L40S vs 32 A100 -> much shorter queue, 48 GB VRAM is ample
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# GPU alternatives on the `gpu` partition (sinfo, 2026-08-23):
#     gpu:A100:1   32 available -- fastest, but more contended
#     gpu:L40S:1   80 available -- the default here
#     gpu:1        whatever frees up first; may land you on a 16 GB T4, which is
#                  tight for a wav2vec2 fine-tune even with checkpointing
# m3g/V100 no longer exists. H100s are on partition m3h and need --qos=m3h.
# The gpu partition allows up to 7 days, so our 12 h request is comfortable.
#
# the GPU cluster enforces a GPU-efficiency policy: jobs that leave the GPU idle get killed,
# so keep num_workers high enough that the loader is not the bottleneck.

set -euo pipefail

# <project> is a shared project -- everything we write stays inside our own
# namespace (rule R1). runs/ lives under PROJECT_DIR, NOT scratch2: <scratch>
# was at 109% of quota (14,561/13,312 GB) while the primary space sat at 32%,
# so setup_env.sh defaults RUNS_DIR to ${PROJECT_DIR}/runs and only symlinks
# elsewhere if RUNS_DIR is explicitly repointed (rule R2).
USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
ENV_NAME="${ENV_NAME:-retro}"
SEED="${SEED:-42}"
# Selectable so the WavLM arm of the matched RQ1 pair can reuse this script:
#   sbatch --export=ALL,EXPERIMENT=gupta2022_finetune_wavlm scripts/cluster/gupta_finetune.slurm
EXPERIMENT="${EXPERIMENT:-gupta2022_finetune}"
# See retro_finetune.slurm: hardcoding this to `timit` made NISP-trained
# models evaluate against the TIMIT test set.
EVAL_CORPUS="${EVAL_CORPUS:-timit}"
RUN_NAME="${EXPERIMENT}_${EVAL_CORPUS}_s${SEED}"

cd "${PROJECT_DIR}"
mkdir -p logs

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true   # miniforge3/24.3.0-0 confirmed on the GPU cluster
# `source activate` is not on PATH in a non-interactive/compute-node shell --
# sourcing conda's own init script is the portable way to get `conda activate`.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# The package is imported from src/ (never `pip install -e .`; pyproject pins numpy<2).
export PYTHONPATH="${PROJECT_DIR}/src"
# Compute nodes have no internet: read the weights cached by setup_env.sh and
# fail loudly rather than hanging on a download attempt.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | ${RUN_NAME} | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- train --------------------------------------------------------------------
# resume_from=auto so a walltime kill can be resumed by resubmitting this script
# unchanged. Note a resumed run's RNG diverges from an uninterrupted one.
python scripts/train.py \
    experiment="${EXPERIMENT}" \
    seed="${SEED}" \
    run_name="${RUN_NAME}" \
    resume_from=auto \
    num_workers=8

# Pick the BEST checkpoint by the val=... field in its filename, not the newest.
#
# This line used to be `ls -t ... | head -1`. That was correct only while
# save_top_k was 1, where newest == best. save_top_k was raised to 3 after the
# E11 selection mistake left a better epoch unrecoverable, and "newest" then
# means "whichever most recently entered the top three" -- which evaluated a
# sub-optimal checkpoint in 7 of 10 runs before this was caught. train.py writes
# filenames as epoch={epoch}-step={step}-val={monitor:.4f}.ckpt and every
# config monitors val/combined_mae, where LOWER IS BETTER.
# 2026-10-10: the lowest-val= scan above picked an ORPHANED PHASE-1 file in a
# two-phase run (training chose epoch 39, evaluation used epoch 16). Ask the
# helper instead: it reads training's own choice (best_checkpoint.txt) and is
# phase-aware for older runs. It exits non-zero rather than guess.
CKPT=$(python scripts/select_checkpoint.py "runs/${RUN_NAME}")
echo "=== evaluating ${CKPT} ==="

# --- evaluate -----------------------------------------------------------------
# --unit utterance is mandatory here: the paper's numbers are over 1,680 clips,
# not 168 speakers. Per-speaker metrics are derived from the SAME predictions,
# so both protocols land in one summary.json and cannot drift apart.
python scripts/evaluate.py \
    --ckpt "${CKPT}" \
    --config-dir "runs/${RUN_NAME}/.hydra" \
    --eval "${EVAL_CORPUS}" \
    --unit utterance \
    --output "runs/${RUN_NAME}/eval_utt"

echo "=== $(date) | done ==="
echo
# the GPU cluster kills jobs that leave the GPU idle. jobstats reports actual CPU/GPU
# utilisation for a finished job -- check it before queueing anything longer.
echo "GPU/CPU utilisation for this job:"
jobstats "${SLURM_JOB_ID}" 2>/dev/null || echo "  run: jobstats ${SLURM_JOB_ID}"
