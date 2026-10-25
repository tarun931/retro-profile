#!/bin/bash
# RETRO-Profile with a FINE-TUNED WavLM encoder.
#
# Unlike the Gupta job this runs FOUR stages, because RETRO is a retrieval
# method and the datastore has to be built from the trained encoder before it
# can be evaluated:
#
#   1. train                  (two-phase curriculum; datastore rebuilt each epoch)
#   2. build_datastore        from the selected checkpoint
#   3. evaluate WITH kNN      -> runs/<run>/eval_utt
#   4. evaluate WITHOUT kNN   -> runs/<run>/eval_utt_noknn
#
# Stage 4 costs one extra inference pass and gives the retrieval ablation
# (SESSION_HANDOFF.md §3.3) within a single checkpoint -- immune to seed and
# checkpoint-selection noise, which is what made that result trustworthy.
#
#   sbatch scripts/cluster/retro_finetune.slurm                       # with retrieval
#   sbatch --export=ALL,EXPERIMENT=retro_finetune_noknn,WITH_RETRIEVAL=0 \
#          scripts/cluster/retro_finetune.slurm                       # the fair ablation
#
# The pair above is the honest kNN ablation: identical budget, retrieval on vs
# off for the WHOLE run. Do not use `eval_utt_noknn` from the retrieval run for
# this -- once the datastore is populated the direct branch stops training, so
# that number measures an abandoned head (see retro_finetune_noknn.yaml).

#SBATCH --job-name=retro_ft
#SBATCH --account=<project>
#SBATCH --time=02:00:00              # measured on L40S: ~15 s/epoch + ~10 s per
                                     # datastore rebuild, so 60 epochs is ~30 min.
                                     # 2 h covers training + datastore + two evals.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1            # 48 GB; bag-mode fine-tuning is memory-hungry
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
ENV_NAME="${ENV_NAME:-retro}"
SEED="${SEED:-42}"
EXPERIMENT="${EXPERIMENT:-retro_finetune}"
WITH_RETRIEVAL="${WITH_RETRIEVAL:-1}"
# Corpus to TRAIN and EVALUATE on. This used to be hardcoded to `timit` at
# every --eval call site, so the four NISP runs of 2026-09-12 trained on NISP
# and were then evaluated on TIMIT -- valid training, wrong test set, and it
# only surfaced because a TIMIT utterance id (SX289) appeared in a NISP run's
# attention CSV. The run name carries it too, so a run directory says which
# corpus produced it.
EVAL_CORPUS="${EVAL_CORPUS:-timit}"
RUN_NAME="${EXPERIMENT}_${EVAL_CORPUS}_s${SEED}"
# EVAL_BAG_SIZE=10: score the speaker bags on up to 10 clips (TIMIT: all, fixed order) instead of the corpus
# default (8 drawn at random). Unset = unchanged.
BAG_FLAG=()
if [ -n "${EVAL_BAG_SIZE:-}" ]; then BAG_FLAG=(--bag-size "${EVAL_BAG_SIZE}"); fi
RUN_DIR="runs/${RUN_NAME}"

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

echo "=== $(date) | ${RUN_NAME} | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- 1. train -----------------------------------------------------------------
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
CKPT=$(python scripts/select_checkpoint.py "${RUN_DIR}")
echo "=== selected checkpoint: ${CKPT} ==="

if [ "${WITH_RETRIEVAL}" = "1" ]; then
    # --- 2. build the datastore ----------------------------------------------
    # Indexes the TRAIN speakers only. Leave-one-out retrieval (bug §6.2) keeps a
    # training speaker from retrieving itself.
    python scripts/build_datastore.py \
        --ckpt "${CKPT}" \
        --config-dir "${RUN_DIR}/.hydra" \
        --output "${RUN_DIR}/datastore" \
        --split train

    # --- 3. evaluate WITH retrieval, per-utterance ---------------------------
    python scripts/evaluate.py \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --datastore "${RUN_DIR}/datastore" \
        --eval "${EVAL_CORPUS}" --unit utterance --output "${RUN_DIR}/eval_utt"

    # --- 4. bag mode, for the MIL attention analysis -------------------------
    # Attention is only meaningful over a real bag; in utterance mode every bag
    # holds one clip and every weight is trivially 1.0.
    python scripts/evaluate.py \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --datastore "${RUN_DIR}/datastore" \
        "${BAG_FLAG[@]}" --eval "${EVAL_CORPUS}" --unit speaker --output "${RUN_DIR}/eval_spk"

    python scripts/analyse_attention.py \
        --attention "${RUN_DIR}/eval_spk/attention_timit.csv" \
        --predictions "${RUN_DIR}/eval_spk/predictions_timit.csv" \
        --output "${RUN_DIR}/eval_spk/attention_report" || true

    # --- 5. OrdCon / retrieval-space analysis --------------------------------
    # Section A (embedding ordering) needs the checkpoint; section B (anchor
    # quality, residual contribution) needs only the predictions CSV.
    python scripts/analyse_ordcon.py \
        --run "${RUN_DIR}" \
        --predictions "${RUN_DIR}/eval_spk/predictions_timit.csv" \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --output "${RUN_DIR}/ordcon_report.json" || true
else
    # No datastore in this configuration -- the direct branch is the model.
    python scripts/evaluate.py \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --eval "${EVAL_CORPUS}" --unit utterance --output "${RUN_DIR}/eval_utt"

    python scripts/evaluate.py \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        "${BAG_FLAG[@]}" --eval "${EVAL_CORPUS}" --unit speaker --output "${RUN_DIR}/eval_spk"
fi

# --- 6. per-epoch diagnostic curves ------------------------------------------
# Reads diagnostics.csv, written every epoch by PerEpochDiagnostics: accuracy,
# n_eff, sentence-type preference and anchor share vs epoch, with the retrieval
# switch-on marked. Markdown always; PNG when matplotlib is installed.
python scripts/plot_diagnostics.py --run "${RUN_DIR}" || true

echo
echo "=== artefacts written ==="
find "${RUN_DIR}" -maxdepth 2 \( -name '*.json' -o -name '*.csv' -o -name '*.md' \
     -o -name '*.png' \) -printf '  %P\n' 2>/dev/null | sort || true

echo "=== $(date) | done ==="
echo
echo "GPU/CPU utilisation (note: jobstats can misreport the device on multi-GPU"
echo "nodes -- srun --jobid=\${SLURM_JOB_ID} --overlap nvidia-smi is authoritative):"
jobstats "${SLURM_JOB_ID}" 2>/dev/null || echo "  run: jobstats ${SLURM_JOB_ID}"
