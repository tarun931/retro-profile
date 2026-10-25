#!/bin/bash
# Re-evaluate finished runs from their BEST checkpoint.  🔵 SUBMIT ON the GPU cluster.
#
# WHY THIS EXISTS. Both training scripts used to select the checkpoint to
# evaluate with `ls -t ... | head -1` -- the NEWEST by modification time. That
# was correct while save_top_k was 1, where newest == best. save_top_k was
# raised to 3 after the E11 selection mistake left a better epoch unrecoverable,
# and from that point "newest" meant "whichever most recently entered the top
# three", which is not the best one. Seven of ten runs were evaluated on a
# sub-optimal checkpoint before this was caught:
#
#     gupta2022_finetune_wavlm_s42   evaluated ep18 (1.3387)  best ep7  (1.3358)
#     gupta2022_finetune_wavlm_s1    evaluated ep20 (1.3030)  best ep9  (1.3011)
#     retro_age_only_budget_s42      evaluated ep38 (0.6224)  best ep33 (0.6111)
#     retro_noknn_budget_s1          evaluated ep39 (1.3806)  best ep35 (1.3578)
#     retro_noknn_budget_s42         evaluated ep59 (1.3391)  best ep16 (1.3382)
#     retro_w2v_noknn_budget_s42     evaluated ep58 (1.3343)  best ep51 (1.3293)
#
# The gaps are small (0.0009-0.023 in normalised units, roughly 0.01-0.07 yr
# equivalent) and do not change any conclusion, but every reported number is
# slightly off and rule R6 requires that a number matches the protocol claimed
# for it. This job re-runs evaluation ONLY -- no training, no new randomness.
#
# The training scripts are fixed; this exists for runs that finished before the
# fix landed.
#
# USAGE
#   sbatch --export=ALL,RUNS="runs/a runs/b" scripts/cluster/reevaluate.slurm
#   sbatch --export=ALL,RUNS="$(ls -d runs/retro_noknn_budget_timit_s*)" \
#          scripts/cluster/reevaluate.slurm
#
# WITH_RETRIEVAL defaults to 0 because every current headline configuration has
# retrieval disabled. Pass WITH_RETRIEVAL=1 for a run that needs a datastore; it
# also writes ordcon_report.json (anchor MAE, residual share), which the
# training script does for retrieval runs and this script previously did not.
#
# Which checkpoint: scripts/select_checkpoint.py -- training's own choice, and
# phase-aware for two-phase runs (see that file for the bug it fixes). Two
# optional overrides, for comparing a specific epoch side by side:
#   CKPT_EPOCH=59     evaluate checkpoints/epoch=59-*.ckpt instead
#   OUT_SUFFIX=_ep59  write eval_utt_ep59/, eval_spk_ep59/, datastore_ep59/,
#                     ordcon_report_ep59.json ... instead of overwriting
#   sbatch --export=ALL,RUNS=runs/x,WITH_RETRIEVAL=1,CKPT_EPOCH=59,OUT_SUFFIX=_ep59 \
#          scripts/cluster/reevaluate.slurm
#
# kNN sweeps (scripts/knn_sweep.py): EVAL_SPLIT=val scores the validation
# speakers (needs a suffix containing 'val'); SAVE_EMBEDDINGS=1 writes each row's
# kNN query so k / weighting can be swept offline.
#
# Existing eval_spk/ and eval_utt/ directories are OVERWRITTEN. That is the
# point: the old numbers came from the wrong checkpoint. `git`-style backups are
# not taken, so if you want the superseded numbers, copy the directories first.

#SBATCH --job-name=reeval
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
WITH_RETRIEVAL="${WITH_RETRIEVAL:-0}"
# Which corpus to evaluate against. Must match what the run was TRAINED on.
EVAL_CORPUS="${EVAL_CORPUS:-timit}"
CKPT_EPOCH="${CKPT_EPOCH:-}"
# CKPT_NAME: a file pattern inside checkpoints/, e.g. 'best_age-*' or last.ckpt.
# Must match exactly one file. Takes precedence over CKPT_EPOCH.
CKPT_NAME="${CKPT_NAME:-}"
SUF="${OUT_SUFFIX:-}"
# FORCE_RETRIEVAL=1 (with WITH_RETRIEVAL=1): switch kNN on for a model trained
# WITHOUT it. Predictions are then the kNN anchor alone. Use an OUT_SUFFIX such
# as _knn so the model's own retrieval-off results are not overwritten.
FORCE_RETRIEVAL="${FORCE_RETRIEVAL:-0}"
if [ "${FORCE_RETRIEVAL}" = "1" ] && [ "${WITH_RETRIEVAL}" != "1" ]; then
  echo "FORCE_RETRIEVAL=1 needs WITH_RETRIEVAL=1"; exit 1
fi
if [ "${FORCE_RETRIEVAL}" = "1" ] && [ -z "${SUF}" ]; then
  echo "FORCE_RETRIEVAL=1 needs an OUT_SUFFIX (e.g. _knn) so retrieval-off results are kept"; exit 1
fi
# EVAL_SPLIT=val scores the 70 VALIDATION speakers -- for choosing settings such as
# the kNN k, never as a result. It must write to its own suffix (e.g. _val_knn):
# without one it would overwrite the test results in eval_spk/ and eval_utt/.
EVAL_SPLIT="${EVAL_SPLIT:-test}"
if [ "${EVAL_SPLIT}" != "test" ] && [[ "${SUF}" != *val* ]]; then
  echo "EVAL_SPLIT=${EVAL_SPLIT} needs an OUT_SUFFIX containing 'val' (e.g. _val_knn)"; exit 1
fi
# SAVE_EMBEDDINGS=1: also write embeddings_<corpus>.npz (the kNN query per row),
# so k and the weighting can be swept offline (scripts/knn_sweep.py).
EMB_FLAG=()
if [ "${SAVE_EMBEDDINGS:-0}" = "1" ]; then EMB_FLAG=(--save-embeddings); fi
# EVAL_BAG_SIZE=10: speaker bags hold up to 10 clips at evaluation (TIMIT: all 10, fixed order, the same
# clips for every model) instead of the corpus default (8 drawn at random). Unset = unchanged.
BAG_FLAG=()
if [ -n "${EVAL_BAG_SIZE:-}" ]; then BAG_FLAG=(--bag-size "${EVAL_BAG_SIZE}"); fi

cd "${PROJECT_DIR}"
mkdir -p logs

if [ -z "${RUNS:-}" ]; then
  echo "RUNS is empty. Pass it, e.g.:"
  echo "  sbatch --export=ALL,RUNS=\"\$(ls -d runs/retro_noknn_budget_timit_s*)\" scripts/cluster/reevaluate.slurm"
  exit 1
fi

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export PYTHONPATH="${PROJECT_DIR}/src"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=== $(date) | reevaluate | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

for RUN_DIR in ${RUNS}; do
  RUN_DIR="${RUN_DIR%/}"
  if [ ! -d "${RUN_DIR}/checkpoints" ]; then
    echo "--- ${RUN_DIR}: no checkpoints/, skipping"
    continue
  fi

  if [ -n "${CKPT_NAME}" ]; then
    # shellcheck disable=SC2206
    matches=( "${RUN_DIR}/checkpoints/"${CKPT_NAME} )
    if [ "${#matches[@]}" -eq 1 ] && [ -f "${matches[0]}" ]; then CKPT="${matches[0]}"; else
      echo "--- ${RUN_DIR}: CKPT_NAME='${CKPT_NAME}' matched ${#matches[@]} files, need exactly 1 -- skipping"; CKPT=""; fi
  elif [ -n "${CKPT_EPOCH}" ]; then
    CKPT=$(ls "${RUN_DIR}/checkpoints/epoch=${CKPT_EPOCH}-"*.ckpt 2>/dev/null | head -1) || CKPT=""
  else
    CKPT=$(python scripts/select_checkpoint.py "${RUN_DIR}") || CKPT=""
  fi

  if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
    echo "--- ${RUN_DIR}: no usable checkpoint, skipping"
    continue
  fi

  echo "--- ${RUN_DIR}"
  echo "    checkpoint: $(basename "${CKPT}")${CKPT_NAME:+  (forced: CKPT_NAME=${CKPT_NAME})}${CKPT_EPOCH:+  (forced: CKPT_EPOCH=${CKPT_EPOCH})}  outputs suffix: '${SUF}'"

  if [ "${WITH_RETRIEVAL}" = "1" ]; then
    python scripts/build_datastore.py \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --output "${RUN_DIR}/datastore${SUF}" --split train
    DS=(--datastore "${RUN_DIR}/datastore${SUF}")
    if [ "${FORCE_RETRIEVAL}" = "1" ]; then DS+=(--force-retrieval); fi
  else
    DS=()
  fi

  python scripts/evaluate.py --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
      "${DS[@]}" "${EMB_FLAG[@]}" --split "${EVAL_SPLIT}" \
      --eval "${EVAL_CORPUS}" --unit utterance --output "${RUN_DIR}/eval_utt${SUF}"

  python scripts/evaluate.py --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
      "${DS[@]}" "${EMB_FLAG[@]}" "${BAG_FLAG[@]}" --split "${EVAL_SPLIT}" \
      --eval "${EVAL_CORPUS}" --unit speaker --output "${RUN_DIR}/eval_spk${SUF}"

  # Attention by sentence type (TIMIT SA/SX/SI; NISP COMMON/SHARED/UNIQUE). Only meaningful over a real bag, so it
  # reads eval_spk; in utterance mode every bag holds one clip and every weight
  # is trivially 1.0. Single-target runs have one block instead of two.
  python scripts/analyse_attention.py \
      --attention "${RUN_DIR}/eval_spk${SUF}/attention_${EVAL_CORPUS}.csv" \
      --predictions "${RUN_DIR}/eval_spk${SUF}/predictions_${EVAL_CORPUS}.csv" \
      --output "${RUN_DIR}/eval_spk${SUF}/attention_report" || true

  # Is the error mean reversion or an information limit? For height this is the
  # diagnostic that showed the models sit ~0.2 cm from a gender-mean lookup.
  for t in age height; do
    python scripts/analyse_calibration.py \
        --predictions "${RUN_DIR}/eval_spk${SUF}/predictions_${EVAL_CORPUS}.csv" --target "${t}" \
        --output "${RUN_DIR}/calibration_${t}${SUF}.json" > /dev/null 2>&1 || true
  done

  # Retrieval runs: anchor MAE before the residual, residual share, and the
  # embedding-ordering metrics -- the same report the training script writes.
  if [ "${WITH_RETRIEVAL}" = "1" ]; then
    python scripts/analyse_ordcon.py --run "${RUN_DIR}" \
        --predictions "${RUN_DIR}/eval_spk${SUF}/predictions_${EVAL_CORPUS}.csv" \
        --ckpt "${CKPT}" --config-dir "${RUN_DIR}/.hydra" \
        --output "${RUN_DIR}/ordcon_report${SUF}.json" || true
  fi
done

echo
echo "=== $(date) | done ==="
echo "Rebuild the results table on the Mac with:"
echo "  python scripts/collect_results.py runs/<pattern>_s*"
