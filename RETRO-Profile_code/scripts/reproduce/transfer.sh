#!/bin/bash
# Table 5 (and A3, A5, A6): transfer to HeightCeleb. Five models x 3 seeds, trained on TIMIT + NISP (4a) or on
# TIMIT + NISP + the 84-speaker HeightCeleb sample (4b), scored on the 249 HeightCeleb test speakers and calibrated
# on the 251 validation speakers (corrected = a x prediction + b, per trait).
#
#   bash scripts/reproduce/transfer.sh 4a
#   bash scripts/reproduce/transfer.sh 4b                       # needs data/heightceleb_split2_age_sample/
#   MODELS="retro mlp" SEEDS="42" bash scripts/reproduce/transfer.sh 4a
#
# Run from RETRO-Profile_code/ after data/timit, data/nisp and data/heightceleb_split2 are prepared. Each run writes
# runs/<configuration>_heightceleb_split2_s<seed>/. A finished run is skipped; an unfinished run folder stops the
# script (delete it and run again).

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}" TOKENIZERS_PARALLELISM=false
VERSION="${1:-}"
case "${VERSION}" in
  4a) PREFIX=timit_nisp ;;
  4b) PREFIX=timit_nisp_hcs
      if [ ! -s data/heightceleb_split2_age_sample/manifests/train.csv ]; then
        echo "STOP: data/heightceleb_split2_age_sample/ is missing -- run scripts/sample_heightceleb_ages.py first."; exit 1
      fi ;;
  *) echo "usage: bash scripts/reproduce/transfer.sh 4a|4b"; exit 1 ;;
esac
SEEDS="${SEEDS:-42 1 2}"
MODELS="${MODELS:-retro mlp gupta gupta_w2v2 jaid}"
WORKERS="${WORKERS:-8}"
CORPUS=heightceleb_split2

for model in ${MODELS}; do
  case "${model}" in
    retro)      cfg="${PREFIX}_b10_l100_s5";       done_file=stage2_b10/summary.json ;;
    mlp)        cfg="${PREFIX}_mlp_b10_e100";      done_file=eval_spk_cal/predictions_${CORPUS}.csv ;;
    gupta)      cfg="${PREFIX}_gupta_lr3e6";       done_file=eval_spk_cal/predictions_${CORPUS}.csv ;;
    gupta_w2v2) cfg="${PREFIX}_gupta_w2v2_lr3e6";  done_file=eval_spk_cal/predictions_${CORPUS}.csv ;;
    jaid)       cfg="${PREFIX}_jaid_lr3e3";        done_file=eval_spk_cal/predictions_${CORPUS}.csv ;;
    *) echo "unknown model in MODELS: ${model}"; exit 1 ;;
  esac
  for seed in ${SEEDS}; do
    name="${cfg}_${CORPUS}_s${seed}"; run="runs/${name}"
    if [ -e "${run}/${done_file}" ]; then echo "=== ${name}: done, skipped"; continue; fi
    if [ -e "${run}" ]; then echo "STOP: ${run} exists but is unfinished -- delete it and run again."; exit 1; fi
    echo "=== ${name}"

    case "${model}" in gupta*|jaid) export TORCH_SHARING_STRATEGY=file_system ;; *) unset TORCH_SHARING_STRATEGY ;; esac
    python scripts/train.py experiment="${cfg}" seed="${seed}" run_name="${name}" num_workers="${WORKERS}"
    ckpt=$(python scripts/select_checkpoint.py "${run}")

    # Test speakers: one prediction per speaker from 10 fixed clips (the same audio for every model).
    python scripts/evaluate.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" --eval "${CORPUS}" \
        --unit speaker --bag-size 10 --output "${run}/eval_spk"
    if [ "${model}" = retro ]; then
      # Stage 2 with the datastore built from the run's own training corpora; k and the residual are chosen on the
      # calibrated validation error, and the calibrated test predictions are written (stage2_b10/*_cal.csv).
      python scripts/fit_residual.py --run "${run}" --corpus "${CORPUS}" --bag-size 10 --suffix _b10 \
          --check-sub eval_spk --num-workers "${WORKERS}" --train-from-run --calibrated
    else
      # Validation speakers, then the two-number calibration fitted on them and applied to test (eval_spk_cal/).
      python scripts/evaluate.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" --eval "${CORPUS}" \
          --unit speaker --bag-size 10 --split val --output "${run}/eval_spk_val"
      python scripts/calibrate_predictions.py "${run}" --corpus "${CORPUS}"
    fi
  done
done
echo "Done. Tables: python scripts/make_tables.py"
