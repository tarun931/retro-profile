#!/bin/bash
# Tables 2 and 3 (and A2, A4): the 35 TIMIT runs -- 7 configurations x 5 seeds -- trained, scored on the 168 test
# speakers, and (RETRO-Profile and its two ablations) given their second stage.
#
#   bash scripts/reproduce/timit.sh                                                   # all 35 runs
#   SEEDS="42" bash scripts/reproduce/timit.sh                                        # one seed of each
#   CONFIGS="retro_ordcon_fixed_l100_s5_b10" SEEDS="42" bash scripts/reproduce/timit.sh   # one run
#
# Run from RETRO-Profile_code/ after data/timit/ is prepared. Runs go one after another on one GPU; each writes
# runs/<configuration>_timit_s<seed>/. A finished run is skipped. A run folder left by an interrupted run stops the
# script: delete it and start again (training does not resume, so a half-finished run is never mixed with a new one).

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}" TOKENIZERS_PARALLELISM=false
SEEDS="${SEEDS:-42 1 2 3 4}"
CONFIGS="${CONFIGS:-retro_ordcon_fixed_l100_s5_b10 retro_ordcon_fixed_l100_s5_b10_nomil retro_ordcon_fixed_l100_s5_b10_noordcon wavlm_direct_budget_b10_e100 gupta2022_finetune_wavlm_lr3e6 gupta2022_finetune_w2v2_lr3e6 jaid2023_lr3e3}"
WORKERS="${WORKERS:-8}"

for cfg in ${CONFIGS}; do
  case "${cfg}" in
    retro_*)  kind=retro; done_file=stage2_b10/summary.json ;;
    wavlm_*)  kind=mlp;   done_file=eval_spk/predictions_timit.csv ;;
    gupta*|jaid*) kind=clip; done_file=eval_utt/predictions_timit_speaker.csv ;;
    *) echo "unknown configuration: ${cfg}"; exit 1 ;;
  esac
  for seed in ${SEEDS}; do
    name="${cfg}_timit_s${seed}"; run="runs/${name}"
    if [ -e "${run}/${done_file}" ]; then echo "=== ${name}: done, skipped"; continue; fi
    if [ -e "${run}" ]; then echo "STOP: ${run} exists but is unfinished -- delete it and run again."; exit 1; fi
    echo "=== ${name}"

    # Stage 1: training. The baselines' data loaders use file-system sharing (avoids worker crashes; no effect on results).
    if [ "${kind}" = clip ]; then export TORCH_SHARING_STRATEGY=file_system; else unset TORCH_SHARING_STRATEGY; fi
    python scripts/train.py experiment="${cfg}" seed="${seed}" run_name="${name}" num_workers="${WORKERS}"
    ckpt=$(python scripts/select_checkpoint.py "${run}")            # the epoch training chose (best validation S)

    if [ "${kind}" = clip ]; then
      # Gupta et al. and Jaid et al. predict per clip; a speaker's estimate is the mean over their 10 clips.
      python scripts/evaluate.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" --eval timit \
          --unit utterance --output "${run}/eval_utt"
    else
      # RETRO-Profile and WavLM + MLP predict per speaker from all 10 clips (kNN off for RETRO-Profile here).
      python scripts/evaluate.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" --eval timit \
          --unit speaker --bag-size 10 --output "${run}/eval_spk"
    fi
    if [ "${kind}" = retro ]; then
      # Stage 2: datastore of the 392 training speakers, k per trait on validation, residual correction.
      python scripts/fit_residual.py --run "${run}" --corpus timit --bag-size 10 --suffix _b10 \
          --check-sub eval_spk --num-workers "${WORKERS}"
    fi
  done
done
echo "Done. Tables: python scripts/make_tables.py"
