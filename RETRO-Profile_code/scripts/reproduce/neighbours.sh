#!/bin/bash
# Table 4, A11 and A12: what RETRO-Profile's neighbours share, and how ordered the retrieval space is.
# Needs the finished TIMIT runs of RETRO-Profile and of its no-OrdCon ablation (scripts/reproduce/timit.sh),
# including their checkpoints; nothing is retrained.
#
#   bash scripts/reproduce/neighbours.sh
#
# A11: neighbour_quality.py on the 5 seeds with OrdCon and the 5 without -> runs/<run>/stage2_b10/neighbour_quality.json
# Table 4, A12: one retrieval pass of seed 42 that saves the speaker embeddings, then neighbour_inspection.py
#               -> runs/retro_ordcon_fixed_l100_s5_b10_timit_s42/neighbour_inspection.json

set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-src}" TOKENIZERS_PARALLELISM=false
SEEDS="${SEEDS:-42 1 2 3 4}"
WORKERS="${WORKERS:-8}"
RETRO=retro_ordcon_fixed_l100_s5_b10

for variant in "" _noordcon; do
  for seed in ${SEEDS}; do
    run="runs/${RETRO}${variant}_timit_s${seed}"
    if [ ! -e "${run}/stage2_b10/summary.json" ]; then echo "STOP: ${run} has no stage 2 -- run timit.sh first."; exit 1; fi
    echo "=== neighbour quality: ${run}"
    python scripts/neighbour_quality.py --run "${run}" --corpus timit --bag-size 10 --suffix _b10 \
        --num-workers "${WORKERS}"
  done
done

run="runs/${RETRO}_timit_s42"
echo "=== neighbour inspection: ${run}"
ckpt=$(python scripts/select_checkpoint.py "${run}")
python scripts/build_datastore.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" \
    --output "${run}/datastore_test_knn_b10" --split train
python scripts/evaluate.py --ckpt "${ckpt}" --config-dir "${run}/.hydra" --eval timit --unit speaker --bag-size 10 \
    --datastore "${run}/datastore_test_knn_b10" --force-retrieval --save-embeddings --split test \
    --output "${run}/eval_spk_test_knn_b10"
python scripts/neighbour_inspection.py --run "${run}"
echo "Done. Tables: python scripts/make_tables.py"
