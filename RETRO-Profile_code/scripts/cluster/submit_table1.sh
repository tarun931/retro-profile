#!/bin/bash
# Table 1 (RQ1: train TIMIT -> test TIMIT, 168 speakers) to FIVE seeds per model: 42, 1, 2, 3, 4.  🔵 RUN ON the GPU cluster.
#
# Submits only (no training on the login node). Already on disk, NOT resubmitted:
#   RETRO-Profile  retro_ordcon_fixed_l100_s5_timit_s42  (base setting fixed 11 Oct; kNN val/test scored)
#   WavLM + MLP    wavlm_direct_budget_timit_s42          (= RETRO minus MIL minus OrdCon, same budget)
#   Gupta 2022     gupta2022_finetune_wavlm_timit_s42 / _s1 / _s2
# New here (15 training runs + 4 scoring jobs = 19 submissions):
#   RETRO s1-4, WavLM+MLP s1-4 (retro_finetune.slurm, kNN off in training)
#   Gupta s3-4, Jaid s42/1/2/3/4 (gupta_finetune.slurm; Jaid had only an old-selection-rule number)
#   t1_knnval / t1_knntest: RETRO s1-4 with kNN forced, validation + test, embeddings saved (knn_sweep.py)
#   t1_evalG / t1_evalJ:    per-speaker scoring of the new Gupta / Jaid runs (their job scores per clip only)
#
#   bash scripts/cluster/submit_table1.sh
#
# Stops before submitting anything if a target run folder exists: resume_from=auto would silently
# resume it (CLAUDE.md rule 5). Job numbers are printed and appended to logs/table1_jobs.txt.

set -euo pipefail
cd "${PROJECT_DIR:-<data-root>/retro-profile}"
mkdir -p logs

SEEDS_NEW="1 2 3 4"
RETRO=(); WAVLM=(); GUPTA=(); JAID=()
for s in ${SEEDS_NEW}; do RETRO+=("runs/retro_ordcon_fixed_l100_s5_timit_s${s}"); WAVLM+=("runs/wavlm_direct_budget_timit_s${s}"); done
for s in 3 4; do GUPTA+=("runs/gupta2022_finetune_wavlm_timit_s${s}"); done
for s in 42 1 2 3 4; do JAID+=("runs/jaid2023_timit_s${s}"); done

for r in "${RETRO[@]}" "${WAVLM[@]}" "${GUPTA[@]}" "${JAID[@]}"; do
  if [ -e "${r}" ]; then echo "STOP: ${r} already exists -- nothing submitted."; exit 1; fi
done

OUT="logs/table1_jobs.txt"
echo "=== $(date) submit_table1.sh ===" >> "${OUT}"
record() { echo "$1 $2" | tee -a "${OUT}"; }
join() { local IFS=:; echo "$*"; }

R_IDS=(); W_IDS=(); G_IDS=(); J_IDS=()
for s in ${SEEDS_NEW}; do
  id=$(sbatch --parsable --job-name="t1_retro_s${s}" --time=04:00:00 \
       --export=ALL,EXPERIMENT=retro_ordcon_fixed_l100_s5,WITH_RETRIEVAL=0,EVAL_CORPUS=timit,SEED="${s}" \
       scripts/cluster/retro_finetune.slurm)
  R_IDS+=("${id}"); record "t1_retro_s${s}" "${id}"
  id=$(sbatch --parsable --job-name="t1_mlp_s${s}" --time=04:00:00 \
       --export=ALL,EXPERIMENT=wavlm_direct_budget,WITH_RETRIEVAL=0,EVAL_CORPUS=timit,SEED="${s}" \
       scripts/cluster/retro_finetune.slurm)
  W_IDS+=("${id}"); record "t1_mlp_s${s}" "${id}"
done
for s in 3 4; do
  id=$(sbatch --parsable --job-name="t1_gupta_s${s}" --time=03:00:00 \
       --export=ALL,EXPERIMENT=gupta2022_finetune_wavlm,EVAL_CORPUS=timit,SEED="${s}" \
       scripts/cluster/gupta_finetune.slurm)
  G_IDS+=("${id}"); record "t1_gupta_s${s}" "${id}"
done
for s in 42 1 2 3 4; do
  id=$(sbatch --parsable --job-name="t1_jaid_s${s}" --time=04:00:00 \
       --export=ALL,EXPERIMENT=jaid2023,EVAL_CORPUS=timit,SEED="${s}",TORCH_SHARING_STRATEGY=file_system \
       scripts/cluster/gupta_finetune.slurm)
  J_IDS+=("${id}"); record "t1_jaid_s${s}" "${id}"
done

id=$(sbatch --parsable --job-name=t1_knnval --dependency="afterok:$(join "${R_IDS[@]}")" \
     --export=ALL,RUNS="${RETRO[*]}",WITH_RETRIEVAL=1,FORCE_RETRIEVAL=1,SAVE_EMBEDDINGS=1,EVAL_SPLIT=val,OUT_SUFFIX=_val_knn,EVAL_CORPUS=timit \
     scripts/cluster/reevaluate.slurm)
record t1_knnval "${id}"
id=$(sbatch --parsable --job-name=t1_knntest --dependency="afterok:$(join "${R_IDS[@]}")" \
     --export=ALL,RUNS="${RETRO[*]}",WITH_RETRIEVAL=1,FORCE_RETRIEVAL=1,SAVE_EMBEDDINGS=1,EVAL_SPLIT=test,OUT_SUFFIX=_test_knn,EVAL_CORPUS=timit \
     scripts/cluster/reevaluate.slurm)
record t1_knntest "${id}"
id=$(sbatch --parsable --job-name=t1_evalG --dependency="afterok:$(join "${G_IDS[@]}")" \
     --export=ALL,RUNS="${GUPTA[*]}",WITH_RETRIEVAL=0,EVAL_CORPUS=timit scripts/cluster/reevaluate.slurm)
record t1_evalG "${id}"
id=$(sbatch --parsable --job-name=t1_evalJ --dependency="afterok:$(join "${J_IDS[@]}")" \
     --export=ALL,RUNS="${JAID[*]}",WITH_RETRIEVAL=0,EVAL_CORPUS=timit scripts/cluster/reevaluate.slurm)
record t1_evalJ "${id}"

echo "19 jobs submitted. Numbers also in ${OUT}."
