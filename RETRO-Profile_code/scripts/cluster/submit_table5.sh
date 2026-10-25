#!/bin/bash
# Table 5 (cross-corpus: train HeightCeleb + NISP -> calibrate on 70 TIMIT validation speakers -> test 168 TIMIT test
# speakers), 3 seeds (42, 1, 2) per model, each model at its final Table 1 setting.  🔵 RUN ON the GPU cluster (submits only).
#
#   RETRO        heightceleb_nisp_b10_l100_s5     retro_finetune.slurm, then fit_residual --train-from-run --calibrated
#   WavLM + MLP  heightceleb_nisp_mlp_b10_e100    retro_finetune.slurm, then TIMIT validation scoring (_val)
#   Gupta 2022   heightceleb_nisp_gupta_lr3e6     gupta_finetune.slurm, then TIMIT test + validation per-speaker scoring
#   Jaid 2023    heightceleb_nisp_jaid_lr3e3      gupta_finetune.slurm, then TIMIT test + validation per-speaker scoring
# Every TIMIT score uses all 10 clips per speaker (EVAL_BAG_SIZE=10). Calibration of the baselines runs on the Mac
# (scripts/calibrate_predictions.py); RETRO's is inside its second stage. 12 training runs + 6 scoring jobs.
#
#   bash scripts/cluster/submit_table5.sh                          # all 4 models (12 training + 6 scoring jobs)
#   ONLY="retro gupta" bash scripts/cluster/submit_table5.sh       # batch A: 6 training + 3 scoring jobs
#   ONLY="mlp jaid" bash scripts/cluster/submit_table5.sh          # batch B: 6 training + 3 scoring jobs
# (user, 16 Oct: at most 6 training runs at a time, so their checkpoints never fill <project>; clean batch A's first.)
#
# Stops before submitting anything if a target run folder exists (resume_from=auto would resume it).
# Job numbers are printed and appended to logs/table5_jobs.txt.

set -euo pipefail
cd "${PROJECT_DIR:-<data-root>/retro-profile}"
mkdir -p logs

SEEDS="42 1 2"
MODELS="${ONLY:-retro mlp gupta jaid}"
for m in ${MODELS}; do case "$m" in retro|mlp|gupta|jaid) ;; *) echo "unknown model in ONLY: $m"; exit 1 ;; esac; done
want() { case " ${MODELS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
exp_of() {
  case "$1" in
    retro) echo heightceleb_nisp_b10_l100_s5 ;;  mlp) echo heightceleb_nisp_mlp_b10_e100 ;;
    gupta) echo heightceleb_nisp_gupta_lr3e6 ;;  jaid) echo heightceleb_nisp_jaid_lr3e3 ;;
  esac
}
runs() { local out=""; for s in ${SEEDS}; do out="${out}runs/$(exp_of "$1")_timit_s${s} "; done; echo "${out% }"; }
IDS_retro=""; IDS_mlp=""; IDS_gupta=""; IDS_jaid=""
add_id() { eval "IDS_$1=\"\${IDS_$1}:$2\""; }
ids_of() { eval "echo \"\${IDS_$1}\""; }

for m in ${MODELS}; do
  for r in $(runs "$m"); do
    if [ -e "${r}" ]; then echo "STOP: ${r} already exists -- nothing submitted."; exit 1; fi
  done
done

OUT="logs/table5_jobs.txt"
echo "=== $(date) submit_table5.sh (models: ${MODELS}) ===" >> "${OUT}"
record() { echo "$1 $2" | tee -a "${OUT}"; }
for s in ${SEEDS}; do
  for m in retro mlp; do
    want "$m" || continue
    t=$([ "$m" = mlp ] && echo 08:00:00 || echo 06:00:00)
    id=$(sbatch --parsable --job-name="t5_${m}_s${s}" --time="${t}" \
         --export=ALL,EXPERIMENT="$(exp_of "$m")",WITH_RETRIEVAL=0,EVAL_CORPUS=timit,EVAL_BAG_SIZE=10,SEED="${s}" \
         scripts/cluster/retro_finetune.slurm)
    add_id "$m" "${id}"; record "t5_${m}_s${s}" "${id}"
  done
  for m in gupta jaid; do
    want "$m" || continue
    t=$([ "$m" = gupta ] && echo 08:00:00 || echo 06:00:00)
    id=$(sbatch --parsable --job-name="t5_${m}_s${s}" --time="${t}" \
         --export=ALL,EXPERIMENT="$(exp_of "$m")",EVAL_CORPUS=timit,SEED="${s}",TORCH_SHARING_STRATEGY=file_system \
         scripts/cluster/gupta_finetune.slurm)
    add_id "$m" "${id}"; record "t5_${m}_s${s}" "${id}"
  done
done

if want retro; then
id=$(sbatch --parsable --job-name=t5_s2_retro --time=10:00:00 --dependency="afterok$(ids_of retro)" \
     --export=ALL,RUNS="$(runs retro)",EVAL_CORPUS=timit,EXTRA_ARGS="--train-from-run --calibrated" \
     scripts/cluster/fit_residual.slurm)
record t5_s2_retro "${id}"
fi
if want mlp; then
id=$(sbatch --parsable --job-name=t5_val_mlp --dependency="afterok$(ids_of mlp)" \
     --export=ALL,RUNS="$(runs mlp)",EVAL_SPLIT=val,OUT_SUFFIX=_val,WITH_RETRIEVAL=0,EVAL_CORPUS=timit,EVAL_BAG_SIZE=10 \
     scripts/cluster/reevaluate.slurm)
record t5_val_mlp "${id}"
fi
for m in gupta jaid; do
  want "$m" || continue
  id=$(sbatch --parsable --job-name="t5_test_${m}" --dependency="afterok$(ids_of "$m")" \
       --export=ALL,RUNS="$(runs "$m")",WITH_RETRIEVAL=0,EVAL_CORPUS=timit,EVAL_BAG_SIZE=10 scripts/cluster/reevaluate.slurm)
  record "t5_test_${m}" "${id}"
  id=$(sbatch --parsable --job-name="t5_val_${m}" --dependency="afterok$(ids_of "$m")" \
       --export=ALL,RUNS="$(runs "$m")",EVAL_SPLIT=val,OUT_SUFFIX=_val,WITH_RETRIEVAL=0,EVAL_CORPUS=timit,EVAL_BAG_SIZE=10 \
       scripts/cluster/reevaluate.slurm)
  record "t5_val_${m}" "${id}"
done
echo "Submitted for: ${MODELS}. Numbers also in ${OUT}."
