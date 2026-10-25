#!/bin/bash
# Table 4 (transfer to HeightCeleb: train TIMIT + NISP [4a] or TIMIT + NISP + an age-balanced HeightCeleb split2
# TRAINING sample [4b] -> calibrate on the 251 split2 dev speakers -> test the 249 split2 test speakers), 3 seeds
# (42, 1, 2) per model, each model at its final Table 1 setting.  🔵 RUN ON the GPU cluster (submits only).
#
#   RETRO        timit_nisp[_hcs]_b10_l100_s5     retro_finetune.slurm, then fit_residual --train-from-run --calibrated
#   WavLM + MLP  timit_nisp[_hcs]_mlp_b10_e100    retro_finetune.slurm, then split2 dev scoring (_val)
#   Gupta 2022   timit_nisp[_hcs]_gupta_lr3e6     gupta_finetune.slurm, then split2 test + dev per-speaker scoring
#   Gupta (w2v2) timit_nisp[_hcs]_gupta_w2v2_lr3e6  as Gupta, with the paper's wav2vec 2.0 encoder (added 2026-10-19)
#   Jaid 2023    timit_nisp[_hcs]_jaid_lr3e3      gupta_finetune.slurm, then split2 test + dev per-speaker scoring
# Every score uses up to 10 clips per speaker (EVAL_BAG_SIZE=10). Calibration of the baselines runs on the Mac
# (scripts/calibrate_predictions.py --corpus heightceleb_split2); RETRO's is inside its second stage.
#
#   VERSION=4a ONLY="retro gupta" bash scripts/cluster/submit_table4.sh    # 6 training + 3 scoring jobs
#   VERSION=4a ONLY="mlp jaid" bash scripts/cluster/submit_table4.sh       # 6 training + 3 scoring jobs
#   VERSION=4b ...                                                     # the same two batches for 4b
# (user, 16 Oct: at most 6 training runs at a time, so their checkpoints never fill <project>; clean each batch first.)
#
# Stops before submitting anything if a target run folder exists (resume_from=auto would resume it), or, for 4b, if
# the sample manifest is missing (scripts/sample_heightceleb_ages.py). Job numbers go to logs/table4_jobs.txt.

set -euo pipefail
cd "${PROJECT_DIR:-<data-root>/retro-profile}"
mkdir -p logs

CORPUS=heightceleb_split2
SEEDS="42 1 2"
case "${VERSION:-}" in
  4a) PREFIX=timit_nisp ;;
  4b) PREFIX=timit_nisp_hcs
      SAMPLE=data/heightceleb_split2_age_sample/manifests/train.csv
      if [ ! -s "${SAMPLE}" ]; then echo "STOP: ${SAMPLE} missing -- run scripts/sample_heightceleb_ages.py first."; exit 1; fi ;;
  *) echo "VERSION must be 4a or 4b"; exit 1 ;;
esac
MODELS="${ONLY:-retro mlp gupta jaid}"
for m in ${MODELS}; do case "$m" in retro|mlp|gupta|gupta_w2v2|jaid) ;; *) echo "unknown model in ONLY: $m"; exit 1 ;; esac; done
want() { case " ${MODELS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
exp_of() {
  case "$1" in
    retro) echo "${PREFIX}_b10_l100_s5" ;;   mlp) echo "${PREFIX}_mlp_b10_e100" ;;
    gupta) echo "${PREFIX}_gupta_lr3e6" ;;   jaid) echo "${PREFIX}_jaid_lr3e3" ;;
    gupta_w2v2) echo "${PREFIX}_gupta_w2v2_lr3e6" ;;
  esac
}
runs() { local out=""; for s in ${SEEDS}; do out="${out}runs/$(exp_of "$1")_${CORPUS}_s${s} "; done; echo "${out% }"; }
IDS_retro=""; IDS_mlp=""; IDS_gupta=""; IDS_gupta_w2v2=""; IDS_jaid=""
add_id() { eval "IDS_$1=\"\${IDS_$1}:$2\""; }
ids_of() { eval "echo \"\${IDS_$1}\""; }

for m in ${MODELS}; do
  for r in $(runs "$m"); do
    if [ -e "${r}" ]; then echo "STOP: ${r} already exists -- nothing submitted."; exit 1; fi
  done
done

OUT="logs/table4_jobs.txt"
echo "=== $(date) submit_table4.sh (version ${VERSION}, models: ${MODELS}) ===" >> "${OUT}"
record() { echo "$1 $2" | tee -a "${OUT}"; }
V="${VERSION}"
for s in ${SEEDS}; do
  for m in retro mlp; do
    want "$m" || continue
    t=$([ "$m" = mlp ] && echo 08:00:00 || echo 06:00:00)
    id=$(sbatch --parsable --job-name="t${V}_${m}_s${s}" --time="${t}" \
         --export=ALL,EXPERIMENT="$(exp_of "$m")",WITH_RETRIEVAL=0,EVAL_CORPUS="${CORPUS}",EVAL_BAG_SIZE=10,SEED="${s}" \
         scripts/cluster/retro_finetune.slurm)
    add_id "$m" "${id}"; record "t${V}_${m}_s${s}" "${id}"
  done
  for m in gupta gupta_w2v2 jaid; do
    want "$m" || continue
    t=$([ "$m" = jaid ] && echo 06:00:00 || echo 08:00:00)
    id=$(sbatch --parsable --job-name="t${V}_${m}_s${s}" --time="${t}" \
         --export=ALL,EXPERIMENT="$(exp_of "$m")",EVAL_CORPUS="${CORPUS}",SEED="${s}",TORCH_SHARING_STRATEGY=file_system \
         scripts/cluster/gupta_finetune.slurm)
    add_id "$m" "${id}"; record "t${V}_${m}_s${s}" "${id}"
  done
done

if want retro; then
id=$(sbatch --parsable --job-name="t${V}_s2_retro" --time=10:00:00 --dependency="afterok$(ids_of retro)" \
     --export=ALL,RUNS="$(runs retro)",EVAL_CORPUS="${CORPUS}",EXTRA_ARGS="--train-from-run --calibrated" \
     scripts/cluster/fit_residual.slurm)
record "t${V}_s2_retro" "${id}"
fi
if want mlp; then
id=$(sbatch --parsable --job-name="t${V}_val_mlp" --dependency="afterok$(ids_of mlp)" \
     --export=ALL,RUNS="$(runs mlp)",EVAL_SPLIT=val,OUT_SUFFIX=_val,WITH_RETRIEVAL=0,EVAL_CORPUS="${CORPUS}",EVAL_BAG_SIZE=10 \
     scripts/cluster/reevaluate.slurm)
record "t${V}_val_mlp" "${id}"
fi
for m in gupta gupta_w2v2 jaid; do
  want "$m" || continue
  id=$(sbatch --parsable --job-name="t${V}_test_${m}" --dependency="afterok$(ids_of "$m")" \
       --export=ALL,RUNS="$(runs "$m")",WITH_RETRIEVAL=0,EVAL_CORPUS="${CORPUS}",EVAL_BAG_SIZE=10 scripts/cluster/reevaluate.slurm)
  record "t${V}_test_${m}" "${id}"
  id=$(sbatch --parsable --job-name="t${V}_val_${m}" --dependency="afterok$(ids_of "$m")" \
       --export=ALL,RUNS="$(runs "$m")",EVAL_SPLIT=val,OUT_SUFFIX=_val,WITH_RETRIEVAL=0,EVAL_CORPUS="${CORPUS}",EVAL_BAG_SIZE=10 \
       scripts/cluster/reevaluate.slurm)
  record "t${V}_val_${m}" "${id}"
done
echo "Submitted ${VERSION} for: ${MODELS}. Numbers also in ${OUT}."
