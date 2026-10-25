#!/bin/bash
# Three-seed sweep for the headline RQ1 table.  🔵 RUN ON the GPU cluster (login node).
#
# WHY. Every result in SESSION_HANDOFF.md §3.8e/§3.8f is single-seed, and the
# measured run-to-run spread on age is ~0.87 yr -- larger than every margin
# being claimed (RETRO vs Gupta on age: 0.019 yr; age-only's gain: 0.19 yr).
# Nothing is defensible until each row carries mean ± std over >= 3 seeds.
#
# WHAT. Seeds 1 and 2 for the three WavLM configurations that make up the
# headline table. Seed 42 already exists for all three, so this completes them:
#
#   retro_noknn_budget         best two-target RETRO (retrieval off)
#   retro_age_only_budget      single-task age (best age result: 3.487)
#   gupta2022_finetune_wavlm   the validated baseline, same encoder and budget
#
# Six jobs, ~25-45 min each, ~3 GPU-hours total. the GPU cluster is fair-share, not a hard
# quota, but <project> is the supervisor's project (rule R1) -- tell them before
# running this.
#
# NOT INCLUDED, deliberately:
#   * wav2vec2 arms. The headline table is WavLM; wav2vec2 is the encoder check
#     and already agrees at seed 42. Add later if a reviewer asks.
#   * Jaid 2023. Its numbers predate the val/combined_mae change, so it needs a
#     re-run for rule R6 -- but it is a different budget and belongs in its own
#     batch, not mixed into this one.
#
# USAGE
#   bash scripts/cluster/seed_sweep.sh          # submit
#   bash scripts/cluster/seed_sweep.sh --dry    # print what would be submitted
#
# AFTERWARDS, on the Mac:
#   python scripts/collect_results.py runs/retro_noknn_budget_timit_s* \
#          runs/retro_age_only_budget_timit_s* runs/gupta2022_finetune_wavlm_timit_s*
#   -> groups the _s* suffixes into mean ± std automatically.

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

cd "${PROJECT_DIR}"

# retrieval-off runs must pass WITH_RETRIEVAL=0, or retro_finetune.slurm builds
# a datastore the heads never consult and runs the wrong eval path.
#   config | slurm script | WITH_RETRIEVAL
# Override on the command line to sweep something else, e.g.
#   JOBS_OVERRIDE="retro_height_only_budget|retro_finetune.slurm|0" SEEDS_OVERRIDE="42 1 2" \
#     bash scripts/cluster/seed_sweep.sh
JOBS=(
  "retro_noknn_budget|retro_finetune.slurm|0"
  "retro_age_only_budget|retro_finetune.slurm|0"
  "gupta2022_finetune_wavlm|gupta_finetune.slurm|1"
)
SEEDS=(1 2)

if [ -n "${JOBS_OVERRIDE:-}" ]; then
  IFS=$'\n' read -r -d '' -a JOBS < <(printf '%s\n' ${JOBS_OVERRIDE} && printf '\0')
fi
if [ -n "${SEEDS_OVERRIDE:-}" ]; then
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE}"
fi

# --- preflight: fail before burning GPU time, not after -----------------------
fail=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r exp script _ <<< "${job}"
  [ -f "configs/experiment/${exp}.yaml" ] || { echo "MISSING config: configs/experiment/${exp}.yaml"; fail=1; }
  [ -f "scripts/cluster/${script}" ]           || { echo "MISSING script: scripts/cluster/${script}"; fail=1; }
done
# The selection fix must actually be on the GPU cluster. A stale rsync has silently
# invalidated runs four times in this project -- check, do not assume.
grep -q combined_mae src/retro_profile/training/metrics.py \
  || { echo "MISSING: val/combined_mae not in src/ -- push from the Mac first"; fail=1; }
grep -q combined_mae configs/training/default.yaml \
  || { echo "MISSING: combined_mae not in configs/ -- push from the Mac first"; fail=1; }
if [ "${fail}" -ne 0 ]; then
  echo
  echo "Preflight failed. Nothing submitted. On the Mac, run:"
  echo "  rsync -avz --exclude '/data/' configs scripts src ${USER_NS}@<cluster-host>:${PROJECT_DIR}/"
  exit 1
fi
echo "preflight OK"
echo

# --- submit -------------------------------------------------------------------
for job in "${JOBS[@]}"; do
  IFS='|' read -r exp script retrieval <<< "${job}"
  for seed in "${SEEDS[@]}"; do
    run_dir="runs/${exp}_timit_s${seed}"
    cmd=(sbatch --export="ALL,EXPERIMENT=${exp},SEED=${seed},WITH_RETRIEVAL=${retrieval}" "scripts/cluster/${script}")
    if [ "${DRY}" -eq 1 ]; then
      echo "would clear ${run_dir} and run: ${cmd[*]}"
      continue
    fi
    # Rule R8: resume_from=auto silently resumes a stale directory, producing a
    # two-minute non-run that evaluates the old checkpoint.
    rm -rf "${run_dir}"
    echo -n "${exp} seed ${seed}: "
    "${cmd[@]}"
  done
done

echo
if [ "${DRY}" -eq 1 ]; then
  echo "DRY RUN -- nothing was submitted and no run directory was cleared."
  echo "re-run without --dry to submit."
else
  echo "submitted. watch with:  squeue -u \$USER"
  echo "verify with:            sacct -u \$USER --starttime today --format=JobID,JobName%22,State,Elapsed | grep -v '\\.'"
fi
