#!/bin/bash
# One-time the GPU cluster setup. RUN THIS ON THE LOGIN NODE, NOT IN A JOB.
#
# Compute nodes generally have no outbound internet, so anything that needs to
# be downloaded (pip packages, HuggingFace weights) must be fetched here first.
# A job that tries to reach huggingface.co will fail several minutes in, after
# it has already sat in the queue.
#
#   bash scripts/cluster/setup_env.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# RULE: <project> is somebody else's project. We are guests on it.
#
# ~/<project> and ~/<scratch> are SHARED with every member of <project>, and the quota
# is shared too. Everything this project writes therefore goes inside a personal
# namespace (<space>/<user>/...) and nowhere else. Never write to the root of
# a shared space, never touch another member's directory.
#
#   <data-root>/retro-profile   everything: code, data, env, runs, checkpoints
#
# The welcome email suggests reproducible data goes in scratch2 -- but as of
# setup, <scratch> is at 109% of its quota (14561/13312 GB) while the
# primary space is at 30% (150/500 GB). Writing to a shared space that is
# already over quota is not on, so everything stays on primary until scratch2
# is back under. Our whole footprint is ~20 GB: env ~5, HF cache ~1, TIMIT ~1,
# and ~2 GB of checkpoints per run.
# ---------------------------------------------------------------------------
USER_NS="${USER_NS:-<user>}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
RUNS_DIR="${RUNS_DIR:-${PROJECT_DIR}/runs}"
ENV_NAME="${ENV_NAME:-retro}"

# Refuse to run if either path would land outside our own namespace.
for _d in "${PROJECT_DIR}" "${RUNS_DIR}"; do
    case "${_d}" in
        */${USER_NS}/*) : ;;
        *) echo "REFUSING: '${_d}' is outside the ${USER_NS}/ namespace." >&2; exit 1 ;;
    esac
done
mkdir -p "${PROJECT_DIR}" "${RUNS_DIR}"

# --- 1. conda -----------------------------------------------------------------
# Confirmed available on the GPU cluster: miniforge3/24.3.0-0 (default) and anaconda/2024.02-1.
module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# Keep envs off the 15 GB home quota -- a torch env is ~5 GB.
mkdir -p "${PROJECT_DIR}/envs"
conda config --add envs_dirs "${PROJECT_DIR}/envs" 2>/dev/null || true

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# --- 2. deps ------------------------------------------------------------------
# Install the PINNED versions from requirements.txt, not "latest". The pins are
# a matched set: transformers 4.42.0 works with torch 2.3.1, whereas current
# transformers refuses to torch.load a .bin checkpoint unless torch >= 2.6
# (CVE-2025-32434) and the HF encoder download dies.
#
# Deliberately NOT `pip install -e .` -- pyproject pins numpy<2 and breaks the
# environment. The package is imported via PYTHONPATH=src instead.
#
# Keep pip's cache off the 20 GB home quota.
export PIP_CACHE_DIR="${PROJECT_DIR}/.pip_cache"
mkdir -p "${PIP_CACHE_DIR}"

cd "${PROJECT_DIR}"
pip install --upgrade pip
# pytorch-lightning 2.3.0 still calls pkg_resources.declare_namespace, which
# setuptools removed in 81. Without this pin, `import pytorch_lightning` dies
# with ModuleNotFoundError: No module named 'pkg_resources'.
pip install "setuptools<81"
# CUDA build of the pinned torch (cu121 covers A100 sm_80, L40S sm_89, H100 sm_90).
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pyarrow matplotlib          # manifest writer; diagnostic plots. Not in requirements.txt
                                        # (kept out of the pinned set: neither affects results)

# --- 3. pre-download the HF encoders into a shared cache ----------------------
# HF_HOME must point somewhere the compute node can read. $HOME works; project
# storage is better if your home quota is small (wav2vec2-base is ~360 MB).
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_cache}"
mkdir -p "${HF_HOME}"
python - <<'PY'
from transformers import Wav2Vec2Model, WavLMModel
# Config A (Gupta reproduction) and Config B (matched-budget comparison).
Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
print("HF encoders cached.")
PY

# --- 4. build the TIMIT manifests --------------------------------------------
# Must be re-run on the GPU cluster: manifests store absolute audio paths, so a set built on
# the Mac is useless here.
cd "${PROJECT_DIR}"
python scripts/prepare_timit.py --raw "${TIMIT_RAW:-$PROJECT_DIR/TIMIT}" --out data/timit

# runs/ holds Lightning's checkpoints. Symlink only when RUNS_DIR was pointed
# somewhere else (e.g. scratch2, once it is back under quota); by default it is
# already inside PROJECT_DIR and just needs to exist.
if [ "${RUNS_DIR}" != "${PROJECT_DIR}/runs" ] && [ ! -e runs ]; then
    ln -s "${RUNS_DIR}" runs
fi

echo
echo "Quota check before you submit anything:"
df -h "${PROJECT_DIR}" 2>/dev/null | tail -1 || true
user_info 2>/dev/null || echo "  (run user_info to see project quotas)"

echo
echo "Setup done. HF_HOME=${HF_HOME}"
echo "Next: sbatch scripts/cluster/gupta_finetune.slurm"
