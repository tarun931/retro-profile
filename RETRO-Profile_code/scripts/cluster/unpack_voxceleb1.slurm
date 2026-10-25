#!/bin/bash
# Verify and unpack VoxCeleb1 audio (HeightCeleb's speakers).  🔵 SUBMIT ON the GPU cluster -- never unzip on a login node.
#
# The zips come from the Hugging Face copy ProgramComputer/voxceleb (unofficial; the
# official VoxCeleb site no longer distributes audio -- heightceleb/HeightCeleb_Summary.pdf).
# Before anything is unpacked, each zip's size and MD5 must equal the original release's,
# as published in clovaai/voxceleb_trainer lists/ -- a match proves the copy is byte-identical.
#
# Result: <data-root>/voxceleb1/wav/<VoxCeleb1 ID>/<YouTube video id>/<nnnnn>.wav
# Expected: 1,251 speaker folders (1,211 dev + 40 test) and 153,516 wav files.
#
#   sbatch scripts/cluster/unpack_voxceleb1.slurm                                # verify + unpack, keep zips
#   sbatch --export=ALL,DELETE_ZIPS=1 scripts/cluster/unpack_voxceleb1.slurm     # ... then delete the zips
#
# The zips are deleted only with DELETE_ZIPS=1 AND only if every check passed.

#SBATCH --job-name=vox1_unpack
#SBATCH --account=<project>
#SBATCH --partition=comp
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
DEST="${DEST:-$HOME/<project>/${USER_NS}/voxceleb1}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/<project>/${USER_NS}/retro-profile}"
ENV_NAME="${ENV_NAME:-retro}"
DELETE_ZIPS="${DELETE_ZIPS:-0}"

# Refuse to write outside our own namespace (rule R1), as setup_env.sh and download_nisp.sh do.
case "$(cd "${DEST}" && pwd -P)" in
  */${USER_NS}/*) ;;
  *) echo "REFUSING: ${DEST} resolves outside ${USER_NS}/"; exit 1 ;;
esac

cd "${DEST}"
echo "=== $(date) | vox1 unpack | $(hostname) | $(pwd -P) ==="

# name  bytes  md5 (original release)
EXPECTED="vox1_test_wav.zip 1072793438 185fdc63c3c739954633d50379a3d102
vox1_dev_wav.zip 32596624183 ae63e55b951748cc486645f532ba230b"

ok=1
while read -r name bytes md5; do
  if [ ! -f "${name}" ]; then echo "MISSING ${name}"; ok=0; continue; fi
  size=$(stat -c %s "${name}")
  if [ "${size}" != "${bytes}" ]; then echo "SIZE MISMATCH ${name}: ${size} != ${bytes}"; ok=0; continue; fi
  got=$(md5sum "${name}" | awk '{print $1}')
  if [ "${got}" = "${md5}" ]; then echo "OK  ${name}  ${got}"; else echo "MD5 MISMATCH ${name}: ${got} != ${md5}"; ok=0; fi
done <<< "${EXPECTED}"
if [ "${ok}" != "1" ]; then echo "Verification FAILED -- nothing unpacked."; exit 1; fi

# Both zips unpack into wav/ (dev and test speakers are disjoint). -n: never overwrite.
for z in vox1_test_wav.zip vox1_dev_wav.zip; do
  echo "--- unzip ${z}  ($(date +%H:%M:%S))"
  unzip -q -n "${z}" -d "${DEST}"
done

n_spk=$(find wav -mindepth 1 -maxdepth 1 -type d | wc -l)
n_wav=$(find wav -type f -name '*.wav' | wc -l)
echo "speaker folders: ${n_spk} (expect 1251) | wav files: ${n_wav} (expect 153516)"
counts_ok=0
if [ "${n_spk}" = "1251" ] && [ "${n_wav}" = "153516" ]; then counts_ok=1; fi

# Measure the audio before choosing max_audio_seconds (as was done for NISP): rate, channels,
# duration on a random sample of 2,000 files.
module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python - <<'PY'
import random, soundfile as sf, statistics as st
from pathlib import Path
files = sorted(Path("wav").rglob("*.wav"))
random.Random(0).shuffle(files)
info = [sf.info(f) for f in files[:2000]]
dur = sorted(i.duration for i in info)
q = lambda p: dur[int(p * (len(dur) - 1))]
print("sample rates:", sorted({i.samplerate for i in info}), "| channels:", sorted({i.channels for i in info}),
      "| subtypes:", sorted({i.subtype for i in info}))
print(f"duration s (n={len(dur)}): min {dur[0]:.1f} median {q(0.5):.1f} p90 {q(0.9):.1f} max {dur[-1]:.1f} "
      f"| share over 8 s: {sum(d > 8 for d in dur) / len(dur):.0%}")
PY

du -sh --apparent-size wav
if [ "${counts_ok}" = "1" ] && [ "${DELETE_ZIPS}" = "1" ]; then
  rm -f vox1_test_wav.zip vox1_dev_wav.zip && echo "zips deleted"
else
  echo "zips kept (DELETE_ZIPS=${DELETE_ZIPS}, counts_ok=${counts_ok})"
fi
echo "=== $(date) | done ==="
