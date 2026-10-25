#!/bin/bash
# VoxCeleb1 leakage check, step c: identical audio files under different speaker IDs.  🔵 SUBMIT ON the GPU cluster.
#
# Steps a/b (same person under two IDs by Wikidata / name; same YouTube video under two IDs) ran on
# the Mac from heightceleb/ metadata -- SESSION_HANDOFF §0.7.14. This job fingerprints every clip
# (MD5 of the file bytes) and reports any fingerprint shared by two or more speaker IDs: the same
# recording filed under two people, which would be real train/test leakage. Read-only on the audio.
#
#   sbatch scripts/cluster/vox1_duplicates.slurm
#
# Writes <data-root>/voxceleb1/clip_md5.csv (one row per clip) and prints a summary in the log.

#SBATCH --job-name=vox1_dups
#SBATCH --account=<project>
#SBATCH --partition=comp
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

USER_NS="${USER_NS:-<user>}"
WAV="${WAV:-$HOME/<project>/${USER_NS}/voxceleb1/wav}"
OUT="${OUT:-$HOME/<project>/${USER_NS}/voxceleb1/clip_md5.csv}"
ENV_NAME="${ENV_NAME:-retro}"

case "$(cd "$(dirname "${OUT}")" && pwd -P)" in
  */${USER_NS}/*|*/${USER_NS}) ;;
  *) echo "REFUSING: ${OUT} resolves outside ${USER_NS}/"; exit 1 ;;
esac

module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== $(date) | vox1 duplicate audio | $(hostname) | ${WAV} ==="
python - "${WAV}" "${OUT}" "${SLURM_CPUS_PER_TASK:-8}" <<'PY'
import hashlib, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

wav, out, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

files = sorted(wav.glob("*/*/*.wav"))
print(f"clips: {len(files):,} (expect 153,516)", flush=True)
# Threads, not processes: hashlib and file reads release the GIL, and threads need no fork/spawn.
with ThreadPoolExecutor(max_workers=workers) as pool:
    hashes = list(pool.map(md5, files))
df = pd.DataFrame({"speaker_id": [f.parts[-3] for f in files], "video_id": [f.parts[-2] for f in files],
                   "clip": [f.stem for f in files], "md5": hashes})
df.to_csv(out, index=False)
print(f"wrote {out}")

per = df.groupby("md5").agg(n=("clip", "size"), speakers=("speaker_id", lambda s: sorted(set(s))),
                            videos=("video_id", lambda s: sorted(set(s))))
dup = per[per.n > 1]
cross = dup[dup.speakers.map(len) > 1]
print(f"fingerprints shared by 2+ clips: {len(dup):,} ({int(dup.n.sum()):,} clips)")
print(f"  ... of which under 2+ DIFFERENT speaker IDs (leakage): {len(cross):,}")
print(f"  ... within one speaker only (duplicate clips): {len(dup) - len(cross):,}")
if len(cross):
    pairs = cross.speakers.map(tuple).value_counts()
    print("speaker groups sharing identical clips (count of shared fingerprints):")
    print(pairs.head(30).to_string())
PY
echo "=== $(date) | done ==="
