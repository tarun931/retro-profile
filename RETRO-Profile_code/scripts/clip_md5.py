"""MD5 fingerprint of every VoxCeleb1 clip, for the HeightCeleb split (paper, Appendix A.3).

``prepare_heightceleb.py --split-mode grouped --md5-csv <this file's output>`` keeps speakers who share an identical
clip in the same split. On VoxCeleb1 (153,516 clips) no clip is shared by two different speaker identities, so the
fingerprints confirm the split rather than change it.

    python scripts/clip_md5.py --wav-root /path/to/voxceleb1/wav --out data/vox1_clip_md5.csv

Expects the VoxCeleb1 layout <wav-root>/<speaker id>/<YouTube video id>/<nnnnn>.wav.
"""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wav-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(args.wav_root.glob("*/*/*.wav"))
    print(f"clips: {len(files):,} (VoxCeleb1 has 153,516)")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = list(pool.map(md5, files))
    df = pd.DataFrame({"speaker_id": [f.parts[-3] for f in files], "video_id": [f.parts[-2] for f in files],
                       "clip": [f.stem for f in files], "md5": hashes})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    shared = df.groupby("md5")["speaker_id"].nunique()
    print(f"wrote {args.out}; fingerprints shared by two or more speakers: {int((shared > 1).sum())}")


if __name__ == "__main__":
    main()
