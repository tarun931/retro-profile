"""Table 4b: an age-balanced sample of HeightCeleb split2 TRAINING speakers to add to TIMIT + NISP (2026-10-17).

TIMIT and NISP hold almost no speakers over 50, while HeightCeleb's test speakers run from 13 to 87. 4b adds a small,
age-BALANCED set of HeightCeleb training speakers -- the same number from every 10-year band (<20, 20s, ..., 70+), half
men and half women where the band allows -- so the missing ages are covered without tilting predictions towards any
age. Only speakers with an age are eligible; only the split2 TRAIN manifest is read, and the script stops if any chosen
speaker, YouTube video or audio file is also in a sibling dev.csv / test.csv (those calibrate and test Table 4).

Per band: ``min(--per-band, smallest band)`` speakers, so every band is exactly equal; within a band, half from each
gender (the odd one to the gender with more speakers); a gender short of its half gives all it has and the other
fills the rest. Draws use ``numpy.random.default_rng(--seed)`` over speaker IDs in sorted order, so the same input and
seed always give the same sample.

Writes the chosen speakers' rows, unchanged (absolute audio paths), to ``<out-root>/manifests/train.csv`` and one row
per speaker to ``--list`` (kept in the repo, so the sample is documented even without the audio).

    python scripts/sample_heightceleb_ages.py --manifest data/heightceleb_split2/manifests/train.csv \\
        --out-root data/heightceleb_split2_age_sample --list heightceleb/table4b_age_sample.csv --per-band 15 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EDGES = [20, 30, 40, 50, 60, 70]


def band_of(age: float, edges=EDGES) -> str:
    """'<20', '20-29', ..., '70+'."""
    if age < edges[0]:
        return f"<{edges[0]}"
    for lo, hi in zip(edges, edges[1:], strict=False):
        if age < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def speakers_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per speaker WITH an age: speaker_id, gender, age, height, n_clips, band (age must be constant)."""
    m = manifest[manifest["age"].notna()]
    g = m.groupby("speaker_id")
    if (g["age"].nunique() > 1).any():
        raise SystemExit(f"speakers with more than one age: {list(g['age'].nunique()[lambda s: s > 1].index[:5])}")
    spk = g.agg(gender=("gender", "first"), age=("age", "first"), height=("height", "first"),
                n_clips=("utt_id", "size")).reset_index()
    spk["speaker_id"] = spk["speaker_id"].astype(str)
    spk["gender"] = spk["gender"].astype(str).str.upper().str[0]
    spk["band"] = spk["age"].map(band_of)
    return spk.sort_values("speaker_id").reset_index(drop=True)


def sample(spk: pd.DataFrame, per_band: int, seed: int) -> tuple[pd.DataFrame, int]:
    """The balanced sample and the per-band count actually used."""
    order = [band_of(e - 1) for e in EDGES] + [band_of(EDGES[-1])]
    counts = spk["band"].value_counts()
    missing = [b for b in order if counts.get(b, 0) == 0]
    if missing:
        raise SystemExit(f"no aged training speaker in band(s) {missing}")
    n = min(per_band, int(min(counts[b] for b in order)))
    rng = np.random.default_rng(seed)
    picked = []
    for b in order:
        pool = {g: sorted(spk.loc[(spk["band"] == b) & (spk["gender"] == g), "speaker_id"]) for g in ("F", "M")}
        more = max(pool, key=lambda g: (len(pool[g]), g))
        want = {g: n // 2 + (n % 2 if g == more else 0) for g in pool}
        for g in pool:                                         # a short gender gives all it has, the other fills
            other = "M" if g == "F" else "F"
            if len(pool[g]) < want[g]:
                want[other] += want[g] - len(pool[g])
                want[g] = len(pool[g])
        for g in ("F", "M"):
            picked += list(rng.choice(pool[g], size=want[g], replace=False)) if want[g] else []
    out = spk[spk["speaker_id"].isin(picked)].sort_values(["band", "gender", "speaker_id"]).reset_index(drop=True)
    return out, n


def check_disjoint(rows: pd.DataFrame, manifests_dir: Path) -> None:
    """Stop if a sampled speaker, YouTube video or audio file also appears in the sibling dev / test manifests."""
    for name in ("dev.csv", "test.csv"):
        path = manifests_dir / name
        if not path.exists():
            continue
        other = pd.read_csv(path, dtype={"speaker_id": str})
        for col in ("speaker_id", "video_id", "audio_path"):
            clash = set(rows[col].astype(str)) & set(other[col].astype(str))
            if clash:
                raise SystemExit(f"{len(clash)} sampled {col} values are also in {path}, e.g. {sorted(clash)[:3]}. Stop.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True, help="split2 TRAIN manifest")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True, help="speaker list CSV (kept in the repo)")
    ap.add_argument("--per-band", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.manifest.name != "train.csv":
        raise SystemExit(f"--manifest must be a TRAIN manifest (train.csv), got {args.manifest}")
    manifest = pd.read_csv(args.manifest, dtype={"speaker_id": str})
    spk = speakers_table(manifest)
    chosen, n = sample(spk, args.per_band, args.seed)
    rows = manifest[manifest["speaker_id"].isin(set(chosen["speaker_id"]))]
    check_disjoint(rows, args.manifest.parent)

    (args.out_root / "manifests").mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_root / "manifests" / "train.csv", index=False)
    args.list.parent.mkdir(parents=True, exist_ok=True)
    chosen.assign(seed=args.seed, per_band=n).to_csv(args.list, index=False)

    print(f"aged training speakers available per band:\n{pd.crosstab(spk['band'], spk['gender'], margins=True)}")
    print(f"\nsample: {n} per band, seed {args.seed} -> {len(chosen)} speakers, {len(rows)} clips")
    print(pd.crosstab(chosen["band"], chosen["gender"], margins=True))
    print(f"wrote {args.out_root / 'manifests' / 'train.csv'} and {args.list}")


if __name__ == "__main__":
    main()
