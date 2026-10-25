"""Prepare TIMIT for retro-profile training.

Reads a raw TIMIT tree plus ``SPKRINFO.TXT`` and emits manifests at:

    <out>/manifests/train.csv
    <out>/manifests/dev.csv
    <out>/manifests/test.csv

Each row has columns:
    speaker_id, utt_id, audio_path, age, height, gender, dialect_region

Layout handling
---------------
Supports both the classic LDC layout (``TRAIN/<DR>/<spk>/<utt>.WAV`` with
``SPKRINFO.TXT`` at the root) and the common "Kaggle" mirror, where:

  * ``TRAIN/`` and ``TEST/`` live under a ``data/`` subdirectory, and
  * each utterance ships **twice** — ``<utt>.WAV`` (NIST SPHERE) *and* a
    pre-converted ``<utt>.WAV.wav`` (vanilla RIFF).

We prefer the RIFF ``.wav`` when present (no SPHERE/libsndfile dependency) and
ignore the SPHERE duplicate, so each utterance appears exactly once.

Label join
----------
TIMIT speaker *folders* carry a sex prefix (``FCJF0``) while ``SPKRINFO.TXT``
keys on the un-prefixed id (``CJF0``). We strip the leading sex char before
joining. Heights in SPKRINFO are feet/inches → converted to cm; age is
"RECDATE" - "BIRTHDATE".

Dev split
---------
Speaker-disjoint, **gender-stratified 85/15** carved from TRAIN (males and
females split separately), matching the bi-encoder baseline preprocessing.
TEST is the standard 168-speaker TIMIT test set.
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

INCH_TO_CM = 2.54


def parse_spkrinfo(path: Path) -> pd.DataFrame:
    """Parse SPKRINFO.TXT into a per-speaker DataFrame keyed by un-prefixed id."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = re.split(r"\s+", line.strip())
            # ID  Sex DR  Use Recdate    Birthdate  Ht    Race Edu  Comments
            if len(parts) < 7:
                continue
            spk_key, sex, dr, use, recdate, birthdate, ht_raw = parts[:7]
            try:
                feet, inches = ht_raw.replace('"', "").split("'")
                height_cm = (int(feet) * 12 + int(inches)) * INCH_TO_CM
            except Exception:
                height_cm = float("nan")
            try:
                age = (
                    datetime.strptime(recdate, "%m/%d/%y")
                    - datetime.strptime(birthdate, "%m/%d/%y")
                ).days / 365.25
                if age < 0:  # 2-digit-year rollover (e.g. birthdate parsed as 2050+)
                    age += 100
            except Exception:
                age = float("nan")
            rows.append({
                "spk_key": spk_key,
                "gender": sex,
                "dr_num": dr,
                "use": use,
                "age": age,
                "height": height_cm,
            })
    return pd.DataFrame(rows)


def resolve_layout(raw: Path) -> tuple[Path, Path]:
    """Locate the audio base (dir containing TRAIN/) and SPKRINFO.TXT."""
    audio_base: Path | None = None
    for cand in (raw, raw / "data"):
        if (cand / "TRAIN").is_dir() or (cand / "train").is_dir():
            audio_base = cand
            break
    if audio_base is None:
        raise FileNotFoundError(
            f"Could not find a TRAIN/ directory under {raw} or {raw / 'data'}."
        )
    for cand in (raw / "SPKRINFO.TXT", raw / "data" / "SPKRINFO.TXT",
                 audio_base / "SPKRINFO.TXT"):
        if cand.exists():
            return audio_base, cand
    hits = list(raw.rglob("SPKRINFO.TXT"))
    if not hits:
        raise FileNotFoundError(f"SPKRINFO.TXT not found anywhere under {raw}.")
    return audio_base, hits[0]


def _strip_sex_prefix(folder: str) -> str:
    """``FCJF0`` -> ``CJF0`` so the folder name joins to SPKRINFO."""
    return folder[1:] if folder[:1] in ("M", "F") else folder


def _utt_id(fname: str) -> str:
    """Strip the audio extension(s): ``SA1.WAV.wav`` / ``SA1.WAV`` -> ``SA1``."""
    name = fname
    if name.endswith(".wav"):
        name = name[:-4]
    if name.endswith(".WAV"):
        name = name[:-4]
    return name


def discover_utterances(audio_base: Path) -> pd.DataFrame:
    """Walk TRAIN/ and TEST/, one row per utterance (RIFF .wav preferred)."""
    rows: list[dict] = []
    for split in ("TRAIN", "TEST"):
        base = audio_base / split
        if not base.is_dir():
            base = audio_base / split.lower()
        if not base.is_dir():
            continue
        wavs = sorted(base.rglob("*.wav"))           # converted RIFF (preferred)
        if not wavs:
            wavs = sorted(base.rglob("*.WAV"))       # classic SPHERE fallback
            warnings.warn(
                f"No RIFF .wav files under {base}; falling back to SPHERE .WAV — "
                "audio reading requires libsndfile with SPHERE support.",
                stacklevel=2,
            )
        for wav in wavs:
            parts = wav.relative_to(audio_base).parts
            if len(parts) < 4:                       # split/DR/speaker/file
                continue
            _split, dr, spk, fname = parts[0], parts[1], parts[2], parts[3]
            rows.append({
                "split_raw": split,
                "dialect_region": dr,
                "speaker_id": spk,
                "spk_key": _strip_sex_prefix(spk),
                "utt_id": _utt_id(fname),
                "audio_path": str(wav.resolve()),
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["speaker_id", "utt_id"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, type=Path,
                   help="TIMIT corpus root (containing TRAIN/TEST or data/TRAIN, plus SPKRINFO.TXT)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dev-frac", type=float, default=0.15,
                   help="Fraction of TRAIN speakers (per gender) reserved as dev.")
    p.add_argument("--seed", type=int, default=1,
                   help="random_state for the male split; females use seed+1 (matches the baseline).")
    return p.parse_args()


def _gender_stratified_dev(train_df: pd.DataFrame, dev_frac: float, seed: int) -> set[str]:
    """Return the set of TRAIN speaker_ids reserved as a gender-stratified dev split."""
    spk_gender = train_df.drop_duplicates("speaker_id").set_index("speaker_id")["gender"]
    dev: set[str] = set()
    for sex, rs in (("M", seed), ("F", seed + 1)):
        ids = sorted(spk_gender[spk_gender == sex].index.tolist())
        if len(ids) < 2:
            continue
        _, val = train_test_split(ids, test_size=dev_frac, random_state=rs)
        dev.update(val)
    return dev


def main() -> None:
    args = parse_args()
    manifest_dir = args.out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    audio_base, spkrinfo_path = resolve_layout(args.raw)
    print(f"[prepare_timit] audio base: {audio_base}")
    print(f"[prepare_timit] SPKRINFO:   {spkrinfo_path}")

    spk_info = parse_spkrinfo(spkrinfo_path)
    print(f"[prepare_timit] Parsed metadata for {len(spk_info)} speakers.")
    utts = discover_utterances(audio_base)
    print(f"[prepare_timit] Found {len(utts)} utterances "
          f"({utts['speaker_id'].nunique()} speakers).")

    df = utts.merge(spk_info[["spk_key", "age", "height", "gender"]],
                    on="spk_key", how="left")

    labelled = int(df["height"].notna().sum())
    if labelled == 0:
        raise RuntimeError(
            "Label join produced 0 rows with a height — speaker ids did not match "
            "SPKRINFO. Check the sex-prefix stripping / SPKRINFO path."
        )
    print(f"[prepare_timit] Label join: {labelled}/{len(df)} rows have height, "
          f"{int(df['age'].notna().sum())}/{len(df)} have age.")

    train = df[df["split_raw"] == "TRAIN"].reset_index(drop=True)
    test = df[df["split_raw"] == "TEST"].reset_index(drop=True)

    dev_ids = _gender_stratified_dev(train, args.dev_frac, args.seed)
    dev = train[train["speaker_id"].isin(dev_ids)].reset_index(drop=True)
    train = train[~train["speaker_id"].isin(dev_ids)].reset_index(drop=True)

    cols = ["speaker_id", "utt_id", "audio_path", "age", "height", "gender", "dialect_region"]
    train[cols].to_csv(manifest_dir / "train.csv", index=False)
    dev[cols].to_csv(manifest_dir / "dev.csv", index=False)
    test[cols].to_csv(manifest_dir / "test.csv", index=False)

    print(f"[prepare_timit] Wrote {manifest_dir}/{{train,dev,test}}.csv")
    print(f"  train: {train['speaker_id'].nunique()} speakers, {len(train)} utts")
    print(f"  dev:   {dev['speaker_id'].nunique()} speakers, {len(dev)} utts")
    print(f"  test:  {test['speaker_id'].nunique()} speakers, {len(test)} utts")


if __name__ == "__main__":
    main()
