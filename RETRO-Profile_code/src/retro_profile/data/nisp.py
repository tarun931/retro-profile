"""NISP multilingual speaker-profile corpus (5 Indian languages).

Manifest produced by ``scripts/prepare_nisp.py`` at
``<root>/manifests/<split>.csv`` with columns:

    speaker_id, utt_id, audio_path, age, height, weight, shoulder_size,
    gender, language
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from retro_profile.data.base import BaseSpeakerDataset


class NispDataset(BaseSpeakerDataset):
    """NISP loader; optionally filters by ``languages``."""

    SPLIT_FILE = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}

    def __init__(self, *args, languages: Sequence[str] | None = None, **kwargs) -> None:
        self._languages: list[str] | None = list(languages) if languages else None
        super().__init__(*args, **kwargs)

    def _load_manifest(self) -> pd.DataFrame:
        # A split may pool several official ones, e.g. "train+test": training on
        # NISP and TESTING ON ANOTHER CORPUS (TIMIT), where NISP's own test
        # speakers are free to train on. The parts must be speaker-disjoint.
        parts = [p.strip() for p in str(self.split).split("+")]
        if any(p not in self.SPLIT_FILE for p in parts) or len(set(parts)) != len(parts):
            raise ValueError(
                f"NISP split must be one of {list(self.SPLIT_FILE)} or several joined "
                f"with '+' (e.g. 'train+test'); got {self.split!r}"
            )
        frames = []
        for part in parts:
            path = self.root / "manifests" / self.SPLIT_FILE[part]
            if not path.exists():
                raise FileNotFoundError(
                    f"NISP manifest not found at {path}. Run scripts/prepare_nisp.py first."
                )
            frames.append(pd.read_csv(path))
        seen: set[str] = set()
        for part, f in zip(parts, frames, strict=True):
            spk = set(f["speaker_id"].astype(str))
            if spk & seen:
                raise ValueError(f"NISP split {part!r} shares speakers with another part of "
                                 f"{self.split!r}: {sorted(spk & seen)[:5]}")
            seen |= spk
        df = pd.concat(frames, ignore_index=True)
        if self._languages is not None and "language" in df.columns:
            # Case-insensitive on purpose. prepare_nisp.py writes the language
            # capitalised ("Hindi") because that is how the release names its
            # directories, while configs/data/nisp.yaml has historically listed
            # them lowercase. A case-sensitive `isin` matches nothing and yields
            # an EMPTY dataset silently, which surfaces much later as a confusing
            # error about zero speakers.
            wanted = {str(x).strip().lower() for x in self._languages}
            available = sorted(set(df["language"].astype(str)))
            before = len(df)
            df = df[df["language"].astype(str).str.lower().isin(wanted)]
            df = df.reset_index(drop=True)
            if before and df.empty:
                raise ValueError(
                    f"language filter {sorted(wanted)} matched none of {available} "
                    f"in split {self.split!r}. Check `languages:` in configs/data/nisp.yaml."
                )
        df["audio_path"] = df["audio_path"].apply(
            lambda p: str(self.root / p) if not Path(p).is_absolute() else p
        )
        return df
