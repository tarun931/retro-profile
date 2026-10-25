"""TIMIT acoustic-phonetic continuous-speech corpus.

We assume ``scripts/prepare_timit.py`` has already produced a manifest at
``<root>/manifests/<split>.csv`` with columns:

    speaker_id, utt_id, audio_path, age, height, gender, dialect_region

TIMIT speaker ages are computed from recording date - birth date and may be
non-integer. Heights are in cm.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retro_profile.data.base import BaseSpeakerDataset


class TimitDataset(BaseSpeakerDataset):
    """TIMIT loader. Provides age + height labels for 630 speakers."""

    SPLIT_FILE = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}

    def _load_manifest(self) -> pd.DataFrame:
        if self.split not in self.SPLIT_FILE:
            raise ValueError(
                f"TIMIT split must be one of {list(self.SPLIT_FILE)}; got {self.split!r}"
            )
        path = self.root / "manifests" / self.SPLIT_FILE[self.split]
        if not path.exists():
            raise FileNotFoundError(
                f"TIMIT manifest not found at {path}. Run scripts/prepare_timit.py first."
            )
        df = pd.read_csv(path)
        # Resolve relative audio paths against ``root`` so manifests stay portable.
        df["audio_path"] = df["audio_path"].apply(
            lambda p: str(self.root / p) if not Path(p).is_absolute() else p
        )
        return df
