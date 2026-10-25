"""HeightCeleb — VoxCeleb1's 1,251 speakers with crowd-sourced heights, plus our ages.

Manifest produced by ``scripts/prepare_heightceleb.py`` at
``<root>/manifests/<split>.csv`` with columns:

    speaker_id, utt_id, audio_path, video_id, age, height, gender

Notes:
    * Height for every speaker; age for 573 (``heightceleb/vox1_height_age.csv``), NaN for the
      rest — the masked loss skips it.
    * A split may pool official splits, e.g. "train+test" when TIMIT is the test set.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from retro_profile.data.base import BaseSpeakerDataset, read_split_manifests


class HeightCelebDataset(BaseSpeakerDataset):
    """HeightCeleb loader: height for all speakers, age where known."""

    def _load_manifest(self) -> pd.DataFrame:
        df = read_split_manifests(self.root, self.split, "HeightCeleb", "scripts/prepare_heightceleb.py")
        df["audio_path"] = df["audio_path"].apply(
            lambda p: str(self.root / p) if not Path(p).is_absolute() else p
        )
        return df
