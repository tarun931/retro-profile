"""Phase 4 data-plumbing tests: gender-as-target mapping + train_unit switch.

TIMIT-independent: uses an in-memory fake manifest, so no audio/corpus needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from retro_profile.data.base import BaseSpeakerDataset
from retro_profile.data.registry import TimitDataModule


class _FakeDS(BaseSpeakerDataset):
    """Concrete dataset backed by an in-memory DataFrame (no audio loading)."""

    def __init__(self, df, targets, mode="utterance"):
        self._df = df
        super().__init__(root=".", split="train", targets=targets, mode=mode)

    def _load_manifest(self):
        return self._df


def _fake_df():
    return pd.DataFrame(
        [
            {"speaker_id": "s0", "utt_id": "u0", "audio_path": "/a.wav", "age": 30.0, "height": 170.0, "gender": "M"},
            {"speaker_id": "s0", "utt_id": "u1", "audio_path": "/b.wav", "age": 30.0, "height": 170.0, "gender": "M"},
            {"speaker_id": "s1", "utt_id": "u0", "audio_path": "/c.wav", "age": 40.0, "height": 180.0, "gender": "F"},
        ]
    )


# ---------- gender M/F -> 0/1 --------------------------------------------


def test_gender_string_mapped_to_0_1():
    ds = _FakeDS(_fake_df(), targets=("age", "height", "gender"))
    m_row = ds._row_labels(ds.manifest.iloc[0])   # M
    f_row = ds._row_labels(ds.manifest.iloc[2])   # F
    assert m_row["gender"] == 0.0
    assert f_row["gender"] == 1.0
    assert m_row["age"] == 30.0 and m_row["height"] == 170.0   # numeric targets unaffected


def test_numeric_gender_still_works():
    df = _fake_df()
    df["gender"] = [0.0, 0.0, 1.0]
    ds = _FakeDS(df, targets=("age", "height", "gender"))
    assert ds._row_labels(ds.manifest.iloc[2])["gender"] == 1.0


def test_get_label_stats_skips_gender():
    ds = _FakeDS(_fake_df(), targets=("age", "height", "gender"))
    stats = ds.get_label_stats()
    assert "age" in stats and "height" in stats
    assert "gender" not in stats                   # categorical -> never z-scored


# ---------- train_unit switch -------------------------------------------


def test_mode_toggles_indexing_unit():
    df = _fake_df()                                # 2 speakers, 3 utterances
    assert len(_FakeDS(df, ("age", "height"), mode="bag")) == 2          # per-speaker
    assert len(_FakeDS(df, ("age", "height"), mode="utterance")) == 3    # per-utterance


def test_datamodule_stores_and_validates_train_unit():
    dm = TimitDataModule(root="data/timit", train_unit="utterance")
    assert dm.train_unit == "utterance"
    with pytest.raises(ValueError):
        TimitDataModule(root="data/timit", train_unit="banana")
