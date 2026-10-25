"""Every DataModule must honour `train_unit`, not hardcode the indexing mode.

`_SpeakerDataModule.setup()` builds the training set with `mode=self.train_unit`
so a baseline can request per-utterance training (Gupta 2022 trains one clip at a
time; RETRO trains per-speaker bags). Validation and test always stay in bag mode
so evaluation is one protocol.

`NispDataModule` overrides `setup()` purely to forward the `languages` filter, and
the override was copied from the base with `mode="bag"` frozen in. That silently
ignored `data.train_unit: utterance`: the dataset yielded `SpeakerBag` objects
while the DataLoader picked its collate function from `train_unit`, so bags
reached a collate expecting single utterances --

    AttributeError: 'SpeakerBag' object has no attribute 'waveform'

-- which killed both Gupta-on-NISP runs on 2026-09-13. The failure surfaced in a
DataLoader worker, several frames from the cause.

These tests pin the contract for every DataModule subclass, so a future override
cannot reintroduce it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from retro_profile.data.registry import NispDataModule, TimitDataModule

_ROWS = [
    {"speaker_id": "Hin_0001", "utt_id": "Hin_0001_Eng_f_0000",
     "audio_path": "/nonexistent/a.wav", "age": 24.2, "height": 163.0,
     "gender": "F", "language": "Hindi", "session_lang": "Eng",
     "item_id": "item0000", "sentence_type": "COMMON"},
    {"speaker_id": "Hin_0002", "utt_id": "Hin_0002_Eng_f_0000",
     "audio_path": "/nonexistent/b.wav", "age": 26.1, "height": 154.5,
     "gender": "F", "language": "Hindi", "session_lang": "Eng",
     "item_id": "item0000", "sentence_type": "COMMON"},
]


@pytest.fixture
def nisp_root(tmp_path):
    m = tmp_path / "manifests"
    m.mkdir()
    for split in ("train", "dev", "test"):
        pd.DataFrame(_ROWS).to_csv(m / f"{split}.csv", index=False)
    return tmp_path


def _dm(cls, root, **kw):
    return cls(root=str(root), sample_rate=16000, max_audio_seconds=8.0,
               bag_size=8, batch_size=2, num_workers=0,
               targets=["age", "height"], **kw)


@pytest.mark.parametrize("unit", ["bag", "utterance"])
def test_nisp_honours_train_unit(nisp_root, unit):
    """The regression: NispDataModule used to force "bag" whatever was asked."""
    dm = _dm(NispDataModule, nisp_root, train_unit=unit)
    dm.setup("fit")
    assert dm.train_set.mode == unit


def test_nisp_eval_sets_stay_in_bag_mode(nisp_root):
    """Evaluation is always per-speaker, so val/test ignore train_unit."""
    dm = _dm(NispDataModule, nisp_root, train_unit="utterance")
    dm.setup("fit")
    dm.setup("test")
    assert dm.val_set.mode == "bag"
    assert dm.test_set.mode == "bag"


def test_nisp_matches_the_base_class(nisp_root):
    """The override exists only to forward `languages` -- nothing else may differ."""
    a = _dm(NispDataModule, nisp_root, train_unit="utterance")
    b = _dm(TimitDataModule, nisp_root, train_unit="utterance")
    a.setup("fit")
    b.setup("fit")
    assert a.train_set.mode == b.train_set.mode
    assert a.val_set.mode == b.val_set.mode


def test_language_filter_still_works(nisp_root):
    """The one thing the override is FOR. Case-insensitive by design."""
    dm = _dm(NispDataModule, nisp_root, train_unit="bag", languages=["hindi"])
    dm.setup("fit")
    assert dm.train_set.mode == "bag"
    assert len(dm.train_set.manifest) == 2      # both rows are Hindi
