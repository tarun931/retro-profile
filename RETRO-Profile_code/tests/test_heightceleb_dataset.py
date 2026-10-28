"""HeightCeleb loader and config: age where known (NaN otherwise), the pooled 'train+test' split
(TIMIT is the test set), and the data config's targets and augmentation."""

from __future__ import annotations


import pandas as pd
import pytest

from retro_profile.data.heightceleb import HeightCelebDataset


def _manifests(root, splits):
    (root / "manifests").mkdir(parents=True)
    for split, speakers in splits.items():
        rows = [{"speaker_id": s, "utt_id": f"vidAAAAAAAA_{i:05d}", "audio_path": f"wav/{s}/{i}.wav",
                 "video_id": "vidAAAAAAAA", "age": 40.0 if s.endswith("1") else float("nan"),
                 "height": 175.0, "gender": "M"} for s in speakers for i in range(3)]
        pd.DataFrame(rows).to_csv(root / "manifests" / f"{split}.csv", index=False)


@pytest.fixture
def root(tmp_path):
    _manifests(tmp_path, {"train": ["id10001", "id10002"], "dev": ["id10003"], "test": ["id10041"]})
    return tmp_path


def test_train_plus_test_pools_both_splits(root):
    ds = HeightCelebDataset(root=root, split="train+test")
    assert ds.speakers == ["id10001", "id10002", "id10041"] and len(ds.manifest) == 9
    assert ds.manifest.audio_path.iloc[0] == str(root / "wav/id10001/0.wav")


def test_age_where_known_nan_otherwise(root):
    ds = HeightCelebDataset(root=root, split="train", targets=["age", "height"])
    rows = ds.manifest.drop_duplicates("speaker_id").set_index("speaker_id")
    assert ds._row_labels(rows.loc["id10001"]) == {"age": 40.0, "height": 175.0}
    assert ds._row_labels(rows.loc["id10002"]) == {"height": 175.0}
    assert ds.get_label_stats()["age"]["count"] == 1


@pytest.mark.parametrize("split", ["train+val", "train+train", "all", ""])
def test_unknown_or_repeated_parts_are_refused(root, split):
    with pytest.raises(ValueError, match="HeightCeleb split must be"):
        HeightCelebDataset(root=root, split=split)


def test_parts_that_share_a_speaker_are_refused(tmp_path):
    _manifests(tmp_path, {"train": ["id10001"], "dev": [], "test": ["id10001"]})
    with pytest.raises(ValueError, match="shares speakers"):
        HeightCelebDataset(root=tmp_path, split="train+test")


def test_missing_manifest_names_the_prepare_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_heightceleb.py"):
        HeightCelebDataset(root=tmp_path, split="train")


