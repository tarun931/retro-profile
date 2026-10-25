"""Training on all of NISP: the pooled 'train+test' split and its experiment config.

When the test corpus is TIMIT, NISP's own test speakers are free to train on. The
loader pools official splits only if they are speaker-disjoint, and the experiment
must differ from the base setting in its training data alone.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from retro_profile.data.nisp import NispDataset


def _manifests(root, splits):
    (root / "manifests").mkdir(parents=True)
    for split, speakers in splits.items():
        rows = [{"speaker_id": s, "utt_id": f"{s}_{i}", "audio_path": f"{s}_{i}.wav",
                 "age": 25.0, "height": 165.0, "gender": "M",
                 "language": "Hindi" if i % 2 else "English"}
                for s in speakers for i in range(3)]
        pd.DataFrame(rows).to_csv(root / "manifests" / f"{split}.csv", index=False)


@pytest.fixture
def nisp_root(tmp_path):
    _manifests(tmp_path, {"train": ["Hin_0001", "Hin_0002"], "dev": ["Hin_0003"],
                          "test": ["Tam_0001", "Tam_0002", "Tam_0003"]})
    return tmp_path


def test_train_plus_test_pools_both_official_splits(nisp_root):
    ds = NispDataset(root=nisp_root, split="train+test")
    assert ds.speakers == ["Hin_0001", "Hin_0002", "Tam_0001", "Tam_0002", "Tam_0003"]
    assert len(ds.manifest) == 15 and len(ds) == 5


def test_single_splits_are_unchanged(nisp_root):
    assert NispDataset(root=nisp_root, split="train").speakers == ["Hin_0001", "Hin_0002"]
    assert NispDataset(root=nisp_root, split="dev").speakers == ["Hin_0003"]


@pytest.mark.parametrize("split", ["train+val", "train+train", "all", ""])
def test_unknown_or_repeated_parts_are_refused(nisp_root, split):
    with pytest.raises(ValueError, match="NISP split must be"):
        NispDataset(root=nisp_root, split=split)


def test_parts_that_share_a_speaker_are_refused(tmp_path):
    _manifests(tmp_path, {"train": ["Hin_0001"], "dev": [], "test": ["Hin_0001"]})
    with pytest.raises(ValueError, match="shares speakers"):
        NispDataset(root=tmp_path, split="train+test")


def test_language_filter_applies_to_the_pooled_split(nisp_root):
    ds = NispDataset(root=nisp_root, split="train+test", languages=["english"])
    assert set(ds.manifest["language"]) == {"English"} and len(ds.speakers) == 5
    with pytest.raises(ValueError, match="matched none"):
        NispDataset(root=nisp_root, split="train+test", languages=["klingon"])


def test_nisp_full_experiment_changes_only_the_training_data():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flat(v, key) if isinstance(v, dict) else {key: v})
        return out

    def cfg(experiment):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            c = compose(config_name="default", overrides=[f"experiment={experiment}"])
        return flat(OmegaConf.to_container(c, resolve=True))

    base, new = cfg("retro_ordcon_fixed_l100_s5"), cfg("nisp_full_l100_s5")
    changed = {k for k in set(base) | set(new) if base.get(k) != new.get(k)}
    assert changed == {"data._target_", "data.name", "data.root", "data.splits.train",
                       "data.splits.test", "run_name", "output_dir"}
    assert new["data.splits.train"] == "train+test" and new["data.splits.val"] == "dev"

    both = cfg("nisp_full_both_l100_s5")
    changed = {k for k in set(new) | set(both) if new.get(k) != both.get(k)}
    assert changed == {"data.root", "run_name", "output_dir"}
    assert both["data.root"] == "data/nisp_both"
