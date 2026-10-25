"""Fixed scoring clips (2026-10-17, Table 4): validation / test bags and 8 s windows identical on every pass.

On HeightCeleb a speaker can have ~100 clips and 35 % of clips run past 8 s, so a bag of 10 was a fresh random draw
and a long clip a fresh random crop each time: two models -- or evaluate.py and fit_residual.py's reproduction check --
saw different audio. ``eval_fixed_clips`` fixes both for validation, test and per-clip scoring; training keeps its
random draw, and with the flag off (every config except heightceleb_split2) nothing changes.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

from retro_profile.data import HeightCelebDataModule, MultiCorpusDataModule
from retro_profile.data.base import _stable_seed
from retro_profile.data.heightceleb import HeightCelebDataset

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SR = 16_000


def _wav(path: Path, seconds: float, seed: int) -> None:
    sf.write(str(path), np.random.default_rng(seed).standard_normal(int(seconds * SR)).astype("float32") * 0.1, SR)


@pytest.fixture
def root(tmp_path):
    """dev: 'many' (25 short clips), 'few' (3 short clips), 'long' (one 12 s clip); train mirrors dev."""
    (tmp_path / "manifests").mkdir()
    rows, k = [], 0
    for spk, n, secs in (("many", 25, 1.0), ("few", 3, 1.0), ("long", 1, 12.0)):
        for i in range(n):
            k += 1
            _wav(tmp_path / f"{spk}_{i}.wav", secs, k)
            rows.append({"speaker_id": spk, "utt_id": f"{spk}_{i}", "audio_path": f"{spk}_{i}.wav", "video_id": "v",
                         "age": 40.0, "height": 175.0, "gender": "M"})
    for split in ("train", "dev", "test"):
        pd.DataFrame(rows).to_csv(tmp_path / "manifests" / f"{split}.csv", index=False)
    return tmp_path


def _ds(root, fixed, mode="bag"):
    return HeightCelebDataset(root=root, split="dev", mode=mode, bag_size=10, max_audio_seconds=8.0, fixed_clips=fixed)


def _bag(ds, spk):
    return ds[ds.speakers.index(spk)]


def test_fixed_bags_are_the_same_on_every_pass_and_in_a_new_dataset(root):
    a, b = _ds(root, True), _ds(root, True)
    first = _bag(a, "many").utt_ids
    assert len(first) == 10 and _bag(a, "many").utt_ids == first and _bag(b, "many").utt_ids == first


def test_without_the_flag_bags_are_redrawn(root):
    ds = _ds(root, False)
    draws = {tuple(_bag(ds, "many").utt_ids) for _ in range(5)}
    assert len(draws) > 1                                  # 25 choose 10: equal draws ~ never


def test_small_bags_and_short_clips_are_untouched_by_the_flag(root):
    """TIMIT (10 clips of < 8 s per speaker, bag of 10) scores exactly as before."""
    on, off = _bag(_ds(root, True), "few"), _bag(_ds(root, False), "few")
    assert on.utt_ids == off.utt_ids == ["few_0", "few_1", "few_2"]
    assert all(torch.equal(x, y) for x, y in zip(on.waveforms, off.waveforms, strict=True))


def test_long_clips_get_one_fixed_8s_window(root):
    on = [_bag(_ds(root, True), "long").waveforms[0] for _ in range(3)]
    assert on[0].shape[-1] == 8 * SR and all(torch.equal(on[0], w) for w in on[1:])
    off = {_bag(_ds(root, False), "long").waveforms[0][:50].sum().item() for _ in range(5)}
    assert len(off) > 1                                    # random crop without the flag
    full, _ = sf.read(str(root / "long_0.wav"), dtype="float32")
    start = int(np.random.default_rng(_stable_seed(str(root / "long_0.wav"))).integers(0, 12 * SR - 8 * SR + 1))
    assert torch.allclose(on[0], torch.from_numpy(full[start:start + 8 * SR]))
    utt = _ds(root, True, mode="utterance")                # per-clip scoring uses the same window
    assert torch.equal(utt[[r.utt_id for r in utt.manifest.itertuples()].index("long_0")].waveform, on[0])


def test_the_seed_is_a_crc32_not_pythons_salted_hash():
    assert _stable_seed("id10001") == zlib.crc32(b"id10001")


def test_datamodules_fix_validation_test_and_per_clip_but_never_training(root):
    dm = HeightCelebDataModule(root=str(root), bag_size=10, num_workers=0, eval_fixed_clips=True)
    dm.setup("fit")
    dm.setup("test")
    assert not dm.train_set.fixed_clips and dm.val_set.fixed_clips and dm.test_set.fixed_clips
    assert dm.utterance_loader("test").dataset.fixed_clips
    off = HeightCelebDataModule(root=str(root), bag_size=10, num_workers=0)
    off.setup("fit")
    off.setup("test")
    assert not (off.val_set.fixed_clips or off.test_set.fixed_clips or off.utterance_loader("test").dataset.fixed_clips)
    multi = MultiCorpusDataModule(corpora={"heightceleb": {"root": str(root), "splits": {"train": "train", "val": "dev",
                                                                                         "test": "test"}}},
                                  bag_size=10, num_workers=0, eval_fixed_clips=True)
    multi.setup("fit")
    multi.setup("test")
    assert not multi.train_set.fixed_clips and multi.val_set.fixed_clips and multi.test_set.fixed_clips
    assert _bag(multi.test_set, "many").utt_ids == _bag(dm.test_set, "many").utt_ids


def test_only_heightceleb_split2_turns_it_on():
    on = {p.stem for p in (CONFIGS / "data").glob("*.yaml") if yaml.safe_load(p.read_text()).get("eval_fixed_clips")}
    assert on == {"heightceleb_split2"}
