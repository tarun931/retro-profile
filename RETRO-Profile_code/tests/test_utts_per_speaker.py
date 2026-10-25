"""8 fresh clips per speaker per epoch for utterance-level baselines (Gupta, Jaid) on HeightCeleb + NISP.

Per-clip training over every clip would give them ~13x RETRO's audio; the sampler caps each speaker
at ``n`` clips per epoch, redrawn each epoch, and the budget is recorded in every result's __meta.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from retro_profile.data import HeightCelebDataModule, MultiCorpusDataModule
from retro_profile.data.sampler import PerSpeakerUtteranceSampler

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SPEAKERS = ["a"] * 20 + ["b"] * 3 + ["c"] * 9


def test_sampler_caps_each_speaker_and_redraws_every_epoch():
    s = PerSpeakerUtteranceSampler(SPEAKERS, 8)
    assert len(s) == 8 + 3 + 8
    torch.manual_seed(0)
    first = list(s)
    per_spk = pd.Series([SPEAKERS[i] for i in first]).value_counts().to_dict()
    assert per_spk == {"a": 8, "b": 3, "c": 8} and len(set(first)) == len(first)
    assert set(first) >= {20, 21, 22}                    # b has only 3 clips: all of them
    assert set(list(s)) != set(first)                    # a fresh draw next epoch
    torch.manual_seed(0)
    assert list(s) == first                              # reproducible under a seed


def test_sampler_rejects_n_below_one():
    with pytest.raises(ValueError, match="n must be >= 1"):
        PerSpeakerUtteranceSampler(SPEAKERS, 0)


@pytest.fixture
def hc_root(tmp_path):
    (tmp_path / "manifests").mkdir()
    for split, spk in (("train", SPEAKERS), ("dev", ["d"] * 2)):
        pd.DataFrame({"speaker_id": spk, "utt_id": [f"u{i}" for i in range(len(spk))],
                      "audio_path": [f"{i}.wav" for i in range(len(spk))], "age": 30.0,
                      "height": 170.0, "gender": "M"}).to_csv(tmp_path / "manifests" / f"{split}.csv",
                                                             index=False)
    return tmp_path


def test_train_loader_uses_the_cap(hc_root):
    dm = HeightCelebDataModule(root=str(hc_root), train_unit="utterance", utts_per_speaker=8,
                               batch_size=2, num_workers=0, splits={"train": "train", "val": "dev"})
    dm.setup("fit")
    loader = dm.train_dataloader()
    assert isinstance(loader.sampler, PerSpeakerUtteranceSampler) and len(loader) == 19 // 2
    full = HeightCelebDataModule(root=str(hc_root), train_unit="utterance", batch_size=2,
                                 num_workers=0, splits={"train": "train", "val": "dev"})
    full.setup("fit")
    assert len(full.train_dataloader()) == len(SPEAKERS) // 2          # null = every clip


def test_cap_is_refused_for_bag_training():
    with pytest.raises(ValueError, match="utterance' only"):
        HeightCelebDataModule(root=".", train_unit="bag", utts_per_speaker=8)
    with pytest.raises(ValueError, match="utterance' only"):
        MultiCorpusDataModule(corpora={"heightceleb": {"root": "."}}, utts_per_speaker=8)


def _cfg(experiment):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.fspath(CONFIGS), version_base=None):
        return OmegaConf.to_container(compose(config_name="default",
                                              overrides=[f"experiment={experiment}"]), resolve=True)


@pytest.mark.parametrize("new,old", [("heightceleb_nisp_gupta_wavlm", "gupta2022_finetune_wavlm"),
                                     ("heightceleb_nisp_jaid", "jaid2023")])
def test_baselines_change_only_the_data_and_the_cap(new, old):
    a, b, retro = _cfg(old), _cfg(new), _cfg("heightceleb_nisp_l100_s5")
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"data", "run_name", "output_dir"}
    for k in ("train_unit", "batch_size", "bag_size", "max_audio_seconds", "targets", "augmentation"):
        assert a["data"][k] == b["data"][k], k
    assert b["data"]["utts_per_speaker"] == 8 == retro["data"]["bag_size"]
    assert b["data"]["corpora"] == retro["data"]["corpora"]               # same speakers as RETRO B
    assert b["data"]["_target_"] == "retro_profile.data.MultiCorpusDataModule"


def test_run_metadata_records_the_cap():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    cfg = OmegaConf.create({"data": {"train_unit": "utterance", "utts_per_speaker": 8},
                            "model": {}, "training": {"checkpoint": {}, "early_stopping": {}}})
    args = argparse.Namespace(unit="speaker", ckpt=Path("runs/x/checkpoints/epoch=3-step=1.ckpt"))
    meta = ev._run_metadata(cfg, args, nn.Linear(2, 1), {"epoch": 3}, None)
    assert meta["budget"]["utts_per_speaker"] == 8 and meta["budget"]["train_unit"] == "utterance"


@pytest.mark.parametrize("new,old,cap", [("heightceleb_only_l100_s5", "retro_ordcon_fixed_l100_s5", None),
                                         ("heightceleb_only_gupta_wavlm", "gupta2022_finetune_wavlm", 8),
                                         ("heightceleb_only_jaid", "jaid2023", 8)])
def test_heightceleb_only_runs_change_only_the_data(new, old, cap):
    a, b = _cfg(old), _cfg(new)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"data", "run_name", "output_dir"}
    for k in ("train_unit", "batch_size", "bag_size", "max_audio_seconds", "targets", "augmentation"):
        assert a["data"].get(k) == b["data"].get(k), k
    assert b["data"]["root"] == "data/heightceleb" and b["data"].get("utts_per_speaker") == cap
    assert b["data"]["splits"] == {"train": "train", "val": "dev", "test": "test"}   # official 40 test speakers


@pytest.mark.parametrize("suf", ["l100_s5", "gupta_wavlm", "jaid"])
def test_split2_runs_differ_from_official_split_only_in_manifests(suf):
    a, b = _cfg(f"heightceleb_only_{suf}"), _cfg(f"heightceleb_split2_{suf}")
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"data", "run_name", "output_dir"}
    # eval_fixed_clips added 17 Oct 2026 (Table 4): split2 validation / test score fixed clips. The E33 runs of 12 Oct
    # were scored before it, on random draws.
    assert {k for k in set(a["data"]) | set(b["data"]) if a["data"].get(k) != b["data"].get(k)} == \
        {"name", "root", "eval_fixed_clips"}
    assert b["data"]["root"] == "data/heightceleb_split2" and b["data"]["name"] == "heightceleb_split2"
    assert b["data"]["eval_fixed_clips"] is True and a["data"].get("eval_fixed_clips") is None


def test_split2_is_an_evaluable_corpus(hc_root):
    from retro_profile.data import build_datamodule
    dm = build_datamodule("heightceleb_split2", root=str(hc_root), splits={"train": "train", "val": "dev"},
                          num_workers=0)
    dm.setup("fit")
    assert dm.train_set.num_speakers == 3
    spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    assert ev.CORPUS_DEFAULTS["heightceleb_split2"]["root"] == "data/heightceleb_split2"
    assert (CONFIGS / "data" / "heightceleb_split2.yaml").exists()


def test_eval_bag_size_scores_every_clip_in_a_fixed_order(hc_root, monkeypatch):
    """--bag-size (15 Oct 2026): at evaluation, a bag at least as large as a speaker's clips holds all of them,
    in manifest order, identically on every call; the corpus default draws a random subset."""
    from retro_profile.data.base import BaseSpeakerDataset
    monkeypatch.setattr(BaseSpeakerDataset, "_load_and_trim", lambda self, p: torch.zeros(160))
    spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    base = {"root": str(hc_root), "splits": {"train": "train", "val": "dev"}, "num_workers": 0, "bag_size": 8}
    assert ev.apply_bag_size(base, None) is base
    with pytest.raises(SystemExit):
        ev.apply_bag_size(base, 0)

    def bag_of_c(cfg):
        dm = HeightCelebDataModule(**cfg)
        dm.setup("fit")
        ds = dm.train_set
        return ds._get_bag(ds.speakers.index("c")).utt_ids

    full = ev.apply_bag_size(base, 10)
    assert full["bag_size"] == 10 and base["bag_size"] == 8                    # the input is not modified
    first, again = bag_of_c(full), bag_of_c(full)
    assert first == again == [f"u{i}" for i in range(23, 32)]                  # all 9 of c's clips, in order
    torch.manual_seed(0)
    assert len(bag_of_c(base)) == 8                                            # default: 8 drawn at random
    args = argparse.Namespace(unit="speaker", split="test", bag_size=10, ckpt=Path("runs/x/checkpoints/last.ckpt"))
    assert ev._run_metadata({"model": {}}, args, nn.Linear(2, 1), {"epoch": 3}, None)["eval"]["bag_size"] == 10
