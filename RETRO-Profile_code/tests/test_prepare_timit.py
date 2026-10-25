"""Regression tests for scripts/prepare_timit.py.

Pins the three real-world bugs the Kaggle TIMIT layout exposed:
  1. labels must join (sex-prefixed folders vs un-prefixed SPKRINFO ids),
  2. each utterance appears once (prefer RIFF .wav, ignore the SPHERE .WAV dup),
  3. the `data/` nesting is auto-detected.

prepare_timit never reads audio, so empty placeholder files are sufficient.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_timit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_timit", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pt = _load_module()

MALE_KEYS = ["AA00", "AB00", "AC00", "AD00"]
FEMALE_KEYS = ["BA00", "BB00", "BC00", "BD00"]
TEST_SPK = [("M", "CA00"), ("F", "CB00")]
N_UTT = 3


def _build_corpus(root: Path) -> None:
    """Write a tiny TIMIT-shaped tree under ``root`` with a `data/` subdir."""
    data = root / "data"
    lines = ["; SPKRINFO header comment"]

    def add(split: str, sex: str, key: str) -> None:
        folder = sex + key                       # sex-prefixed folder, like FCJF0
        d = data / split / "DR1" / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(N_UTT):
            (d / f"SA{i}.WAV").write_bytes(b"")        # SPHERE duplicate (must be ignored)
            (d / f"SA{i}.WAV.wav").write_bytes(b"")    # RIFF (preferred)
        use = "TRN" if split == "TRAIN" else "TST"
        lines.append(f"{key}  {sex}  1  {use}  03/03/86  06/17/60  5'10\"  WHT  BS")

    for k in MALE_KEYS:
        add("TRAIN", "M", k)
    for k in FEMALE_KEYS:
        add("TRAIN", "F", k)
    for sex, k in TEST_SPK:
        add("TEST", sex, k)

    (root / "SPKRINFO.TXT").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "TIMIT"
    _build_corpus(root)
    return root


def test_layout_and_spkrinfo_resolved(corpus: Path):
    audio_base, spkrinfo = pt.resolve_layout(corpus)
    assert audio_base == corpus / "data"          # data/ nesting auto-detected
    assert spkrinfo == corpus / "SPKRINFO.TXT"
    info = pt.parse_spkrinfo(spkrinfo)
    assert len(info) == len(MALE_KEYS) + len(FEMALE_KEYS) + len(TEST_SPK)
    assert info["height"].notna().all()
    assert set(info["spk_key"]) >= set(MALE_KEYS)  # keys are un-prefixed


def test_no_duplicate_utterances_and_prefers_riff(corpus: Path):
    audio_base, _ = pt.resolve_layout(corpus)
    utts = pt.discover_utterances(audio_base)
    n_speakers = len(MALE_KEYS) + len(FEMALE_KEYS) + len(TEST_SPK)
    assert len(utts) == n_speakers * N_UTT         # NOT 2x (SPHERE dup dropped)
    assert utts["audio_path"].str.endswith(".WAV.wav").all()
    assert "SA0" in set(utts["utt_id"])            # extension stripped


def test_labels_join_after_prefix_strip(corpus: Path):
    audio_base, spkrinfo = pt.resolve_layout(corpus)
    info = pt.parse_spkrinfo(spkrinfo)
    utts = pt.discover_utterances(audio_base)
    merged = utts.merge(info[["spk_key", "age", "height", "gender"]], on="spk_key", how="left")
    assert merged["height"].notna().all()          # 0 -> all rows labelled
    assert merged["age"].notna().all()


def test_main_writes_disjoint_gender_stratified_splits(corpus: Path, tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv",
                        ["prepare_timit.py", "--raw", str(corpus), "--out", str(out)])
    pt.main()

    tr = pd.read_csv(out / "manifests" / "train.csv")
    dv = pd.read_csv(out / "manifests" / "dev.csv")
    te = pd.read_csv(out / "manifests" / "test.csv")

    s_tr, s_dv, s_te = set(tr.speaker_id), set(dv.speaker_id), set(te.speaker_id)
    assert s_tr.isdisjoint(s_dv) and s_tr.isdisjoint(s_te) and s_dv.isdisjoint(s_te)
    assert s_te == {sex + key for sex, key in TEST_SPK}        # standard TEST set
    assert s_tr | s_dv == {"M" + k for k in MALE_KEYS} | {"F" + k for k in FEMALE_KEYS}
    assert len(dv) > 0                                          # dev is non-empty
    assert tr["height"].notna().all()
