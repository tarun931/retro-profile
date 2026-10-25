"""scripts/prepare_heightceleb.py on a tiny VoxCeleb1-shaped tree (empty wav files: it never reads audio).

Pins: an aged speaker keeps every clip in train but only its dated video's clips in dev and test;
everyone else keeps all clips with age NaN; TEST stays test; dev is carved from TRAIN,
speaker-disjoint, with both genders and aged speakers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_heightceleb.py"
_SPEC = importlib.util.spec_from_file_location("prepare_heightceleb", _SCRIPT)
ph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ph)

N_TRAIN = 40                                    # id10001..id10040 TRAIN, id10041..id10044 TEST


def _speakers() -> list[str]:
    return [f"id{10001 + i}" for i in range(N_TRAIN + 4)]


@pytest.fixture
def corpus(tmp_path):
    wav = tmp_path / "wav"
    for s in _speakers():
        for v, n in (("vidAAAAAAAA", 3), ("vidBBBBBBBB", 5)):
            d = wav / s / v
            d.mkdir(parents=True)
            for k in range(1, n + 1):
                (d / f"{k:05d}.wav").touch()
    heights = pd.DataFrame({
        "VoxCeleb1 ID": _speakers(),
        "height": [160 + i for i in range(len(_speakers()))],
        "sex": ["M" if i % 2 else "F" for i in range(len(_speakers()))],
        "split": ["TRAIN"] * N_TRAIN + ["TEST"] * 4,
    })
    heights.to_csv(tmp_path / "heights.csv", index=False)
    aged = heights.iloc[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, N_TRAIN]]       # 12 TRAIN + 1 TEST
    ages = pd.DataFrame({"speaker_id": aged["VoxCeleb1 ID"], "sex": aged["sex"],
                         "height": aged["height"], "video_id": "vidBBBBBBBB", "age": 40.0})
    ages.loc[ages.speaker_id == "id10002", "video_id"] = "vidGONEGONE"     # dated video not in audio
    ages.to_csv(tmp_path / "ages.csv", index=False)
    return tmp_path


def _run(corpus, *extra):
    ph.main(["--wav-root", str(corpus / "wav"), "--out", str(corpus / "out"),
             "--heights", str(corpus / "heights.csv"), "--ages", str(corpus / "ages.csv"), *extra])
    return {s: pd.read_csv(corpus / "out" / "manifests" / f"{s}.csv") for s in ("train", "dev", "test")}


def test_columns_and_utt_ids(corpus):
    m = _run(corpus)
    df = pd.concat(m.values())
    assert list(df.columns) == ph.COLS
    assert not df.duplicated(["speaker_id", "utt_id"]).any()
    assert df.utt_id.iloc[0].startswith(df.video_id.iloc[0] + "_")
    assert set(df.gender) == {"M", "F"}


def _aged_clip_counts(m):
    """split -> {aged speaker: clips}."""
    return {s: d[d.age.notna()].groupby("speaker_id").size().to_dict() for s, d in m.items()}


def test_aged_speakers_all_clips_in_train_dated_video_in_dev_and_test(corpus):
    m = _run(corpus)
    n = _aged_clip_counts(m)
    assert n["train"] and n["dev"] and n["test"] == {"id10041": 5}
    assert set(n["train"].values()) == {8} and set(n["dev"].values()) == {5}
    df = pd.concat(m.values())
    eval_aged = df[df.age.notna() & ~df.speaker_id.isin(n["train"])]
    assert set(eval_aged.video_id) == {"vidBBBBBBBB"}
    assert (df[df.age.notna()].age == 40).all()
    plain = df[df.speaker_id == "id10020"]
    assert len(plain) == 8 and plain.age.isna().all() and (plain.height == 179).all()
    gone = df[df.speaker_id == "id10002"]                       # dated video missing -> no age
    assert len(gone) == 8 and gone.age.isna().all()


def test_age_clip_options(corpus):
    n = _aged_clip_counts(_run(corpus, "--train-age-clips", "video"))
    assert set(n["train"].values()) == {5}
    n = _aged_clip_counts(_run(corpus, "--test-age-clips", "all"))      # test pooled into training
    assert set(n["dev"].values()) == {5} and n["test"] == {"id10041": 8}
    n = _aged_clip_counts(_run(corpus, "--dev-age-clips", "all"))
    assert set(n["dev"].values()) == {8} and n["test"] == {"id10041": 5}


def test_splits(corpus):
    m = _run(corpus)
    spk = {s: set(d.speaker_id) for s, d in m.items()}
    assert spk["test"] == set(_speakers()[N_TRAIN:])
    assert not (spk["train"] & spk["dev"]) and not ((spk["train"] | spk["dev"]) & spk["test"])
    assert spk["train"] | spk["dev"] == set(_speakers()[:N_TRAIN])
    dev = m["dev"].drop_duplicates("speaker_id")
    assert set(dev.gender) == {"M", "F"} and dev.age.notna().any()
    assert 2 <= len(dev) <= 8                                    # ~10% of 40, per stratum
    assert set(m["dev"].speaker_id) != set(_run(corpus, "--seed", "43")["dev"].speaker_id)


def test_labelled_speaker_without_audio_is_an_error(corpus):
    for f in (corpus / "wav" / "id10044").rglob("*.wav"):
        f.unlink()
    with pytest.raises(SystemExit, match="no audio"):
        _run(corpus)


def test_ages_must_agree_with_heights(corpus):
    ages = pd.read_csv(corpus / "ages.csv")
    ages.loc[0, "height"] += 1
    ages.to_csv(corpus / "ages.csv", index=False)
    with pytest.raises(SystemExit, match="disagrees"):
        _run(corpus)


# ---- grouped split (--split-mode grouped, 2026-10-12) ---------------------------------------------

def _grouped_corpus(root, n=50, shared=(("id10001", "id10046"), ("id10002", "id10003"), ("id10003", "id10004"))):
    """n speakers (the last 10 official TEST), each with two own videos of 4 clips; ``shared`` pairs
    also appear together in one extra video. Every 2nd speaker has an age, dated on its first video."""
    spk = [f"id{10001 + i}" for i in range(n)]
    for s in spk:
        for v in (f"v{s}A", f"v{s}B"):
            (root / "wav" / s / v).mkdir(parents=True)
            for k in range(1, 5):
                (root / "wav" / s / v / f"{k:05d}.wav").touch()
    for j, pair in enumerate(shared):
        for s in pair:
            d = root / "wav" / s / f"shared{j:05d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "00001.wav").touch()
    h = pd.DataFrame({"VoxCeleb1 ID": spk, "height": [155 + (i * 7) % 40 for i in range(n)],
                      "sex": ["M" if i % 2 else "F" for i in range(n)],
                      "split": ["TRAIN"] * (n - 10) + ["TEST"] * 10})
    h.to_csv(root / "heights.csv", index=False)
    aged = h.iloc[::2]
    pd.DataFrame({"speaker_id": aged["VoxCeleb1 ID"], "sex": aged["sex"], "height": aged["height"],
                  "video_id": [f"v{s}A" for s in aged["VoxCeleb1 ID"]], "age": 40.0}).to_csv(root / "ages.csv",
                                                                                            index=False)
    return root


def _run_grouped(root, *extra):
    ph.main(["--wav-root", str(root / "wav"), "--out", str(root / "out"), "--heights", str(root / "heights.csv"),
             "--ages", str(root / "ages.csv"), "--split-mode", "grouped", *extra])
    m = {s: pd.read_csv(root / "out" / "manifests" / f"{s}.csv") for s in ("train", "dev", "test")}
    return m, {sp: s for s, d in m.items() for sp in d.speaker_id}


def test_grouped_split_keeps_co_appearing_speakers_together(tmp_path):
    m, split = _run_grouped(_grouped_corpus(tmp_path))
    assert split["id10001"] == split["id10046"]                       # official TRAIN + TEST, one video
    assert split["id10002"] == split["id10003"] == split["id10004"]   # a chain is one group
    videos = pd.concat(m.values())
    assert (videos.assign(s=videos.speaker_id.map(split)).groupby("video_id").s.nunique() == 1).all()
    sizes = {s: d.speaker_id.nunique() for s, d in m.items()}
    assert sum(sizes.values()) == 50 and 28 <= sizes["train"] <= 32 and 8 <= sizes["test"] <= 12
    assert m["test"].drop_duplicates("speaker_id").age.notna().sum() >= 3
    groups = pd.read_csv(tmp_path / "out" / "groups.csv")
    assert groups.groupby("group").split.nunique().max() == 1


def test_grouped_split_uses_clip_fingerprints(tmp_path):
    root = _grouped_corpus(tmp_path, shared=())
    pd.DataFrame({"speaker_id": ["id10010", "id10041"], "md5": ["abc", "abc"]}).to_csv(root / "md5.csv", index=False)
    _, split = _run_grouped(root, "--md5-csv", str(root / "md5.csv"))
    assert split["id10010"] == split["id10041"]


def test_grouped_split_is_seeded(tmp_path):
    root = _grouped_corpus(tmp_path)
    _, a = _run_grouped(root, "--seed", "42")
    _, b = _run_grouped(root, "--seed", "42")
    _, c = _run_grouped(root, "--seed", "1")
    assert a == b and a != c


def test_speaker_groups_union_find():
    clips = pd.DataFrame({"speaker_id": ["a", "b", "b", "c", "d"], "video_id": ["v1", "v1", "v2", "v2", "v3"]})
    g = ph.speaker_groups(clips)
    assert g["a"] == g["b"] == g["c"] == "a" and g["d"] == "d"
    g = ph.speaker_groups(clips, pd.DataFrame({"speaker_id": ["d", "a"], "md5": ["x", "x"]}))
    assert g["d"] == "a"
