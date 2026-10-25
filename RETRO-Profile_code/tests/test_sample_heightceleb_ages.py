"""scripts/sample_heightceleb_ages.py: Table 4b's age-balanced HeightCeleb training sample.

Pins: equal count per 10-year band, half men / half women where possible, only aged speakers, the same seed gives the
same sample, rows are copied unchanged, and a speaker also in dev / test stops the script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sample_heightceleb_ages", _ROOT / "scripts" / "sample_heightceleb_ages.py")
sh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sh)


def _train(seed=0, per_band_gender=(6, 6), aged=True):
    """A train manifest: per band, F / M speakers with 3 clips each, plus speakers without an age."""
    rng = np.random.default_rng(seed)
    rows, i = [], 0
    for lo in (15, 20, 30, 40, 50, 60, 70):
        for g, n in zip("FM", per_band_gender, strict=True):
            for _ in range(n if lo != 70 or g == "M" else 2):          # few women over 70, as in HeightCeleb
                i += 1
                age = lo + int(rng.integers(0, 5))
                rows += [{"speaker_id": f"id{i:05d}", "utt_id": f"v_{i}_{k}", "audio_path": f"/wav/id{i:05d}/{k}.wav",
                          "video_id": "v", "age": age, "height": 170.0, "gender": g} for k in range(3)]
    for j in range(5):                                                  # no age: never eligible
        rows += [{"speaker_id": f"nx{j}", "utt_id": f"n_{j}", "audio_path": f"/wav/nx{j}.wav", "video_id": "w",
                  "age": np.nan, "height": 180.0, "gender": "M"}]
    return pd.DataFrame(rows)


def test_equal_per_band_half_and_half_and_only_aged():
    spk = sh.speakers_table(_train())
    chosen, n = sh.sample(spk, per_band=4, seed=0)
    assert n == 4 and len(chosen) == 28 and chosen["age"].notna().all()
    assert set(chosen["band"].value_counts()) == {4}
    by = pd.crosstab(chosen["band"], chosen["gender"])
    assert (by.drop(index="70+") == 2).all().all()
    assert by.loc["70+", "F"] == 2 and by.loc["70+", "M"] == 2


def test_per_band_is_capped_by_the_smallest_band_and_a_short_gender_is_filled():
    spk = sh.speakers_table(_train())
    chosen, n = sh.sample(spk, per_band=15, seed=0)          # smallest band: 70+ with 2 F + 6 M = 8
    assert n == 8 and set(chosen["band"].value_counts()) == {8}
    by = pd.crosstab(chosen["band"], chosen["gender"])
    assert by.loc["70+", "F"] == 2 and by.loc["70+", "M"] == 6  # women short: men fill
    assert (by.drop(index="70+") == 4).all().all()


def test_same_seed_same_sample_other_seed_differs():
    spk = sh.speakers_table(_train())
    a, _ = sh.sample(spk, per_band=3, seed=0)
    b, _ = sh.sample(spk.sample(frac=1, random_state=1), per_band=3, seed=0)   # input order does not matter
    c, _ = sh.sample(spk, per_band=3, seed=1)
    assert a["speaker_id"].tolist() == b["speaker_id"].tolist() and set(a["speaker_id"]) != set(c["speaker_id"])


def test_main_copies_rows_unchanged_and_refuses_dev_or_test_speakers(tmp_path):
    src = tmp_path / "split2" / "manifests"
    src.mkdir(parents=True)
    m = _train()
    m.to_csv(src / "train.csv", index=False)
    out, lst = tmp_path / "sample", tmp_path / "list.csv"
    sh.main(["--manifest", str(src / "train.csv"), "--out-root", str(out), "--list", str(lst), "--per-band", "4"])
    rows = pd.read_csv(out / "manifests" / "train.csv", dtype={"speaker_id": str})
    listed = pd.read_csv(lst, dtype={"speaker_id": str})
    assert set(rows["speaker_id"]) == set(listed["speaker_id"]) and len(listed) == 28
    want = m[m["speaker_id"].isin(set(listed["speaker_id"]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(rows.reset_index(drop=True), want, check_dtype=False)
    assert (listed["seed"] == 0).all() and (listed["per_band"] == 4).all()

    m[m["speaker_id"].isin(set(listed["speaker_id"]))].to_csv(src / "test.csv", index=False)
    with pytest.raises(SystemExit, match="speaker_id values are also in"):
        sh.main(["--manifest", str(src / "train.csv"), "--out-root", str(out), "--list", str(lst), "--per-band", "4"])
    pd.DataFrame({"speaker_id": ["zz"], "utt_id": ["u"], "audio_path": ["/x.wav"], "video_id": ["v"], "age": [40],
                  "height": [170.0], "gender": ["M"]}).to_csv(src / "test.csv", index=False)     # another speaker, same video
    with pytest.raises(SystemExit, match="video_id values are also in"):
        sh.main(["--manifest", str(src / "train.csv"), "--out-root", str(out), "--list", str(lst), "--per-band", "4"])
    with pytest.raises(SystemExit, match="TRAIN manifest"):
        sh.main(["--manifest", str(src / "test.csv"), "--out-root", str(out), "--list", str(lst)])


def test_band_edges():
    assert [sh.band_of(a) for a in (13, 19.9, 20, 29, 30, 69, 70, 87)] == \
        ["<20", "<20", "20-29", "20-29", "30-39", "60-69", "70+", "70+"]
