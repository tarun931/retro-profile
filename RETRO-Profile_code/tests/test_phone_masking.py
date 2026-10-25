"""scripts/phone_masking.py: TIMIT phone-class masking for the Table 1 models (Gupta et al. 2022's Table 4 analysis).

Pins: the 7 classes split TIMIT's 61 test phone symbols exactly once each; .PHN beside the audio; only the chosen
class is silenced and the input is not modified; 'none' loads the audio untouched; a clip that would be cropped at
random stops the script; the reproduction check stops on a mismatch; the masked share of audio is measured right.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

pytest.importorskip("faiss")
_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phone_masking", _ROOT / "scripts" / "phone_masking.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)

# Every phone symbol in TIMIT's 1,680 test .PHN files (counted 18 Oct 2026).
TIMIT_TEST_PHONES = [
    "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay", "b", "bcl", "ch", "d", "dcl", "dh", "dx", "eh", "el", "em",
    "en", "eng", "epi", "er", "ey", "f", "g", "gcl", "h#", "hh", "hv", "ih", "ix", "iy", "jh", "k", "kcl", "l", "m", "n",
    "ng", "nx", "ow", "oy", "p", "pau", "pcl", "q", "r", "s", "sh", "t", "tcl", "th", "uh", "uw", "ux", "v", "w", "y",
    "z", "zh",
]


def test_the_seven_classes_split_timits_phones_exactly_once():
    seen = [ph for phones in pm.PHONE_CLASSES.values() for ph in phones]
    assert len(seen) == len(set(seen)) == 61 and set(seen) == set(TIMIT_TEST_PHONES)
    assert len(pm.PHONE_CLASSES["vowels"]) == 20 and pm.CONDITIONS[0] == "none" and len(pm.CONDITIONS) == 8


def test_phn_path_sits_beside_the_audio():
    assert pm.phn_path("/x/DR1/FAKS0/SA1.WAV.wav") == Path("/x/DR1/FAKS0/SA1.PHN")
    assert pm.phn_path("/x/SA1.wav") == Path("/x/SA1.PHN") and pm.phn_path("/x/SA1.WAV") == Path("/x/SA1.PHN")


def _utt(root: Path, spk: str, utt: str, n: int, phones):
    d = root / "TEST" / spk
    d.mkdir(parents=True, exist_ok=True)
    wav = (np.arange(n) % 97 + 1).astype("float32") / 1000.0          # non-zero everywhere
    sf.write(str(d / f"{utt}.WAV.wav"), wav, 16000)
    (d / f"{utt}.PHN").write_text("".join(f"{s} {e} {ph}\n" for s, e, ph in phones))
    return str(d / f"{utt}.WAV.wav")


@pytest.fixture
def timit(tmp_path):
    rows = []
    segs = [(0, 800, "h#"), (800, 2400, "iy"), (2400, 3200, "s"), (3200, 4800, "ae"), (4800, 6000, "h#")]
    for spk, age, h in (("FAKS0", 30.0, 165.0), ("MABC0", 40.0, 180.0)):
        for utt in ("SA1", "SA2"):
            rows.append({"speaker_id": spk, "utt_id": utt, "audio_path": _utt(tmp_path, spk, utt, 6000, segs),
                         "age": age, "height": h, "gender": spk[0], "dialect_region": "DR1"})
    (tmp_path / "manifests").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "manifests" / "test.csv", index=False)
    return tmp_path, segs


def _ds(root, mask, mode="utterance", max_s=8.0):
    return pm.MaskedTimitDataset(root=root, split="test", mode=mode, transforms=None, targets=pm.TARGETS,
                                 sample_rate=16000, max_audio_seconds=max_s, min_audio_seconds=0.1, bag_size=10, mask=mask)


def test_mask_silences_only_the_chosen_class_and_leaves_the_input_alone(timit):
    root, segs = timit
    wav = torch.arange(1, 6001, dtype=torch.float32)
    out = pm.mask_waveform(wav, segs, pm.PHONE_CLASSES["vowels"])
    assert torch.equal(wav, torch.arange(1, 6001, dtype=torch.float32))             # input untouched
    assert (out[800:2400] == 0).all() and (out[3200:4800] == 0).all()               # iy, ae
    assert torch.equal(out[:800], wav[:800]) and torch.equal(out[2400:3200], wav[2400:3200])
    assert torch.equal(out[4800:], wav[4800:])


def test_none_is_the_untouched_audio_and_a_class_silences_its_phones(timit):
    root, _ = timit
    plain, vow, fric = _ds(root, None), _ds(root, "vowels"), _ds(root, "fricatives")
    for i in range(len(plain)):
        a = plain[i].waveform
        assert torch.equal(a, pm.load_audio(plain.manifest.audio_path.iloc[i], target_sr=16000))
        v, f = vow[i].waveform, fric[i].waveform
        assert (v[800:2400] == 0).all() and (v[3200:4800] == 0).all() and torch.equal(v[2400:3200], a[2400:3200])
        assert (f[2400:3200] == 0).all() and torch.equal(f[:2400], a[:2400])
    bag = _ds(root, "vowels", mode="bag")[0]
    assert len(bag.waveforms) == 2 and all((w[800:2400] == 0).all() for w in bag.waveforms)


def test_a_clip_that_would_be_cropped_at_random_stops(timit):
    root, _ = timit
    with pytest.raises(SystemExit, match="cropped at random"):
        _ds(root, None, max_s=0.2)[0]


def test_unknown_phone_symbol_stops(tmp_path):
    (tmp_path / "x.PHN").write_text("0 10 h#\n10 20 zz\n")
    with pytest.raises(ValueError, match="outside every class"):
        pm.read_phn(tmp_path / "x.PHN")


def test_reproduction_check_stops_on_a_mismatch(tmp_path):
    ours = pd.DataFrame({"speaker_id": ["A", "A"], "target": ["age", "height"], "prediction": [30.0, 170.0]})
    ref = tmp_path / "ref.csv"
    ours.assign(prediction=[30.001, 170.0]).to_csv(ref, index=False)
    assert pm.check_reproduces(ours, ref) == pytest.approx(0.001, abs=1e-9)
    ours.assign(prediction=[31.0, 170.0]).to_csv(ref, index=False)
    with pytest.raises(SystemExit, match="differ"):
        pm.check_reproduces(ours, ref)


def test_masked_fraction_is_the_share_of_samples(timit):
    root, _ = timit
    f = pm.masked_fraction({"root": str(root)}).set_index("condition")["masked_fraction"]
    assert f["vowels"] == pytest.approx(3200 / 6000) and f["fricatives"] == pytest.approx(800 / 6000)
    assert f["others"] == pytest.approx(2000 / 6000) and f["nasals"] == 0.0
