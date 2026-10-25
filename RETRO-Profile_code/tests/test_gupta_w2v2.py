"""Gupta 2022 as published (wav2vec 2.0) for Table 1, 2026-10-18.

Pins: each of the four wav2vec 2.0 candidates differs from its WavLM twin (the Table 1 Gupta row and its tuning
candidates) in the encoder alone -- same learning rates, batch, clips, epochs, validation bag, losses and checkpoint
rule -- so the two Gupta rows compare encoders and nothing else; and the 1e-6 candidate is the 2026-08-30 wav2vec2
reproduction setting under a new run name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_multi_corpus import _cfg  # noqa: E402

ENCODER_KEYS = {"model.encoder._target_", "model.encoder.pretrained_name"}
NAME_KEYS = {"run_name", "output_dir"}


@pytest.mark.parametrize("w2v2, wavlm", [
    ("gupta2022_finetune_w2v2_lr1e6", "gupta2022_finetune_wavlm"),
    ("gupta2022_finetune_w2v2_lr3e6", "gupta2022_finetune_wavlm_lr3e6"),
    ("gupta2022_finetune_w2v2_lr1e5", "gupta2022_finetune_wavlm_lr1e5"),
    ("gupta2022_finetune_w2v2_retrorecipe", "gupta2022_finetune_wavlm_retrorecipe"),
])
def test_each_candidate_differs_from_its_wavlm_twin_in_the_encoder_alone(w2v2, wavlm):
    a, b = _cfg(w2v2), _cfg(wavlm)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == ENCODER_KEYS | NAME_KEYS, changed
    assert a["model.encoder.pretrained_name"] == "facebook/wav2vec2-base"
    assert a["model.encoder._target_"].endswith("Wav2Vec2FrameEncoder")
    assert a["model.encoder.freeze"] is False and a["model.encoder.freeze_feature_extractor"] is True
    assert a["run_name"] == f"{w2v2}_timit"


def test_the_paper_recipe_candidates_keep_the_table1_gupta_settings():
    for name, lr in (("gupta2022_finetune_w2v2_lr1e6", 1e-6), ("gupta2022_finetune_w2v2_lr3e6", 3e-6),
                     ("gupta2022_finetune_w2v2_lr1e5", 1e-5)):
        c = _cfg(name)
        assert c["training.gupta.encoder_lr"] == c["training.gupta.head_lr"] == lr
        assert c["data.batch_size"] == 8 and c["data.bag_size"] == 8 and c["data.train_unit"] == "utterance"
        assert c["training.max_epochs"] == 50 and c["training.accumulate_grad_batches"] == 1
        assert c["training.checkpoint.monitor"] == "val/combined_mae"
        assert not c.get("training.mixup.enabled", False)


def test_the_1e6_candidate_is_the_old_wav2vec2_reproduction_under_a_new_name():
    a, b = _cfg("gupta2022_finetune_w2v2_lr1e6"), _cfg("gupta2022_finetune")
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == NAME_KEYS, changed


@pytest.mark.parametrize("w2v2, wavlm", [
    ("timit_nisp_gupta_w2v2_lr3e6", "timit_nisp_gupta_lr3e6"),            # Table 4a
    ("timit_nisp_hcs_gupta_w2v2_lr3e6", "timit_nisp_hcs_gupta_lr3e6"),    # Table 4b
])
def test_table4_w2v2_configs_differ_from_their_wavlm_twins_in_the_encoder_alone(w2v2, wavlm):
    a, b = _cfg(w2v2), _cfg(wavlm)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == ENCODER_KEYS | NAME_KEYS, changed
    assert a["model.encoder.pretrained_name"] == "facebook/wav2vec2-base" and a["data.utts_per_speaker"] == 10


def test_masking_only_exp_scores_just_the_named_experiment(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pm", Path(__file__).resolve().parents[1] / "scripts/phone_masking.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    assert pm.TABLE1["Gupta 2022 (wav2vec 2.0)"] == ("gupta2022_finetune_w2v2_lr3e6", "clip_mean",
                                                     "eval_utt/predictions_timit_speaker.csv")
    loaded = []

    class StopError(Exception):
        pass

    class FR:
        def corpus_data_cfg(self, *a):
            return {"root": "x"}

        def load_frozen_model(self, run):
            loaded.append(str(run))
            raise StopError

    monkeypatch.setattr(pm, "_script", lambda name: FR())
    monkeypatch.setattr(pm, "masked_fraction", lambda cfg: __import__("pandas").DataFrame({"condition": [], "masked_fraction": []}))
    with pytest.raises(StopError):
        pm.main(["--out", str(tmp_path), "--only-exp", "gupta2022_finetune_w2v2_lr3e6"])
    assert loaded == ["runs/gupta2022_finetune_w2v2_lr3e6_timit_s42"]
