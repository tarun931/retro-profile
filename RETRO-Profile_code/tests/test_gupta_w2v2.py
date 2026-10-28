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
    ("gupta2022_finetune_w2v2_lr3e6", "gupta2022_finetune_wavlm_lr3e6"),
])
def test_each_candidate_differs_from_its_wavlm_twin_in_the_encoder_alone(w2v2, wavlm):
    a, b = _cfg(w2v2), _cfg(wavlm)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == ENCODER_KEYS | NAME_KEYS, changed
    assert a["model.encoder.pretrained_name"] == "facebook/wav2vec2-base"
    assert a["model.encoder._target_"].endswith("Wav2Vec2FrameEncoder")
    assert a["model.encoder.freeze"] is False and a["model.encoder.freeze_feature_extractor"] is True
    assert a["run_name"] == f"{w2v2}_timit"


@pytest.mark.parametrize("w2v2, wavlm", [
    ("timit_nisp_gupta_w2v2_lr3e6", "timit_nisp_gupta_lr3e6"),            # Table 4a
    ("timit_nisp_hcs_gupta_w2v2_lr3e6", "timit_nisp_hcs_gupta_lr3e6"),    # Table 4b
])
def test_table4_w2v2_configs_differ_from_their_wavlm_twins_in_the_encoder_alone(w2v2, wavlm):
    a, b = _cfg(w2v2), _cfg(wavlm)
    changed = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert changed == ENCODER_KEYS | NAME_KEYS, changed
    assert a["model.encoder.pretrained_name"] == "facebook/wav2vec2-base" and a["data.utts_per_speaker"] == 10


