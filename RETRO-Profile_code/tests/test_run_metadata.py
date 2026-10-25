"""The provenance block written into every summary.json.

A result is not comparable to another result without the budget and protocol
that produced it. Two of these fields have already invalidated runs in this
project: `selection.monitor` (monitoring val/height/mae alone early-stops
before age converges) and `eval.unit` (per-speaker bags are ~sqrt(8) easier
than the per-utterance protocol published TIMIT papers report).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_script", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
)
evaluate_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluate_script)


CFG = OmegaConf.create({
    "run_name": "gupta2022_finetune_timit_s42",
    "seed": 42,
    "precision": "16-mixed",
    "model": {
        "name": "baseline_gupta2022",
        "encoder": {
            "_target_": "retro_profile.models.encoders.Wav2Vec2FrameEncoder",
            "pretrained_name": "facebook/wav2vec2-base",
            "freeze": False,
        },
    },
    "data": {
        "batch_size": 8, "bag_size": 8, "max_audio_seconds": 8.0,
        "train_unit": "utterance", "augmentation": {"enabled": False},
    },
    "training": {
        "max_epochs": 50, "accumulate_grad_batches": 1,
        "optimizer": {"lr": 3.0e-4},
        "gupta": {"encoder_lr": 1.0e-6},
        "checkpoint": {"monitor": "val/total"},
        "early_stopping": {"enabled": False, "monitor": "val/height/mae"},
    },
})


def _meta(unit="utterance", datastore=None):
    model = nn.Linear(4, 2)
    args = argparse.Namespace(unit=unit, ckpt=Path("runs/x/checkpoints/epoch=14-step=7350.ckpt"))
    return evaluate_script._run_metadata(CFG, args, model, {"epoch": 14}, datastore)


def test_records_the_selection_protocol():
    """The field whose absence cost three runs."""
    m = _meta()
    assert m["selection"]["monitor"] == "val/total"
    assert m["selection"]["early_stopping"] is False


def test_records_the_eval_unit():
    assert _meta(unit="utterance")["eval"]["unit"] == "utterance"
    assert _meta(unit="speaker")["eval"]["unit"] == "speaker"


def test_records_the_budget():
    b = _meta()["budget"]
    assert b["max_audio_seconds"] == 8.0
    assert b["batch_size"] == 8
    assert b["effective_batch"] == 8          # batch_size * accumulate
    assert b["max_epochs"] == 50
    assert b["best_epoch"] == 14
    assert b["precision"] == "16-mixed"
    assert b["augmentation"] is False


def test_encoder_lr_falls_back_to_the_baseline_block():
    """RETRO puts encoder_lr at training.encoder_lr; Gupta nests it under training.gupta."""
    assert _meta()["budget"]["encoder_lr"] == 1.0e-6


def test_records_model_identity_and_param_counts():
    m = _meta()["model"]
    assert m["name"] == "baseline_gupta2022"
    assert m["encoder_frozen"] is False
    assert m["trainable_params"] == sum(p.numel() for p in nn.Linear(4, 2).parameters())
    assert m["total_params"] >= m["trainable_params"]


def test_records_whether_retrieval_was_used():
    assert _meta(datastore=None)["eval"]["retrieval"] is False
    ds = [object()] * 392
    assert _meta(datastore=ds)["eval"]["retrieval"] is True
    assert _meta(datastore=ds)["eval"]["datastore_size"] == 392


def test_meta_is_json_serialisable():
    import json
    json.dumps(_meta())          # must not raise: it is written into summary.json
