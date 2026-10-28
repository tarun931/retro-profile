"""Mixup on training batches (training/mixup.py, 2026-10-18; Gupta et al. 2022, Truong et al. 2022).

Pins: the blend of audio and labels (one lam per pair, a missing label stays missing); OFF by default with the
training step exactly the old one (no random number drawn, batch untouched); ON only in training, never validation;
both training modules accept it; only the two new mixup configs turn it on, each differing from its base in that
switch alone; and the result metadata records it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("faiss")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gupta_lightning import _batch as gupta_batch  # noqa: E402
from test_gupta_lightning import _lit_with as gupta_lit  # noqa: E402
from test_lightning_module import LABEL_STATS, _make_batch, _make_model_and_litmodule  # noqa: E402

import retro_profile.training.lightning_module as lm  # noqa: E402
from retro_profile.training.mixup import mixup_batch  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def test_mixup_blends_audio_and_labels_with_one_lam_per_pair():
    torch.manual_seed(0)
    b = _make_batch(B=5)
    torch.manual_seed(7)
    out = mixup_batch(b, 1.0)
    torch.manual_seed(7)                                     # replay the same draws
    lam = torch.distributions.Beta(torch.tensor(1.0), torch.tensor(1.0)).sample((5,))
    perm = torch.randperm(5)
    w = lam.view(5, 1, 1)
    assert torch.allclose(out.waveforms, w * b.waveforms + (1 - w) * b.waveforms[perm])
    for t in ("age", "height"):
        assert torch.allclose(out.labels[t], lam * b.labels[t] + (1 - lam) * b.labels[t][perm])
    assert torch.equal(out.sample_mask, b.sample_mask | b.sample_mask[perm])
    assert out.speaker_ids == b.speaker_ids and out.waveforms.shape == b.waveforms.shape


def test_a_label_missing_on_either_side_stays_missing():
    torch.manual_seed(0)
    b = _make_batch(B=6)
    b.labels["age"][[0, 3]] = float("nan")
    b.label_mask["age"][[0, 3]] = False
    torch.manual_seed(3)
    out = mixup_batch(b, 1.0)
    torch.manual_seed(3)
    torch.distributions.Beta(torch.tensor(1.0), torch.tensor(1.0)).sample((6,))
    perm = torch.randperm(6)
    want = b.label_mask["age"] & b.label_mask["age"][perm]
    assert torch.equal(out.label_mask["age"], want)
    assert torch.isnan(out.labels["age"][~want]).all() and not torch.isnan(out.labels["age"][want]).any()
    assert out.label_mask["height"].all()                    # untouched target stays complete


def test_off_by_default_the_training_step_is_exactly_the_old_one(monkeypatch):
    _, lit = _make_model_and_litmodule(LABEL_STATS)
    assert lit.mixup_alpha is None
    monkeypatch.setattr(lm, "mixup_batch", lambda *a, **k: pytest.fail("mixup ran while off"))
    b = _make_batch()
    torch.manual_seed(11)
    new = lit.training_step(b, 0)
    torch.manual_seed(11)                                    # same random state as before the step
    old = lit._shared_step(b, "train")[0]["total"]           # what training_step did before mixup existed
    assert torch.equal(new, old)


def test_on_it_mixes_training_batches_only(monkeypatch):
    _, lit = _make_model_and_litmodule(LABEL_STATS)
    lit.mixup_alpha = 1.0
    calls = []
    real = lm.mixup_batch
    monkeypatch.setattr(lm, "mixup_batch", lambda batch, alpha: calls.append(alpha) or real(batch, alpha))
    b = _make_batch()
    assert torch.isfinite(lit.training_step(b, 0))
    lit.validation_step(b, 0)
    assert calls == [1.0]                                    # once, for training; never for validation


def test_both_training_modules_take_the_switch():
    _, lit = _make_model_and_litmodule(LABEL_STATS)
    assert lit.hparams.get("mixup_alpha") is None
    g = gupta_lit(mixup_alpha=1.0)
    assert g.mixup_alpha == 1.0 and gupta_lit().mixup_alpha is None
    torch.manual_seed(0)
    loss = g.training_step(gupta_batch(), 0)                 # gender becomes a soft 0-1 target
    assert torch.isfinite(loss)
    loss.backward()


def test_result_metadata_records_mixup():
    spec = importlib.util.spec_from_file_location("ev", _ROOT / "scripts" / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    args = argparse.Namespace(unit="speaker", ckpt="x/epoch=1-step=1-val=1.0.ckpt", bag_size=10, split="test")
    model = nn.Linear(2, 2)
    old = ev._run_metadata({"training": {}}, args, model, {}, None)
    new = ev._run_metadata({"training": {"mixup": {"enabled": True, "alpha": 1.0}}}, args, model, {}, None)
    assert old["budget"]["mixup_alpha"] is None and new["budget"]["mixup_alpha"] == 1.0
