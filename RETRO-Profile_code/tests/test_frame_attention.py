"""Frame-level attention pooling (hierarchical attention, level 1).

Pins the properties the comparison against the base model depends on: at
initialisation the module IS the masked mean (so the hierarchical model starts as
the base model), padded frames never count, a clip with no real frames gives
zeros rather than NaN, the default encoder path is unchanged, and the new
parameters train at the head LR.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from retro_profile.models.encoders.base import BaseEncoder
from retro_profile.models.encoders.wavlm import Wav2Vec2Encoder, WavLMEncoder
from retro_profile.models.pooling import FrameAttentionPooling


def _frames(n=3, t=7, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, t, d, generator=g)
    mask = torch.ones(n, t, dtype=torch.bool)
    mask[1, 4:] = False          # clip 1: 4 real frames
    mask[2, 2:] = False          # clip 2: 2 real frames
    return feats, mask


# ---- the module ------------------------------------------------------------
def test_at_init_it_is_exactly_the_masked_mean():
    feats, mask = _frames()
    pool = FrameAttentionPooling(input_dim=8, hidden_dim=4).eval()
    pooled, weights = pool(feats, mask)
    expected = BaseEncoder._masked_mean_time(feats, mask)
    assert torch.allclose(pooled, expected, atol=1e-6)
    real = mask.sum(dim=1, keepdim=True).float()
    assert torch.allclose(weights, mask.float() / real, atol=1e-6)


def test_padded_frames_get_zero_weight_and_real_frames_sum_to_one():
    feats, mask = _frames()
    pool = FrameAttentionPooling(input_dim=8, hidden_dim=4).eval()
    nn.init.normal_(pool.w.weight)                 # move it off uniform
    _, weights = pool(feats, mask)
    assert (weights[~mask] == 0).all()
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_a_clip_with_no_real_frames_gives_zeros_not_nan():
    """Padded bag slots: softmax over all -inf is NaN, and NaN * 0 is still NaN."""
    feats, mask = _frames()
    mask[0] = False
    pool = FrameAttentionPooling(input_dim=8, hidden_dim=4)
    nn.init.normal_(pool.w.weight)
    feats.requires_grad_(True)
    pooled, weights = pool(feats, mask)
    assert torch.isfinite(pooled).all() and torch.isfinite(weights).all()
    assert (pooled[0] == 0).all() and (weights[0] == 0).all()
    pooled.sum().backward()
    assert torch.isfinite(feats.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in pool.parameters() if p.grad is not None)


def test_the_scorer_receives_gradient_at_init_so_it_can_leave_uniform():
    feats, mask = _frames()
    pool = FrameAttentionPooling(input_dim=8, hidden_dim=4)
    pooled, _ = pool(feats, mask)
    (pooled * torch.randn_like(pooled)).sum().backward()
    assert pool.w.weight.grad is not None and pool.w.weight.grad.abs().sum() > 0


def test_learned_weights_are_not_uniform():
    feats, mask = _frames()
    pool = FrameAttentionPooling(input_dim=8, hidden_dim=4).eval()
    nn.init.normal_(pool.w.weight, std=5.0)
    pooled, weights = pool(feats, mask)
    assert not torch.allclose(weights[0], torch.full((7,), 1 / 7), atol=1e-3)
    assert not torch.allclose(pooled, BaseEncoder._masked_mean_time(feats, mask), atol=1e-3)


# ---- inside the encoder ----------------------------------------------------
class _StubHF(nn.Module):
    """Minimal stand-in for an HF SSL model (as in test_encoder_attention_mask)."""

    def __init__(self, hidden: int = 8, layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, num_hidden_layers=layers,
                                      feat_extract_norm="layer")
        self.proj = nn.Linear(1, hidden)

    def forward(self, input_values, attention_mask=None):
        h = self.proj(input_values[:, : max(1, input_values.shape[1] // 4)].unsqueeze(-1))
        return SimpleNamespace(hidden_states=(h, h, h), last_hidden_state=h)


def _encoder(monkeypatch, cls=WavLMEncoder, stub=None, **kw):
    stub = stub or _StubHF()
    monkeypatch.setattr("retro_profile.models.encoders.wavlm.AutoModel",
                        SimpleNamespace(from_pretrained=lambda *a, **k: stub))
    return cls(freeze=False, **kw).eval(), stub


def _bag(b=2, k=3, t=16):
    torch.manual_seed(1)
    wav = torch.randn(b, k, t)
    sm = torch.ones(b, k, t, dtype=torch.bool)
    sm[0, 1, 8:] = False                           # a shorter clip
    um = torch.ones(b, k, dtype=torch.bool)
    return wav, sm, um


def test_default_is_mean_and_adds_no_parameters(monkeypatch):
    enc, _ = _encoder(monkeypatch)
    assert enc.frame_pooling == "mean" and enc.frame_pool is None
    assert not any("frame_pool" in n for n, _ in enc.named_parameters())


def test_attention_encoder_equals_mean_encoder_at_init(monkeypatch):
    stub = _StubHF()
    mean_enc, _ = _encoder(monkeypatch, stub=stub)
    att_enc, _ = _encoder(monkeypatch, stub=stub, frame_pooling="attention",
                          frame_attention_hidden=4)
    att_enc.layer_weights.data.copy_(mean_enc.layer_weights.data)
    wav, sm, um = _bag()
    assert torch.allclose(att_enc(wav, sm, um), mean_enc(wav, sm, um), atol=1e-6)


def test_padded_utterance_stays_finite_with_attention(monkeypatch):
    enc, _ = _encoder(monkeypatch, frame_pooling="attention", frame_attention_hidden=4)
    nn.init.normal_(enc.frame_pool.w.weight)
    wav, sm, um = _bag()
    sm[1, 2] = False                               # an empty bag slot ...
    um[1, 2] = False                               # ... marked as padding
    out = enc(wav, sm, um)
    assert torch.isfinite(out).all()
    assert (out[1, 2] == 0).all()


def test_unknown_frame_pooling_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="frame_pooling"):
        _encoder(monkeypatch, frame_pooling="max")


def test_wav2vec2_encoder_passes_the_option_through(monkeypatch):
    enc, _ = _encoder(monkeypatch, cls=Wav2Vec2Encoder, frame_pooling="attention")
    assert isinstance(enc.frame_pool, FrameAttentionPooling)


def test_an_encoder_without_frame_pool_falls_back_to_the_mean(monkeypatch):
    """HuBERTEncoder bypasses WavLMEncoder.__init__, so it never sets frame_pool."""
    enc, _ = _encoder(monkeypatch)
    expected = enc(*_bag())
    del enc.frame_pool
    assert torch.allclose(enc(*_bag()), expected)


def test_frame_attention_trains_at_the_head_lr_even_when_frozen(monkeypatch):
    """Its params must not match the slow-LR backbone prefix, and must stay trainable."""
    faiss = pytest.importorskip("faiss")  # noqa: F841 -- lightning_module imports retrieval
    from retro_profile.training.lightning_module import RetroProfileLitModule
    stub = _StubHF()
    monkeypatch.setattr("retro_profile.models.encoders.wavlm.AutoModel",
                        SimpleNamespace(from_pretrained=lambda *a, **k: stub))
    enc = WavLMEncoder(freeze=True, frame_pooling="attention")
    prefix = RetroProfileLitModule._BACKBONE_PREFIX
    names = [f"model.encoder.{n}" for n, p in enc.named_parameters() if "frame_pool" in n]
    assert names and not any(n.startswith(prefix) for n in names)
    assert all(p.requires_grad for n, p in enc.named_parameters() if "frame_pool" in n)


# ---- saving the frame weights at evaluation --------------------------------
def test_encoder_keeps_frame_weights_in_eval_mode_only(monkeypatch):
    enc, _ = _encoder(monkeypatch, frame_pooling="attention", frame_attention_hidden=4)
    wav, sm, um = _bag()
    enc.train()
    enc(wav, sm, um)
    assert enc.last_frame_attention is None
    enc.eval()
    enc(wav, sm, um)
    w, m = enc.last_frame_attention
    assert w.shape == m.shape and w.shape[0] == 2 * 3 and not w.requires_grad


def _tiny_model(monkeypatch, frame_pooling):
    from retro_profile.models.heads.mlp import MLPHead
    from retro_profile.models.pooling import MILAttentionPooling
    from retro_profile.models.retro_profile import RetroProfile
    enc, _ = _encoder(monkeypatch, frame_pooling=frame_pooling, frame_attention_hidden=4)
    return RetroProfile(
        encoder=enc,
        pooling=MILAttentionPooling(8, 4, 1, target_names=["age", "height"]),
        heads={t: MLPHead(input_dim=4, hidden_dim=4, output_dim=1) for t in ("age", "height")},
        projector=MLPHead(input_dim=8, hidden_dim=8, output_dim=4, l2_normalize=True),
    )


def _loader():
    from retro_profile.data.base import SpeakerBag
    from retro_profile.data.collate import collate_bags
    torch.manual_seed(2)
    bags = [SpeakerBag([torch.randn(64), torch.randn(40), torch.randn(64)][: 3 - i], 16000,
                       f"S{i}", ["SA1", "SX2", "SI3"][: 3 - i], {"age": 30.0 + i, "height": 170.0})
            for i in range(2)]                     # speaker S1 has a padded bag slot
    return [collate_bags(bags)]


def test_evaluator_writes_one_row_per_real_clip(monkeypatch):
    from retro_profile.evaluation.evaluator import evaluate_with_explanations
    model = _tiny_model(monkeypatch, "attention")
    rows: list[dict] = []
    evaluate_with_explanations(model, _loader(), ["age", "height"], device="cpu", frame_attention=rows)
    assert [(r["speaker_id"], r["utt_id"]) for r in rows] == [
        ("S0", "SA1"), ("S0", "SX2"), ("S0", "SI3"), ("S1", "SA1"), ("S1", "SX2")]
    assert rows[1]["n_frames"] < rows[0]["n_frames"]          # the shorter clip
    for r in rows:                                             # zero-init scorer: plain mean
        assert r["eff_share"] == pytest.approx(1.0) and r["entropy"] == pytest.approx(1.0)
        assert len(r["weights"]) == r["n_frames"]
        assert float(r["weights"].astype("float32").sum()) == pytest.approx(1.0, abs=1e-2)


def test_evaluator_collects_nothing_for_a_mean_model(monkeypatch):
    from retro_profile.evaluation.evaluator import evaluate_with_explanations
    rows: list[dict] = []
    evaluate_with_explanations(_tiny_model(monkeypatch, "mean"), _loader(), ["age", "height"],
                               device="cpu", frame_attention=rows)
    assert rows == []


def test_evaluator_saves_the_exact_knn_query(monkeypatch):
    """The embeddings the kNN sweep uses must be the model's own retrieval query."""
    from retro_profile.evaluation.evaluator import evaluate_with_explanations
    model = _tiny_model(monkeypatch, "mean").eval()
    loader = _loader()
    rows: list[dict] = []
    evaluate_with_explanations(model, loader, ["age", "height"], device="cpu", embeddings=rows)
    with torch.no_grad():
        expected = model.embed_speaker(loader[0]).numpy()
    assert [r["speaker_id"] for r in rows] == ["S0", "S1"]
    assert np.allclose(np.stack([r["embedding"] for r in rows]), expected, atol=1e-6)


def test_frame_attention_files(tmp_path):
    import numpy as np
    import pandas as pd
    rows = [{"speaker_id": "S0", "utt_id": "SA1", "n_frames": 2, "n_eff": 2.0, "eff_share": 1.0,
             "max_weight": 0.5, "entropy": 1.0, "weights": np.array([0.5, 0.5], dtype="float16")}]
    _evaluate_script()._write_frame_attention(rows, tmp_path, "timit")
    table = pd.read_csv(tmp_path / "frame_attention_timit.csv")
    assert "weights" not in table.columns and table.loc[0, "eff_share"] == 1.0
    with np.load(tmp_path / "frame_attention_timit.npz") as z:
        assert list(z.keys()) == ["S0|SA1"] and z["S0|SA1"].tolist() == [0.5, 0.5]


# ---- config and provenance -------------------------------------------------
CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))


def _compose(experiment):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default", overrides=[f"experiment={experiment}"])
    return OmegaConf.to_container(cfg, resolve=True)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out.update(_flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def _evaluate_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_script", Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("encoder,expected", [
    ({"_target_": "retro_profile.models.WavLMEncoder", "frame_pooling": "attention"}, "attention"),
    ({"_target_": "retro_profile.models.WavLMEncoder"}, "mean"),   # pre-2026-10-11 config
    ({"_target_": "retro_profile.models.encoders.Wav2Vec2FrameEncoder"}, None),  # Gupta
])
def test_summary_meta_records_frame_pooling(encoder, expected):
    cfg = {"model": {"name": "retro_profile", "encoder": encoder}}
    args = argparse.Namespace(unit="speaker", ckpt=Path("runs/x/checkpoints/last.ckpt"))
    meta = _evaluate_script()._run_metadata(cfg, args, nn.Linear(2, 1), {"epoch": 3}, None)
    assert meta["model"]["frame_pooling"] == expected
