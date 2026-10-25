"""Tests for the Jaid & AbdulHassan 2023 1D-CNN baseline.

Runs on raw waveforms with no SSL encoder, so these are fast and need no download.
"""

from __future__ import annotations

import pytest
import torch

from retro_profile.data.base import SpeakerBag
from retro_profile.data.collate import collate_bags
from retro_profile.models.baselines.jaid2023 import Jaid2023Baseline
from retro_profile.models.filterbanks import gammatone_filterbank

LABEL_STATS = {"age": {"mean": 30.1, "std": 7.6}, "height": {"mean": 175.5, "std": 9.8}}


def _model(**kw):
    kw.setdefault("label_stats", LABEL_STATS)
    return Jaid2023Baseline(**kw)


def _batch(B=3, K=2, lengths=(4000, 2500)):
    bags = [
        SpeakerBag([torch.randn(lengths[j % len(lengths)]) for j in range(K)],
                   16_000, f"spk_{i}", [f"u{j}" for j in range(K)],
                   {"age": 25.0 + 5 * i, "height": 168.0 + 4 * i})
        for i in range(B)
    ]
    return collate_bags(bags, targets=("age", "height"))


# ---------- architecture ---------------------------------------------------


def test_parameter_count_matches_the_papers_architecture():
    """Compact by design. We get 71,203 against the paper's reported 77,603 --
    the difference is documented in the module docstring (the paper does not
    specify padding, the dense connection, or the gender head's input)."""
    n = sum(p.numel() for p in _model().parameters())
    assert 60_000 < n < 90_000, n           # same order: a ~77 K compact model
    assert n == 71_203                       # pin our exact reading of Tables 1-2


def test_layer_stack_matches_tables_1_and_2():
    m = _model()
    assert m.frontend.out_channels == 64
    assert m.frontend.kernel_size[0] == 100 and m.frontend.stride[0] == 100
    assert len(m.blocks) == 3
    assert [b.shortcut.out_channels for b in m.blocks] == [32, 64, 64]
    for b in m.blocks:                      # Table 2: k=1 shortcut, two k=3 convs
        assert b.shortcut.kernel_size[0] == 1
        assert b.conv_a.kernel_size[0] == 3 and b.conv_b.kernel_size[0] == 3
        assert b.pool.kernel_size == 2
    assert m.avgpool.kernel_size[0] == 3


# ---------- filter-bank initialisation -------------------------------------


def test_frontend_is_initialised_from_the_gammatone_bank():
    m = _model(filterbank="gammatone", n_filters=64, kernel_size=100)
    expected = gammatone_filterbank(64, 100, sample_rate=16_000)
    assert torch.allclose(m.frontend.weight.squeeze(1), expected, atol=1e-6)


def test_frontend_stays_trainable_after_initialisation():
    """The paper's key distinction from SincNet: the bank is a starting point,
    not a constraint."""
    m = _model(filterbank="gammatone")
    assert m.frontend.weight.requires_grad


def test_morlet_and_random_initialisations_differ():
    g = _model(filterbank="gammatone").frontend.weight
    r = _model(filterbank="random").frontend.weight
    mo = _model(filterbank="morlet", n_filters=40).frontend.weight
    assert not torch.allclose(g, r)
    assert mo.shape[0] == 40


# ---------- input conditioning ---------------------------------------------


def test_short_clips_are_repeated_not_zero_padded():
    """Their protocol duplicates short audio; zero padding would feed silence to
    a network that has no masking anywhere in its stack."""
    m = _model(n_samples=48_000)
    wav = torch.arange(1, 101, dtype=torch.float).unsqueeze(0)      # 100 real samples
    padded = torch.zeros(1, 4000)
    padded[:, :100] = wav
    mask = torch.zeros(1, 4000, dtype=torch.bool)
    mask[:, :100] = True
    out = m._fix_length(padded, mask)
    assert out.shape == (1, 48_000)
    assert torch.all(out != 0)                                       # no silence
    assert torch.allclose(out[0, :100], wav[0])                      # starts with the real audio
    assert torch.allclose(out[0, 100:200], wav[0])                   # then repeats it


def test_long_clips_are_truncated():
    m = _model(n_samples=2_400)                                      # stack minimum
    wav = torch.randn(1, 4000)
    mask = torch.ones(1, 4000, dtype=torch.bool)
    out = m._fix_length(wav, mask)
    assert out.shape == (1, 2_400)
    assert torch.allclose(out[0], wav[0, :2_400])


def test_rejects_input_too_short_for_the_stack():
    with pytest.raises(ValueError):
        _model(n_samples=100)


# ---------- forward --------------------------------------------------------


def test_forward_returns_age_height_gender_per_speaker():
    out = _model().eval()(_batch(B=3, K=2))
    assert set(out) == {"age", "height", "gender"}
    for k in out:
        assert out[k].shape == (3,) and torch.isfinite(out[k]).all()


def test_predictions_start_near_the_label_means():
    """Zero-init weights + mean bias, so training starts from the dataset centre
    rather than from 0 (the degenerate ReLU start)."""
    out = _model().eval()(_batch())
    assert torch.allclose(out["age"], torch.full((3,), 30.1), atol=1e-3)
    assert torch.allclose(out["height"], torch.full((3,), 175.5), atol=1e-3)


def test_gender_is_a_logit_not_a_probability():
    """Returned as a logit so the trainer can use AMP-safe BCE-with-logits."""
    out = _model().eval()(_batch())
    gate = torch.sigmoid(out["gender"])
    assert torch.isfinite(out["gender"]).all()
    assert torch.all(gate >= 0) and torch.all(gate <= 1)


def test_per_speaker_aggregation_is_mean_over_the_bag():
    torch.manual_seed(0)
    m = _model().eval()
    wav = torch.randn(4000)
    one = collate_bags([SpeakerBag([wav], 16_000, "s", ["u0"], {"age": 30.0, "height": 170.0})],
                       targets=("age", "height"))
    three = collate_bags([SpeakerBag([wav, wav, wav], 16_000, "s", ["u0", "u1", "u2"],
                                     {"age": 30.0, "height": 170.0})], targets=("age", "height"))
    assert torch.allclose(m(one)["age"], m(three)["age"], atol=1e-4)


def test_a_training_step_updates_the_filterbank_frontend():
    """The bank must actually train (the paper's whole premise).

    Note the regressor's weight is zero-initialised so predictions start at the
    label means; at step 0 the regression path therefore contributes no gradient
    to the trunk (the gender head does), and it bootstraps from step 1. What
    matters is that an optimiser step moves the front end -- assert that directly.
    """
    torch.manual_seed(0)
    m = _model()
    before = m.frontend.weight.detach().clone()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)

    batch = _batch()
    for _ in range(2):                                  # step 0 bootstraps the regressor
        out = m(batch)
        loss = (out["age"] - 40.0).abs().mean() + out["gender"].abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert torch.isfinite(m.frontend.weight).all()
    assert not torch.allclose(before, m.frontend.weight)          # front end moved
    assert m.regressor.weight.abs().sum() > 0                     # bootstrapped off zero


def test_regression_path_feeds_the_trunk_once_bootstrapped():
    """After one update the regressor is non-zero, so the age/height loss alone
    then propagates gradient all the way back to the filter bank."""
    torch.manual_seed(0)
    m = _model()
    with torch.no_grad():                                # simulate a completed step
        m.regressor.weight.normal_(std=0.01)
    m(_batch())["age"].sum().backward()
    g = m.frontend.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_embed_speaker_not_implemented():
    with pytest.raises(NotImplementedError):
        _model().embed_speaker(_batch())


# ---------- config composition ---------------------------------------------


def _cfg(experiment: str):
    import os

    from hydra import compose, initialize_config_dir
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="default", overrides=[f"experiment={experiment}"])


def test_jaid_config_builds_the_intended_model():
    from retro_profile.utils import instantiate
    cfg = _cfg("jaid2023")
    m = instantiate(cfg.model)
    assert isinstance(m, Jaid2023Baseline)
    assert m.filterbank == "gammatone"
    assert m.frontend.out_channels == 64
    assert m.n_samples == 48_000


def test_morlet_variant_uses_forty_wavelet_filters():
    from retro_profile.utils import instantiate
    m = instantiate(_cfg("jaid2023_morlet").model)
    assert m.filterbank == "morlet"
    assert m.frontend.out_channels == 40          # paper: 40 scales for the CWT bank


def test_jaid_training_recipe_matches_the_paper():
    """MAE + plain sum (not uncertainty weighting), per-utterance, 3 s, batch 64."""
    cfg = _cfg("jaid2023")
    assert cfg.training.lit_module == "multitask"
    assert cfg.training.baseline.regression_kind == "l1"
    assert cfg.training.baseline.loss_combination == "sum"
    assert int(cfg.training.max_epochs) == 200
    assert int(cfg.data.batch_size) == 64
    assert int(cfg.data.max_audio_seconds) == 3
    assert cfg.data.train_unit == "utterance"
    assert "gender" in list(cfg.data.targets)     # gender supervised as an aux task


def test_audio_budget_and_model_input_length_agree():
    """max_audio_seconds * sample_rate must equal the model's fixed n_samples,
    otherwise the model would silently repeat or discard audio."""
    cfg = _cfg("jaid2023")
    assert int(cfg.data.max_audio_seconds) * int(cfg.model.sample_rate) == int(cfg.model.n_samples)
