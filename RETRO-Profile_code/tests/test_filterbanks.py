"""Tests for the auditory filter banks (Jaid & AbdulHassan 2023 initialisation).

These check the *signal-processing* properties, not just shapes -- a bank with the
right shape but the wrong frequency content would silently produce a baseline that
looks implemented but isn't.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from retro_profile.models.filterbanks import (
    build_filterbank,
    erb_bandwidth,
    erb_spaced_frequencies,
    erb_to_hz,
    gammatone_filterbank,
    hz_to_erb,
    morlet_filterbank,
    morlet_scales,
)

# ---------- ERB scale ------------------------------------------------------


def test_erb_conversions_roundtrip():
    freqs = np.array([100.0, 500.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(erb_to_hz(hz_to_erb(freqs)), freqs, rtol=1e-6)


def test_erb_spacing_is_monotonic_and_spans_the_band():
    f = erb_spaced_frequencies(64, 100.0, 8000.0)
    assert f.shape == (64,)
    assert np.all(np.diff(f) > 0)                       # strictly increasing
    assert np.isclose(f[0], 100.0) and np.isclose(f[-1], 8000.0)


def test_erb_spacing_is_denser_at_low_frequencies():
    """The whole point of the ERB scale: fine resolution low, coarse high."""
    f = erb_spaced_frequencies(32, 100.0, 8000.0)
    gaps = np.diff(f)
    assert gaps[0] < gaps[-1] / 5                       # far denser at the bottom


def test_erb_bandwidth_grows_with_frequency():
    assert erb_bandwidth(200.0) < erb_bandwidth(2000.0) < erb_bandwidth(8000.0)


def test_erb_spacing_rejects_bad_range():
    with pytest.raises(ValueError):
        erb_spaced_frequencies(8, 8000.0, 100.0)        # f_min > f_max


# ---------- gammatone ------------------------------------------------------


def test_gammatone_shape_and_dtype():
    fb = gammatone_filterbank(n_filters=64, length=100)
    assert fb.shape == (64, 100)
    assert fb.dtype == torch.float32
    assert torch.isfinite(fb).all()


def test_gammatone_filters_are_unit_norm():
    fb = gammatone_filterbank(n_filters=16, length=100)
    assert torch.allclose(fb.norm(dim=1), torch.ones(16), atol=1e-5)


def test_gammatone_peak_frequency_tracks_centre_frequency():
    """Each filter must actually be band-pass at its own centre frequency."""
    sr, length, n = 16_000, 400, 8                      # long kernel -> sharp FFT
    fb = gammatone_filterbank(n_filters=n, length=length, sample_rate=sr,
                              f_min=200.0, f_max=4000.0)
    centers = erb_spaced_frequencies(n, 200.0, 4000.0)
    freqs = np.fft.rfftfreq(length, d=1.0 / sr)
    peaks = [freqs[int(np.argmax(np.abs(np.fft.rfft(fb[i].numpy()))))] for i in range(n)]
    assert all(np.diff(peaks) >= 0)                     # ordered like the centres
    # each spectral peak should land near its design centre frequency
    for peak, fc in zip(peaks, centers, strict=True):
        assert abs(peak - fc) < max(0.35 * fc, 100.0), (peak, fc)


def test_gammatone_envelope_peaks_earlier_for_higher_frequencies():
    """The envelope t^(p-1)*exp(-2*pi*b*t) peaks at t=(p-1)/(2*pi*b), and b grows
    with centre frequency -- so high-frequency filters peak (and decay) sooner.
    """
    fb = gammatone_filterbank(n_filters=4, length=200, f_min=300.0, f_max=3000.0)
    peaks = [int(f.abs().argmax()) for f in fb]
    assert peaks == sorted(peaks, reverse=True), peaks      # 300Hz latest, 3kHz earliest


def test_gammatone_decays_after_its_peak():
    """Once past the envelope peak the response dies away."""
    fb = gammatone_filterbank(n_filters=1, length=200, f_min=3000.0, f_max=3000.0)
    f = fb[0].abs()
    peak = int(f.argmax())
    assert peak < 100                                       # 3 kHz peaks early
    assert f[peak:peak + 20].sum() > 10 * f[-20:].sum()     # decayed by the end


def test_short_kernels_truncate_low_frequency_gammatones():
    """Documents a real property of the paper's setting (length=100 @ 16 kHz =
    6.25 ms): the lowest-frequency filters peak *beyond* the kernel, so they are
    captured mid-rise rather than as complete impulse responses."""
    fb = gammatone_filterbank(n_filters=64, length=100, f_min=100.0, f_max=8000.0)
    assert int(fb[0].abs().argmax()) > 90                   # still rising at the end
    assert int(fb[-1].abs().argmax()) < 10                  # top filter peaks at once


# ---------- morlet ---------------------------------------------------------


def test_morlet_shape_and_dtype():
    fb = morlet_filterbank(n_filters=40, length=100)
    assert fb.shape == (40, 100)
    assert fb.dtype == torch.float32
    assert torch.isfinite(fb).all()


def test_morlet_filters_are_zero_mean_admissible():
    """Wavelet admissibility: each filter must integrate to ~zero."""
    fb = morlet_filterbank(n_filters=40, length=100)
    assert torch.allclose(fb.mean(dim=1), torch.zeros(40), atol=1e-6)


def test_morlet_scales_are_fractional_powers_of_two():
    s = morlet_scales(n_filters=8, signal_length=100)
    assert s.shape == (8,)
    assert np.all(np.diff(s) > 0)                       # increasing scale
    ratios = s[1:] / s[:-1]
    assert np.allclose(ratios, ratios[0])               # geometric progression


def test_morlet_larger_scale_means_lower_frequency():
    """Higher scales stretch the wavelet -> lower centre frequency."""
    sr, length = 16_000, 512
    fb = morlet_filterbank(n_filters=6, length=length)
    freqs = np.fft.rfftfreq(length, d=1.0 / sr)
    peaks = [freqs[int(np.argmax(np.abs(np.fft.rfft(fb[i].numpy()))))] for i in range(6)]
    assert peaks[0] > peaks[-1]                         # scale up -> frequency down


# ---------- factory --------------------------------------------------------


def test_build_filterbank_dispatches():
    assert build_filterbank("gammatone", 64, 100).shape == (64, 100)
    assert build_filterbank("morlet", 40, 100).shape == (40, 100)


def test_build_filterbank_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_filterbank("sincnet", 64, 100)


def test_banks_are_not_degenerate():
    """Filters must differ from one another -- a bank of identical filters would
    train like a single filter and silently defeat the initialisation."""
    for name, n in (("gammatone", 64), ("morlet", 40)):
        fb = build_filterbank(name, n, 100)
        assert fb.std(dim=1).min() > 1e-6               # none is constant
        assert not torch.allclose(fb[0], fb[n // 2], atol=1e-3)
