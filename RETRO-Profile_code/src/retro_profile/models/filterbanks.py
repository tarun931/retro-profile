"""Auditory filter banks used to initialise the first conv layer of a 1D CNN.

Implements the two initialisation schemes from Jaid & AbdulHassan 2023,
"End-to-End Speaker Profiling Using 1D CNN Architectures and Filter Bank
Initialization" (iJOE 19(10), 2023):

  * ``gammatone_filterbank`` -- 64 filters whose centre frequencies are spaced
    on the ERB scale between 100 Hz and the Nyquist frequency (their Sec. 2.3),
  * ``morlet_filterbank``    -- 40 scaled Morlet wavelets, with the scales taken
    as fractional powers of two (their Eq. 7).

Unlike SincNet-style layers -- which *constrain* the layer to a parametric filter
for the whole of training -- these banks only provide the **initial** weights; the
convolution is then trained freely. That is the paper's central claim: a middle
ground between handcrafted features and fully learned representations.

Each function returns a real-valued ``(n_filters, length)`` float32 tensor, ready
to be copied into a ``nn.Conv1d`` weight of shape ``(n_filters, 1, length)``.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# ERB helpers (Glasberg & Moore scale, as used by the paper's Eq. 4)
# ---------------------------------------------------------------------------
def hz_to_erb(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """ERB-rate of a frequency in Hz."""
    return 21.4 * np.log10(1.0 + 0.00437 * np.asarray(freq_hz, dtype=np.float64))


def erb_to_hz(erb: np.ndarray | float) -> np.ndarray | float:
    """Inverse of :func:`hz_to_erb`."""
    return (10.0 ** (np.asarray(erb, dtype=np.float64) / 21.4) - 1.0) / 0.00437


def erb_bandwidth(center_hz: np.ndarray | float) -> np.ndarray | float:
    """Equivalent rectangular bandwidth of a filter centred at ``center_hz``.

    The paper's Eq. 4: ERB(f_c) = 24.7 + 0.108 * f_c  (equivalently 24.7*(4.37f/1000 + 1)).
    """
    return 24.7 + 0.108 * np.asarray(center_hz, dtype=np.float64)


def erb_spaced_frequencies(n_filters: int, f_min: float, f_max: float) -> np.ndarray:
    """``n_filters`` centre frequencies spaced equally on the ERB scale."""
    if n_filters < 1:
        raise ValueError("n_filters must be >= 1")
    if f_min <= 0:
        raise ValueError(f"f_min must be > 0, got {f_min}")
    # A single filter may sit at one frequency; several must span a real range,
    # otherwise the bank would be a set of identical (degenerate) filters.
    if f_max < f_min or (n_filters > 1 and f_max == f_min):
        raise ValueError(f"require f_min < f_max for n_filters>1, got {f_min} and {f_max}")
    erbs = np.linspace(hz_to_erb(f_min), hz_to_erb(f_max), n_filters)
    return np.asarray(erb_to_hz(erbs), dtype=np.float64)


# ---------------------------------------------------------------------------
# Gammatone
# ---------------------------------------------------------------------------
def gammatone_filterbank(
    n_filters: int = 64,
    length: int = 100,
    sample_rate: int = 16_000,
    f_min: float = 100.0,
    f_max: float | None = None,
    order: int = 4,
    phase: float = 0.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Gammatone impulse responses, one per ERB-spaced centre frequency.

    g(t) = t^(order-1) * exp(-2*pi*b*t) * cos(2*pi*f_c*t + phase)

    where the bandwidth ``b`` is 1.019 * ERB(f_c) -- the standard human-auditory
    fit. ``f_max`` defaults to the Nyquist frequency, as in the paper.

    Returns:
        (n_filters, length) float32 tensor.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if f_max is None:
        f_max = sample_rate / 2.0

    centers = erb_spaced_frequencies(n_filters, f_min, f_max)      # (n,)
    bandwidths = 1.019 * erb_bandwidth(centers)                    # (n,)

    t = np.arange(length, dtype=np.float64) / sample_rate          # (length,)
    t = t[None, :]                                                 # (1, length)
    fc = centers[:, None]
    b = bandwidths[:, None]

    envelope = t ** (order - 1) * np.exp(-2.0 * np.pi * b * t)
    filters = envelope * np.cos(2.0 * np.pi * fc * t + phase)      # (n, length)

    if normalize:
        filters = _unit_norm(filters)
    return torch.from_numpy(filters.astype(np.float32))


# ---------------------------------------------------------------------------
# Morlet (continuous wavelet transform)
# ---------------------------------------------------------------------------
def morlet_scales(
    n_filters: int, signal_length: int, dt: float = 1.0, s0: float | None = None,
    dj: float | None = None,
) -> np.ndarray:
    """Wavelet scales as fractional powers of two (the paper's Eq. 7).

    ``s_j = s0 * 2^(j * dj)`` for j = 0..n_filters-1. ``s0`` defaults to ``2*dt``
    (the smallest resolvable scale) and ``dj`` is chosen so the largest scale
    spans the signal, mirroring their Eq. 8.
    """
    if n_filters < 1:
        raise ValueError("n_filters must be >= 1")
    if s0 is None:
        s0 = 2.0 * dt
    if dj is None:
        if n_filters == 1:
            dj = 0.0
        else:
            j_max = np.log2(max(signal_length, 2) * dt / s0)
            dj = j_max / (n_filters - 1)
    j = np.arange(n_filters, dtype=np.float64)
    return s0 * 2.0 ** (j * dj)


def morlet_filterbank(
    n_filters: int = 40,
    length: int = 100,
    omega0: float = 6.0,
    dt: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Real parts of ``n_filters`` scaled Morlet wavelets.

    psi(t) = pi^(-1/4) * exp(i*omega0*t) * exp(-t^2/2), evaluated at t/s for each
    scale s (their Eq. 6). We take the real part so the bank can initialise a
    real-valued Conv1d. ``omega0`` defaults to 6, as the paper specifies.

    Returns:
        (n_filters, length) float32 tensor.
    """
    if length < 1:
        raise ValueError("length must be >= 1")

    scales = morlet_scales(n_filters, length, dt=dt)               # (n,)
    # Centre the time axis so each wavelet sits in the middle of the kernel.
    t = (np.arange(length, dtype=np.float64) - (length - 1) / 2.0) * dt
    t_scaled = t[None, :] / scales[:, None]                        # (n, length)

    gauss = np.exp(-0.5 * t_scaled ** 2)
    filters = np.pi ** -0.25 * np.cos(omega0 * t_scaled) * gauss   # real part

    # Admissibility: a wavelet must integrate to ~0. The finite window leaves a
    # small DC offset at large scales, so remove it explicitly.
    filters = filters - filters.mean(axis=1, keepdims=True)

    if normalize:
        filters = _unit_norm(filters)
    return torch.from_numpy(filters.astype(np.float32))


# ---------------------------------------------------------------------------
def _unit_norm(filters: np.ndarray) -> np.ndarray:
    """Scale each filter to unit L2 norm (keeps activations comparable)."""
    norms = np.linalg.norm(filters, axis=1, keepdims=True)
    return filters / np.clip(norms, 1e-12, None)


def build_filterbank(name: str, n_filters: int, length: int, **kwargs) -> torch.Tensor:
    """Build a filter bank by name: 'gammatone' | 'morlet'."""
    if name == "gammatone":
        return gammatone_filterbank(n_filters=n_filters, length=length, **kwargs)
    if name == "morlet":
        return morlet_filterbank(n_filters=n_filters, length=length, **kwargs)
    raise ValueError(f"Unknown filter bank {name!r}; expected 'gammatone' or 'morlet'.")
