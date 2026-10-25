"""Audio I/O helpers shared by every dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


def load_audio(
    path: str | Path,
    target_sr: int = 16_000,
    mono: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load an audio file as a 1-D tensor at ``target_sr``.

    Args:
        path: Path to the audio file (wav / flac / ogg).
        target_sr: Target sample rate. The file is resampled if it differs.
        mono: If True, average multi-channel audio down to mono.
        dtype: Output tensor dtype.

    Returns:
        1-D tensor of shape ``(num_samples,)``.
    """
    waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform).T  # (channels, samples)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)

    return waveform.squeeze(0).to(dtype)


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a 1-D or 2-D tensor (last dim = time)."""
    if orig_sr == target_sr:
        return waveform
    return AF.resample(waveform, orig_sr, target_sr)


def normalize_loudness(waveform: torch.Tensor, target_dbfs: float = -23.0) -> torch.Tensor:
    """Peak-normalize then scale to a target dBFS RMS level.

    Cheap, suitable for training-time normalization. For mastering-grade EBU R128,
    use pyloudnorm — overkill for our purposes.
    """
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    rms = waveform.pow(2).mean().sqrt().clamp_min(1e-8)
    target_rms = 10 ** (target_dbfs / 20)
    return waveform * (target_rms / rms)


def crop_or_pad(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Center-crop or zero-pad a 1-D waveform to exactly ``target_length`` samples."""
    n = waveform.shape[-1]
    if n == target_length:
        return waveform
    if n > target_length:
        start = (n - target_length) // 2
        return waveform[..., start : start + target_length]
    pad = target_length - n
    left = pad // 2
    right = pad - left
    return torch.nn.functional.pad(waveform, (left, right))


def random_crop(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Random-crop a 1-D waveform to ``target_length``; pads if shorter."""
    n = waveform.shape[-1]
    if n <= target_length:
        return crop_or_pad(waveform, target_length)
    start = int(np.random.randint(0, n - target_length + 1))
    return waveform[..., start : start + target_length]
