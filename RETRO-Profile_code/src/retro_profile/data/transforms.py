"""Train-time waveform augmentations.

Speed perturbation is restricted to a narrow range (0.95-1.05) to avoid biasing
pitch — a wider range hurts height regression because height correlates with
formant frequencies, and speed perturbation shifts those.
"""

from __future__ import annotations

import math
import random

import torch
import torchaudio.functional as AF


class AugmentationPipeline:
    """Composable waveform augmentation pipeline.

    Each augmentation is applied independently with its configured probability.
    Designed to be cheap and safe at training time; not used during eval.
    """

    def __init__(
        self,
        noise_prob: float = 0.3,
        noise_snr_db_range: tuple[float, float] = (10.0, 25.0),
        speed_perturb: tuple[float, float] | None = (0.95, 1.05),
        speed_prob: float = 0.5,
        spec_augment: bool = True,
        spec_time_mask: int = 25,           # max consecutive samples masked (in % of length)
        spec_n_time_masks: int = 2,
        gain_jitter_db: float = 3.0,
    ) -> None:
        self.noise_prob = noise_prob
        self.noise_snr_db_range = noise_snr_db_range
        self.speed_perturb = speed_perturb
        self.speed_prob = speed_prob
        self.spec_augment = spec_augment
        self.spec_time_mask = spec_time_mask
        self.spec_n_time_masks = spec_n_time_masks
        self.gain_jitter_db = gain_jitter_db

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # 1) Speed perturbation - changes both length and pitch slightly
        if self.speed_perturb is not None and random.random() < self.speed_prob:
            lo, hi = self.speed_perturb
            factor = random.uniform(lo, hi)
            waveform = self._speed_perturb(waveform, sample_rate, factor)

        # 2) Additive Gaussian noise at a random SNR
        if random.random() < self.noise_prob:
            snr_db = random.uniform(*self.noise_snr_db_range)
            waveform = self._add_noise(waveform, snr_db)

        # 3) Random gain jitter
        if self.gain_jitter_db > 0:
            db = random.uniform(-self.gain_jitter_db, self.gain_jitter_db)
            waveform = waveform * (10 ** (db / 20))

        # 4) SpecAugment-style masking applied directly on the waveform (zero out spans).
        if self.spec_augment:
            waveform = self._time_mask(waveform)

        return waveform

    @staticmethod
    def _speed_perturb(waveform: torch.Tensor, sample_rate: int, factor: float) -> torch.Tensor:
        """Resample waveform by ``factor`` to mimic tempo+pitch change cheaply."""
        # waveform is 1-D (samples,); AF.resample expects (..., samples)
        new_sr = int(round(sample_rate * factor))
        return AF.resample(waveform.unsqueeze(0), sample_rate, new_sr).squeeze(0)

    @staticmethod
    def _add_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = waveform.pow(2).mean().clamp_min(1e-10)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * noise_power.sqrt()
        return waveform + noise

    def _time_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[-1]
        mask_len_max = max(1, int(n * self.spec_time_mask / 100))
        out = waveform.clone()
        for _ in range(self.spec_n_time_masks):
            length = random.randint(0, mask_len_max)
            if length == 0:
                continue
            start = random.randint(0, max(0, n - length))
            out[..., start : start + length] = 0.0
        return out


class NoOpAugmentation:
    """Identity augmentation, used at val/test time."""

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return waveform


def build_augmentation(cfg: dict | None) -> AugmentationPipeline | NoOpAugmentation:
    """Construct an augmentation pipeline from the data-config ``augmentation`` block."""
    if not cfg or not cfg.get("enabled", False):
        return NoOpAugmentation()
    return AugmentationPipeline(
        noise_prob=cfg.get("noise_prob", 0.3),
        speed_perturb=tuple(cfg["speed_perturb"]) if cfg.get("speed_perturb") else None,
        spec_augment=cfg.get("spec_augment", True),
    )


# Re-export for convenience
__all__ = ["AugmentationPipeline", "NoOpAugmentation", "build_augmentation", "math"]
