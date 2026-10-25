"""Base dataset abstractions used by every speaker-profile corpus.

Speech-trait regression has a *bag-of-utterances* structure that pure utterance-
indexed datasets ignore. We expose two indexing modes:

    * ``SpeakerBag``  — one item per speaker, returning K utterances drawn from
      that speaker. Used during training so that MIL pooling sees the bag.
    * ``SpeakerExample`` — one item per utterance. Used at inference when we need
      to embed every utterance once.
"""

from __future__ import annotations

import abc
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from retro_profile.utils.audio import load_audio, random_crop


@dataclass(slots=True)
class SpeakerExample:
    """A single utterance with its speaker-level labels."""

    waveform: torch.Tensor          # 1-D float tensor (samples,)
    sample_rate: int
    speaker_id: str
    utt_id: str
    labels: dict[str, float]        # e.g. {"age": 32.0, "height": 178.0}
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpeakerBag:
    """A bag of utterances from a single speaker."""

    waveforms: list[torch.Tensor]   # K x (samples_i,) — variable length
    sample_rate: int
    speaker_id: str
    utt_ids: list[str]
    labels: dict[str, float]        # speaker-level labels are constant across the bag
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.waveforms)


SPLIT_FILES = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}


def _stable_seed(text: str) -> int:
    """A seed that depends on ``text`` only -- the same in every process and run (unlike ``hash``)."""
    return zlib.crc32(text.encode("utf-8"))


def read_split_manifests(root: Path, split: str, corpus: str, prepare_script: str) -> pd.DataFrame:
    """Read ``<root>/manifests/<split>.csv``. ``split`` may pool several official splits joined with
    '+' (e.g. "train+test" when another corpus is the test set); the parts must be speaker-disjoint."""
    parts = [p.strip() for p in str(split).split("+")]
    if any(p not in SPLIT_FILES for p in parts) or len(set(parts)) != len(parts):
        raise ValueError(f"{corpus} split must be one of {list(SPLIT_FILES)} or several joined "
                         f"with '+' (e.g. 'train+test'); got {split!r}")
    frames, seen = [], set()
    for part in parts:
        path = Path(root) / "manifests" / SPLIT_FILES[part]
        if not path.exists():
            raise FileNotFoundError(f"{corpus} manifest not found at {path}. Run {prepare_script} first.")
        f = pd.read_csv(path)
        spk = set(f["speaker_id"].astype(str))
        if spk & seen:
            raise ValueError(f"{corpus} split {part!r} shares speakers with another part of "
                             f"{split!r}: {sorted(spk & seen)[:5]}")
        seen |= spk
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


class BaseSpeakerDataset(Dataset, abc.ABC):
    """Common base for TIMIT / NISP / HeightCeleb.

    Concrete subclasses must populate ``self.manifest``, a ``pandas.DataFrame``
    with at minimum the columns:

        speaker_id (str), utt_id (str), audio_path (str), <target columns ...>

    The dataset can be indexed in two ways:

        * ``mode='bag'``     -> returns ``SpeakerBag`` (one per speaker)
        * ``mode='utterance'`` -> returns ``SpeakerExample`` (one per row)
    """

    REQUIRED_COLUMNS = ("speaker_id", "utt_id", "audio_path")
    # Categorical -> numeric mapping so 'gender' can be used as a target
    # (the Gupta 2022 baseline supervises gender). Manifests store 'M'/'F'.
    GENDER_MAP = {"M": 0.0, "F": 1.0}

    def __init__(
        self,
        root: str | Path,
        split: str,
        targets: Sequence[str] = ("age", "height"),
        sample_rate: int = 16_000,
        max_audio_seconds: float = 8.0,
        min_audio_seconds: float = 0.5,
        bag_size: int = 8,
        mode: str = "bag",
        transforms: Any | None = None,
        fixed_clips: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.targets = list(targets)
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.min_samples = int(min_audio_seconds * sample_rate)
        self.bag_size = bag_size
        if mode not in {"bag", "utterance"}:
            raise ValueError(f"mode must be 'bag' or 'utterance', got {mode!r}")
        self.mode = mode
        self.transforms = transforms
        # Scoring only (2026-10-17): the same clips and the same 8 s window every time, for every model. A speaker
        # with more than bag_size clips otherwise gets a fresh random draw on each pass, and a long clip a fresh
        # random crop -- harmless on TIMIT (10 short clips each) but on HeightCeleb (median ~100 clips, 35 % longer
        # than 8 s) two models, or two passes over one model, would be scored on different audio.
        self.fixed_clips = fixed_clips

        self.manifest: pd.DataFrame = self._load_manifest()
        self._validate_manifest()
        self.speakers: list[str] = sorted(self.manifest["speaker_id"].unique().tolist())
        self._by_speaker: dict[str, pd.DataFrame] = {
            sid: g.reset_index(drop=True) for sid, g in self.manifest.groupby("speaker_id")
        }

    # ---- subclass hooks -----------------------------------------------------
    @abc.abstractmethod
    def _load_manifest(self) -> pd.DataFrame:
        """Return a manifest DataFrame for ``self.split``."""

    # ---- introspection ------------------------------------------------------
    @property
    def num_speakers(self) -> int:
        return len(self.speakers)

    def get_label_stats(self) -> dict[str, dict[str, float]]:
        """Per-target mean/std over speakers (not utterances) — used for normalization."""
        stats: dict[str, dict[str, float]] = {}
        # one row per speaker
        unique = self.manifest.drop_duplicates(subset="speaker_id")
        for t in self.targets:
            if t not in unique.columns or t == "gender":
                continue                              # gender is categorical, never z-scored
            vals = unique[t].dropna().to_numpy(dtype=float)
            stats[t] = {
                "mean": float(vals.mean()),
                "std": float(vals.std() + 1e-8),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "count": int(vals.size),
            }
        return stats

    # ---- pytorch hooks ------------------------------------------------------
    def __len__(self) -> int:
        return len(self.speakers) if self.mode == "bag" else len(self.manifest)

    def __getitem__(self, idx: int) -> SpeakerBag | SpeakerExample:
        return self._get_bag(idx) if self.mode == "bag" else self._get_example(idx)

    # ---- concrete loaders ---------------------------------------------------
    def _load_and_trim(self, path: str | Path) -> torch.Tensor:
        wav = load_audio(path, target_sr=self.sample_rate)
        if wav.shape[-1] > self.max_samples:
            if self.fixed_clips:                    # a window fixed by the file path
                start = int(np.random.default_rng(_stable_seed(str(path))).integers(
                    0, wav.shape[-1] - self.max_samples + 1))
                wav = wav[..., start:start + self.max_samples]
            else:
                wav = random_crop(wav, self.max_samples)
        return wav

    def _row_labels(self, row: pd.Series) -> dict[str, float]:
        labels: dict[str, float] = {}
        for t in self.targets:
            if t not in row or pd.isna(row[t]):
                continue
            val = row[t]
            if t == "gender" and isinstance(val, str):
                mapped = self.GENDER_MAP.get(val.strip().upper())
                if mapped is not None:              # skip unknown gender strings
                    labels[t] = mapped
            else:
                labels[t] = float(val)              # numeric targets (incl. 0/1 gender)
        return labels

    def _get_example(self, idx: int) -> SpeakerExample:
        row = self.manifest.iloc[idx]
        wav = self._load_and_trim(row["audio_path"])
        if self.transforms is not None:
            wav = self.transforms(wav, self.sample_rate)
        return SpeakerExample(
            waveform=wav,
            sample_rate=self.sample_rate,
            speaker_id=str(row["speaker_id"]),
            utt_id=str(row["utt_id"]),
            labels=self._row_labels(row),
        )

    def _get_bag(self, idx: int) -> SpeakerBag:
        sid = self.speakers[idx]
        utts = self._by_speaker[sid]
        if len(utts) > self.bag_size:
            seed = _stable_seed(str(sid)) if self.fixed_clips else None     # fixed: the same clips every pass
            utts = utts.sample(n=self.bag_size, random_state=seed).reset_index(drop=True)
        waveforms = [self._load_and_trim(p) for p in utts["audio_path"]]
        if self.transforms is not None:
            waveforms = [self.transforms(w, self.sample_rate) for w in waveforms]
        first = utts.iloc[0]
        return SpeakerBag(
            waveforms=waveforms,
            sample_rate=self.sample_rate,
            speaker_id=str(sid),
            utt_ids=utts["utt_id"].astype(str).tolist(),
            labels=self._row_labels(first),
        )

    # ---- validation ---------------------------------------------------------
    def _validate_manifest(self) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self.manifest.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        if self.manifest.empty:
            raise RuntimeError(
                f"Manifest for {type(self).__name__}({self.split}) is empty. "
                f"Did you run the corresponding scripts/prepare_*.py?"
            )
