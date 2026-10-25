"""Lightning DataModules for each speaker corpus.

Each DataModule owns the train/val/test ``BaseSpeakerDataset`` instances and the
bag-aware collate. We register one DataModule per corpus so configs can refer
to them by name (e.g. ``_target_: retro_profile.data.TimitDataModule``).

Why a registry as well as Hydra ``_target_``? Because some scripts (build_datastore,
evaluate.py) need to look up DataModules by short name without a Hydra config.
"""

from __future__ import annotations

from typing import Any, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Sampler

from retro_profile.data.base import BaseSpeakerDataset
from retro_profile.data.collate import make_collate_fn
from retro_profile.data.combined import CombinedSpeakerDataset
from retro_profile.data.heightceleb import HeightCelebDataset
from retro_profile.data.nisp import NispDataset
from retro_profile.data.sampler import PerSpeakerUtteranceSampler
from retro_profile.data.timit import TimitDataset
from retro_profile.data.transforms import build_augmentation
from retro_profile.utils.registry import Registry

datamodules = Registry("datamodules")


class _SpeakerDataModule(pl.LightningDataModule):
    """Generic DataModule that wraps a ``BaseSpeakerDataset`` subclass.

    Subclasses set ``DATASET_CLS``; the rest of the wiring is identical.
    """

    DATASET_CLS: type[BaseSpeakerDataset]

    def __init__(
        self,
        root: str,
        sample_rate: int = 16_000,
        max_audio_seconds: float = 8.0,
        min_audio_seconds: float = 0.5,
        bag_size: int = 8,
        batch_size: int = 16,
        num_workers: int = 4,
        targets: Sequence[str] = ("age", "height"),
        normalize_targets: bool = True,
        augmentation: dict[str, Any] | None = None,
        splits: dict[str, str] | None = None,
        train_unit: str = "bag",
        utts_per_speaker: int | None = None,
        eval_fixed_clips: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["augmentation"])
        if train_unit not in {"bag", "utterance"}:
            raise ValueError(f"train_unit must be 'bag' or 'utterance', got {train_unit!r}")
        # Utterance-level training only: draw this many fresh clips per speaker each epoch
        # instead of every clip (data/sampler.py). null = every clip, the TIMIT recipes.
        if utts_per_speaker is not None and train_unit != "utterance":
            raise ValueError("utts_per_speaker applies to train_unit 'utterance' only; a bag "
                             "already holds bag_size clips per speaker")
        self.train_unit = train_unit
        self.utts_per_speaker = utts_per_speaker
        # Validation, test and per-clip loaders score the SAME clips and 8 s windows on every pass and for every
        # model (BaseSpeakerDataset.fixed_clips). Training keeps its fresh random draw. Off by default; on in
        # configs/data/heightceleb_split2.yaml (Table 4, 2026-10-17).
        self.eval_fixed_clips = eval_fixed_clips
        self.root = root
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.min_audio_seconds = min_audio_seconds
        self.bag_size = bag_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.targets = list(targets)
        self.normalize_targets = normalize_targets
        self.aug_cfg = augmentation
        self.splits = splits or {"train": "train", "val": "dev", "test": "test"}

        self.train_set: BaseSpeakerDataset | None = None
        self.val_set: BaseSpeakerDataset | None = None
        self.test_set: BaseSpeakerDataset | None = None
        self.label_stats: dict[str, dict[str, float]] = {}

    # ---- LightningDataModule hooks -----------------------------------------
    def setup(self, stage: str | None = None) -> None:
        train_aug = build_augmentation(self.aug_cfg)
        eval_aug = build_augmentation(None)

        common = dict(
            root=self.root,
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
            min_audio_seconds=self.min_audio_seconds,
            bag_size=self.bag_size,
            targets=self.targets,
        )
        if stage in (None, "fit"):
            # Training unit: 'bag' (per-speaker, MIL) or 'utterance' (per-clip,
            # e.g. the faithful Gupta 2022 per-utterance training). Eval stays bag.
            self.train_set = self.DATASET_CLS(
                split=self.splits["train"], mode=self.train_unit, transforms=train_aug, **common
            )
            self.val_set = self.DATASET_CLS(
                split=self.splits["val"], mode="bag", transforms=eval_aug,
                fixed_clips=self.eval_fixed_clips, **common
            )
            # Label statistics computed from train speakers only.
            self.label_stats = self.train_set.get_label_stats()
        if stage in (None, "test", "predict"):
            self.test_set = self.DATASET_CLS(
                split=self.splits["test"], mode="bag", transforms=eval_aug,
                fixed_clips=self.eval_fixed_clips, **common
            )

    # ---- DataLoaders -------------------------------------------------------
    def _loader(self, ds: BaseSpeakerDataset, shuffle: bool, unit: str = "bag",
                sampler: Sampler[int] | None = None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=shuffle,
            collate_fn=make_collate_fn(self.targets, mode=unit),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None, "Call setup() first."
        sampler = None
        if self.utts_per_speaker:
            sampler = PerSpeakerUtteranceSampler(self.train_set.manifest["speaker_id"].tolist(),
                                                 self.utts_per_speaker)
        return self._loader(self.train_set, shuffle=True, unit=self.train_unit, sampler=sampler)

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None, "Call setup() first."
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None, "Call setup() first."
        return self._loader(self.test_set, shuffle=False)

    # ---- For datastore building -------------------------------------------
    def utterance_loader(self, split: str = "train") -> DataLoader:
        """Yield one *utterance* per sample. Used by build_datastore.py."""
        ds_split = self.splits.get(split, split)
        ds = self.DATASET_CLS(
            split=ds_split,
            mode="utterance",
            transforms=build_augmentation(None),
            root=self.root,
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
            min_audio_seconds=self.min_audio_seconds,
            bag_size=self.bag_size,
            targets=self.targets,
            fixed_clips=self.eval_fixed_clips,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.targets, mode="utterance"),
        )


@datamodules.register("timit")
class TimitDataModule(_SpeakerDataModule):
    DATASET_CLS = TimitDataset


@datamodules.register("nisp")
class NispDataModule(_SpeakerDataModule):
    DATASET_CLS = NispDataset

    def __init__(self, languages: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.languages = languages

    def setup(self, stage: str | None = None) -> None:
        # Override to forward the languages filter to the dataset constructor.
        train_aug = build_augmentation(self.aug_cfg)
        eval_aug = build_augmentation(None)
        common = dict(
            root=self.root,
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
            min_audio_seconds=self.min_audio_seconds,
            bag_size=self.bag_size,
            targets=self.targets,
            languages=self.languages,
        )
        if stage in (None, "fit"):
            # mode=self.train_unit, NOT a hardcoded "bag". This override exists
            # only to forward `languages`, and was copied from the base class
            # with the mode frozen -- which silently ignored
            # `data.train_unit: utterance` and handed SpeakerBag objects to a
            # collate function expecting single utterances:
            #   AttributeError: 'SpeakerBag' object has no attribute 'waveform'
            # It killed both Gupta runs on NISP (2026-09-13). Eval stays bag.
            self.train_set = self.DATASET_CLS(
                split=self.splits["train"], mode=self.train_unit,
                transforms=train_aug, **common
            )
            self.val_set = self.DATASET_CLS(
                split=self.splits["val"], mode="bag", transforms=eval_aug,
                fixed_clips=self.eval_fixed_clips, **common
            )
            self.label_stats = self.train_set.get_label_stats()
        if stage in (None, "test", "predict"):
            self.test_set = self.DATASET_CLS(
                split=self.splits["test"], mode="bag", transforms=eval_aug,
                fixed_clips=self.eval_fixed_clips, **common
            )

    def utterance_loader(self, split: str = "train") -> DataLoader:
        ds_split = self.splits.get(split, split)
        ds = self.DATASET_CLS(
            split=ds_split,
            mode="utterance",
            transforms=build_augmentation(None),
            root=self.root,
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
            min_audio_seconds=self.min_audio_seconds,
            bag_size=self.bag_size,
            targets=self.targets,
            languages=self.languages,
            fixed_clips=self.eval_fixed_clips,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.targets, mode="utterance"),
        )


@datamodules.register("heightceleb")
class HeightCelebDataModule(_SpeakerDataModule):
    DATASET_CLS = HeightCelebDataset


# HeightCeleb's 60/20/20 grouped split (prepare_heightceleb.py --split-mode grouped ->
# data/heightceleb_split2): the same corpus with other manifests. Its own name lets
# `evaluate.py --eval heightceleb_split2` score that split's test set (configs/data/heightceleb_split2.yaml).
datamodules.register("heightceleb_split2")(HeightCelebDataModule)


CORPUS_DATASETS: dict[str, type[BaseSpeakerDataset]] = {
    "timit": TimitDataset, "nisp": NispDataset, "heightceleb": HeightCelebDataset,
}


@datamodules.register("multi")
class MultiCorpusDataModule(_SpeakerDataModule):
    """Several corpora at once (e.g. HeightCeleb + NISP, tested on TIMIT).

    ``corpora`` maps a corpus name to its ``root``, the split that fills each role
    (``splits: {train, val, test}``; a missing or null role takes none of its speakers) and any
    loader option (NISP's ``languages``). Budget, targets and augmentation are shared.
    """

    ROLES = {"train": "train", "val": "val", "dev": "val", "test": "test"}

    def __init__(self, corpora: dict[str, dict[str, Any]], root: str = ".", **kwargs: Any) -> None:
        unknown = sorted(set(corpora) - set(CORPUS_DATASETS))
        if unknown:
            raise ValueError(f"unknown corpora {unknown}; known: {sorted(CORPUS_DATASETS)}")
        super().__init__(root=root, **kwargs)
        self.corpora = {name: dict(spec) for name, spec in corpora.items()}

    def _dataset(self, role: str, mode: str, transforms: Any, fixed_clips: bool = False) -> CombinedSpeakerDataset:
        common = dict(
            sample_rate=self.sample_rate,
            max_audio_seconds=self.max_audio_seconds,
            min_audio_seconds=self.min_audio_seconds,
            bag_size=self.bag_size,
            targets=self.targets,
        )
        parts: dict[str, BaseSpeakerDataset] = {}
        for name, spec in self.corpora.items():
            split = (spec.get("splits") or {}).get(role)
            if split:
                extra = {k: v for k, v in spec.items() if k not in ("root", "splits")}
                parts[name] = CORPUS_DATASETS[name](root=spec["root"], split=split, **extra, **common)
        if not parts:
            raise ValueError(f"no corpus supplies the {role!r} role: {self.corpora}")
        return CombinedSpeakerDataset(parts, mode=mode, transforms=transforms, fixed_clips=fixed_clips, **common)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_set = self._dataset("train", self.train_unit, build_augmentation(self.aug_cfg))
            self.val_set = self._dataset("val", "bag", build_augmentation(None), self.eval_fixed_clips)
            self.label_stats = self.train_set.get_label_stats()
        if stage in (None, "test", "predict"):
            self.test_set = self._dataset("test", "bag", build_augmentation(None), self.eval_fixed_clips)

    def utterance_loader(self, split: str = "train") -> DataLoader:
        if split not in self.ROLES:
            raise ValueError(f"split must be one of {sorted(self.ROLES)}; got {split!r}")
        ds = self._dataset(self.ROLES[split], "utterance", build_augmentation(None), self.eval_fixed_clips)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=make_collate_fn(self.targets, mode="utterance"),
        )


def build_datamodule(name: str, **kwargs: Any) -> _SpeakerDataModule:
    """Build a DataModule by short name (used by non-Hydra scripts)."""
    return datamodules.build(name, **kwargs)
