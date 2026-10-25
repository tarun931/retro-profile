"""Data layer: datasets, augmentation, bag-aware collation, DataModules."""

from retro_profile.data.base import BaseSpeakerDataset, SpeakerBag, SpeakerExample
from retro_profile.data.collate import BagBatch, collate_bags, collate_examples, make_collate_fn
from retro_profile.data.combined import CombinedSpeakerDataset
from retro_profile.data.heightceleb import HeightCelebDataset
from retro_profile.data.nisp import NispDataset
from retro_profile.data.registry import (
    HeightCelebDataModule,
    MultiCorpusDataModule,
    NispDataModule,
    TimitDataModule,
    build_datamodule,
    datamodules,
)
from retro_profile.data.timit import TimitDataset
from retro_profile.data.transforms import AugmentationPipeline, NoOpAugmentation, build_augmentation

__all__ = [
    "BaseSpeakerDataset",
    "SpeakerBag",
    "SpeakerExample",
    "BagBatch",
    "collate_bags",
    "collate_examples",
    "make_collate_fn",
    "TimitDataset",
    "NispDataset",
    "HeightCelebDataset",
    "CombinedSpeakerDataset",
    "TimitDataModule",
    "NispDataModule",
    "HeightCelebDataModule",
    "MultiCorpusDataModule",
    "build_datamodule",
    "datamodules",
    "AugmentationPipeline",
    "NoOpAugmentation",
    "build_augmentation",
]
