"""Several corpora as one dataset, e.g. HeightCeleb + NISP for training, TIMIT for testing.

Each corpus is read by its own loader (its split rules, NISP's language filter, path resolution);
this dataset is the union of their manifests, tagged with a ``corpus`` column. Speaker IDs must not
repeat across corpora -- a repeat would merge two people into one bag.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from retro_profile.data.base import BaseSpeakerDataset


class CombinedSpeakerDataset(BaseSpeakerDataset):
    """Union of already-built corpus datasets (``parts``: corpus name -> dataset)."""

    def __init__(self, parts: dict[str, BaseSpeakerDataset], **kwargs: Any) -> None:
        if not parts:
            raise ValueError("CombinedSpeakerDataset needs at least one corpus")
        self._parts = parts
        split = "+".join(f"{name}:{ds.split}" for name, ds in parts.items())
        super().__init__(root=".", split=split, **kwargs)

    def _load_manifest(self) -> pd.DataFrame:
        owner: dict[str, str] = {}
        for name, ds in self._parts.items():
            for spk in ds.speakers:
                if spk in owner:
                    raise ValueError(f"speaker {spk!r} appears in both {owner[spk]!r} and {name!r}")
                owner[spk] = name
        return pd.concat([ds.manifest.assign(corpus=name) for name, ds in self._parts.items()],
                         ignore_index=True)
