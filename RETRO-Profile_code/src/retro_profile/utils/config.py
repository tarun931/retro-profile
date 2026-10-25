"""Hydra-friendly config loading and recursive target instantiation.

We don't depend on ``hydra.utils.instantiate`` directly so that scripts /
notebooks can build models from plain dicts too — useful for unit tests and
for the datastore-build script which only needs a Hydra config to look up the
model definition.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file via OmegaConf."""
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a DictConfig at the top level of {path}")
    return cfg


def _is_target_dict(value: Any) -> bool:
    return isinstance(value, dict) and "_target_" in value


def _resolve_class(dotted_path: str) -> Any:
    module_path, _, attr = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"_target_ must be a dotted path; got {dotted_path!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def instantiate(cfg: Any, **overrides: Any) -> Any:
    """Recursively instantiate a config tree.

    Rules:
      * dicts with a ``_target_`` key are instantiated by importing the target
        and calling it with the remaining keys as kwargs;
      * dicts without ``_target_`` are recursed into (each value instantiated
        in place) and returned as a plain dict;
      * lists / tuples are recursed into;
      * everything else is returned as-is.

    Top-level overrides only apply to the outermost dict.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "_target_" in node:
                target = node["_target_"]
                kwargs = {k: _walk(v) for k, v in node.items() if k != "_target_"}
                cls = _resolve_class(target)
                return cls(**kwargs)
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    if isinstance(cfg, dict):
        cfg = {**cfg, **overrides}
    return _walk(cfg)


# Backwards-compatible alias.
def instantiate_from_config(cfg: Any, **overrides: Any) -> Any:
    return instantiate(cfg, **overrides)
