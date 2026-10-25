"""Tests for the small utility modules."""

from __future__ import annotations

import pytest

from retro_profile.utils.config import instantiate
from retro_profile.utils.registry import Registry


def test_registry_register_and_build():
    reg = Registry("test")

    @reg.register("foo")
    class Foo:
        def __init__(self, x: int = 0):
            self.x = x

    assert "foo" in reg
    obj = reg.build("foo", x=42)
    assert isinstance(obj, Foo)
    assert obj.x == 42


def test_registry_duplicate_key_raises():
    reg = Registry("test")

    @reg.register("foo")
    class A:
        pass

    with pytest.raises(KeyError):
        @reg.register("foo")
        class B:
            pass


def test_instantiate_recursive_resolves_nested_targets():
    cfg = {
        "_target_": "retro_profile.models.heads.MLPHead",
        "input_dim": 8,
        "hidden_dim": 16,
        "output_dim": 4,
    }
    obj = instantiate(cfg)
    assert obj.__class__.__name__ == "MLPHead"
    assert obj.input_dim == 8
    assert obj.output_dim == 4


def test_instantiate_passes_through_non_target_dicts():
    cfg = {"a": 1, "b": {"c": 2}}
    out = instantiate(cfg)
    assert out == {"a": 1, "b": {"c": 2}}
