"""Lightweight registry pattern for swappable components.

Usage:
    encoders = Registry("encoders")

    @encoders.register("wavlm")
    class WavLMEncoder(BaseEncoder):
        ...

    encoder = encoders.build("wavlm", pretrained_name="microsoft/wavlm-large")
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """A name -> class registry with simple build helpers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, type] = {}

    def register(self, key: str | None = None) -> Callable[[type[T]], type[T]]:
        """Decorator to register a class under ``key`` (defaults to the class name)."""

        def decorator(cls: type[T]) -> type[T]:
            name = key or cls.__name__
            if name in self._registry:
                raise KeyError(f"{self.name} already has '{name}' registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, key: str) -> type:
        if key not in self._registry:
            raise KeyError(
                f"Unknown {self.name} '{key}'. Available: {sorted(self._registry)}"
            )
        return self._registry[key]

    def build(self, key: str, **kwargs: Any) -> Any:
        return self.get(key)(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._registry)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, entries={self.keys()})"
