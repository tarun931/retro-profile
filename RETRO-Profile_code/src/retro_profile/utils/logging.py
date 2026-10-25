"""Logging setup using loguru with sensible defaults."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger as _logger


def setup_logging(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    rotation: str = "100 MB",
) -> None:
    """Configure loguru: pretty stderr sink + optional rotating file sink."""
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _logger.add(
            log_dir / "run_{time:YYYYMMDD_HHmmss}.log",
            level=level,
            rotation=rotation,
            enqueue=True,
            backtrace=True,
            diagnose=False,  # never log local variables — they may contain secrets / PII
        )


def get_logger(name: str | None = None):
    """Return a loguru logger bound to ``name``."""
    return _logger.bind(name=name) if name else _logger
