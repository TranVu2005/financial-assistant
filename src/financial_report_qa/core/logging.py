"""Application logging configuration owned by process entry points."""

from __future__ import annotations

import logging
from typing import TextIO

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configure_logging(level: str, *, stream: TextIO | None = None) -> None:
    """Configure deterministic process-wide logging for CLI and service entry points."""
    normalized = level.upper()
    if normalized not in _LEVELS:
        supported = ", ".join(_LEVELS)
        raise ValueError(f"Unsupported log level {level!r}; choose one of: {supported}")

    logging.basicConfig(
        level=_LEVELS[normalized],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=stream,
        force=True,
    )
