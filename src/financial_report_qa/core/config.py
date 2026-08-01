"""Typed application settings loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings shared by product entry points."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["local", "test", "staging"] = "local"
    data_root: Path = Path("data")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @classmethod
    def load(cls, *, env_file: str | Path | None = ".env") -> Self:
        """Load settings with an optional dotenv file and environment precedence."""
        # BaseSettings accepts this runtime control, but Pydantic's synthesized subclass
        # constructor exposed to mypy contains only declared model fields.
        return cls(_env_file=env_file)  # type: ignore[call-arg]
