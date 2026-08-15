"""Typed application settings loaded from environment variables and YAML."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    """Day 17 LLM-planner client settings, layered from `configs/*.yaml`.

    Deliberately strict (`extra="forbid"`): the `llm:` block in
    `configs/base.yaml`/`configs/local_rtx3050.yaml` previously had zero
    Python readers, so a stray or misspelled key would have gone unnoticed
    forever (Day 17 plan §1.3).
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    timeout_seconds: float
    max_output_tokens: int
    temperature: float
    context_length: int
    json_schema_constrained: bool


def load_llm_settings(paths: Sequence[str | Path]) -> LLMSettings:
    """Merge the `llm:` block of one or more YAML files, later files winning.

    Mirrors the `configs/base.yaml` + `configs/local_rtx3050.yaml` layering
    already used to run this project locally; a later path overrides keys an
    earlier path also sets, and keys unique to either path pass through.
    """
    merged: dict[str, object] = {}
    for path in paths:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        merged.update(document.get("llm", {}))
    return LLMSettings.model_validate(merged)


class ExecutionSettings(BaseModel):
    """Day 18 deterministic-compiler settings, layered from `configs/*.yaml`.

    Deliberately strict (`extra="forbid"`): the `execution:` block in
    `configs/local_rtx3050.yaml` previously had zero Python readers (Day 18
    plan §1.7), so a stray or misspelled key would have gone unnoticed forever.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float
    max_rows: int
    allow_operations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_allow_operations_non_empty(self) -> Self:
        if not self.allow_operations:
            raise ValueError("allow_operations must not be empty")
        return self


def load_execution_settings(paths: Sequence[str | Path]) -> ExecutionSettings:
    """Merge the `execution:` block of one or more YAML files, later files winning."""
    merged: dict[str, object] = {}
    for path in paths:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        merged.update(document.get("execution", {}))
    return ExecutionSettings.model_validate(merged)


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
