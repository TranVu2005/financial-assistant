"""Tests for environment-backed product settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.core.config import Settings


def test_environment_overrides_local_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ignoring environment values would make test and staging use local paths."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dataset"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings.load(env_file=None)

    assert settings.app_env == "test"
    assert settings.data_root == tmp_path / "dataset"
    assert settings.log_level == "DEBUG"


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepting a misspelled level would silently suppress operational logs."""
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings.load(env_file=None)
