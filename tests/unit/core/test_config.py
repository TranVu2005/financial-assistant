"""Tests for environment-backed product settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.core.config import LLMSettings, Settings, load_llm_settings


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


def test_load_llm_settings_merges_layered_yaml_files(tmp_path: Path) -> None:
    """`configs/local_rtx3050.yaml` is an overlay: it must win over `base.yaml`
    for keys present in both (e.g. `max_output_tokens`), while keys unique to
    the base file (e.g. `base_url`) must still come through unchanged."""
    base = tmp_path / "base.yaml"
    base.write_text(
        "llm:\n"
        "  base_url: http://127.0.0.1:8080/v1\n"
        "  model: qwen3-4b-instruct-2507-q4_k_m\n"
        "  timeout_seconds: 30\n"
        "  max_output_tokens: 160\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "local.yaml"
    overlay.write_text(
        "llm:\n"
        "  temperature: 0.0\n"
        "  context_length: 4096\n"
        "  max_output_tokens: 96\n"
        "  json_schema_constrained: true\n",
        encoding="utf-8",
    )

    settings = load_llm_settings((base, overlay))

    assert settings.base_url == "http://127.0.0.1:8080/v1"
    assert settings.model == "qwen3-4b-instruct-2507-q4_k_m"
    assert settings.timeout_seconds == 30
    assert settings.max_output_tokens == 96
    assert settings.temperature == 0.0
    assert settings.context_length == 4096
    assert settings.json_schema_constrained is True


def test_load_llm_settings_rejects_missing_required_field(tmp_path: Path) -> None:
    """A config missing `base_url` must fail loudly, not silently default to
    an unreachable placeholder that only surfaces as a confusing connection error."""
    base = tmp_path / "base.yaml"
    base.write_text("llm:\n  model: qwen3-4b\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_llm_settings((base,))


def test_llm_settings_rejects_unknown_field() -> None:
    """Extra keys usually mean a typo in the YAML; catch it instead of ignoring it."""
    with pytest.raises(ValidationError):
        LLMSettings.model_validate(
            {
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "qwen3-4b",
                "timeout_seconds": 30,
                "max_output_tokens": 160,
                "temperature": 0.0,
                "context_length": 4096,
                "json_schema_constrained": True,
                "unexpected_field": "oops",
            }
        )
