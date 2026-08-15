"""Tests for the product command-line dispatcher."""

from collections.abc import Sequence

import pytest

from financial_report_qa.cli import main


def test_download_data_forwards_arguments_without_starting_transfer() -> None:
    """The dispatcher must not add --download behind the user's back."""
    received: list[str] = []

    def fake_download_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 0

    exit_code = main(
        ["download-data", "--target", "data/raw/example"],
        download_main_fn=fake_download_main,
    )

    assert exit_code == 0
    assert received == ["--target", "data/raw/example"]


def test_inventory_data_forwards_arguments() -> None:
    received: list[str] = []

    def fake_inventory_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 0

    exit_code = main(
        ["inventory-data", "--root", "data/raw/vifinqa"],
        inventory_main_fn=fake_inventory_main,
    )

    assert exit_code == 0
    assert received == ["--root", "data/raw/vifinqa"]


def test_execution_cli_does_not_hide_unexpected_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the product boundary turns programmer bugs into exit code 2."""
    import financial_report_qa.execution.cli as execution_cli

    def unexpected_resolver(*_: object, **__: object) -> object:
        raise ValueError("programmer bug")

    monkeypatch.setattr(execution_cli, "resolve_retrieval_release", unexpected_resolver)

    with pytest.raises(ValueError, match="programmer bug"):
        main(
            [
                "execution",
                "compile-plans",
                "--release-lock",
                "lock.json",
                "--gold-path",
                "gold.jsonl",
                "--output-dir",
                "out",
                "--execution-config",
                "execution.yaml",
            ]
        )


def test_verification_cli_does_not_hide_unexpected_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the product boundary turns programmer bugs into exit code 2."""
    import financial_report_qa.verification.cli as verification_cli

    def unexpected_resolver(*_: object, **__: object) -> object:
        raise ValueError("programmer bug")

    monkeypatch.setattr(verification_cli, "resolve_retrieval_release", unexpected_resolver)

    with pytest.raises(ValueError, match="programmer bug"):
        main(
            [
                "verification",
                "verify-answers",
                "--release-lock",
                "lock.json",
                "--gold-path",
                "gold.jsonl",
                "--output-dir",
                "out",
                "--execution-config",
                "execution.yaml",
            ]
        )


def test_retrieval_cli_does_not_hide_unexpected_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the product boundary turns programmer bugs into exit code 2."""
    import financial_report_qa.retrieval.cli as retrieval_cli

    def unexpected_resolver(*_: object, **__: object) -> object:
        raise ValueError("programmer bug")

    monkeypatch.setattr(retrieval_cli, "resolve_retrieval_release", unexpected_resolver)

    with pytest.raises(ValueError, match="programmer bug"):
        main(
            [
                "retrieval",
                "validate-gold",
                "--release-lock",
                "lock.json",
                "--gold-path",
                "gold.jsonl",
            ]
        )


def test_pipeline_cli_does_not_hide_unexpected_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if the product boundary turns programmer bugs into exit code 2."""
    import financial_report_qa.pipeline.cli as pipeline_cli

    def unexpected_resolver(*_: object, **__: object) -> object:
        raise ValueError("programmer bug")

    monkeypatch.setattr(pipeline_cli, "resolve_retrieval_release", unexpected_resolver)

    with pytest.raises(ValueError, match="programmer bug"):
        main(
            [
                "pipeline",
                "run-e2e",
                "--release-lock",
                "lock.json",
                "--gold-path",
                "gold.jsonl",
                "--rankings-path",
                "rankings.json",
                "--output-dir",
                "out",
                "--execution-config",
                "execution.yaml",
            ]
        )
