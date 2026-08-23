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


def test_export_tables_forwards_arguments_and_propagates_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dispatcher must hand raw arguments to the export CLI untouched."""
    import financial_report_qa.export.cli as export_cli

    received: list[str] = []

    def fake_export_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 7

    monkeypatch.setattr(export_cli, "main", fake_export_main)

    exit_code = main(
        [
            "export-tables",
            "--release-dir",
            "data/release/example",
            "--snapshot-root",
            "data/raw/vifinqa",
        ]
    )

    assert exit_code == 7
    assert received == [
        "--release-dir",
        "data/release/example",
        "--snapshot-root",
        "data/raw/vifinqa",
    ]


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
