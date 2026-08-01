"""Tests for the product command-line dispatcher."""

from collections.abc import Sequence

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
