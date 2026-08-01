"""Smoke checks for the project environment."""

import financial_report_qa


def test_package_is_importable() -> None:
    """The editable project package is available to the test runner."""
    assert financial_report_qa.__name__ == "financial_report_qa"
