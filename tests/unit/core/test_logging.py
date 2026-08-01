"""Tests for application logging configuration."""

import logging
from io import StringIO

import pytest

from financial_report_qa.core.logging import configure_logging


def test_configure_logging_emits_consistent_operational_context() -> None:
    """Dropping level or logger name would make production diagnostics ambiguous."""
    stream = StringIO()

    configure_logging("INFO", stream=stream)
    logging.getLogger("financial_report_qa.test").info("pipeline ready")

    output = stream.getvalue()
    assert " INFO financial_report_qa.test pipeline ready" in output


def test_configure_logging_rejects_unknown_level() -> None:
    """An unknown level must fail at startup instead of misconfiguring logging."""
    with pytest.raises(ValueError, match="Unsupported log level"):
        configure_logging("VERBOSE")
