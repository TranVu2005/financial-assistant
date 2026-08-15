"""Tests for the Day 19 execution sandbox (ADR 0008 decisions B2/C1/D3).

`sandbox.py` is the sole gateway to `replay_pandas_query`: it converts every
exception the replayer can raise into a typed `SandboxResult`, and measures
elapsed time against a caller-supplied budget instead of a preemptive
timeout (unavailable on win32 -- Day 19 plan Sec 1.6).
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from financial_report_qa.execution import sandbox
from financial_report_qa.execution.sandbox import replay_in_sandbox

FRAME = pd.DataFrame(
    {
        "company_code": ["ACB"],
        "row_label_canonical": ["cash_and_cash_equivalents"],
        "row_label_raw": ["Tien mat"],
        "unit": ["VND"],
        "value": [Decimal("100")],
        "period": pd.array([2023], dtype="Int64"),
    }
)


def test_sandbox_returns_value_on_success() -> None:
    result = replay_in_sandbox(
        'df1[(df1.period == 2023)]["value"].iloc[0]', FRAME, timeout_seconds=5.0
    )
    assert result.value == Decimal("100")
    assert result.error_code is None
    assert result.error_message is None


def test_sandbox_converts_replay_valueerror_to_query_rejected() -> None:
    result = replay_in_sandbox("__import__('os')", FRAME, timeout_seconds=5.0)
    assert result.value is None
    assert result.error_code == "query_rejected"
    assert result.error_message is not None and "ValueError" in result.error_message


def test_sandbox_converts_replay_syntaxerror_to_query_rejected() -> None:
    """Day 19 plan Sec 1.3: a real corpus label with an unescaped quote used to
    raise an uncaught SyntaxError. Sandbox must deny-by-default on ANY
    exception type, not just ValueError."""
    result = replay_in_sandbox('df1[df1.x == "a"b"]', FRAME, timeout_seconds=5.0)
    assert result.value is None
    assert result.error_code == "query_rejected"
    assert result.error_message is not None and "SyntaxError" in result.error_message


def test_sandbox_converts_keyerror_to_query_rejected() -> None:
    result = replay_in_sandbox("df1.nope", FRAME, timeout_seconds=5.0)
    assert result.error_code == "query_rejected"


def test_sandbox_reports_budget_exceeded_when_over_time_budget() -> None:
    result = replay_in_sandbox(
        'df1[(df1.period == 2023)]["value"].iloc[0]', FRAME, timeout_seconds=-1.0
    )
    assert result.value is None
    assert result.error_code == "budget_exceeded"


def test_sandbox_does_not_catch_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR 0008 decision C1: sandbox catches `Exception`, never `BaseException`
    -- KeyboardInterrupt/SystemExit must propagate, not be swallowed as a
    typed error."""

    class _Boom(BaseException):
        pass

    def _raise_boom(query: str, frame: pd.DataFrame) -> Decimal:
        raise _Boom("should not be caught")

    monkeypatch.setattr(sandbox, "replay_pandas_query", _raise_boom)
    with pytest.raises(_Boom):
        replay_in_sandbox("df1", FRAME, timeout_seconds=5.0)


def test_sandbox_measures_elapsed_time() -> None:
    result = replay_in_sandbox(
        'df1[(df1.period == 2023)]["value"].iloc[0]', FRAME, timeout_seconds=5.0
    )
    assert result.elapsed_seconds >= 0.0
