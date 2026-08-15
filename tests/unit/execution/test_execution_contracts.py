"""Tests for Day 18 deterministic-compiler contracts."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.execution.contracts import (
    CellMatch,
    CompiledQuery,
    ExecutionIssueCode,
)


def _cell_match(**overrides: object) -> CellMatch:
    defaults: dict[str, object] = {
        "table_id": "tbl_" + "a" * 64,
        "cell_ids": ("cell_" + "b" * 64,),
        "value": Decimal("100"),
        "unit": "VND",
        "period": 2023,
        "period_inferred": False,
    }
    defaults.update(overrides)
    return CellMatch.model_validate(defaults)


def _replay_row(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "company_code": "ACB",
        "row_label_canonical": "cash_and_cash_equivalents",
        "row_label_raw": None,
        "period": 2023,
        "value": Decimal("100"),
    }
    defaults.update(overrides)
    return defaults


def test_cell_match_requires_at_least_one_cell_id() -> None:
    """An evidence record with zero cell_ids cannot be traced back to a source
    cell, defeating the audit trail plan.md requires."""
    with pytest.raises(ValidationError):
        _cell_match(cell_ids=())


def test_cell_match_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        CellMatch.model_validate(
            {
                "table_id": "tbl_" + "a" * 64,
                "cell_ids": ("cell_" + "b" * 64,),
                "value": Decimal("100"),
                "unit": "VND",
                "period": 2023,
                "period_inferred": False,
                "unexpected": "oops",
            }
        )


def test_compiled_query_answered_forbids_error_fields() -> None:
    """A status="answered" result carrying an error_code would let a caller miss
    that the answer is actually invalid — the two must be mutually exclusive."""
    with pytest.raises(ValidationError):
        CompiledQuery.model_validate(
            {
                "operation": "lookup",
                "status": "answered",
                "answer": Decimal("100"),
                "unit": "VND",
                "evidence": (_cell_match().model_dump(mode="json"),),
                "pandas_query": "df1['value'].iloc[0]",
                "error_code": "metric_not_found",
                "error_message": None,
                "replay_rows": (_replay_row(),),
            }
        )


def test_compiled_query_answered_requires_replay_rows() -> None:
    """Day 22 plan §2 decision A: a submission exporter reads the exact
    replayed DataFrame from `replay_rows` -- an answered result without it
    would silently break that contract."""
    with pytest.raises(ValidationError):
        CompiledQuery.model_validate(
            {
                "operation": "lookup",
                "status": "answered",
                "answer": Decimal("100"),
                "unit": "VND",
                "evidence": (_cell_match().model_dump(mode="json"),),
                "pandas_query": "df1['value'].iloc[0]",
                "error_code": None,
                "error_message": None,
                "replay_rows": (),
            }
        )


def test_compiled_query_error_forbids_replay_rows() -> None:
    with pytest.raises(ValidationError):
        CompiledQuery.model_validate(
            {
                "operation": "lookup",
                "status": "error",
                "answer": None,
                "unit": None,
                "evidence": (),
                "pandas_query": "df1['value'].iloc[0]",
                "error_code": "metric_not_found",
                "error_message": "no matching row",
                "replay_rows": (_replay_row(),),
            }
        )


def test_compiled_query_error_forbids_answer() -> None:
    with pytest.raises(ValidationError):
        CompiledQuery.model_validate(
            {
                "operation": "lookup",
                "status": "error",
                "answer": Decimal("100"),
                "unit": None,
                "evidence": (),
                "pandas_query": "df1['value'].iloc[0]",
                "error_code": "metric_not_found",
                "error_message": "no matching row",
            }
        )


def test_compiled_query_error_requires_message() -> None:
    with pytest.raises(ValidationError):
        CompiledQuery.model_validate(
            {
                "operation": "lookup",
                "status": "error",
                "answer": None,
                "unit": None,
                "evidence": (),
                "pandas_query": "df1['value'].iloc[0]",
                "error_code": "metric_not_found",
                "error_message": None,
            }
        )


def test_compiled_query_answered_accepts_valid_result() -> None:
    result = CompiledQuery.model_validate(
        {
            "operation": "lookup",
            "status": "answered",
            "answer": Decimal("100"),
            "unit": "VND",
            "evidence": (_cell_match().model_dump(mode="json"),),
            "pandas_query": "df1['value'].iloc[0]",
            "error_code": None,
            "error_message": None,
            "replay_rows": (_replay_row(),),
        }
    )
    assert result.answer == Decimal("100")
    assert result.replay_rows[0].company_code == "ACB"


def test_execution_issue_code_includes_day19_sandbox_codes() -> None:
    """ADR 0008 decision G1: four new codes for the sandbox hardening layer."""
    from typing import get_args

    codes = set(get_args(ExecutionIssueCode))
    assert {"plan_rejected", "query_rejected", "budget_exceeded", "row_limit_exceeded"} <= codes


def test_execution_issue_code_includes_unit_missing() -> None:
    """ADR 0009 decision C1: a distinct code from `unit_incompatible` -- 'no
    unit was recorded' is a different failure than 'units could not convert'."""
    from typing import get_args

    assert "unit_missing" in get_args(ExecutionIssueCode)


def test_cell_match_rejects_fabricated_unit_string() -> None:
    """Day 20 plan Sec 1.3: `str(float('nan'))` == 'nan' must never pass as a
    unit -- ADR 0009 decision C1 constrains CellMatch.unit to the 6 real
    CanonicalUnit values."""
    with pytest.raises(ValidationError):
        _cell_match(unit="nan")
