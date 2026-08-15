"""Tests for Day 18 deterministic-compiler contracts."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery


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
        }
    )
    assert result.answer == Decimal("100")
