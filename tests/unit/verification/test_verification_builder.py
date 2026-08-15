"""Tests for the Day 20 `build_answer_package` orchestrator (ADR 0009).

Ties `checks.py` + `templates.py` together the way `execution/compiler.py`
ties `locator`/`operations`/`pandas_query` together for Day 18: one function,
never a guessed or half-verified package.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.verification.builder import build_answer_package

TABLE_ID = "tbl_" + "1" * 64
QUESTION_ID = "retq_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_CITATION_LOOKUP: dict[str, dict[str, object]] = {
    CELL_ID: {
        "doc_relative_path": "ACB/2023/report.txt",
        "source_line_start": 12,
        "source_line_end": 12,
        "table_title": "Bang can doi ke toan",
    }
}


def _cell(**overrides: object) -> CellMatch:
    defaults: dict[str, object] = {
        "table_id": TABLE_ID,
        "cell_ids": ("cell_" + "a" * 64,),
        "value": Decimal("100"),
        "unit": "VND",
        "period": 2023,
        "period_inferred": False,
    }
    defaults.update(overrides)
    return CellMatch.model_validate(defaults)


def _plan(**overrides: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("ACB",),
        "periods": ("2023",),
        "candidate_table_ids": (TABLE_ID,),
        "metric": MetricSelector(canonical="cash_and_cash_equivalents"),
    }
    defaults.update(overrides)
    return FinancialQueryPlan(**defaults)  # type: ignore[arg-type]


def _compiled(**overrides: object) -> CompiledQuery:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "status": "answered",
        "answer": Decimal("100"),
        "unit": "VND",
        "evidence": (_cell(),),
        "pandas_query": 'df1[(df1.period == 2023)]["value"].iloc[0]',
        "error_code": None,
        "error_message": None,
        "replay_rows": (
            {
                "company_code": "ACB",
                "row_label_canonical": "cash_and_cash_equivalents",
                "row_label_raw": None,
                "period": 2023,
                "value": Decimal("100"),
            },
        ),
    }
    defaults.update(overrides)
    return CompiledQuery.model_validate(defaults)


def test_build_answer_package_verified_on_clean_input() -> None:
    plan = _plan()
    compiled = _compiled()
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "verified"
    assert package.verification_issues == ()
    assert package.answer == Decimal("100")
    assert package.evidence[0].table_id == TABLE_ID
    assert "ACB" in package.answer_text
    assert package.display in package.answer_text


def test_build_answer_package_rejected_when_evidence_outside_retrieval() -> None:
    plan = _plan()
    other_table = "tbl_" + "9" * 64
    compiled = _compiled(evidence=(_cell(table_id=other_table),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "evidence_outside_retrieval" in codes


def test_build_answer_package_rejected_on_recompute_mismatch() -> None:
    plan = _plan()
    compiled = _compiled(answer=Decimal("999"))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "recompute_mismatch" in codes


def test_build_answer_package_carries_period_inferred_warning_but_stays_verified() -> None:
    """Day 20 plan Sec 1.5: 6/30 gold70 answers rely on an inferred period --
    this must warn, not block."""
    plan = _plan()
    compiled = _compiled(evidence=(_cell(period_inferred=True),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "verified"
    assert package.period_inferred is True
    codes = {issue.code for issue in package.verification_issues}
    assert "period_inferred_warning" in codes


def test_build_answer_package_rejected_when_scope_inferred() -> None:
    """Day 21 plan §1.5/ADR 0010 decision B1: unlike a merely-inferred period,
    a `CompiledQuery.scope_inferred=True` result must block, not just warn --
    the compiler resolved a real value conflict using a default the plan
    never stated."""
    plan = _plan()
    compiled = _compiled(scope_inferred=True)
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "scope_inferred" in codes


def test_build_answer_package_raises_on_non_answered_compiled_query() -> None:
    plan = _plan()
    compiled = CompiledQuery.model_validate(
        {
            "operation": "lookup",
            "status": "error",
            "answer": None,
            "unit": None,
            "evidence": (),
            "pandas_query": "<plan rejected before rendering>",
            "error_code": "metric_not_found",
            "error_message": "no match",
        }
    )
    with pytest.raises(ValueError):
        build_answer_package(
            question_id=QUESTION_ID,
            question="Tra cứu tiền mặt của ACB năm 2023.",
            plan=plan,
            compiled=compiled,
            retrieved_table_ids=frozenset({TABLE_ID}),
        )


def test_build_answer_package_citation_fields_populated_from_evidence() -> None:
    plan = _plan()
    compiled = _compiled()
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup={
            "cell_" + "a" * 64: {
                "doc_relative_path": "ACB/2023/report.txt",
                "source_line_start": 12,
                "source_line_end": 12,
                "table_title": "Bang can doi ke toan",
            }
        },
    )
    citation = package.evidence[0]
    assert citation.doc_relative_path == "ACB/2023/report.txt"
    assert citation.source_line_start == 12
