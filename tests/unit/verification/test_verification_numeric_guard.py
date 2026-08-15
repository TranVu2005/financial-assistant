"""Tests for the Day 20 numeric guard (ADR 0009 decision F1): the optional
LLM-paraphrase path must never introduce a number outside the whitelist of
already-known numbers (locked answer, plan periods/top_k, evidence values).

Day 20 plan Sec 1.8 measured that every numeric token in gold70 questions is
a year, and that naive tokenization captures trailing punctuation (`'2023.'`)
-- both are exercised below.
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.verification.numeric_guard import (
    build_number_whitelist,
    extract_number_tokens,
    guard_generated_text,
)

TABLE_ID = "tbl_" + "1" * 64


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
        "pandas_query": "df1",
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


# --- extract_number_tokens ------------------------------------------------


def test_extract_number_tokens_strips_trailing_sentence_punctuation() -> None:
    """Day 20 plan Sec 1.8: naive tokenization on
    '... giữa năm 2022 và năm 2023.' captures '2023.' with the period
    attached -- must be normalized away."""
    tokens = extract_number_tokens("So sánh ... giữa năm 2022 và năm 2023.")
    assert "2023" in tokens
    assert "2023." not in tokens


def test_extract_number_tokens_finds_multiple_numbers() -> None:
    tokens = extract_number_tokens("Từ 100 lên 250, tăng 150.")
    assert set(tokens) >= {"100", "250", "150"}


# --- build_number_whitelist ------------------------------------------------


def test_whitelist_includes_answer_periods_and_evidence_values() -> None:
    plan = _plan(periods=("2023",))
    compiled = _compiled(answer=Decimal("100"), evidence=(_cell(value=Decimal("100")),))
    whitelist = build_number_whitelist(plan, compiled)
    assert Decimal("100") in whitelist
    assert Decimal("2023") in whitelist


def test_whitelist_includes_top_k() -> None:
    plan = _plan(operation="rank", companies=("ACB", "MBB"), top_k=2)
    compiled = _compiled(operation="rank", answer=Decimal("500"))
    whitelist = build_number_whitelist(plan, compiled)
    assert Decimal("2") in whitelist


# --- guard_generated_text --------------------------------------------------


def test_guard_allows_text_using_only_whitelisted_numbers() -> None:
    plan = _plan(periods=("2023",))
    compiled = _compiled(answer=Decimal("100"))
    whitelist = build_number_whitelist(plan, compiled)
    result = guard_generated_text("Tiền mặt của ACB năm 2023 là 100 VND.", whitelist=whitelist)
    assert result.allowed is True
    assert result.disallowed_numbers == ()


def test_guard_rejects_text_with_fabricated_number() -> None:
    plan = _plan(periods=("2023",))
    compiled = _compiled(answer=Decimal("100"))
    whitelist = build_number_whitelist(plan, compiled)
    result = guard_generated_text(
        "Tiền mặt của ACB năm 2023 là 100 VND, tăng 15% so với dự báo.",
        whitelist=whitelist,
    )
    assert result.allowed is False
    assert "15" in result.disallowed_numbers


def test_guard_allows_year_only_question_style_text() -> None:
    """Every numeric token in gold70 questions is a year (Day 20 plan Sec
    1.8) -- a two-year comparison sentence must pass cleanly."""
    plan = _plan(operation="difference", periods=("2022", "2023"))
    compiled = _compiled(operation="difference", answer=Decimal("-1000000"), unit="VND")
    whitelist = build_number_whitelist(plan, compiled)
    result = guard_generated_text(
        "Chỉ số của ACB thay đổi -1,000,000 VND từ năm 2022 đến năm 2023.",
        whitelist=whitelist,
    )
    assert result.allowed is True
