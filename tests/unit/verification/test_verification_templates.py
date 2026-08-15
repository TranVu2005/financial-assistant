"""Tests for the Day 20 template renderer (ADR 0009 decision F1): the default,
LLM-free answer path. Every operation must render a display string and a
Vietnamese sentence containing only numbers already present in the locked
answer or the plan's periods -- never a number invented by the template."""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.verification.templates import render_answer, render_sentence

TABLE_ID = "tbl_" + "1" * 64

_DUMMY_EVIDENCE = (
    CellMatch.model_validate(
        {
            "table_id": TABLE_ID,
            "cell_ids": ("cell_" + "a" * 64,),
            "value": Decimal("1"),
            "unit": "VND",
            "period": 2023,
            "period_inferred": False,
        }
    ),
)


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
        "answer": Decimal("1000000"),
        "unit": "VND",
        "evidence": _DUMMY_EVIDENCE,
        "pandas_query": "df1",
        "error_code": None,
        "error_message": None,
    }
    defaults.update(overrides)
    return CompiledQuery.model_validate(defaults)


def test_render_answer_currency_uses_zero_precision() -> None:
    display, precision = render_answer(_plan(), _compiled())
    assert precision == 0
    assert "1,000,000" in display or "1000000" in display
    assert "VND" in display


def test_render_answer_ratio_without_expected_percent_stays_ratio() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"))
    compiled = _compiled(operation="growth_rate", answer=Decimal("0.0523"), unit="ratio")
    display, precision = render_answer(plan, compiled)
    assert precision == 4
    assert "0.0523" in display


def test_render_answer_ratio_presented_as_percent_multiplies_by_100() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"), expected_unit="percent")
    compiled = _compiled(operation="growth_rate", answer=Decimal("0.0523"), unit="ratio")
    display, precision = render_answer(plan, compiled)
    assert precision == 2
    assert "5.23" in display
    assert "%" in display


def test_render_answer_negative_ratio_as_percent() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"), expected_unit="percent")
    compiled = _compiled(
        operation="growth_rate",
        answer=Decimal("-0.01932846513079090948136258551"),
        unit="ratio",
    )
    display, precision = render_answer(plan, compiled)
    assert "-1.93" in display
    assert "%" in display


def test_render_sentence_lookup_mentions_company_period_and_display() -> None:
    plan = _plan()
    display, _ = render_answer(plan, _compiled())
    sentence = render_sentence(plan, _compiled(), display)
    assert "ACB" in sentence
    assert "2023" in sentence
    assert display in sentence


def test_render_sentence_difference_mentions_both_periods() -> None:
    plan = _plan(operation="difference", periods=("2022", "2023"))
    compiled = _compiled(operation="difference", answer=Decimal("-1000000"), unit="VND")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "2022" in sentence
    assert "2023" in sentence


def test_render_sentence_growth_rate_uses_tang_when_positive() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"), expected_unit="percent")
    compiled = _compiled(operation="growth_rate", answer=Decimal("0.05"), unit="ratio")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "tăng" in sentence


def test_render_sentence_growth_rate_uses_giam_when_negative() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"), expected_unit="percent")
    compiled = _compiled(operation="growth_rate", answer=Decimal("-0.05"), unit="ratio")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "giảm" in sentence


def test_render_sentence_compare_companies_mentions_both_companies() -> None:
    plan = _plan(
        operation="compare_companies",
        companies=("ACB", "MBB"),
        periods=("2023",),
    )
    compiled = _compiled(operation="compare_companies", answer=Decimal("500"), unit="VND")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "ACB" in sentence
    assert "MBB" in sentence


def test_render_sentence_ratio_mentions_both_metrics() -> None:
    plan = _plan(
        operation="ratio",
        metric=None,
        numerator_metric=MetricSelector(canonical="cash_and_cash_equivalents"),
        denominator_metric=MetricSelector(canonical="total_assets"),
    )
    compiled = _compiled(operation="ratio", answer=Decimal("0.25"), unit="ratio")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "cash_and_cash_equivalents" in sentence
    assert "total_assets" in sentence


def test_render_sentence_rank_mentions_top_k() -> None:
    plan = _plan(
        operation="rank",
        companies=("ACB", "MBB", "VCB"),
        periods=("2023",),
        top_k=2,
    )
    compiled = _compiled(operation="rank", answer=Decimal("500"), unit="VND")
    display, _ = render_answer(plan, compiled)
    sentence = render_sentence(plan, compiled, display)
    assert "2" in sentence


def test_render_answer_rejects_error_status() -> None:
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
    try:
        render_answer(_plan(), compiled)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
