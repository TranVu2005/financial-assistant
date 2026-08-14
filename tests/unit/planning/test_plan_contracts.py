"""Unit tests for the Day 15 FinancialQueryPlan schema and metric selector shape."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def test_minimal_lookup_plan_constructs() -> None:
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("NVL",),
        periods=("2023",),
        metric=MetricSelector(canonical="revenue"),
        candidate_table_ids=(_table_id("a"),),
    )
    assert plan.operation == "lookup"
    assert plan.companies == ("NVL",)
    assert plan.metric == MetricSelector(canonical="revenue")


def test_metric_selector_rejects_both_canonical_and_raw_text() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        MetricSelector(canonical="revenue", raw_text="Doanh thu thuần")


def test_metric_selector_rejects_neither_canonical_nor_raw_text() -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        MetricSelector()


def test_metric_selector_rejects_unknown_canonical_metric() -> None:
    with pytest.raises(ValidationError, match="not a canonical metric"):
        MetricSelector(canonical="not_a_real_metric")


def test_metric_selector_rejects_blank_raw_text() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        MetricSelector(raw_text="   ")


def _plan(**overrides: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("NVL",),
        "periods": ("2023",),
        "metric": MetricSelector(canonical="revenue"),
        "candidate_table_ids": (_table_id("a"),),
    }
    defaults.update(overrides)
    return FinancialQueryPlan(**defaults)  # type: ignore[arg-type]


def test_plan_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="extra"):
        FinancialQueryPlan(
            operation="lookup",
            companies=("NVL",),
            periods=("2023",),
            metric=MetricSelector(canonical="revenue"),
            candidate_table_ids=(_table_id("a"),),
            unexpected_field="x",  # type: ignore[call-arg]
        )


def test_plan_rejects_empty_companies() -> None:
    with pytest.raises(ValidationError, match="companies must not be empty"):
        _plan(companies=())


def test_plan_rejects_empty_periods() -> None:
    with pytest.raises(ValidationError, match="periods must not be empty"):
        _plan(periods=())


def test_plan_rejects_duplicate_companies() -> None:
    with pytest.raises(ValidationError, match="companies must not contain duplicates"):
        _plan(companies=("NVL", "NVL"))


def test_plan_rejects_duplicate_periods() -> None:
    with pytest.raises(ValidationError, match="periods must not contain duplicates"):
        _plan(periods=("2022", "2022"))


def test_plan_rejects_empty_candidate_table_ids() -> None:
    with pytest.raises(ValidationError, match="candidate_table_ids"):
        _plan(candidate_table_ids=())


_HEX_DIGITS = "0123456789abcdef"


def test_plan_rejects_more_than_twelve_candidate_table_ids() -> None:
    ids = tuple(_table_id(_HEX_DIGITS[i]) for i in range(13))
    with pytest.raises(ValidationError, match="candidate_table_ids"):
        _plan(candidate_table_ids=ids)


def test_plan_rejects_duplicate_candidate_table_ids() -> None:
    with pytest.raises(ValidationError, match="candidate_table_ids must not contain duplicates"):
        _plan(candidate_table_ids=(_table_id("a"), _table_id("a")))


def test_plan_rejects_full_date_period_grammar() -> None:
    with pytest.raises(ValidationError, match="periods must be canonical 'YYYY'"):
        _plan(periods=("2024-12-31",))


def test_plan_accepts_twelve_candidate_table_ids() -> None:
    ids = tuple(_table_id(_HEX_DIGITS[i]) for i in range(12))
    plan = _plan(candidate_table_ids=ids)
    assert len(plan.candidate_table_ids) == 12
