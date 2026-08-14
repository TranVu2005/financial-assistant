"""Unit tests for the Day 15 FinancialQueryPlan semantic validator (arity table)."""

from __future__ import annotations

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.planning.plan_validator import validate_plan_semantics


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


_KNOWN_TABLES = frozenset({_table_id(c) for c in "abcdef"})
_M = MetricSelector(canonical="revenue")


def _plan(**overrides: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("NVL",),
        "periods": ("2023",),
        "metric": _M,
        "candidate_table_ids": (_table_id("a"),),
    }
    defaults.update(overrides)
    return FinancialQueryPlan(**defaults)  # type: ignore[arg-type]


def _codes(plan: FinancialQueryPlan) -> tuple[str, ...]:
    return tuple(
        issue.code for issue in validate_plan_semantics(plan, known_table_ids=_KNOWN_TABLES)
    )


def test_valid_lookup_plan_has_no_issues() -> None:
    assert _codes(_plan()) == ()


def test_lookup_missing_metric_is_rejected() -> None:
    assert "metric_arity_invalid" in _codes(_plan(metric=None))


def test_lookup_with_metric_a_set_is_rejected() -> None:
    assert "metric_pair_arity_invalid" in _codes(_plan(metric_a=_M))


def test_lookup_with_two_companies_is_rejected() -> None:
    assert "companies_arity_invalid" in _codes(_plan(companies=("NVL", "VHM")))


def test_lookup_with_two_periods_is_rejected() -> None:
    assert "periods_arity_invalid" in _codes(_plan(periods=("2022", "2023")))


def test_valid_compare_plan_has_no_issues() -> None:
    plan = _plan(
        operation="compare",
        metric=None,
        metric_a=MetricSelector(canonical="current_assets"),
        metric_b=MetricSelector(canonical="non_current_assets"),
    )
    assert _codes(plan) == ()


def test_compare_missing_metric_b_is_rejected() -> None:
    plan = _plan(
        operation="compare",
        metric=None,
        metric_a=MetricSelector(canonical="current_assets"),
    )
    assert "metric_pair_arity_invalid" in _codes(plan)


def test_valid_difference_plan_has_no_issues() -> None:
    plan = _plan(operation="difference", periods=("2022", "2023"))
    assert _codes(plan) == ()


def test_difference_with_one_period_is_rejected() -> None:
    assert "periods_arity_invalid" in _codes(_plan(operation="difference"))


def test_difference_with_descending_periods_is_rejected() -> None:
    plan = _plan(operation="difference", periods=("2023", "2022"))
    assert "periods_not_chronological" in _codes(plan)


def test_valid_growth_rate_plan_has_no_issues() -> None:
    plan = _plan(
        operation="growth_rate", periods=("2022", "2023"), expected_unit="percent"
    )
    assert _codes(plan) == ()


def test_growth_rate_expected_unit_mismatch_is_rejected() -> None:
    plan = _plan(operation="growth_rate", periods=("2022", "2023"), expected_unit="VND")
    assert "expected_unit_mismatch" in _codes(plan)


def test_valid_ratio_plan_has_no_issues() -> None:
    plan = _plan(
        operation="ratio",
        metric=None,
        numerator_metric=MetricSelector(canonical="net_revenue"),
        denominator_metric=MetricSelector(canonical="total_assets"),
        expected_unit="ratio",
    )
    assert _codes(plan) == ()


def test_ratio_missing_numerator_is_rejected() -> None:
    plan = _plan(
        operation="ratio",
        metric=None,
        denominator_metric=MetricSelector(canonical="total_assets"),
    )
    assert "numerator_denominator_arity_invalid" in _codes(plan)


def test_ratio_with_plain_metric_set_is_rejected() -> None:
    plan = _plan(
        operation="ratio",
        numerator_metric=MetricSelector(canonical="net_revenue"),
        denominator_metric=MetricSelector(canonical="total_assets"),
    )
    assert "metric_arity_invalid" in _codes(plan)


def test_valid_average_over_periods_has_no_issues() -> None:
    plan = _plan(operation="average", periods=("2022", "2023", "2024"))
    assert _codes(plan) == ()


def test_valid_sum_over_companies_has_no_issues() -> None:
    plan = _plan(operation="sum", companies=("NVL", "VHM"))
    assert _codes(plan) == ()


def test_average_with_single_company_and_single_period_is_rejected() -> None:
    assert "companies_arity_invalid" in _codes(_plan(operation="average"))


def test_average_with_both_dimensions_varying_is_rejected() -> None:
    plan = _plan(operation="average", companies=("NVL", "VHM"), periods=("2022", "2023"))
    assert "companies_arity_invalid" in _codes(plan)


def test_valid_rank_plan_has_no_issues() -> None:
    plan = _plan(operation="rank", companies=("NVL", "VHM", "DXG"), top_k=2)
    assert _codes(plan) == ()


def test_rank_missing_top_k_is_rejected() -> None:
    plan = _plan(operation="rank", companies=("NVL", "VHM", "DXG"))
    assert "top_k_arity_invalid" in _codes(plan)


def test_rank_top_k_out_of_range_is_rejected() -> None:
    plan = _plan(operation="rank", companies=("NVL", "VHM", "DXG"), top_k=3)
    assert "top_k_out_of_range" in _codes(plan)


def test_lookup_with_top_k_set_is_rejected() -> None:
    assert "top_k_arity_invalid" in _codes(_plan(top_k=1))


def test_unknown_candidate_table_id_is_rejected() -> None:
    plan = _plan(candidate_table_ids=(_table_id("9"),))
    assert "candidate_table_ids_unknown" in _codes(plan)
