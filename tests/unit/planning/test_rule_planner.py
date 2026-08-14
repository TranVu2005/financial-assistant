"""Unit tests for the Day 16 deterministic rule planner.

`build_plan` never returns a plan that fails `validate_plan_semantics` — every
test either asserts a concrete, valid `FinancialQueryPlan` or an abstain code,
never a semantic issue leaking through.
"""

from __future__ import annotations

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.planning.rule_planner import RulePlanResult, build_plan

_TABLE_A = "tbl_" + "a" * 64
_TABLE_B = "tbl_" + "b" * 64
_KNOWN_TABLES = frozenset({_TABLE_A, _TABLE_B})


def _plan_for(
    question: str, *, candidate_table_ids: tuple[str, ...] = (_TABLE_A,)
) -> RulePlanResult:
    entities = parse_query_entities(question)
    return build_plan(
        entities, candidate_table_ids=candidate_table_ids, known_table_ids=_KNOWN_TABLES
    )


def test_single_company_single_period_single_metric_is_lookup() -> None:
    result = _plan_for("Tra cứu doanh thu thuần của NVL năm 2023.")
    assert result.abstain_codes == ()
    assert result.plan is not None
    assert result.plan.operation == "lookup"
    assert result.plan.companies == ("NVL",)
    assert result.plan.periods == ("2023",)
    assert result.plan.metric is not None
    assert result.plan.metric.canonical == "net_revenue"
    assert validate_plan_semantics(result.plan, known_table_ids=_KNOWN_TABLES) == ()


def test_growth_wording_over_two_periods_is_growth_rate() -> None:
    result = _plan_for("Tính tốc độ tăng trưởng doanh thu thuần của NVL từ năm 2022 đến năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "growth_rate"
    assert result.plan.periods == ("2022", "2023")


def test_compare_wording_over_two_periods_is_difference() -> None:
    result = _plan_for("So sánh doanh thu thuần của CTG giữa năm 2022 và năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "difference"
    assert result.plan.periods == ("2022", "2023")


def test_two_companies_one_period_is_compare_companies() -> None:
    result = _plan_for("So sánh doanh thu thuần giữa NVL và CTG năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "compare_companies"
    assert result.plan.companies == ("CTG", "NVL")


def test_missing_company_abstains_with_entity_ambiguous() -> None:
    result = _plan_for("Doanh thu thuần năm 2023 là bao nhiêu?")
    assert result.plan is None
    assert result.abstain_codes == ("entity_ambiguous",)


def test_unknown_metric_abstains_with_entity_ambiguous() -> None:
    result = _plan_for("Tra cứu tổng lợi thế cạnh tranh của DBC năm 2023.")
    assert result.plan is None
    assert result.abstain_codes == ("entity_ambiguous",)


def test_two_metrics_abstains_with_multi_metric_unsupported() -> None:
    result = _plan_for("So sánh doanh thu thuần và giá vốn hàng bán của DBC năm 2023.")
    assert result.plan is None
    assert result.abstain_codes == ("multi_metric_unsupported",)


def test_date_period_abstains_with_period_grammar_unsupported() -> None:
    result = _plan_for("Tra cứu doanh thu thuần của DBC tại ngày 31/12/2023.")
    assert result.plan is None
    assert result.abstain_codes == ("period_grammar_unsupported",)


def test_banking_metric_uses_raw_text_selector_not_canonical() -> None:
    """`loans_to_customers` is a question-side quasi-canonical id (Day 16 §1.6/1.7,
    ADR 0004) that never entered `CANONICAL_METRICS` — the plan must locate it
    via `raw_text`, matching the verbatim question phrase, not `.canonical`."""
    result = _plan_for("Tra cứu cho vay khách hàng của STB tại cuối năm 2024.")
    assert result.plan is not None
    assert result.plan.metric is not None
    assert result.plan.metric.canonical is None
    assert result.plan.metric.raw_text == "cho vay khách hàng"


def test_three_or_more_periods_abstains_with_operation_unknown() -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của NVL năm 2023.")
    entities = entities.model_copy(update={"periods": ("2021", "2022", "2023")})
    result = build_plan(entities, candidate_table_ids=(_TABLE_A,), known_table_ids=_KNOWN_TABLES)
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


def test_candidate_table_id_unknown_to_release_abstains() -> None:
    result = _plan_for("Tra cứu doanh thu thuần của NVL năm 2023.", candidate_table_ids=(_TABLE_B,))
    # _TABLE_B is deliberately outside _KNOWN_TABLES passed by _plan_for's default;
    # override known_table_ids directly to exercise the semantic-issue fallback.
    entities = parse_query_entities("Tra cứu doanh thu thuần của NVL năm 2023.")
    result = build_plan(
        entities,
        candidate_table_ids=("tbl_" + "9" * 64,),
        known_table_ids=_KNOWN_TABLES,
    )
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)
