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


def test_plan_defaults_statement_scope_to_none_when_unstated() -> None:
    result = _plan_for("Tra cứu doanh thu thuần của NVL năm 2023.")
    assert result.plan is not None
    assert result.plan.statement_scope is None


def test_plan_carries_statement_scope_parsed_from_question() -> None:
    """Day 21 plan §1.3/ADR 0010 decision A1: the entity parser resolves
    'công ty mẹ' to `separate`; that value must reach the plan, not be
    dropped between parser output and plan construction."""
    result = _plan_for("Tra cứu doanh thu thuần của công ty mẹ NVL năm 2023.")
    assert result.plan is not None
    assert result.plan.statement_scope == "separate"


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


def test_three_companies_one_period_does_not_silently_become_compare_companies() -> None:
    """Regression: `compile_compare_companies` only ever reads
    `companies[0]`/`companies[1]` (execution/compiler.py) -- routing a
    3+-company question there silently drops every company past the first
    two and answers a difference the question never asked for. Confirmed
    live in the Day 22/23 submission export: id 931 ("average across 5
    companies") and id 973 ("how many of 3 companies...") were both answered
    with a GEE-GEX / HPG-HSG style two-company subtraction before this fix."""
    result = _plan_for("So sánh doanh thu thuần giữa NVL, CTG và VNM năm 2023.")
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


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


def test_two_metrics_with_ratio_keyword_is_ratio_in_reading_order() -> None:
    """Day 23 plan Step 2: measured 25/26 real 2-metric questions are this
    'A trên B' = A/B ratio shape (e.g. ROA phrasing)."""
    result = _plan_for(
        "Lợi nhuận sau thuế trên tổng tài sản của DBC năm 2023 là bao nhiêu phần trăm?"
    )
    assert result.plan is not None
    assert result.plan.operation == "ratio"
    assert result.plan.numerator_metric is not None
    assert result.plan.numerator_metric.canonical == "profit_after_tax"
    assert result.plan.denominator_metric is not None
    assert result.plan.denominator_metric.canonical == "total_assets"


def test_two_metrics_ratio_keyword_but_two_periods_abstains() -> None:
    """The ratio fast path requires exactly 1 company and 1 period -- a
    2-metric, 2-period question is a shape Step 2 does not model; it must
    fall back to the existing multi_metric_unsupported abstain, not guess."""
    result = _plan_for(
        "Tỷ lệ lợi nhuận sau thuế trên tổng tài sản của DBC năm 2022 và 2023 "
        "là bao nhiêu phần trăm?"
    )
    assert result.plan is None
    assert result.abstain_codes == ("multi_metric_unsupported",)


def test_one_metric_three_periods_average_keyword_is_average() -> None:
    # "năm" repeated before each year: a bare comma-separated list ("năm
    # 2021, 2022 và 2023") is a separate, pre-existing entity-parser period
    # gap (only 2/3 years extracted) out of scope for Step 2 -- see
    # docs/plans/day23-coverage-and-evidence-table.md Step 3.
    result = _plan_for("Tính trung bình doanh thu thuần của DBC năm 2021, năm 2022 và năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "average"
    assert result.plan.companies == ("DBC",)
    assert result.plan.periods == ("2021", "2022", "2023")


def test_one_metric_three_periods_sum_keyword_is_sum() -> None:
    result = _plan_for("Tính tổng doanh thu thuần của DBC năm 2021, năm 2022 và năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "sum"


def test_one_metric_three_companies_average_keyword_is_average() -> None:
    result = _plan_for("Tính trung bình doanh thu thuần của DBC, NVL và CTG năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "average"
    assert result.plan.companies == ("CTG", "DBC", "NVL")
    assert result.plan.periods == ("2023",)


def test_one_metric_three_companies_sum_keyword_is_sum() -> None:
    result = _plan_for("Tính tổng doanh thu thuần của DBC, NVL và CTG năm 2023.")
    assert result.plan is not None
    assert result.plan.operation == "sum"


def test_one_metric_three_companies_no_aggregate_keyword_abstains() -> None:
    result = _plan_for("Doanh thu thuần của DBC, NVL và CTG năm 2023 là bao nhiêu?")
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


def test_one_metric_three_companies_superlative_keyword_does_not_become_average() -> None:
    """Guards against silently answering a composite/rank question (needs a
    metric-per-company ranking, not a flat aggregate) as if it were a plain
    average -- Day 23 plan Step 2 measured 6/35 real multi-company questions
    are this superlative shape and must stay unsupported, not guessed."""
    result = _plan_for(
        "Trung bình doanh thu thuần của DBC, NVL và CTG có giá trị cao nhất năm 2023 là bao nhiêu?"
    )
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


def test_one_metric_three_companies_count_question_does_not_become_sum() -> None:
    result = _plan_for("Tổng doanh thu thuần của DBC, NVL và CTG có bao nhiêu, năm 2023?")
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


def test_quarter_period_abstains_with_period_grammar_unsupported() -> None:
    """Date phrasings now normalize to bare fiscal years (spec §6.4), so the
    period-grammar gate is pinned with a quarter phrasing instead — "2023-Q4"
    is still a non-year period `FinancialQueryPlan` cannot accept."""
    result = _plan_for("Doanh thu thuần của DBC quý IV năm 2023 là bao nhiêu?")
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


def test_overlong_raw_metric_label_abstains_instead_of_raising() -> None:
    """Grounding recovery (cell_grounding.ground_with_recovery) injects raw
    corpus row labels into `entities.metrics` directly -- unbounded, unlike
    a question-parsed span. `MetricSelector.raw_text` caps at 512 chars; an
    OCR-merged label over that cap must abstain, not raise `ValidationError`
    uncaught out of `build_plan` (regression: this crashed a full export)."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của NVL năm 2023.")
    # Mirrors ground_with_recovery's candidate-switching rebuild: the metric
    # span is dropped (no span field="metric" survives an unrecognized-metric
    # parse) and the raw corpus row label is substituted directly for
    # `metrics`, bypassing `_metric_selector`'s canonical/span shortcuts.
    entities = entities.model_copy(
        update={
            "metrics": ("x" * 600,),
            "spans": tuple(span for span in entities.spans if span.field != "metric"),
        }
    )
    result = build_plan(entities, candidate_table_ids=(_TABLE_A,), known_table_ids=_KNOWN_TABLES)
    assert result.plan is None
    assert result.abstain_codes == ("operation_unknown",)


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


def test_compare_two_metrics_difference() -> None:
    result = _plan_for(
        "Chênh lệch giữa tài sản ngắn hạn và nợ phải trả của FPT năm 2021 là bao nhiêu?"
    )
    assert result.plan is not None
    assert result.plan.operation == "compare"
    assert result.plan.companies == ("FPT",)
    assert result.plan.periods == ("2021",)
    assert result.plan.metric_a is not None
    assert result.plan.metric_a.canonical == "current_assets"
    assert result.plan.metric_b is not None
    assert result.plan.metric_b.canonical == "total_liabilities"
