"""Unit tests for the Day 16 plan-case labels (16.1).

Every expected label is a pure function of `EntityCase.template_id` — never
derived by running `rule_planner.build_plan` and copying its output. That
keeps this an evaluation set instead of a mirror, matching the discipline
`entity_cases.py` already uses for Day 10.
"""

from __future__ import annotations

from financial_report_qa.planning.entity_cases import EntityCase
from financial_report_qa.planning.plan_cases import (
    generate_plan_cases,
    plan_case_set_sha256,
    write_plan_cases,
)


def _case(template_id: str, **overrides: object) -> EntityCase:
    defaults: dict[str, object] = {
        "case_id": f"ecase_{template_id}",
        "template_id": template_id,
        "category": "lookup",
        "question": "irrelevant text for this test",
        "expected_company_codes": ("NVL",),
        "expected_periods": ("2023",),
        "expected_metrics": ("net_revenue",),
    }
    defaults.update(overrides)
    return EntityCase(**defaults)  # type: ignore[arg-type]


def test_single_period_lookup_templates_expect_lookup() -> None:
    for template_id in ("lookup_ticker", "lookup_name", "statement_lookup"):
        plan_cases = generate_plan_cases((_case(template_id),))
        assert plan_cases[0].expected_operation == "lookup"
        assert plan_cases[0].expected_abstain_code is None


def test_compare_years_expects_difference_not_compare() -> None:
    """Day 16 finding #2: this shape (1 company, 2 periods, 1 metric) maps to
    `difference`, never the Day 15 `compare` operation (two metrics)."""
    case = _case(
        "compare_years",
        category="compare",
        expected_periods=("2022", "2023"),
    )
    plan_cases = generate_plan_cases((case,))
    assert plan_cases[0].expected_operation == "difference"


def test_growth_years_expects_growth_rate() -> None:
    case = _case("growth_years", category="growth", expected_periods=("2022", "2023"))
    plan_cases = generate_plan_cases((case,))
    assert plan_cases[0].expected_operation == "growth_rate"


def test_two_companies_expects_compare_companies() -> None:
    case = _case(
        "two_companies",
        category="adversarial",
        expected_company_codes=("CTG", "NVL"),
    )
    plan_cases = generate_plan_cases((case,))
    assert plan_cases[0].expected_operation == "compare_companies"


def test_quarter_and_date_lookup_expect_period_grammar_abstain() -> None:
    for template_id in ("quarter_lookup", "date_lookup"):
        case = _case(template_id, expected_periods=("2023-Q4",))
        plan_cases = generate_plan_cases((case,))
        assert plan_cases[0].expected_operation is None
        assert plan_cases[0].expected_abstain_code == "period_grammar_unsupported"


def test_ambiguous_templates_expect_entity_ambiguous_abstain() -> None:
    for template_id in (
        "missing_company",
        "company_conflict",
        "relative_period",
        "incomplete_quarter",
        "unknown_metric",
        "missing_period",
    ):
        case = _case(
            template_id,
            category="adversarial",
            expected_company_codes=(),
            expected_ambiguity=("period_missing",),
        )
        plan_cases = generate_plan_cases((case,))
        assert plan_cases[0].expected_operation is None
        assert plan_cases[0].expected_abstain_code == "entity_ambiguous"


def test_plan_case_set_sha256_is_deterministic() -> None:
    cases = (_case("lookup_ticker"), _case("growth_years", expected_periods=("2022", "2023")))
    plan_cases = generate_plan_cases(cases)
    assert plan_case_set_sha256(plan_cases) == plan_case_set_sha256(plan_cases)


def test_write_plan_cases_round_trips(tmp_path: object) -> None:
    from pathlib import Path

    from financial_report_qa.planning.plan_cases import load_plan_cases

    cases = (_case("lookup_ticker"),)
    plan_cases = generate_plan_cases(cases)
    path = Path(str(tmp_path)) / "plan-cases-v1.jsonl"
    write_plan_cases(plan_cases, path)
    loaded = load_plan_cases(path)
    assert loaded == plan_cases
