"""Day 15 semantic validator: per-operation arity table and release-bound checks.

Structural well-formedness (types, enums, `extra="forbid"`) is already guaranteed
by `FinancialQueryPlan` at construction time — everything here assumes a
structurally valid plan and only asks "is this plan consistent with what its
`operation` requires, and does `candidate_table_ids` exist in this release?"
See ADR 0004 for why this is a separate layer from the pydantic schema.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.retrieval.contracts import NonEmptyString, _FrozenModel

PlanField = Literal[
    "companies",
    "periods",
    "metric",
    "metric_pair",
    "numerator_denominator",
    "top_k",
    "expected_unit",
    "candidate_table_ids",
]

PlanErrorCode = Literal[
    "companies_arity_invalid",
    "periods_arity_invalid",
    "periods_not_chronological",
    "metric_arity_invalid",
    "metric_pair_arity_invalid",
    "numerator_denominator_arity_invalid",
    "top_k_arity_invalid",
    "top_k_out_of_range",
    "expected_unit_mismatch",
    "candidate_table_ids_unknown",
]

PLAN_ISSUE_FIELD_BY_CODE: dict[PlanErrorCode, PlanField] = {
    "companies_arity_invalid": "companies",
    "periods_arity_invalid": "periods",
    "periods_not_chronological": "periods",
    "metric_arity_invalid": "metric",
    "metric_pair_arity_invalid": "metric_pair",
    "numerator_denominator_arity_invalid": "numerator_denominator",
    "top_k_arity_invalid": "top_k",
    "top_k_out_of_range": "top_k",
    "expected_unit_mismatch": "expected_unit",
    "candidate_table_ids_unknown": "candidate_table_ids",
}

_CURRENCY_UNITS = frozenset({"VND", "VND_thousand", "VND_million", "VND_billion"})
_RATIO_UNITS = frozenset({"ratio", "percent"})


class PlanValidationIssue(_FrozenModel):
    """One arity or provenance violation, typed for LLM-repair prompts (Day 17)."""

    code: PlanErrorCode
    field: PlanField
    message: NonEmptyString

    @model_validator(mode="after")
    def validate_code_field_pair(self) -> Self:
        expected = PLAN_ISSUE_FIELD_BY_CODE[self.code]
        if self.field != expected:
            raise ValueError(f"issue code {self.code} requires field {expected}")
        return self


def _issue(code: PlanErrorCode, message: str) -> PlanValidationIssue:
    return PlanValidationIssue(code=code, field=PLAN_ISSUE_FIELD_BY_CODE[code], message=message)


def _check_single_metric(
    plan: FinancialQueryPlan, *, required: bool
) -> tuple[PlanValidationIssue, ...]:
    present = plan.metric is not None
    if required and not present:
        return (_issue("metric_arity_invalid", "operation requires metric"),)
    if not required and present:
        return (_issue("metric_arity_invalid", "operation forbids metric"),)
    return ()


def _check_forbidden_extras(
    plan: FinancialQueryPlan,
    *,
    allow_metric_pair: bool = False,
    allow_ratio_pair: bool = False,
    allow_top_k: bool = False,
) -> tuple[PlanValidationIssue, ...]:
    """Every operation-specific field not explicitly allowed must be absent.

    Declarative on purpose: a per-operation function that forgets to call
    `_check_forbidden` for one field silently lets that field through. Listing
    what IS allowed instead means a new field defaults to forbidden everywhere.
    """
    issues: tuple[PlanValidationIssue, ...] = ()
    if not allow_metric_pair:
        if plan.metric_a is not None:
            issues = (*issues, _issue("metric_pair_arity_invalid", "operation forbids metric_a"))
        if plan.metric_b is not None:
            issues = (*issues, _issue("metric_pair_arity_invalid", "operation forbids metric_b"))
    if not allow_ratio_pair:
        if plan.numerator_metric is not None:
            issues = (
                *issues,
                _issue("numerator_denominator_arity_invalid", "operation forbids numerator_metric"),
            )
        if plan.denominator_metric is not None:
            issues = (
                *issues,
                _issue(
                    "numerator_denominator_arity_invalid", "operation forbids denominator_metric"
                ),
            )
    if not allow_top_k and plan.top_k is not None:
        issues = (*issues, _issue("top_k_arity_invalid", "operation forbids top_k"))
    return issues


def _check_periods_count(
    plan: FinancialQueryPlan, *, expected: int
) -> tuple[PlanValidationIssue, ...]:
    if len(plan.periods) != expected:
        return (
            _issue(
                "periods_arity_invalid",
                f"operation requires exactly {expected} period(s), got {len(plan.periods)}",
            ),
        )
    return ()


def _check_companies_count(
    plan: FinancialQueryPlan, *, expected: int
) -> tuple[PlanValidationIssue, ...]:
    if len(plan.companies) != expected:
        return (
            _issue(
                "companies_arity_invalid",
                f"operation requires exactly {expected} compan(y/ies), got {len(plan.companies)}",
            ),
        )
    return ()


def _check_chronological_periods(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    if len(plan.periods) == 2 and tuple(sorted(plan.periods)) != plan.periods:
        return (
            _issue(
                "periods_not_chronological",
                "periods must be ordered earliest first for a directional operation",
            ),
        )
    return ()


def _check_expected_unit(
    plan: FinancialQueryPlan, *, allowed: frozenset[str]
) -> tuple[PlanValidationIssue, ...]:
    if plan.expected_unit is not None and plan.expected_unit not in allowed:
        return (
            _issue(
                "expected_unit_mismatch",
                f"expected_unit '{plan.expected_unit}' is not valid for "
                f"operation '{plan.operation}'",
            ),
        )
    return ()


def _validate_lookup(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    return (
        *_check_companies_count(plan, expected=1),
        *_check_periods_count(plan, expected=1),
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )


def _validate_compare(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    issues = (
        *_check_companies_count(plan, expected=1),
        *_check_periods_count(plan, expected=1),
        *_check_single_metric(plan, required=False),
        *_check_forbidden_extras(plan, allow_metric_pair=True),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )
    if plan.metric_a is None or plan.metric_b is None:
        issues = (
            *issues,
            _issue("metric_pair_arity_invalid", "operation requires both metric_a and metric_b"),
        )
    return issues


def _validate_difference(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    return (
        *_check_companies_count(plan, expected=1),
        *_check_periods_count(plan, expected=2),
        *_check_chronological_periods(plan),
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )


def _validate_growth_rate(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    return (
        *_check_companies_count(plan, expected=1),
        *_check_periods_count(plan, expected=2),
        *_check_chronological_periods(plan),
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan),
        # ADR 0009 decision B1: `ratio` is the computed unit (executor
        # hardcodes it), `percent` is its presentation form -- both are valid
        # declarations, not a contradiction to resolve at validation time.
        *_check_expected_unit(plan, allowed=frozenset({"percent", "ratio"})),
    )


def _validate_ratio(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    issues = (
        *_check_companies_count(plan, expected=1),
        *_check_periods_count(plan, expected=1),
        *_check_single_metric(plan, required=False),
        *_check_forbidden_extras(plan, allow_ratio_pair=True),
        *_check_expected_unit(plan, allowed=_RATIO_UNITS),
    )
    if plan.numerator_metric is None or plan.denominator_metric is None:
        issues = (
            *issues,
            _issue(
                "numerator_denominator_arity_invalid",
                "operation requires both numerator_metric and denominator_metric",
            ),
        )
    return issues


def _validate_aggregate(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    """Shared arity for `average`/`sum`: exactly one dimension varies."""
    varying = (len(plan.companies) > 1, len(plan.periods) > 1)
    issues: tuple[PlanValidationIssue, ...] = (
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )
    if varying.count(True) != 1:
        issues = (
            *issues,
            _issue(
                "companies_arity_invalid",
                "operation requires exactly one of companies/periods to vary (len > 1) "
                "and the other to be a single value",
            ),
        )
    return issues


def _validate_compare_companies(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    """`(exactly 2 companies, 1 period, 1 metric)` — one metric's difference
    between exactly two companies.

    Distinct from `compare` (two metrics, one company/period) and `rank`
    (needs `top_k`, and genuinely generalizes to N companies). Day 23 plan
    Step 2 tightened this from ADR 0005 decision A1's original ">= 2
    companies": `compile_compare_companies` (execution/compiler.py) only
    ever reads `companies[0]`/`companies[1]`, so a 3+-company plan would
    silently drop every company past the first two and answer a two-way
    difference the question never asked for -- confirmed live in the Day
    22/23 submission export (question ids 931, 973).
    """
    issues = (
        *_check_periods_count(plan, expected=1),
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )
    if len(plan.companies) != 2:
        issues = (
            *issues,
            _issue("companies_arity_invalid", "compare_companies requires exactly 2 companies"),
        )
    return issues


def _validate_rank(plan: FinancialQueryPlan) -> tuple[PlanValidationIssue, ...]:
    issues = (
        *_check_periods_count(plan, expected=1),
        *_check_single_metric(plan, required=True),
        *_check_forbidden_extras(plan, allow_top_k=True),
        *_check_expected_unit(plan, allowed=_CURRENCY_UNITS),
    )
    if len(plan.companies) < 2:
        issues = (
            *issues,
            _issue("companies_arity_invalid", "rank requires at least 2 companies"),
        )
    if plan.top_k is None:
        issues = (*issues, _issue("top_k_arity_invalid", "rank requires top_k"))
    elif not 1 <= plan.top_k < len(plan.companies):
        issues = (
            *issues,
            _issue(
                "top_k_out_of_range",
                f"top_k must satisfy 1 <= top_k < {len(plan.companies)} (companies count)",
            ),
        )
    return issues


_VALIDATORS = {
    "lookup": _validate_lookup,
    "compare": _validate_compare,
    "compare_companies": _validate_compare_companies,
    "difference": _validate_difference,
    "growth_rate": _validate_growth_rate,
    "ratio": _validate_ratio,
    "average": _validate_aggregate,
    "sum": _validate_aggregate,
    "rank": _validate_rank,
}


def validate_plan_semantics(
    plan: FinancialQueryPlan, *, known_table_ids: frozenset[str]
) -> tuple[PlanValidationIssue, ...]:
    """Reject a structurally valid plan whose fields do not match its operation."""
    issues = _VALIDATORS[plan.operation](plan)
    unknown = tuple(sorted(set(plan.candidate_table_ids) - known_table_ids))
    if unknown:
        issues = (
            *issues,
            _issue(
                "candidate_table_ids_unknown",
                f"candidate_table_ids not found in release: {', '.join(unknown)}",
            ),
        )
    return issues
