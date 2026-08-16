"""Property tests for the Day 15 plan validator.

`_expected_ok` is an independent, hand-written restatement of the arity table
from docs/plans/day15-financial-query-plan.md — deliberately *not* sharing code
with `plan_validator.py` — so these tests catch divergence between the two,
not just confirm the validator agrees with itself.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.planning.plan_validator import validate_plan_semantics

_OPERATIONS = (
    "lookup",
    "compare",
    "compare_companies",
    "difference",
    "growth_rate",
    "ratio",
    "average",
    "sum",
    "rank",
)
_COMPANY_POOL = ("AAA", "BBB", "CCC", "DDD")
_PERIOD_POOL = ("2020", "2021", "2022", "2023")
_TABLE_ID = "tbl_" + "a" * 64
_KNOWN_TABLES = frozenset({_TABLE_ID})
_M = MetricSelector(canonical="revenue")


def _expected_ok(
    operation: str,
    n_companies: int,
    n_periods: int,
    periods_ascending: bool,
    has_metric: bool,
    has_metric_a: bool,
    has_metric_b: bool,
    has_numerator: bool,
    has_denominator: bool,
    top_k: int | None,
) -> bool:
    """Independent restatement of the operation arity table."""
    if operation in {"lookup", "compare", "difference", "growth_rate", "ratio"}:
        if n_companies != 1:
            return False
    elif operation == "compare_companies":
        # Day 23 plan Step 2: exactly 2, not >= 2 -- `compile_compare_companies`
        # only ever reads companies[0]/companies[1] (execution/compiler.py).
        if n_companies != 2:
            return False
    elif operation == "rank":
        if n_companies < 2:
            return False
    elif n_companies > 1 and n_periods > 1:  # average, sum
        return False
    elif n_companies == 1 and n_periods == 1:
        return False

    if operation in {"difference", "growth_rate"}:
        if n_periods != 2 or not periods_ascending:
            return False
    elif operation in {"lookup", "compare", "ratio", "rank", "compare_companies"}:
        if n_periods != 1:
            return False

    wants_metric = operation in {
        "lookup",
        "difference",
        "growth_rate",
        "average",
        "sum",
        "rank",
        "compare_companies",
    }
    if wants_metric != has_metric:
        return False

    wants_pair = operation == "compare"
    if wants_pair != (has_metric_a and has_metric_b):
        return False
    if has_metric_a != wants_pair or has_metric_b != wants_pair:
        return False

    wants_ratio_pair = operation == "ratio"
    if wants_ratio_pair != (has_numerator and has_denominator):
        return False
    if has_numerator != wants_ratio_pair or has_denominator != wants_ratio_pair:
        return False

    wants_top_k = operation == "rank"
    has_top_k = top_k is not None
    if wants_top_k != has_top_k:
        return False
    if wants_top_k and top_k is not None and not 1 <= top_k < n_companies:
        return False

    return True


@st.composite
def _plan_candidates(draw: st.DrawFn) -> FinancialQueryPlan | None:
    operation = draw(st.sampled_from(_OPERATIONS))
    companies = tuple(
        draw(st.permutations(_COMPANY_POOL))[: draw(st.integers(min_value=1, max_value=4))]
    )
    periods_all = draw(st.permutations(_PERIOD_POOL))[: draw(st.integers(min_value=1, max_value=4))]
    ascending = draw(st.booleans())
    periods = tuple(sorted(periods_all)) if ascending else tuple(sorted(periods_all, reverse=True))
    if len(periods) < 2:
        ascending = True  # a single period is trivially "ordered"

    has_metric = draw(st.booleans())
    has_metric_a = draw(st.booleans())
    has_metric_b = draw(st.booleans())
    has_numerator = draw(st.booleans())
    has_denominator = draw(st.booleans())
    top_k = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=5)))

    kwargs: dict[str, object] = {
        "operation": operation,
        "companies": companies,
        "periods": periods,
        "candidate_table_ids": (_TABLE_ID,),
    }
    if has_metric:
        kwargs["metric"] = _M
    if has_metric_a:
        kwargs["metric_a"] = _M
    if has_metric_b:
        kwargs["metric_b"] = _M
    if has_numerator:
        kwargs["numerator_metric"] = _M
    if has_denominator:
        kwargs["denominator_metric"] = _M
    if top_k is not None:
        kwargs["top_k"] = top_k

    try:
        plan = FinancialQueryPlan(**kwargs)  # type: ignore[arg-type]
    except ValidationError:
        return None
    return plan


@given(_plan_candidates())
@settings(max_examples=300)
def test_validator_agrees_with_independent_arity_spec(plan: FinancialQueryPlan | None) -> None:
    if plan is None:
        return
    issues = validate_plan_semantics(plan, known_table_ids=_KNOWN_TABLES)
    expected = _expected_ok(
        operation=plan.operation,
        n_companies=len(plan.companies),
        n_periods=len(plan.periods),
        periods_ascending=tuple(sorted(plan.periods)) == plan.periods,
        has_metric=plan.metric is not None,
        has_metric_a=plan.metric_a is not None,
        has_metric_b=plan.metric_b is not None,
        has_numerator=plan.numerator_metric is not None,
        has_denominator=plan.denominator_metric is not None,
        top_k=plan.top_k,
    )
    assert (issues == ()) == expected, (
        f"operation={plan.operation} companies={plan.companies} periods={plan.periods} "
        f"top_k={plan.top_k} issues={issues} expected_ok={expected}"
    )


@given(_plan_candidates())
@settings(max_examples=200)
def test_validation_is_deterministic(plan: FinancialQueryPlan | None) -> None:
    if plan is None:
        return
    first = validate_plan_semantics(plan, known_table_ids=_KNOWN_TABLES)
    second = validate_plan_semantics(plan, known_table_ids=_KNOWN_TABLES)
    assert first == second


@given(_plan_candidates())
@settings(max_examples=200)
def test_plan_round_trips_through_json(plan: FinancialQueryPlan | None) -> None:
    if plan is None:
        return
    dumped = plan.model_dump(mode="json")
    assert FinancialQueryPlan.model_validate(dumped) == plan
