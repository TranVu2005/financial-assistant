"""Day 17 LLM-facing plan contracts (§17.5).

`LLMPlanOutput` is `FinancialQueryPlan` minus `candidate_table_ids`: ADR 0006
decision A/B and Day 17 plan §1.5 measured that 12 candidate table ids cost
192-400 tokens against a 160-token output budget, and that a 64-hex-char id
is exactly the kind of string an LLM fabricates worst. The caller (retrieval,
already run) injects `candidate_table_ids` after parsing, the same way
`rule_planner.build_plan` already does.
"""

from __future__ import annotations

from financial_report_qa.planning.plan_contracts import (
    ExpectedUnit,
    FinancialQueryPlan,
    MetricSelector,
    PlanOperation,
)
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel


class LLMPlanOutput(_FrozenModel):
    """`FinancialQueryPlan` fields the LLM is responsible for producing."""

    operation: PlanOperation
    companies: tuple[NonEmptyString, ...]
    periods: tuple[NonEmptyString, ...]
    metric: MetricSelector | None = None
    metric_a: MetricSelector | None = None
    metric_b: MetricSelector | None = None
    numerator_metric: MetricSelector | None = None
    denominator_metric: MetricSelector | None = None
    top_k: int | None = None
    expected_unit: ExpectedUnit | None = None


def to_financial_query_plan(
    output: LLMPlanOutput, *, candidate_table_ids: tuple[TableId, ...]
) -> FinancialQueryPlan:
    """Inject caller-supplied `candidate_table_ids` and construct the full plan.

    Raises the same `ValueError` `FinancialQueryPlan` itself would raise for a
    structurally invalid combination of fields — this function adds no
    validation of its own beyond what the schema already enforces.
    """
    return FinancialQueryPlan(
        operation=output.operation,
        companies=output.companies,
        periods=output.periods,
        candidate_table_ids=candidate_table_ids,
        metric=output.metric,
        metric_a=output.metric_a,
        metric_b=output.metric_b,
        numerator_metric=output.numerator_metric,
        denominator_metric=output.denominator_metric,
        top_k=output.top_k,
        expected_unit=output.expected_unit,
    )
