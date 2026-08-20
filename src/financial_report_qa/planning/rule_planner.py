"""Day 16 deterministic rule planner: `QueryEntities` -> `FinancialQueryPlan` | abstain.

Pure and side-effect-free: `candidate_table_ids` is injected by the caller
(retrieval already ran) rather than fetched here, keeping this module testable
without a corpus and preserving the module boundary from ADR 0001.

Never guesses. Every abstain carries a `PlanAbstainCode` explaining why, and
every returned plan has already passed `validate_plan_semantics` — a plan that
would fail semantic validation is never returned (see ADR 0005 §Hệ quả).
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.entity_parser import ordered_metric_canonicals
from financial_report_qa.planning.plan_contracts import (
    _PERIOD_PATTERN,
    CANONICAL_METRICS,
    FinancialQueryPlan,
    MetricSelector,
    PlanOperation,
    map_requested_unit,
)
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.retrieval.contracts import TableId, _FrozenModel

PlanAbstainCode = Literal[
    "operation_unknown",
    "metric_role_unassignable",
    "multi_metric_unsupported",
    "period_grammar_unsupported",
    "entity_ambiguous",
    # Three distinct client failures that must stay distinct: the endpoint
    # was unreachable, it rejected the request (4xx -- wrong model name or
    # an unsupported `response_format`), or it answered 200 with an
    # envelope that did not parse. Collapsing them into "unavailable"
    # pointed a Day 26 diagnosis at the server being down when its log
    # showed every request returning 200.
    "llm_unavailable",
    "llm_request_rejected",
    "llm_bad_response",
    "llm_invalid_json",
    "llm_plan_invalid",
]

_GROWTH_KEYWORDS = ("tăng trưởng", "biến động", "tốc độ")
_AGGREGATE_AVERAGE_KEYWORDS = ("trung bình", "bình quân")
_AGGREGATE_SUM_KEYWORDS = ("tổng cộng", "tổng giá trị", "tổng số", "tổng ")
_AGGREGATE_EXCLUDE_KEYWORDS = (
    "cao nhất",
    "thấp nhất",
    "nhiều nhất",
    "lớn nhất",
    "nhỏ nhất",
    "vượt",
    "trung vị",
    "đứng đầu",
    "dẫn đầu",
    "có bao nhiêu",
    "số công ty",
    "số doanh nghiệp",
)


def _infer_aggregate_operation(question: str) -> PlanOperation | None:
    lowered = question.lower()
    if any(keyword in lowered for keyword in _AGGREGATE_EXCLUDE_KEYWORDS):
        return None
    has_average = any(keyword in lowered for keyword in _AGGREGATE_AVERAGE_KEYWORDS)
    has_sum = any(keyword in lowered for keyword in _AGGREGATE_SUM_KEYWORDS)
    if has_average and not has_sum:
        return "average"
    if has_sum and not has_average:
        return "sum"
    return None


def _infer_operation(question: str, *, n_companies: int, n_periods: int) -> PlanOperation | None:
    if n_companies == 2 and n_periods == 1:
        return "compare_companies"
    if n_companies == 1 and n_periods == 1:
        return "lookup"
    if n_companies == 1 and n_periods == 2:
        lowered = question.lower()
        if any(keyword in lowered for keyword in _GROWTH_KEYWORDS):
            return "growth_rate"
        return "difference"
    if n_companies == 1 and n_periods >= 3:
        return _infer_aggregate_operation(question)
    if n_companies >= 3 and n_periods == 1:
        return _infer_aggregate_operation(question)
    return None


class RulePlanResult(_FrozenModel):
    """Exactly one of `plan` (fully valid) or `abstain_codes` (non-empty) is set.

    `repaired` is always `False` here (the rule planner never retries); the
    Day 17 LLM planner sets it `True` when a result required its one repair
    round trip, so `llm_evaluation.py` can report `repair_success_rate`
    without a parallel result type.
    """

    plan: FinancialQueryPlan | None = None
    abstain_codes: tuple[PlanAbstainCode, ...] = ()
    repaired: bool = False

    @model_validator(mode="after")
    def validate_exactly_one(self) -> Self:
        if (self.plan is None) == (not self.abstain_codes):
            raise ValueError("exactly one of plan or abstain_codes must be set")
        return self


def _abstain(code: PlanAbstainCode) -> RulePlanResult:
    return RulePlanResult(abstain_codes=(code,))


def _build_ratio_plan(
    entities: QueryEntities, *, candidate_table_ids: tuple[TableId, ...]
) -> FinancialQueryPlan | None:
    """(1 company, 1 period, 2 metrics, ratio keyword) -> numerator/denominator
    in question reading order -- "A trên B" means A/B (Day 23 plan Step 2).
    Returns `None` on any shape/keyword/construction mismatch; the caller
    falls back to the existing `multi_metric_unsupported` abstain, never a
    guess."""
    if len(entities.company_codes) != 1 or len(entities.periods) != 1:
        return None
    ordered = ordered_metric_canonicals(entities)
    if len(ordered) != 2:
        return None
    numerator, denominator = ordered
    try:
        return FinancialQueryPlan(
            operation="ratio",
            companies=entities.company_codes,
            periods=entities.periods,
            candidate_table_ids=candidate_table_ids,
            numerator_metric=MetricSelector(canonical=numerator),
            denominator_metric=MetricSelector(canonical=denominator),
            statement_scope=entities.statement_scope,
        )
    except ValueError:
        return None


def _metric_selector(entities: QueryEntities, metric: str) -> MetricSelector:
    if metric in CANONICAL_METRICS:
        return MetricSelector(canonical=metric)
    span = next((s for s in entities.spans if s.field == "metric"), None)
    return MetricSelector(raw_text=span.surface if span is not None else metric)


def build_plan(
    entities: QueryEntities,
    *,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
) -> RulePlanResult:
    """Deterministically compile parsed entities into a plan, or abstain.

    `candidate_table_ids` must be non-empty (supplied by retrieval upstream);
    this function does not call retrieval itself.
    """
    if entities.ambiguity:
        return _abstain("entity_ambiguous")

    periods = entities.periods
    metrics = entities.metrics
    metric_phrases = entities.metric_phrases

    if any(not _PERIOD_PATTERN.match(period) for period in periods):
        return _abstain("period_grammar_unsupported")

    target_metrics = metrics if metrics else metric_phrases
    n_metrics = len(target_metrics)

    operation = entities.operation
    if operation is None:
        operation = _infer_operation(
            entities.question, n_companies=len(entities.company_codes), n_periods=len(periods)
        )
    if operation is None:
        return _abstain("operation_unknown")

    # Map difference of 2 metrics to compare
    if operation == "difference" and n_metrics == 2:
        operation = "compare"

    # Build ratio plan
    if operation == "ratio":
        if len(metrics) == 2:
            ratio_plan = _build_ratio_plan(entities, candidate_table_ids=candidate_table_ids)
            if ratio_plan is not None and not validate_plan_semantics(
                ratio_plan, known_table_ids=known_table_ids
            ):
                return RulePlanResult(plan=ratio_plan)
        return _abstain("multi_metric_unsupported")

    # Build compare plan
    if operation == "compare":
        if n_metrics != 2:
            return _abstain("multi_metric_unsupported")
        metric_a_val = metrics[0] if metrics else metric_phrases[0]
        metric_b_val = metrics[1] if len(metrics) > 1 else metric_phrases[1]
        try:
            plan = FinancialQueryPlan(
                operation="compare",
                companies=entities.company_codes,
                periods=periods,
                candidate_table_ids=candidate_table_ids,
                metric_a=_metric_selector(entities, metric_a_val),
                metric_b=_metric_selector(entities, metric_b_val),
                expected_unit=map_requested_unit(entities.requested_unit),
                statement_scope=entities.statement_scope,
            )
        except ValueError:
            return _abstain("operation_unknown")

        if validate_plan_semantics(plan, known_table_ids=known_table_ids):
            return _abstain("operation_unknown")
        return RulePlanResult(plan=plan)

    # For other operations, we expect exactly 1 metric
    if n_metrics != 1:
        return _abstain("multi_metric_unsupported")

    metric_val = metrics[0] if metrics else metric_phrases[0]
    metric_selector = _metric_selector(entities, metric_val)
    try:
        plan = FinancialQueryPlan(
            operation=operation,
            companies=entities.company_codes,
            periods=periods,
            candidate_table_ids=candidate_table_ids,
            metric=metric_selector,
            expected_unit=map_requested_unit(entities.requested_unit),
            statement_scope=entities.statement_scope,
        )
    except ValueError:
        return _abstain("operation_unknown")

    if validate_plan_semantics(plan, known_table_ids=known_table_ids):
        return _abstain("operation_unknown")
    return RulePlanResult(plan=plan)
