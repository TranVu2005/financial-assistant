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
from financial_report_qa.planning.plan_contracts import (
    _PERIOD_PATTERN,
    CANONICAL_METRICS,
    FinancialQueryPlan,
    MetricSelector,
    PlanOperation,
)
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.retrieval.contracts import TableId, _FrozenModel

PlanAbstainCode = Literal[
    "operation_unknown",
    "metric_role_unassignable",
    "multi_metric_unsupported",
    "period_grammar_unsupported",
    "entity_ambiguous",
    "llm_unavailable",
    "llm_invalid_json",
    "llm_plan_invalid",
]

_GROWTH_KEYWORDS = ("tăng trưởng", "biến động", "tốc độ")


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


def _infer_operation(question: str, *, n_companies: int, n_periods: int) -> PlanOperation | None:
    if n_companies >= 2 and n_periods == 1:
        return "compare_companies"
    if n_companies == 1 and n_periods == 1:
        return "lookup"
    if n_companies == 1 and n_periods == 2:
        lowered = question.lower()
        if any(keyword in lowered for keyword in _GROWTH_KEYWORDS):
            return "growth_rate"
        return "difference"
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

    if any(not _PERIOD_PATTERN.match(period) for period in periods):
        return _abstain("period_grammar_unsupported")
    if len(metrics) > 1:
        return _abstain("multi_metric_unsupported")

    operation = _infer_operation(
        entities.question, n_companies=len(entities.company_codes), n_periods=len(periods)
    )
    if operation is None:
        return _abstain("operation_unknown")

    metric_selector = _metric_selector(entities, metrics[0])
    try:
        plan = FinancialQueryPlan(
            operation=operation,
            companies=entities.company_codes,
            periods=periods,
            candidate_table_ids=candidate_table_ids,
            metric=metric_selector,
            statement_scope=entities.statement_scope,
        )
    except ValueError:
        return _abstain("operation_unknown")

    if validate_plan_semantics(plan, known_table_ids=known_table_ids):
        return _abstain("operation_unknown")
    return RulePlanResult(plan=plan)
