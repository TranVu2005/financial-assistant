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

# Day 23 plan Step 2: measured 25/26 real 2-metric, 1-company, 1-period
# questions in the official 1.012-question set are this exact "A trên/trong
# B" ratio shape (e.g. ROA phrasing); 0/26 are the no-keyword `compare`
# shape, so `compare` is deliberately not inferred here -- no measured
# evidence to route on.
_RATIO_KEYWORDS = ("tỷ lệ", "tỷ trọng", "tỷ số", "phần trăm", "%")

_AGGREGATE_AVERAGE_KEYWORDS = ("trung bình", "bình quân")
_AGGREGATE_SUM_KEYWORDS = ("tổng cộng", "tổng giá trị", "tổng số", "tổng ")
# Composite/rank ("company with the highest X") and count ("how many
# companies satisfy...") questions are a different shape than a flat
# average/sum: Day 23 plan Step 2 measured 6/35 and 4/35 real multi-company
# candidates are these shapes respectively -- silently averaging/summing
# anyway would answer a different question than asked.
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


def _infer_aggregate_operation(question: str) -> PlanOperation | None:
    """average vs sum, from keyword; `None` for neither, both (ambiguous), or
    a composite/rank/count shape this operation does not model (Day 23 plan
    Step 2)."""
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
    # Exactly 2, not >= 2: `compile_compare_companies` (execution/compiler.py)
    # only ever reads `companies[0]`/`companies[1]` -- routing a 3+-company
    # question here would silently drop every company past the first two and
    # answer a two-way difference the question never asked for.
    if n_companies == 2 and n_periods == 1:
        return "compare_companies"
    if n_companies == 1 and n_periods == 1:
        return "lookup"
    if n_companies == 1 and n_periods == 2:
        lowered = question.lower()
        if any(keyword in lowered for keyword in _GROWTH_KEYWORDS):
            return "growth_rate"
        return "difference"
    # `_validate_aggregate` requires exactly one of (companies, periods) to
    # vary (len > 1) with the other pinned to a single value -- both shapes
    # below satisfy that and `execution/compiler.py`'s average/sum dispatch
    # implements both (Day 23 plan Step 2).
    if n_companies == 1 and n_periods >= 3:
        return _infer_aggregate_operation(question)
    if n_companies >= 3 and n_periods == 1:
        return _infer_aggregate_operation(question)
    return None


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
    if not any(keyword in entities.question.lower() for keyword in _RATIO_KEYWORDS):
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

    if any(not _PERIOD_PATTERN.match(period) for period in periods):
        return _abstain("period_grammar_unsupported")

    if len(metrics) == 2:
        ratio_plan = _build_ratio_plan(entities, candidate_table_ids=candidate_table_ids)
        if ratio_plan is None or validate_plan_semantics(
            ratio_plan, known_table_ids=known_table_ids
        ):
            return _abstain("multi_metric_unsupported")
        return RulePlanResult(plan=ratio_plan)
    if len(metrics) != 1:
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
