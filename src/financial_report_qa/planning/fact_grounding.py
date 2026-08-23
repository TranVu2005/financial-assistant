"""plan.md §9/§14: turn label-based selectors into positional ones.

Two pure functions, no filesystem and no retrieval calls:

- ``bind_plan_to_rows`` takes a plan whose selectors still name rows by label
  and the row-fusion candidates that produced those labels, and returns a copy
  whose selectors are pinned to ``(table_id, row_index)``. That is the moment
  semantic matching stops: from there the locator and the rendered
  ``df.loc[...]`` do positional extraction only (§14).
- ``grounded_facts`` reads the compiled result back out as §9 ``GroundedFact``
  provenance -- one fact per evidence cell, each carrying the exact row it came
  from alongside its label, column, period, value and unit.

Binding is all-or-nothing per plan. A plan with one selector pinned and another
still matched by label would answer from two different grounding regimes at
once, which is exactly the ambiguity §14 exists to remove -- so a plan that
cannot be fully bound is reported as unbindable and the caller keeps the
label-based plan unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

_SELECTOR_FIELDS = (
    "metric",
    "metric_a",
    "metric_b",
    "numerator_metric",
    "denominator_metric",
)


def is_position_bound_plan(plan: FinancialQueryPlan) -> bool:
    """Whether every selector `plan` carries is already pinned to a row.

    Such a plan has nothing left for `bind_plan_to_rows` to decide, and
    re-deciding is not neutral: binding picks by label plus lowest rank, so it
    can only *replace* a position that was chosen deliberately.
    """
    selectors = [
        selector
        for selector in (getattr(plan, field) for field in _SELECTOR_FIELDS)
        if selector is not None
    ]
    return bool(selectors) and all(selector.is_position_bound for selector in selectors)


def _matches(selector: MetricSelector, candidate: RowFusedCandidate) -> bool:
    """Whether `candidate` is the row this selector was built from.

    Matching is normalization-tolerant on the raw branch for the same reason
    `locator._metric_mask` is: a selector's `raw_text` is copied out of a row
    label and can differ from it in casing or whitespace while naming the same
    row. This is the *last* place a label comparison happens -- afterwards the
    selector carries a position and nothing re-derives it.
    """
    if selector.canonical is not None:
        return candidate.metadata.row_label_canonical == selector.canonical
    assert selector.raw_text is not None
    label = candidate.metadata.row_label_raw
    return isinstance(label, str) and normalized_key(label) == normalized_key(selector.raw_text)


def _best_candidate(
    selector: MetricSelector,
    candidates: Sequence[RowFusedCandidate],
) -> RowFusedCandidate | None:
    matching = [candidate for candidate in candidates if _matches(selector, candidate)]
    return min(matching, key=lambda candidate: candidate.rank) if matching else None


def bind_plan_to_rows(
    plan: FinancialQueryPlan,
    fusion_rows: Sequence[RowFusedCandidate],
) -> FinancialQueryPlan | None:
    """Return `plan` with every selector pinned to a physical row, or `None`.

    `None` means "this plan cannot be grounded positionally from these
    candidates" -- not an error. The caller keeps its label-based plan.
    """
    # A single selector serving several companies cannot be a single position:
    # each company's figure lives in its own table, at its own row index.
    # plan.md §12's per-fact operand model is what generalizes this; faking it
    # here would silently answer every company from one company's row.
    if len(plan.companies) > 1:
        return None
    company = plan.companies[0]
    scoped = [
        candidate
        for candidate in fusion_rows
        if candidate.table_id in plan.candidate_table_ids
        and candidate.metadata.company_code in (None, company)
    ]
    if not scoped:
        return None

    updates: dict[str, MetricSelector] = {}
    for field in _SELECTOR_FIELDS:
        selector = getattr(plan, field)
        if selector is None:
            continue
        candidate = _best_candidate(selector, scoped)
        if candidate is None:
            return None
        updates[field] = selector.model_copy(
            update={"table_id": candidate.table_id, "row_index": candidate.row_idx}
        )
    if not updates:
        return None
    return plan.model_copy(update=updates)


def grounded_facts(
    compiled: CompiledQuery,
    *,
    grounding_score: float | None,
) -> tuple[GroundedFact, ...]:
    """The §9 provenance record for one compiled answer, one fact per cell.

    `evidence` and `replay_rows` are produced side by side and in the same
    order by every branch of `compiler._dispatch`, so they zip: the cell
    supplies position, value, unit and period, the replay row supplies the
    label the question was grounded through.
    """
    if compiled.status != "answered":
        return ()
    facts: list[GroundedFact] = []
    for cell, replay in zip(compiled.evidence, compiled.replay_rows, strict=False):
        if cell.row_index is None:
            continue
        label = replay.row_label_raw or replay.row_label_canonical
        if not label:
            continue
        facts.append(
            GroundedFact(
                fact_id=f"F{len(facts) + 1}",
                table_id=cell.table_id,
                row_index=cell.row_index,
                row_label=label,
                column=cell.column_label,
                period=cell.period,
                raw_value=cell.value,
                unit=cell.unit,
                grounding_score=grounding_score,
            )
        )
    return tuple(facts)
