"""Day 26 column-refinement retry: add the missing cell dimension, once.

`locate()` narrows by row and period. A Vietnamese statement row routinely
spans several semantically distinct amounts -- the real PC1 2025 tax note puts
"Số phải nộp đầu năm", "Số phải nộp trong năm", "Số đã thực nộp" and "Số phải
nộp cuối năm" on the "Thuế giá trị gia tăng" row -- and two of those resolve to
the same closing year. Row plus period cannot separate them, so the locator
correctly abstains with `cell_ambiguous`: 117 of the 1,012 official questions
end there.

Deterministic column grounding was measured first and rejected: only 5 of those
117 questions name a column in their text, and inspection showed most of those
were row labels that happen to recur as headers. The column is genuinely absent
from the question, so it has to be read off the real headers -- a reading task,
not a rule, and the one shape small local models are measured to do well
(`llm_cell_grounding`).

This module holds only the plan surgery, with the chooser injected, so the
retry policy stays testable without a model and the module stays free of any
`submission`/`execution` import.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

# Every selector field a plan can carry a row selection in. `compare` and
# `ratio` hold two, so refining only `metric` would silently leave half of
# those plans un-narrowed.
_SELECTOR_FIELDS = ("metric", "metric_a", "metric_b", "numerator_metric", "denominator_metric")

ColumnChooser = Callable[[str, tuple[str, ...]], str | None]


def _selector_label(selector: MetricSelector) -> str:
    return selector.canonical if selector.canonical is not None else (selector.raw_text or "")


def plan_with_column(
    plan: FinancialQueryPlan,
    chooser: ColumnChooser,
    *,
    columns_for: Callable[[str], Sequence[str]] | None = None,
) -> FinancialQueryPlan | None:
    """Return `plan` with a column chosen for each selector, or `None`.

    `None` whenever the retry must not happen or did not help: a selector
    already carries a `column_text` (one bounded retry, never a loop that keeps
    re-narrowing), or the chooser declined for every selector. Declining leaves
    the original `cell_ambiguous` standing rather than filtering on a column
    nobody actually picked.
    """
    selectors = {
        name: getattr(plan, name)
        for name in _SELECTOR_FIELDS
        if getattr(plan, name, None) is not None
    }
    if not selectors:
        return None
    if any(selector.column_text is not None for selector in selectors.values()):
        return None

    updates: dict[str, MetricSelector] = {}
    for name, selector in selectors.items():
        label = _selector_label(selector)
        columns = tuple(columns_for(label)) if columns_for is not None else ()
        chosen = chooser(label, columns)
        if chosen is None:
            continue
        updates[name] = selector.model_copy(update={"column_text": chosen})

    if not updates:
        return None
    return plan.model_copy(update=updates)
