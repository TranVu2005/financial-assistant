"""Day 18 locator: (MetricSelector, period) -> CellMatch (ADR 0007 decision D1).

Never guesses. Four outcomes only:

- no row matches the metric selector at all -> `metric_not_found`
- the metric matches somewhere, but not at the requested period -> `period_unresolved`
- exactly one distinct `(value, unit)` pair matches -> a `CellMatch`, evidence
  carrying every cell_id that agrees (Day 18 plan §1.4: duplicate rows are
  common — 35,766/766,710 groups — and most of them (93.2%) actually disagree)
- more than one distinct `(value, unit)` pair matches -> `cell_ambiguous`,
  never averaged, summed, or arbitrarily picked
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from financial_report_qa.execution.contracts import CellMatch, ExecutionIssueCode
from financial_report_qa.planning.plan_contracts import MetricSelector


@dataclass(frozen=True)
class LocateResult:
    """Either a resolved `CellMatch`, or a typed reason it could not resolve."""

    match: CellMatch | None
    error_code: ExecutionIssueCode | None
    error_message: str | None


def _metric_mask(frame: pd.DataFrame, selector: MetricSelector) -> pd.Series:
    if selector.canonical is not None:
        return frame["row_label_canonical"] == selector.canonical
    return frame["row_label_raw"] == selector.raw_text


def _selector_label(selector: MetricSelector) -> str | None:
    return selector.canonical if selector.canonical is not None else selector.raw_text


def locate(
    frame: pd.DataFrame,
    selector: MetricSelector,
    period: int,
    *,
    company_code: str | None = None,
) -> LocateResult:
    """Resolve one metric selector at one period within an already-scoped frame."""
    scoped = frame
    if company_code is not None:
        scoped = scoped[scoped["company_code"] == company_code]

    metric_rows = scoped[_metric_mask(scoped, selector)]
    if metric_rows.empty:
        return LocateResult(
            match=None,
            error_code="metric_not_found",
            error_message=f"no cell matches metric selector '{_selector_label(selector)}'",
        )

    period_rows = metric_rows[metric_rows["period"] == period]
    if period_rows.empty:
        return LocateResult(
            match=None,
            error_code="period_unresolved",
            error_message=(
                f"metric '{_selector_label(selector)}' has no cell resolved to period {period}"
            ),
        )

    distinct = period_rows.drop_duplicates(subset=["value", "unit"])
    if len(distinct) > 1:
        candidates = ", ".join(
            f"{row.value} {row.unit} (cell_id={row.cell_id})" for row in distinct.itertuples()
        )
        return LocateResult(
            match=None,
            error_code="cell_ambiguous",
            error_message=(
                f"metric '{_selector_label(selector)}' at period {period} "
                f"has conflicting values: {candidates}"
            ),
        )

    return LocateResult(
        match=CellMatch(
            table_id=str(period_rows["table_id"].iloc[0]),
            cell_ids=tuple(period_rows["cell_id"].tolist()),
            value=Decimal(str(distinct["value"].iloc[0])),
            unit=str(distinct["unit"].iloc[0]),
            period=period,
            period_inferred=bool(period_rows["period_inferred"].iloc[0]),
        ),
        error_code=None,
        error_message=None,
    )
