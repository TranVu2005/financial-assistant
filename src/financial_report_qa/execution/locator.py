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
from functools import lru_cache
from typing import cast

import pandas as pd

from financial_report_qa.execution.contracts import CellMatch, ExecutionIssueCode
from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.normalization.metrics import normalize_metric
from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import MetricSelector


@dataclass(frozen=True)
class LocateResult:
    """Either a resolved `CellMatch`, or a typed reason it could not resolve."""

    match: CellMatch | None
    error_code: ExecutionIssueCode | None
    error_message: str | None


@lru_cache(maxsize=100_000)
def _canonical_of_raw_label(label: str) -> str | None:
    """Re-derive a cell's canonical metric from its raw label at query time.

    The locked release baked `row_label_canonical` in at ingestion, before
    `normalize_metric` learned to ignore ordinal/note decoration -- so 57,466
    numeric cells whose label is an exact alias under a "1. "/"I. "/"(1)"
    wrapper are still NULL there (Day 23 measurement). Re-deriving here
    recovers them without re-ingesting, which would change
    `dataset_fingerprint` and invalidate every pinned baseline (ADR 0004
    §1.7). Cached because a candidate frame repeats the same few labels
    across companies, periods, and columns.
    """
    return normalize_metric(label).value


def _position_mask(frame: pd.DataFrame, selector: MetricSelector) -> pd.Series:
    """plan.md §14: select the one grounded row by position, nothing else.

    No label is consulted. Grounding already decided which row the question
    means; re-deriving that decision here from `row_label_raw` is exactly the
    semantic search this design moved out of Pandas.
    """
    assert selector.table_id is not None and selector.row_index is not None
    return (frame["table_id"] == selector.table_id) & (frame["row_idx"] == selector.row_index)


def _metric_mask(frame: pd.DataFrame, selector: MetricSelector) -> pd.Series:
    if selector.is_position_bound:
        return _position_mask(frame, selector)
    if selector.canonical is not None:
        by_column = frame["row_label_canonical"] == selector.canonical
        by_raw_label = frame["row_label_raw"].map(
            lambda value: (
                _canonical_of_raw_label(value) == selector.canonical
                if isinstance(value, str)
                else False
            )
        )
        return by_column | by_raw_label
    # Raw-text matching is normalization-tolerant (NFKC/casefold/whitespace
    # collapse), not byte-for-byte equality: Day 23 grounding sources
    # `raw_text` straight from a corpus `row_label_raw`, which can differ
    # from the selector's exact casing/spacing while naming the same cell.
    assert selector.raw_text is not None
    target = normalized_key(selector.raw_text)
    return frame["row_label_raw"].map(
        lambda value: normalized_key(value) == target if isinstance(value, str) else False
    )


def _column_mask(frame: pd.DataFrame, column_text: str) -> pd.Series:
    """Keep only cells whose column header names the requested column.

    Containment, not equality: extraction concatenates a header with the row
    above and with its unit row, so the real corpus label is "Số phải nộpcuối
    năm" or "Số cuối nămVND" rather than a clean caption.

    Whitespace is removed entirely, not merely collapsed, because that
    concatenation drops the separator: the real PC1 2025 header reads
    "nộpcuối", so a question saying "phải nộp cuối năm" would not otherwise
    match the very cell it names. Column headers are short and few per table,
    so the cross-word matches this admits are far cheaper than missing the
    dominant real-world form.
    """
    target = normalized_key(column_text).replace(" ", "")
    return frame["column_label"].map(
        lambda value: (
            target in normalized_key(value).replace(" ", "") if isinstance(value, str) else False
        )
    )


def _selector_label(selector: MetricSelector) -> str | None:
    return selector.canonical if selector.canonical is not None else selector.raw_text


def _prefer_statutory_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Narrow a conflicting row set to its Circular 200 statement rows.

    A note table can repeat a statement line's label with a different, narrower
    figure -- the CEO 2018 "Thu nhập khác" case in the Day 24 notes, where the
    statement line and a note breakdown row disagree. Only statement rows carry
    a "Mã số"; note tables have no such column at all (8,449 of 146,011 tables
    do). So a code is positive evidence that a row is the statement line the
    question means.

    Deliberately narrow: it fires only when the coded rows agree among
    themselves, so it resolves main-versus-note conflicts without ever picking
    between two genuine statement values (consolidated versus separate, which
    ADR 0010 measured disagreeing in 92.8% of cross-scope groups). Everything
    else falls through to `cell_ambiguous` unchanged.
    """
    if rows["value"].nunique() <= 1:
        return rows
    coded = rows[rows["statutory_code"].notna()]
    if coded.empty or coded["value"].nunique() != 1:
        return rows
    return coded


def _first_optional_int(rows: pd.DataFrame, column: str) -> int | None:
    if column not in rows.columns:
        return None
    value = rows[column].iloc[0]
    return None if pd.isna(value) else int(value)


def _first_optional_str(rows: pd.DataFrame, column: str) -> str | None:
    if column not in rows.columns:
        return None
    value = rows[column].iloc[0]
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def locate(
    frame: pd.DataFrame,
    selector: MetricSelector,
    period: int,
    *,
    company_code: str | None = None,
    prefer_statutory_rows: bool = False,
) -> LocateResult:
    """Resolve one metric selector at one period within an already-scoped frame.

    Statutory rows are only preferred through an explicit opt-in from a caller
    that has independent evidence the question targets a primary statement.
    A code identifies the source row, but cannot establish question intent.
    """
    scoped = frame
    if company_code is not None:
        scoped = scoped[scoped["company_code"] == company_code]

    metric_rows = scoped[_metric_mask(scoped, selector)]
    if selector.column_text is not None and not metric_rows.empty:
        metric_rows = metric_rows[_column_mask(metric_rows, selector.column_text)]
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

    if prefer_statutory_rows:
        period_rows = _prefer_statutory_rows(period_rows)
    distinct = period_rows.drop_duplicates(subset=["value", "unit"])
    known_units = distinct["unit"].dropna().drop_duplicates()
    if len(distinct) > 1:
        distinct_values = distinct["value"].drop_duplicates()
        # ADR 0010 decision C1: `(X, None)` and `(X, "VND")` are not a real
        # conflict when every row agrees on the value -- the NULL just means
        # one physical cell didn't record its unit. Only >=2 *known* units
        # disagreeing is a genuine conflict (measured: 859/868 false
        # positives were this NULL-split case, 9/868 were real).
        if not (len(distinct_values) == 1 and len(known_units) <= 1):
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

    resolved_unit = known_units.iloc[0] if len(known_units) == 1 else distinct["unit"].iloc[0]
    if pd.isna(resolved_unit):
        # ADR 0009 decision C1: a missing unit is a different failure than an
        # incompatible one. `str(resolved_unit)` here would be either 'None'
        # or the fabricated 'nan' (Day 20 plan Sec 1.3) -- neither is a real
        # CanonicalUnit, so this must be caught before CellMatch construction.
        return LocateResult(
            match=None,
            error_code="unit_missing",
            error_message=(
                f"metric '{_selector_label(selector)}' at period {period} has no recorded unit"
            ),
        )

    return LocateResult(
        match=CellMatch(
            table_id=str(period_rows["table_id"].iloc[0]),
            cell_ids=tuple(period_rows["cell_id"].tolist()),
            value=Decimal(str(distinct["value"].iloc[0])),
            unit=cast(CanonicalUnit, str(resolved_unit)),
            period=period,
            period_inferred=bool(period_rows["period_inferred"].iloc[0]),
            row_index=_first_optional_int(period_rows, "row_idx"),
            column_label=_first_optional_str(period_rows, "column_label"),
        ),
        error_code=None,
        error_message=None,
    )
