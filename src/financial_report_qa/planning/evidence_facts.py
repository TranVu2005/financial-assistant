"""plan.md §12: enumerate the grounded candidate facts a planner is shown.

The Evidence-Aware Planner's input is not the question plus a pile of table
text -- it is a short numbered list of cells that genuinely exist, each already
carrying its label, period, value and unit. This module builds that list by
reading the actual release for the rows row retrieval ranked, so the planner
has nothing left to invent and can only point at what it was given.

Ordering is by retrieval rank, not by table position: the planner reads the
list top-down and the best-retrieved row should be `F1`.

Two exclusions are deliberate:

- a cell with no resolved period cannot be addressed by an executable plan
  (`FinancialQueryPlan.periods` is required), so it is not offered;
- a cell with no recorded unit is dropped for the reason ADR 0009 decision C1
  gives -- a unitless figure cannot be arithmetic-checked downstream, and
  showing it to the planner invites an answer nothing can verify.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import SupportsInt, cast

import pandas as pd

from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.normalization._shared import sanitize_selector_text
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DEFAULT_MAX_FACTS = 24
"""Prompt budget, in facts. Deliberately far smaller than the 60-row label
menu `llm_cell_grounding` uses: each fact renders as a whole line with label,
period, value and unit, and §12's premise is that a small model does better
with a short menu than with a reasoning task."""


def _as_int(value: object) -> int:
    """`DataFrame.itertuples` is typed as a broad scalar union, so narrow once
    here rather than repeating a cast at every use site."""
    return int(cast(SupportsInt, value))


def enumerate_candidate_facts(
    release_dir: Path,
    fusion_rows: Sequence[RowFusedCandidate],
    *,
    company_code: str | None = None,
    periods: Sequence[int] = (),
    max_facts: int = DEFAULT_MAX_FACTS,
) -> tuple[GroundedFact, ...]:
    """The §12 fact menu for one question, ordered by row-retrieval rank."""
    if not fusion_rows:
        return ()

    rank_by_position: dict[tuple[str, int], int] = {}
    for candidate in fusion_rows:
        key = (candidate.table_id, candidate.row_idx)
        # A row can be retrieved more than once across weight branches; the
        # best rank it achieved is the one that should order its facts.
        if key not in rank_by_position or candidate.rank < rank_by_position[key]:
            rank_by_position[key] = candidate.rank

    table_ids = tuple(dict.fromkeys(candidate.table_id for candidate in fusion_rows))
    frame = build_cell_frame(release_dir, table_ids)
    if company_code is not None:
        frame = frame[frame["company_code"] == company_code]
    if periods:
        frame = frame[frame["period"].isin(list(periods))]

    rows = [
        (rank_by_position[key], row)
        for row in frame.itertuples()
        if (key := (str(row.table_id), _as_int(row.row_idx))) in rank_by_position
    ]
    rows.sort(key=lambda item: (item[0], _as_int(item[1].row_idx), _as_int(item[1].col_idx)))

    facts: list[GroundedFact] = []
    for rank, row in rows:
        if len(facts) >= max_facts:
            break
        if pd.isna(row.period) or row.unit is None or pd.isna(row.unit):
            continue
        label = row.row_label_raw if isinstance(row.row_label_raw, str) else None
        label = label or (
            row.row_label_canonical if isinstance(row.row_label_canonical, str) else None
        )
        if not label or not label.strip():
            continue
        column = row.column_label if isinstance(row.column_label, str) else None
        # A real corpus header can concatenate two source lines with an
        # embedded newline (cell_frame.py: extraction joins a header with the
        # row above and its unit row); plan_contracts.RawMetricText forbids
        # control characters outright, so this fact's `column`/`row_label`
        # must already be safe to drop straight into a MetricSelector.
        sanitized_column = sanitize_selector_text(column) if column else None
        facts.append(
            GroundedFact(
                fact_id=f"F{len(facts) + 1}",
                table_id=str(row.table_id),
                row_index=_as_int(row.row_idx),
                row_label=sanitize_selector_text(label),
                column=sanitized_column or None,
                company_code=str(row.company_code),
                period=_as_int(row.period),
                raw_value=Decimal(str(row.value)),
                unit=str(row.unit),  # type: ignore[arg-type]
                # `fused_score` mixes branch weights and is not comparable
                # across candidates (see `evidence_rendering.
                # plan_grounding_rank`); rank is. Reported as a reciprocal so
                # a higher number still reads as "more confident".
                grounding_score=1.0 / rank,
            )
        )
    return tuple(facts)
