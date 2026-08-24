"""Build the numbered cell list the masked-PAL decision step chooses from.

Order is the contract: `ProgramDecision.cells` are positions in this list, so
two runs that produce a different order produce different decisions. Cells are
emitted in row-fusion rank order, then by `col_idx`, so the highest-ranked row
gets the lowest indices and truncation drops the least-likely rows first.

No candidate carries a value. `build_cell_frame` already filters out cells
with no `value_numeric`, so anything in the frame is bindable; this module
must not add a cell that is not in the frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsInt, cast

import pandas as pd

from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DEFAULT_MAX_CANDIDATES = 200


def _row_path(label: str, group: str | None) -> str:
    if group and group.strip():
        return f"{group.strip()} > {label}"
    return label


def _optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text else None


def _frame_int(value: object) -> int:
    """Coerce a numeric frame scalar to ``int``.

    pandas-stubs types ``itertuples`` fields as a broad scalar union, so the
    direct ``int(...)`` the runtime needs does not type-check; every field
    coerced here is a numpy integer in practice.
    """
    return int(cast(SupportsInt, value))


def build_cell_candidates(
    frame: pd.DataFrame,
    row_candidates: Sequence[RowFusedCandidate],
    *,
    periods: Sequence[str] = (),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[CellCandidate, ...]:
    """Number every numeric cell of the ranked rows, best-ranked row first."""
    wanted_periods = {int(period) for period in periods if str(period).isdigit()}
    candidates: list[CellCandidate] = []
    for row_candidate in sorted(row_candidates, key=lambda item: item.rank):
        rows = frame[
            (frame["table_id"] == row_candidate.table_id)
            & (frame["row_idx"] == row_candidate.row_idx)
        ].sort_values("col_idx")
        for row in rows.itertuples():
            period = None if pd.isna(row.period) else _frame_int(row.period)
            if wanted_periods and period not in wanted_periods:
                continue
            if len(candidates) >= max_candidates:
                return tuple(candidates)
            label_raw = str(row.row_label_raw)
            candidates.append(
                CellCandidate(
                    index=len(candidates),
                    table_id=str(row.table_id),
                    company_code=_optional_str(row.company_code),
                    row_idx=_frame_int(row.row_idx),
                    col_idx=_frame_int(row.col_idx),
                    row_path=_row_path(
                        label_raw, row_candidate.metadata.row_group_context_raw
                    ),
                    row_label_raw=label_raw,
                    row_label_canonical=_optional_str(row.row_label_canonical),
                    col_path=str(row.column_label or ""),
                    period=period,
                    statement_type=_optional_str(getattr(row, "statement_type", None)),
                    unit=_optional_str(row.unit),
                )
            )
    return tuple(candidates)
