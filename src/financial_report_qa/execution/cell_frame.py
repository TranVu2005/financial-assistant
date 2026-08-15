"""Day 18 long-format cell projection (ADR 0007 decision B1).

Reads `cells.parquet` directly (ADR 0007 decision A1) rather than going
through the placement-join machinery in `data/table_frame.py`: placements
exist to reconstruct positional grids (colspan duplication), while the
compiler only needs each numeric cell's own row/column labels, period, and
unit, which live entirely on the cell row itself.

Two lossy filters are applied unconditionally (Day 18 plan §1.4/§1.5):
`col_idx > 0` (excludes row-label cells) and `value_numeric IS NOT NULL`
(excludes placeholder cells such as `-` or empty strings).

Period resolution follows ADR 0007 decision C2: an explicit `period` wins
(normalized to a 4-digit year, since ~37.7% of stored periods are ISO dates);
otherwise a column label naming "số cuối năm"/"số đầu năm"/"năm trước" infers
the year from `documents.report_year`. Unresolvable periods stay null.

Every DuckDB connection this module opens is hardened (ADR 0008 decision F1):
`enable_external_access`/`autoinstall_known_extensions`/
`autoload_known_extensions` are all disabled. Day 19 plan Sec 1.7 measured
that `duckdb.connect(":memory:")` defaults to `enable_external_access=True`,
and that `_QUERY` below already binds every plan-derived value (table ids,
which are schema-constrained to `^tbl_[0-9a-f]{64}$`) through parameterized
placeholders, so there is no SQL-injection path from a plan today -- this is
depth hardening against a future query that concatenates a string, not a fix
for a live hole.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from financial_report_qa.core.errors import ExecutionInputError

CELL_FRAME_COLUMNS = (
    "table_id",
    "cell_id",
    "company_code",
    "row_idx",
    "col_idx",
    "row_label_raw",
    "row_label_canonical",
    "column_label",
    "unit",
    "value",
    "period",
    "period_inferred",
)

_QUERY = """
SELECT
    c.table_id,
    c.cell_id,
    d.company_code,
    c.row_idx,
    c.col_idx,
    c.row_label_raw,
    c.row_label_canonical,
    c.column_label_raw AS column_label,
    c.unit,
    c.value_numeric AS value,
    CASE
        WHEN c.period IS NOT NULL THEN TRY_CAST(LEFT(c.period, 4) AS INTEGER)
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%số cuối năm%' THEN d.report_year
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%số đầu năm%' THEN d.report_year - 1
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%năm trước%' THEN d.report_year - 1
        ELSE NULL
    END AS period,
    (c.period IS NULL) AS period_inferred
FROM read_parquet(?) AS c
JOIN read_parquet(?) AS t USING (table_id)
JOIN read_parquet(?) AS d USING (doc_id)
WHERE c.table_id IN (SELECT UNNEST(?))
  AND c.col_idx > 0
  AND c.value_numeric IS NOT NULL
ORDER BY c.table_id, c.row_idx, c.col_idx
"""


def _hardened_connection() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with network access disabled
    (ADR 0008 decision F1). `enable_external_access=false` is deliberately
    NOT set here: it disables local filesystem access too, which would break
    the local `read_parquet` calls this module depends on. Disabling
    extension autoinstall/autoload is sufficient on its own -- `httpfs`
    (required for any http/https path) is not bundled, so without it a
    network read fails outright instead of silently reaching the network.
    """
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    return connection


def build_cell_frame(release_dir: Path, table_ids: Sequence[str]) -> pd.DataFrame:
    """Return the long-format numeric-cell frame for a plan's candidate tables."""
    if not table_ids:
        raise ExecutionInputError("build_cell_frame requires at least one table_id")
    connection = _hardened_connection()
    try:
        frame = connection.execute(
            _QUERY,
            [
                str(release_dir / "cells.parquet"),
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                list(table_ids),
            ],
        ).fetchdf()
    finally:
        connection.close()
    frame["period"] = frame["period"].astype("Int64")
    frame["period_inferred"] = frame["period_inferred"].astype(bool)
    # Day 20 plan Sec 1.3 / ADR 0009 decision C1: when a table mixes a cell
    # with an explicit unit and a cell with SQL NULL unit, DuckDB's pandas
    # conversion for that column turns the null into a genuine float NaN
    # rather than None -- `str(nan)` is the fabricated unit string `'nan'`.
    # Force it back to a real missing value so `locator.py` can detect it.
    frame["unit"] = frame["unit"].astype(object).where(frame["unit"].notna(), None)
    return frame.loc[:, list(CELL_FRAME_COLUMNS)]
