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
otherwise a literal 4-digit year in the extracted column header is used before
falling back to relative labels inferred from `documents.report_year`.
Prior-year markers ("năm trước", "kỳ trước", "đầu năm", "đầu kỳ") are matched
before current-year ones ("năm nay", "kỳ này", "cuối năm", "cuối kỳ") because
"Số dư cuối năm trước" carries both and only the prior-year reading is right.
The bare forms subsume the "Số " variants and add 34,150 numeric cells that
the original three patterns left unresolved. Unresolvable periods stay null.

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
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd

from financial_report_qa.core.errors import ExecutionInputError
from financial_report_qa.normalization.periods import normalize_period

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
    "statutory_code",
    "normalized_period",
    "period_type",
)

# Circular 200/2014 numbers every statement row ("Mã số"), identically across
# all 100 companies and 10 years, in its own column. The release stores it as
# an ordinary cell rather than as row metadata, so it is invisible to the
# locator until joined back onto the row. Measured on the locked release:
# 175,198 cells carry this exact header, 172,966 rows get a code, and 336,654
# numeric cells gain one -- 2.27x the 148,527 cells that have a canonical
# metric label. None of them carry `value_numeric`, so the code column cannot
# leak into the numeric frame as a data point.
_STATUTORY_CODE_HEADER = "Mã số"

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
        -- OCR/header normalization does not always populate `c.period` even
        -- when the raw header still names the year explicitly ("Năm 2023",
        -- "31/12/2023", "2023"). Recover that evidence before interpreting
        -- relative opening/closing labels from the document report year.
        WHEN REGEXP_MATCHES(COALESCE(c.column_label_raw, ''), '(19|20)[0-9]{2}')
            THEN TRY_CAST(
                REGEXP_EXTRACT(
                    COALESCE(c.column_label_raw, ''), '(19|20)[0-9]{2}', 0
                ) AS INTEGER
            )
        -- Prior-year markers are tested first: "Số dư cuối năm trước" contains
        -- both a closing and a prior-year marker, and only the prior-year
        -- reading is correct. Bare forms subsume the "Số " variants.
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%năm trước%' THEN d.report_year - 1
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%kỳ trước%' THEN d.report_year - 1
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%đầu năm%' THEN d.report_year - 1
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%đầu kỳ%' THEN d.report_year - 1
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%năm nay%' THEN d.report_year
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%kỳ này%' THEN d.report_year
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%cuối năm%' THEN d.report_year
        WHEN LOWER(COALESCE(c.column_label_raw, '')) LIKE '%cuối kỳ%' THEN d.report_year
        ELSE NULL
    END AS period,
    (c.period IS NULL) AS period_inferred,
    code.statutory_code,
    d.report_year,
    t.statement_type
FROM read_parquet(?) AS c
JOIN read_parquet(?) AS t USING (table_id)
JOIN read_parquet(?) AS d USING (doc_id)
LEFT JOIN statutory_codes AS code
    ON code.table_id = c.table_id AND code.row_idx = c.row_idx
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


_MATERIALIZE_CODES = """
CREATE TABLE statutory_codes AS
SELECT table_id, row_idx, MIN(TRIM(value_raw)) AS statutory_code
FROM read_parquet(?)
WHERE column_label_raw = ?
  AND value_raw IS NOT NULL
  AND TRIM(value_raw) <> ''
GROUP BY table_id, row_idx
"""


@lru_cache(maxsize=2)
def _statutory_code_connection(cells_path: str) -> duckdb.DuckDBPyConnection:
    """One connection per release holding the statutory-code lookup in memory.

    A frame is built per question, so expressing the code join as
    a second `read_parquet` meant scanning the same 520 MB `cells.parquet`
    twice per question. The code table is small -- 172,966 rows on the locked
    release, against 2.58M numeric cells -- so it is the half worth holding in
    memory; the main scan stays a streaming parquet read. Releases are
    immutable by contract (plan.md §3), so this cannot go stale in normal use.
    """
    connection = _hardened_connection()
    connection.execute(_MATERIALIZE_CODES, [cells_path, _STATUTORY_CODE_HEADER])
    return connection


def build_cell_frame(release_dir: Path, table_ids: Sequence[str]) -> pd.DataFrame:
    """Return the long-format numeric-cell frame for a plan's candidate tables."""
    if not table_ids:
        raise ExecutionInputError("build_cell_frame requires at least one table_id")
    connection = _statutory_code_connection(str(release_dir / "cells.parquet"))
    frame = connection.execute(
        _QUERY,
        [
            str(release_dir / "cells.parquet"),
            str(release_dir / "tables.parquet"),
            str(release_dir / "documents.parquet"),
            list(table_ids),
        ],
    ).fetchdf()
    frame["period"] = frame["period"].astype("Int64")
    frame["period_inferred"] = frame["period_inferred"].astype(bool)
    # Day 20 plan Sec 1.3 / ADR 0009 decision C1: when a table mixes a cell
    # with an explicit unit and a cell with SQL NULL unit, DuckDB's pandas
    # conversion for that column turns the null into a genuine float NaN
    # rather than None -- `str(nan)` is the fabricated unit string `'nan'`.
    # Force it back to a real missing value so `locator.py` can detect it.
    frame["unit"] = frame["unit"].astype(object).where(frame["unit"].notna(), None)
    frame["statutory_code"] = (
        frame["statutory_code"].astype(object).where(frame["statutory_code"].notna(), None)
    )

    # V2 column post-processing
    norm_periods = []
    period_types = []
    for _, row in frame.iterrows():
        col_label = row["column_label"]
        rep_year = row["report_year"]
        pr = row["period"]
        stmt_type = row["statement_type"]

        norm_val = None
        if col_label and pd.notna(col_label):
            norm_val = normalize_period(str(col_label), int(rep_year)).value
        if norm_val is None and pd.notna(pr):
            norm_val = str(pr)

        p_type = None
        if norm_val:
            if stmt_type == "balance_sheet":
                p_type = "point_in_time"
            elif stmt_type in ("income_statement", "cash_flow_statement"):
                p_type = "duration"
            elif len(norm_val) == 10 and "-" in norm_val:
                p_type = "point_in_time"
            else:
                col_lower = str(col_label).lower() if col_label and pd.notna(col_label) else ""
                if "ngày" in col_lower or "tại" in col_lower:
                    p_type = "point_in_time"
                else:
                    p_type = "duration"

        norm_periods.append(norm_val)
        period_types.append(p_type)

    frame["normalized_period"] = norm_periods
    frame["period_type"] = period_types

    return frame.loc[:, list(CELL_FRAME_COLUMNS)]
