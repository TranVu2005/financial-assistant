"""Reconstruct extracted tables as pandas DataFrames from a dataset release.

Two views are produced per table, both derived from `placements.parquet` and
`cells.parquet` without touching the immutable release itself:

- ``grid``: the positional row x column text layout, matching the source table.
- ``values``: a tidy, one-row-per-cell frame of the normalized cell fields
  (metric, period, unit, numeric value, group context), ready for pandas
  filtering, pivoting, and analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from financial_report_qa.core.errors import TableFrameError

_VALUES_COLUMNS = (
    "cell_id",
    "row_idx",
    "col_idx",
    "row_label_raw",
    "row_label_canonical",
    "row_group_context_raw",
    "column_label_raw",
    "column_label_canonical",
    "value_raw",
    "value_numeric",
    "period",
    "unit",
    "source_line_start",
    "source_line_end",
    "extraction_confidence",
)

_JOIN_SELECT = """
    SELECT
        p.table_id, p.row_idx, p.col_idx, c.cell_id,
        c.row_label_raw, c.row_label_canonical, c.row_group_context_raw,
        c.column_label_raw, c.column_label_canonical,
        c.value_raw, c.value_numeric, c.period, c.unit,
        c.source_line_start, c.source_line_end, c.extraction_confidence
    FROM read_parquet(?) AS p
    JOIN read_parquet(?) AS c USING (cell_id)
"""

_TABLE_META_SELECT = """
    SELECT t.table_id, t.doc_id, d.relative_path, t.line_start, t.line_end,
           t.title_raw, t.statement_type, t.unit_normalized
    FROM read_parquet(?) AS t
    JOIN read_parquet(?) AS d USING (doc_id)
"""

_JOIN_COLUMNS = (
    "table_id",
    "row_idx",
    "col_idx",
    "cell_id",
    "row_label_raw",
    "row_label_canonical",
    "row_group_context_raw",
    "column_label_raw",
    "column_label_canonical",
    "value_raw",
    "value_numeric",
    "period",
    "unit",
    "source_line_start",
    "source_line_end",
    "extraction_confidence",
)


@dataclass(frozen=True)
class TableFrame:
    """One reconstructed table alongside its release provenance."""

    table_id: str
    doc_id: str
    relative_path: str
    line_start: int
    line_end: int
    title_raw: str | None
    statement_type: str | None
    unit_normalized: str | None
    grid: pd.DataFrame
    values: pd.DataFrame


@dataclass(frozen=True)
class CsvExportManifest:
    """Result of exporting reconstructed table grids as CSV files."""

    output_dir: Path
    manifest_path: Path
    table_count: int


def build_values_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the tidy, one-row-per-cell normalized frame for one table's rows."""
    missing = [name for name in _VALUES_COLUMNS if name not in rows.columns]
    if missing:
        raise TableFrameError(f"cell rows are missing required columns: {missing}")
    ordered = rows.loc[:, list(_VALUES_COLUMNS)].sort_values(
        ["row_idx", "col_idx", "cell_id"], kind="stable"
    )
    return ordered.reset_index(drop=True)


def build_grid(rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot (row_idx, col_idx, value_raw) rows into the original positional grid."""
    if rows.empty:
        return pd.DataFrame()
    pivoted = rows.pivot(index="row_idx", columns="col_idx", values="value_raw")
    pivoted = pivoted.reindex(
        index=range(0, int(pivoted.index.max()) + 1),
        columns=range(0, int(pivoted.columns.max()) + 1),
    )
    pivoted.index.name = None
    pivoted.columns.name = None
    return pivoted.reset_index(drop=True)


def _table_frame_from_rows(
    table_id: str, meta: Mapping[Any, Any], rows: pd.DataFrame
) -> TableFrame:
    values = (
        build_values_frame(rows) if not rows.empty else rows.reindex(columns=list(_VALUES_COLUMNS))
    )
    grid = (
        build_grid(rows.loc[:, ["row_idx", "col_idx", "value_raw"]])
        if not rows.empty
        else pd.DataFrame()
    )

    def _optional_str(name: str) -> str | None:
        value = meta[name]
        return None if value is None or pd.isna(value) else str(value)

    return TableFrame(
        table_id=table_id,
        doc_id=str(meta["doc_id"]),
        relative_path=str(meta["relative_path"]),
        line_start=int(meta["line_start"]),
        line_end=int(meta["line_end"]),
        title_raw=_optional_str("title_raw"),
        statement_type=_optional_str("statement_type"),
        unit_normalized=_optional_str("unit_normalized"),
        grid=grid,
        values=values,
    )


def load_table_frame(release_dir: Path, table_id: str) -> TableFrame:
    """Reconstruct one table's grid and normalized values by table_id."""
    connection = duckdb.connect(":memory:")
    try:
        meta_df = connection.execute(
            _TABLE_META_SELECT + " WHERE t.table_id = ?",
            [
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                table_id,
            ],
        ).fetchdf()
        if meta_df.empty:
            raise TableFrameError(f"table_id not found in release: {table_id}")
        rows = connection.execute(
            _JOIN_SELECT + " WHERE p.table_id = ?",
            [
                str(release_dir / "placements.parquet"),
                str(release_dir / "cells.parquet"),
                table_id,
            ],
        ).fetchdf()
    finally:
        connection.close()
    meta: Mapping[Any, Any] = meta_df.iloc[0].to_dict()
    return _table_frame_from_rows(table_id, meta, rows)


def iter_table_frames(release_dir: Path) -> Iterator[TableFrame]:
    """Stream a TableFrame for every table in a release, ordered by table_id."""
    connection = duckdb.connect(":memory:")
    try:
        meta_df = connection.execute(
            _TABLE_META_SELECT + " ORDER BY t.table_id",
            [str(release_dir / "tables.parquet"), str(release_dir / "documents.parquet")],
        ).fetchdf()
        rows = connection.execute(
            _JOIN_SELECT + " ORDER BY p.table_id, p.row_idx, p.col_idx",
            [str(release_dir / "placements.parquet"), str(release_dir / "cells.parquet")],
        ).fetchdf()
    finally:
        connection.close()

    meta_by_id: dict[str, dict[Any, Any]] = {
        str(table_id): meta
        for table_id, meta in meta_df.set_index("table_id").to_dict(orient="index").items()
    }
    groups: dict[str, pd.DataFrame] = (
        {str(key): group for key, group in rows.groupby("table_id", sort=False)}
        if not rows.empty
        else {}
    )
    empty_rows = pd.DataFrame(columns=list(_JOIN_COLUMNS))

    for table_id in meta_df["table_id"]:
        table_id_str = str(table_id)
        group = groups.get(table_id_str, empty_rows)
        yield _table_frame_from_rows(table_id_str, meta_by_id[table_id_str], group)


def export_table_csvs(
    release_dir: Path,
    output_dir: Path,
    *,
    table_ids: Iterable[str] | None = None,
) -> CsvExportManifest:
    """Write each table's reconstructed grid as a CSV file, plus a lookup manifest.

    This is a derived, rebuildable artifact separate from the immutable release;
    it does not populate ``TableRecord.csv_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: Iterable[TableFrame] = (
        (load_table_frame(release_dir, table_id) for table_id in table_ids)
        if table_ids is not None
        else iter_table_frames(release_dir)
    )

    entries: list[dict[str, object]] = []
    for table_frame in frames:
        csv_path = output_dir / f"{table_frame.table_id}.csv"
        table_frame.grid.to_csv(csv_path, index=False, header=False, encoding="utf-8-sig")
        entries.append(
            {
                "table_id": table_frame.table_id,
                "doc_id": table_frame.doc_id,
                "relative_path": table_frame.relative_path,
                "csv_path": csv_path.name,
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"table_count": len(entries), "tables": entries},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CsvExportManifest(
        output_dir=output_dir, manifest_path=manifest_path, table_count=len(entries)
    )
