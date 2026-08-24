"""Normalized-table CSV and metadata export from an immutable release.

Reads ``{documents,tables,cells,placements}.parquet`` from a release directory
with one DuckDB join, reconstructs each table's grid, flattens multi-level
headers into underscore paths, prefixes data rows with their group context,
and writes one UTF-8-sig CSV per table plus a JSONL metadata manifest.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path, PurePosixPath

import duckdb
import orjson

from financial_report_qa.core.errors import ExportError


@dataclass(frozen=True)
class CellRow:
    """Minimal mirror of one ``cells.parquet`` row used by the export."""

    cell_id: str
    table_id: str
    row_idx: int
    col_idx: int
    value_raw: str
    value_numeric: Decimal | None
    row_label_raw: str | None
    row_group_context_raw: str | None
    column_label_raw: str | None


@dataclass(frozen=True)
class PlacementRow:
    """Minimal mirror of one ``placements.parquet`` row."""

    table_id: str
    row_idx: int
    col_idx: int
    cell_id: str


@dataclass(frozen=True)
class NormalizedTable:
    """One export-ready table: flattened header tuple plus string grid rows."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class TableExportMetadata:
    """Sidecar record for one exported CSV (one JSONL line)."""

    table_id: str
    company: str  # documents.company_code
    year: int  # documents.report_year
    report_type: str  # documents.statement_scope
    statement: str | None  # tables.statement_type
    unit: str | None  # tables.unit_normalized, falling back to unit_raw
    csv_path: str  # bare file name of the exported CSV


@dataclass(frozen=True)
class CsvExportManifest:
    """Result summary of one ``export_normalized_csvs`` run."""

    output_dir: Path
    manifest_path: Path
    table_count: int
    entries: tuple[TableExportMetadata, ...]


def flatten_header(levels: list[str]) -> str:
    """Join deduplicated header levels with ``_``; whitespace becomes ``_``.

    Empty or whitespace-only levels are dropped first; every remaining level is
    compacted via ``"_".join(level.split())`` so runs of whitespace collapse to
    a single underscore. Example: ``["Tổng cộng", "31/12/2022"]`` ->
    ``"Tổng_cộng_31/12/2022"``.
    """
    cleaned = ["_".join(level.split()) for level in levels if level.strip()]
    return "_".join(cleaned)


def detect_header_row_count(cells: list[CellRow], placements: list[PlacementRow]) -> int:
    """Infer how many leading grid rows are header rows.

    A grid row counts as header when it holds at least one placed cell, every
    placed cell has both ``column_label_raw`` and ``row_label_raw`` unset, and
    at least one placed cell has non-blank text. Scanning stops at the first
    row that fails any condition.
    """
    cells_by_id = {cell.cell_id: cell for cell in cells}
    placed_by_row: dict[int, list[CellRow]] = {}
    for placement in placements:
        cell = cells_by_id.get(placement.cell_id)
        if cell is None:
            raise ExportError(f"placement references unknown cell_id: {placement.cell_id}")
        placed_by_row.setdefault(placement.row_idx, []).append(cell)

    header_rows = 0
    row_idx = 0
    while True:
        row_cells = placed_by_row.get(row_idx)
        if not row_cells:
            break
        labels_absent = all(
            cell.column_label_raw is None and cell.row_label_raw is None for cell in row_cells
        )
        if not labels_absent or all(cell.value_raw.strip() == "" for cell in row_cells):
            break
        header_rows += 1
        row_idx += 1
    return header_rows


def build_normalized_table(
    cells: list[CellRow], placements: list[PlacementRow], header_rows: int
) -> NormalizedTable:
    """Build the normalized grid for one table from its cells and placements.

    Column ``c``'s header collects ``value_raw`` of the cells placed at
    ``(r, c)`` for ``r < header_rows``, merges consecutive duplicates, and
    flattens. Each data row starts with the combined label
    ``"{row_group_context_raw} > {row_label_raw}"`` of its leftmost placed
    cell; remaining columns carry the cell value at that grid position --
    normalized numeric text when ``value_numeric`` is set, else stripped
    ``value_raw`` -- or an empty string when nothing is placed there.
    """
    cells_by_id = {cell.cell_id: cell for cell in cells}
    cell_at: dict[tuple[int, int], CellRow] = {}
    for placement in placements:
        cell = cells_by_id.get(placement.cell_id)
        if cell is None:
            raise ExportError(f"placement references unknown cell_id: {placement.cell_id}")
        cell_at[(placement.row_idx, placement.col_idx)] = cell

    if not cell_at:
        return NormalizedTable(headers=(), rows=())

    row_count = max(row_idx for row_idx, _ in cell_at) + 1
    column_count = max(col_idx for _, col_idx in cell_at) + 1

    def _header_for_column(col_idx: int) -> str:
        levels = [
            cell_at[(row_idx, col_idx)].value_raw
            for row_idx in range(header_rows)
            if (row_idx, col_idx) in cell_at
        ]
        return flatten_header(_dedup_consecutive(levels))

    headers = tuple(_header_for_column(col_idx) for col_idx in range(column_count))

    def _cell_output(row_idx: int, col_idx: int) -> str:
        cell = cell_at.get((row_idx, col_idx))
        if cell is None:
            return ""
        if cell.value_numeric is not None:
            return format(cell.value_numeric.normalize(), "f")
        return cell.value_raw.strip()

    metric_col_idx = _detect_metric_column(cell_at, header_rows, row_count, column_count)

    rows: list[tuple[str, ...]] = []
    for row_idx in range(header_rows, row_count):
        row_columns = sorted(col for (r, col) in cell_at if r == row_idx)
        first_cell = cell_at[(row_idx, row_columns[0])] if row_columns else None
        parts = [
            part
            for part in (
                first_cell.row_group_context_raw if first_cell else None,
                first_cell.row_label_raw if first_cell else None,
            )
            if part
        ]
        combined_label = " > ".join(parts)
        rows.append(
            tuple(
                combined_label if col_idx == metric_col_idx else _cell_output(row_idx, col_idx)
                for col_idx in range(column_count)
            )
            if combined_label
            else tuple(_cell_output(row_idx, col_idx) for col_idx in range(column_count))
        )
    return NormalizedTable(headers=headers, rows=tuple(rows))


def _detect_metric_column(
    cell_at: dict[tuple[int, int], CellRow],
    header_rows: int,
    row_count: int,
    column_count: int,
) -> int:
    """Locate the grid column that actually holds the row label text.

    The metric/label column is not always column 0 -- an "STT"/"Mã số" ordinal
    column, or a "Thuyết minh" note column, is often placed to its left (see
    `ingestion/table_extractor.py::_metric_column_index`, which the source
    extraction already runs to avoid this same assumption). Every cell in a
    data row shares the same `row_label_raw`, but only the cell holding the
    label itself has `value_raw == row_label_raw`; the majority vote across
    data rows makes this table-wide and resistant to one row's incidental
    text collision. Falls back to column 0 when no row yields a match (e.g. a
    table with no captured row labels at all), preserving the prior behavior
    for that edge case.
    """
    votes: dict[int, int] = {}
    for row_idx in range(header_rows, row_count):
        row_label = next(
            (
                cell_at[(row_idx, col_idx)].row_label_raw
                for col_idx in range(column_count)
                if (row_idx, col_idx) in cell_at
                and cell_at[(row_idx, col_idx)].row_label_raw is not None
            ),
            None,
        )
        if row_label is None:
            continue
        match_col = next(
            (
                col_idx
                for col_idx in range(column_count)
                if (row_idx, col_idx) in cell_at
                and cell_at[(row_idx, col_idx)].value_raw == row_label
            ),
            None,
        )
        if match_col is not None:
            votes[match_col] = votes.get(match_col, 0) + 1
    if not votes:
        return 0
    return min(votes.items(), key=lambda item: (-item[1], item[0]))[0]


@dataclass
class _TableAccumulator:
    """Per-table grouping bucket for one contiguous run of the ordered stream."""

    table_id: str
    doc_id: str
    base_name: str
    table_number: int
    company: str
    year: int
    report_type: str
    statement_type: str | None
    unit_normalized: str | None
    unit_raw: str | None
    cells: list[CellRow] = field(default_factory=list)
    placements: list[PlacementRow] = field(default_factory=list)


def _dedup_consecutive(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    return deduped


def _safe_doc_base_name(relative_path: str) -> str:
    candidate = PurePosixPath(relative_path).parent.name
    if not candidate or candidate in {".", ".."} or "/" in candidate or "\\" in candidate:
        raise ExportError(
            f"document path does not yield a safe POSIX base name: {relative_path!r}"
        )
    return candidate


def _write_csv_atomically(path: Path, table: NormalizedTable) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(table.headers)
            writer.writerows(table.rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_manifest_jsonl(path: Path, entries: tuple[TableExportMetadata, ...]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for entry in entries:
                payload = orjson.dumps(
                    asdict(entry),
                    option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
                )
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_expected_table_count(release_dir: Path) -> int:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ExportError(f"release manifest not found: {manifest_path}")
    release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = release_manifest.get("table_count")
    if not isinstance(expected_count, int):
        raise ExportError(f"release manifest has no integer table_count: {manifest_path}")
    return expected_count


#: Rows pulled from DuckDB per `fetchmany` call. Bounds peak memory to roughly
#: one chunk of raw tuples plus the single table currently being accumulated,
#: instead of the whole release (6.7M placements at ~146K tables).
_FETCH_CHUNK_SIZE = 200_000


def export_normalized_csvs(release_dir: Path, output_dir: Path) -> CsvExportManifest:
    """Export every release table as a normalized CSV plus a JSONL manifest.

    Tables are grouped per document in ``(line_start, source_ordinal)`` order
    and numbered from 1, producing ``{doc_base_name}__table_{N}.csv`` where
    ``doc_base_name`` is the parent directory segment of the document's
    ``relative_path``. The exported table count must match ``table_count`` in
    the release manifest.

    Reads the join result in bounded chunks and flushes one CSV as soon as its
    table's contiguous run of rows ends, rather than materializing every cell
    and placement in the release before writing anything -- the DuckDB
    `ORDER BY` groups one table's rows together, but does not include
    `table_id` itself, so a change in `(doc_id, line_start, source_ordinal)`
    -- or, defensively, a `table_id` change within an unchanged key -- is what
    actually marks a table boundary.
    """
    connection = duckdb.connect(":memory:")
    try:
        result = connection.execute(
            """
            SELECT
                d.doc_id, d.relative_path, d.company_code, d.report_year,
                d.statement_scope, t.table_id, t.source_ordinal, t.statement_type,
                t.unit_raw, t.unit_normalized, t.line_start,
                p.row_idx, p.col_idx, p.cell_id,
                c.value_raw, c.value_numeric, c.row_label_raw,
                c.row_group_context_raw, c.column_label_raw
            FROM read_parquet(?) AS t
            JOIN read_parquet(?) AS d USING (doc_id)
            JOIN read_parquet(?) AS p USING (table_id)
            JOIN read_parquet(?) AS c USING (cell_id)
            ORDER BY d.relative_path, t.line_start, t.source_ordinal, p.row_idx, p.col_idx
            """,
            [
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                str(release_dir / "placements.parquet"),
                str(release_dir / "cells.parquet"),
            ],
        )
    except duckdb.Error as error:
        raise ExportError(f"cannot read release parquet in {release_dir}: {error}") from error

    try:
        expected_count = _read_expected_table_count(release_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seen_file_names: set[str] = set()
        entries: list[TableExportMetadata] = []
        current_table: _TableAccumulator | None = None
        current_key: tuple[str, int, int] | None = None
        current_doc_id: str | None = None
        current_base_name = ""
        table_number = 0
        seen_spans: set[tuple[int, int]] = set()

        def flush_table() -> None:
            nonlocal current_table
            if current_table is None:
                return
            file_name = f"{current_table.base_name}__table_{current_table.table_number}.csv"
            if file_name in seen_file_names:
                raise ExportError(f"duplicate file name across exports: {file_name}")
            seen_file_names.add(file_name)
            header_rows = detect_header_row_count(current_table.cells, current_table.placements)
            normalized = build_normalized_table(
                current_table.cells, current_table.placements, header_rows
            )
            _write_csv_atomically(output_dir / file_name, normalized)
            entries.append(
                TableExportMetadata(
                    table_id=current_table.table_id,
                    company=current_table.company,
                    year=current_table.year,
                    report_type=current_table.report_type,
                    statement=current_table.statement_type,
                    unit=current_table.unit_normalized or current_table.unit_raw,
                    csv_path=file_name,
                )
            )
            current_table = None

        while True:
            try:
                batch = result.fetchmany(_FETCH_CHUNK_SIZE)
            except duckdb.Error as error:
                raise ExportError(
                    f"cannot read release parquet in {release_dir}: {error}"
                ) from error
            if not batch:
                break
            for row in batch:
                (
                    doc_id,
                    relative_path,
                    company_code,
                    report_year,
                    statement_scope,
                    table_id,
                    source_ordinal,
                    statement_type,
                    unit_raw,
                    unit_normalized,
                    line_start,
                    row_idx,
                    col_idx,
                    cell_id,
                    value_raw,
                    value_numeric,
                    row_label_raw,
                    row_group_context_raw,
                    column_label_raw,
                ) = row
                doc_id = str(doc_id)
                table_id = str(table_id)
                key = (doc_id, int(line_start), int(source_ordinal))

                if key != current_key:
                    flush_table()
                    current_key = key
                    if doc_id != current_doc_id:
                        current_doc_id = doc_id
                        current_base_name = _safe_doc_base_name(str(relative_path))
                        table_number = 0
                        seen_spans = set()
                    span = (int(line_start), int(source_ordinal))
                    if span in seen_spans:
                        raise ExportError(
                            "duplicate (line_start, source_ordinal) within one document: "
                            f"doc_id={doc_id} span={span}"
                        )
                    seen_spans.add(span)
                    table_number += 1
                    current_table = _TableAccumulator(
                        table_id=table_id,
                        doc_id=doc_id,
                        base_name=current_base_name,
                        table_number=table_number,
                        company=str(company_code),
                        year=int(report_year),
                        report_type=str(statement_scope),
                        statement_type=None if statement_type is None else str(statement_type),
                        unit_normalized=(
                            None if unit_normalized is None else str(unit_normalized)
                        ),
                        unit_raw=None if unit_raw is None else str(unit_raw),
                    )
                elif current_table is not None and table_id != current_table.table_id:
                    # Same (doc_id, line_start, source_ordinal) but a different
                    # table_id: two distinct tables claim the same source span.
                    raise ExportError(
                        "duplicate (line_start, source_ordinal) within one document: "
                        f"doc_id={doc_id} span={key[1:]}"
                    )

                assert current_table is not None  # set unconditionally just above
                current_table.placements.append(
                    PlacementRow(
                        table_id=table_id,
                        row_idx=int(row_idx),
                        col_idx=int(col_idx),
                        cell_id=str(cell_id),
                    )
                )
                current_table.cells.append(
                    CellRow(
                        cell_id=str(cell_id),
                        table_id=table_id,
                        row_idx=int(row_idx),
                        col_idx=int(col_idx),
                        value_raw=str(value_raw),
                        value_numeric=value_numeric,
                        row_label_raw=None if row_label_raw is None else str(row_label_raw),
                        row_group_context_raw=(
                            None if row_group_context_raw is None else str(row_group_context_raw)
                        ),
                        column_label_raw=(
                            None if column_label_raw is None else str(column_label_raw)
                        ),
                    )
                )

        flush_table()
    finally:
        connection.close()

    if expected_count != len(entries):
        raise ExportError(
            "release manifest table count differs from exported table count: "
            f"expected={expected_count} found={len(entries)}"
        )

    manifest_path = output_dir / "manifest.jsonl"
    _write_manifest_jsonl(manifest_path, tuple(entries))
    return CsvExportManifest(
        output_dir=output_dir,
        manifest_path=manifest_path,
        table_count=len(entries),
        entries=tuple(entries),
    )
