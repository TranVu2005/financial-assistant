"""Bounded, deterministic Parquet-to-document conversion for row retrieval."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel


@dataclass(frozen=True)
class RowDocumentBuildResult:
    row_count: int


class RowMetadata(_FrozenModel):
    table_id: TableId
    row_idx: int
    company_code: str | None = None
    statement_type: str | None = None
    title: str | None = None
    row_label_raw: str | None = None
    row_label_canonical: str | None = None
    row_group_context_raw: str | None = None
    periods: tuple[str, ...] = ()
    units: tuple[str, ...] = ()


class RowDocument(_FrozenModel):
    row_id: NonEmptyString  # e.g., "{table_id}|row_{row_idx}"
    table_id: TableId
    row_idx: int
    text: NonEmptyString
    metadata: RowMetadata


def _normalize(value: object | None, *, display_token: bool = False) -> str:
    if value is None:
        return ""
    text = str(value).replace("_", " ") if display_token else str(value)
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _tokens(values: list[object | None], *, display_tokens: bool = False) -> tuple[str, ...]:
    normalized = {
        token for value in values if (token := _normalize(value, display_token=display_tokens))
    }
    return tuple(sorted(normalized))


def _line(label: str, values: tuple[str, ...] | str | None) -> str | None:
    if values is None:
        return None
    value = " | ".join(values) if isinstance(values, tuple) else values
    return f"{label}: {value}" if value else None


def build_row_documents(
    documents_path: Path, tables_path: Path, cells_path: Path
) -> tuple[RowDocument, ...]:
    """Aggregate once per table row using DuckDB; serializes each row to rich context text."""
    cell_columns = set(pq.read_schema(cells_path).names)  # type: ignore[no-untyped-call]
    group_context_expression = (
        "c.row_group_context_raw" if "row_group_context_raw" in cell_columns else "NULL"
    )
    row_idx_expression = (
        "c.row_idx" if "row_idx" in cell_columns else "0"
    )

    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            f"""
            SELECT
                c.table_id,
                {row_idx_expression} AS row_idx,
                ANY_VALUE(c.row_label_raw) AS row_label_raw,
                ANY_VALUE(c.row_label_canonical) AS row_label_canonical,
                ANY_VALUE({group_context_expression}) AS row_group_context_raw,
                list(DISTINCT c.period) FILTER (WHERE c.period IS NOT NULL) AS periods,
                list(DISTINCT c.unit) FILTER (WHERE c.unit IS NOT NULL) AS units,
                ANY_VALUE(t.doc_id) AS doc_id,
                ANY_VALUE(d.company_code) AS company_code,
                ANY_VALUE(t.statement_type) AS statement_type,
                ANY_VALUE(t.title_raw) AS title
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS t USING (table_id)
            JOIN read_parquet(?) AS d USING (doc_id)
            WHERE {row_idx_expression} IS NOT NULL
            GROUP BY c.table_id, row_idx
            ORDER BY c.table_id, row_idx
            """,
            [str(cells_path), str(tables_path), str(documents_path)],
        ).fetchall()
    finally:
        connection.close()

    # Group rows by table_id to calculate neighbor context
    rows_by_table: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        (
            table_id,
            row_idx,
            row_label_raw,
            row_label_canonical,
            row_group_context_raw,
            cell_periods,
            cell_units,
            doc_id,
            company_code,
            statement_type,
            title,
        ) = row

        table_id_str = str(table_id)
        rows_by_table.setdefault(table_id_str, []).append(
            {
                "row_idx": int(row_idx),
                "row_label_raw": row_label_raw,
                "row_label_canonical": row_label_canonical,
                "row_group_context_raw": row_group_context_raw,
                "periods": cell_periods or [],
                "units": cell_units or [],
                "company_code": company_code,
                "statement_type": statement_type,
                "title": title,
            }
        )

    result: list[RowDocument] = []
    for table_id, table_rows in rows_by_table.items():
        # Ensure rows are sorted by row_idx
        sorted_rows = sorted(table_rows, key=lambda r: r["row_idx"])
        n_rows = len(sorted_rows)

        for i, row_data in enumerate(sorted_rows):
            row_idx = row_data["row_idx"]
            row_label_raw = row_data["row_label_raw"]
            row_label_canonical = row_data["row_label_canonical"]
            row_group_context_raw = row_data["row_group_context_raw"]
            cell_periods = row_data["periods"]
            cell_units = row_data["units"]
            company_code = row_data["company_code"]
            statement_type = row_data["statement_type"]
            title = row_data["title"]

            # Neighbor row labels (up to 2 preceding and 2 succeeding)
            neighbors: list[str] = []
            for offset in (-2, -1, 1, 2):
                idx = i + offset
                if 0 <= idx < n_rows:
                    label = sorted_rows[idx]["row_label_raw"]
                    if label and isinstance(label, str):
                        normalized_label = _normalize(label)
                        if normalized_label:
                            neighbors.append(normalized_label)

            periods = _tokens(cell_periods)
            units = _tokens(cell_units, display_tokens=True)

            metadata = RowMetadata(
                table_id=table_id,
                row_idx=row_idx,
                company_code=_normalize(company_code) or None,
                statement_type=_normalize(statement_type) or None,
                title=_normalize(title) or None,
                row_label_raw=_normalize(row_label_raw) or None,
                row_label_canonical=_normalize(row_label_canonical) or None,
                row_group_context_raw=_normalize(row_group_context_raw) or None,
                periods=periods,
                units=units,
            )

            text_lines = (
                _line("company", _normalize(company_code)),
                _line("statement", _normalize(statement_type, display_token=True)),
                _line("table", _normalize(title)),
                _line("section", _normalize(row_group_context_raw)),
                _line("row_label", _normalize(row_label_raw)),
                _line("row_label_canonical", _normalize(row_label_canonical)),
                _line("periods", periods),
                _line("units", units),
                _line("neighbor_rows", tuple(neighbors) if neighbors else None),
            )

            text = "\n".join(line for line in text_lines if line is not None)
            if not text:
                text = "empty row"

            result.append(
                RowDocument(
                    row_id=f"{table_id}|row_{row_idx}",
                    table_id=table_id,
                    row_idx=row_idx,
                    text=text,
                    metadata=metadata,
                )
            )

    return tuple(result)
