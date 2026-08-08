"""Deterministic conversion of canonical Parquet tables into BM25 documents."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from financial_report_qa.retrieval.contracts import TableDocument, TableMetadata


def _optional(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    return None if value is None else str(value)


def build_table_documents(
    documents_path: Path, tables_path: Path, cells_path: Path
) -> tuple[TableDocument, ...]:
    """Build one canonical text document per table with stable row/column ordering."""
    document_rows = pq.read_table(documents_path).to_pylist()  # type: ignore[no-untyped-call]
    documents_by_id = {row["doc_id"]: row for row in document_rows}
    cells_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in pq.read_table(cells_path).to_pylist():  # type: ignore[no-untyped-call]
        cells_by_table[str(cell["table_id"])].append(cell)

    result: list[TableDocument] = []
    for table in sorted(
        pq.read_table(tables_path).to_pylist(),  # type: ignore[no-untyped-call]
        key=lambda row: str(row["table_id"]),
    ):
        table_id = str(table["table_id"])
        document = documents_by_id.get(table["doc_id"])
        if document is None:
            raise ValueError(f"No document metadata for table {table_id}")
        metadata = TableMetadata(
            table_id=table_id,
            doc_id=str(table["doc_id"]),
            company_code=_optional(document, "company_code"),
            period=_optional(document, "report_year"),
            statement_type=_optional(table, "statement_type"),
            title=_optional(table, "title_raw"),
            source_path=str(document["relative_path"]),
            line_start=int(table["line_start"]),
            line_end=int(table["line_end"]),
        )
        lines = [
            f"table_id: {table_id}",
            f"company_code: {metadata.company_code or ''}",
            f"period: {metadata.period or ''}",
            f"statement_type: {metadata.statement_type or ''}",
            f"title: {metadata.title or ''}",
        ]
        for cell in sorted(
            cells_by_table.get(table_id, []),
            key=lambda row: (int(row["row_idx"]), int(row["col_idx"])),
        ):
            row_label = (
                _optional(cell, "row_label_canonical") or _optional(cell, "row_label_raw") or ""
            )
            column_label = (
                _optional(cell, "column_label_canonical")
                or _optional(cell, "column_label_raw")
                or ""
            )
            lines.append(f"{row_label} | {column_label} | {cell['value_raw']}")
        result.append(
            TableDocument(
                table_id=table_id,
                doc_id=str(table["doc_id"]),
                text="\n".join(lines),
                metadata=metadata,
            )
        )
    return tuple(result)
