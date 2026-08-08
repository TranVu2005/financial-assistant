"""Bounded, deterministic Parquet-to-document conversion for BM25."""

from __future__ import annotations

from pathlib import Path

import duckdb

from financial_report_qa.retrieval.contracts import TableDocument, TableMetadata


def _tokens(values: list[str | None]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(value.split()) for value in values if value and value.strip()}))


def build_table_documents(
    documents_path: Path, tables_path: Path, cells_path: Path
) -> tuple[TableDocument, ...]:
    """Aggregate once per table in DuckDB; numeric values are intentionally excluded."""
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            """
            SELECT
                t.table_id, t.doc_id, d.company_code, CAST(d.report_year AS VARCHAR),
                t.statement_type, t.title_raw, d.relative_path, t.line_start, t.line_end,
                list(DISTINCT c.row_label_canonical)
                    FILTER (WHERE c.row_label_canonical IS NOT NULL),
                list(DISTINCT c.row_label_raw) FILTER (WHERE c.row_label_raw IS NOT NULL),
                list(DISTINCT c.period) FILTER (WHERE c.period IS NOT NULL),
                list(DISTINCT c.unit) FILTER (WHERE c.unit IS NOT NULL)
            FROM read_parquet(?) AS t
            JOIN read_parquet(?) AS d USING (doc_id)
            LEFT JOIN read_parquet(?) AS c USING (table_id)
            GROUP BY ALL
            ORDER BY t.table_id
            """,
            [str(tables_path), str(documents_path), str(cells_path)],
        ).fetchall()
    finally:
        connection.close()

    result: list[TableDocument] = []
    for row in rows:
        (
            table_id,
            doc_id,
            company_code,
            report_year,
            statement_type,
            title,
            relative_path,
            line_start,
            line_end,
            canonical_labels,
            raw_labels,
            periods,
            units,
        ) = row
        metadata = TableMetadata(
            table_id=str(table_id),
            doc_id=str(doc_id),
            company_code=company_code,
            period=report_year,
            statement_type=statement_type,
            title=title,
            source_path=str(relative_path),
            line_start=int(line_start),
            line_end=int(line_end),
        )
        lines = (
            f"title: {title or ''}",
            f"statement_type: {statement_type or ''}",
            f"company_code: {company_code or ''}",
            f"report_year: {report_year or ''}",
            f"periods: {' | '.join(_tokens(periods or []))}",
            f"units: {' | '.join(_tokens(units or []))}",
            f"metrics: {' | '.join(_tokens(canonical_labels or []))}",
            f"metric_aliases: {' | '.join(_tokens(raw_labels or []))}",
        )
        result.append(
            TableDocument(
                table_id=str(table_id), doc_id=str(doc_id), text="\n".join(lines), metadata=metadata
            )
        )
    return tuple(result)
