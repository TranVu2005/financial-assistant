"""Bounded, deterministic Parquet-to-document conversion for BM25."""

from __future__ import annotations

import hashlib
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import orjson
import pyarrow.parquet as pq

from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)

if TYPE_CHECKING:
    from financial_report_qa.retrieval.release import ResolvedRetrievalRelease


@dataclass(frozen=True)
class DocumentBuildResult:
    table_count: int
    sha256: str


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


def _metric_labels(values: list[object] | None) -> tuple[MetricLabelObservation, ...]:
    observations: set[tuple[str, str | None]] = set()
    for value in values or []:
        if not isinstance(value, dict):
            raise ValueError("metric label aggregation must produce structs")
        canonical = _normalize(value.get("canonical"))
        if not canonical:
            continue
        raw = _normalize(value.get("raw")) or None
        observations.add((canonical, raw))
    return tuple(
        MetricLabelObservation(canonical=canonical, raw=raw)
        for canonical, raw in sorted(observations, key=lambda item: (item[0], item[1] or ""))
    )


def _line(label: str, values: tuple[str, ...] | str) -> str | None:
    value = " | ".join(values) if isinstance(values, tuple) else values
    return f"{label}: {value}" if value else None


def build_table_documents(
    documents_path: Path, tables_path: Path, cells_path: Path
) -> tuple[TableDocument, ...]:
    """Aggregate once per table in DuckDB; raw numeric values are never selected."""
    table_columns = set(pq.read_schema(tables_path).names)  # type: ignore[no-untyped-call]
    table_unit_expression = "t.unit_normalized" if "unit_normalized" in table_columns else "NULL"
    cell_columns = set(pq.read_schema(cells_path).names)  # type: ignore[no-untyped-call]
    group_context_expression = (
        "c.row_group_context_raw" if "row_group_context_raw" in cell_columns else "NULL"
    )
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            f"""
            SELECT
                t.table_id, t.doc_id, d.company_code, CAST(d.report_year AS VARCHAR),
                t.statement_type, t.title_raw, {table_unit_expression},
                d.relative_path, t.line_start, t.line_end,
                list(DISTINCT struct_pack(
                    canonical := c.row_label_canonical,
                    raw := c.row_label_raw
                )) FILTER (WHERE c.row_label_canonical IS NOT NULL),
                list(DISTINCT c.period) FILTER (WHERE c.period IS NOT NULL),
                list(DISTINCT c.unit) FILTER (WHERE c.unit IS NOT NULL),
                list(DISTINCT {group_context_expression})
                    FILTER (WHERE {group_context_expression} IS NOT NULL)
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
            table_unit,
            relative_path,
            line_start,
            line_end,
            metric_label_values,
            cell_periods,
            cell_units,
            group_context_values,
        ) = row
        periods = _tokens([*(cell_periods or []), report_year])
        metric_labels = _metric_labels(metric_label_values)
        group_contexts = _tokens(group_context_values or [])
        metadata = TableMetadata(
            table_id=str(table_id),
            doc_id=str(doc_id),
            company_code=_normalize(company_code) or None,
            periods=periods,
            statement_type=_normalize(statement_type) or None,
            title=_normalize(title) or None,
            source_path=str(relative_path),
            line_start=int(line_start),
            line_end=int(line_end),
            group_contexts=group_contexts,
        )
        text_lines = (
            _line("title", _normalize(title)),
            _line("statement", _normalize(statement_type, display_token=True)),
            _line(
                "metrics",
                _tokens([label.canonical for label in metric_labels], display_tokens=True),
            ),
            _line(
                "metric aliases",
                _tokens([label.raw for label in metric_labels]),
            ),
            _line("group context", group_contexts),
            _line("company", _normalize(company_code)),
            _line("periods", periods),
            _line(
                "units",
                _tokens([table_unit, *(cell_units or [])], display_tokens=True),
            ),
        )
        result.append(
            TableDocument(
                table_id=str(table_id),
                doc_id=str(doc_id),
                text="\n".join(line for line in text_lines if line is not None),
                metadata=metadata,
                metric_labels=metric_labels,
            )
        )
    return tuple(result)


def iter_table_documents(
    release: ResolvedRetrievalRelease,
) -> tuple[TableDocument, ...]:
    """Build documents from the three immutable release Parquet artifacts."""
    return build_table_documents(
        release.release_dir / "documents.parquet",
        release.release_dir / "tables.parquet",
        release.release_dir / "cells.parquet",
    )


def write_table_documents(
    release: ResolvedRetrievalRelease, output_path: Path
) -> DocumentBuildResult:
    """Write canonical UTF-8 JSONL through a sibling temporary file and replace."""
    documents = iter_table_documents(release)
    expected_count = release.manifest.get("table_count")
    if not isinstance(expected_count, int) or len(documents) != expected_count:
        raise ValueError("retrieval document table count differs from release manifest")
    table_ids = tuple(document.table_id for document in documents)
    if table_ids != tuple(sorted(set(table_ids))):
        raise ValueError("retrieval documents require sorted unique table IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for document in documents:
                line = orjson.dumps(
                    document.model_dump(mode="json"),
                    option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
                )
                stream.write(line)
                digest.update(line)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return DocumentBuildResult(table_count=len(documents), sha256=digest.hexdigest())
