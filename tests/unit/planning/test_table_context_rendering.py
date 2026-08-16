"""Tests for the Day 23 grounded-LLM-fallback table renderer.

Real table content (row labels, values) must reach the LLM prompt verbatim
-- Day 22 measured that a vocabulary-free prompt caused 23.4% of LLM plans
to invent plausible-sounding metric names instead of copying real ones.
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.planning.table_context_rendering import render_table_context

DOC_ID = "doc_" + "a" * 64
TABLE_ID = "tbl_" + "1" * 64


def _write_release(
    tmp_path: Path, *, title_raw: str | None = "18. Quỹ bình ổn giá xăng dầu"
) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "PLX/2015/report.txt",
            "company_code": "PLX",
            "report_year": 2015,
            "statement_scope": "separate",
            "sha256": "0" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "1",
            "normalization_fingerprint": "0" * 64,
        }
    ]
    tables = [
        {
            "table_id": TABLE_ID,
            "doc_id": DOC_ID,
            "source_ordinal": 0,
            "title_raw": title_raw,
            "statement_type": None,
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1078,
            "line_end": 1090,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    cells = [
        {
            "cell_id": "cell_a",
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 0,
            "row_label_raw": None,
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": None,
            "column_label_canonical": None,
            "value_raw": "Số dư cuối năm",
            "value_numeric": None,
            "period": None,
            "unit": None,
            "source_line_start": 1084,
            "source_line_end": 1084,
            "extraction_confidence": 0.9,
        },
        {
            "cell_id": "cell_b",
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Số dư cuối năm",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "2015VND",
            "column_label_canonical": None,
            "value_raw": "2.377.393.168.988",
            "value_numeric": Decimal("2377393168988"),
            "period": "2015",
            "unit": "VND",
            "source_line_start": 1084,
            "source_line_end": 1084,
            "extraction_confidence": 0.9,
        },
    ]
    placements = [
        {"table_id": TABLE_ID, "row_idx": 0, "col_idx": 0, "cell_id": "cell_a"},
        {"table_id": TABLE_ID, "row_idx": 0, "col_idx": 1, "cell_id": "cell_b"},
    ]

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
    )
    return release_dir


def test_render_table_context_includes_real_cell_values(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    rendered = render_table_context(release_dir, [TABLE_ID])
    assert "Số dư cuối năm" in rendered
    assert "2.377.393.168.988" in rendered


def test_render_table_context_includes_source_document_and_title(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    rendered = render_table_context(release_dir, [TABLE_ID])
    assert "PLX/2015/report.txt" in rendered
    assert "18. Quỹ bình ổn giá xăng dầu" in rendered


def test_render_table_context_handles_missing_title(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, title_raw=None)
    rendered = render_table_context(release_dir, [TABLE_ID])
    assert TABLE_ID in rendered


def test_render_table_context_joins_multiple_tables(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    rendered = render_table_context(release_dir, [TABLE_ID, TABLE_ID])
    assert rendered.count("Số dư cuối năm") >= 2
