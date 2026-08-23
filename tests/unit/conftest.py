"""Shared unit-test fixtures: one tiny real release plus the retrieval
outputs that point into it.

`release_dir`/`execution_settings`/`fusion_rows`/`table_ids` describe a single
consistent world -- an ACB 2023 income-statement table whose row 0 is
"Doanh thu thuan" = 100 and row 1 is "Gia von hang ban" = 60 -- so tests of
the answering path can run end to end without each module rebuilding its own
corpus.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64


@pytest.fixture
def execution_settings() -> ExecutionSettings:
    return ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2023/ACB_financial_statements_2023_consolidated_extracted.txt",
            "company_code": "ACB",
            "report_year": 2023,
            "statement_scope": "consolidated",
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
            "title_raw": "Bao cao ket qua kinh doanh",
            "statement_type": "income_statement",
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1,
            "line_end": 10,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    # >= 2 numeric cells so the table never reads as a hardcoded answer (the
    # same compliance shape `test_submission_exporter._write_release` pins).
    cells = [
        {
            "cell_id": "cell_" + "a" * 64,
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 0.9,
        },
        {
            "cell_id": "cell_" + "d" * 64,
            "table_id": TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Gia von hang ban",
            "row_label_canonical": "cost_of_goods_sold",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": "60",
            "value_numeric": Decimal("60"),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 6,
            "source_line_end": 6,
            "extraction_confidence": 0.9,
        },
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
    placements = [
        {
            "table_id": TABLE_ID,
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
    )
    return release_dir


@pytest.fixture
def table_ids() -> tuple[str, ...]:
    return (TABLE_ID,)


@pytest.fixture
def fusion_rows() -> tuple[RowFusedCandidate, ...]:
    """The rank-1 fusion candidate pointing at row 0 ("Doanh thu thuan")."""
    return (
        RowFusedCandidate(
            row_id=f"{TABLE_ID}|row_0",
            table_id=TABLE_ID,
            row_idx=0,
            rank=1,
            snippet="Doanh thu thuan | Năm 2023 | 100",
            metadata=RowMetadata(
                table_id=TABLE_ID,
                row_idx=0,
                company_code="ACB",
                row_label_raw="Doanh thu thuan",
                row_label_canonical="net_revenue",
                periods=("2023",),
            ),
            fused_score=0.9,
            bm25_score=0.9,
            dense_score=0.0,
        ),
    )
