from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_index import (
    build_row_bm25_index,
    load_row_bm25_index,
    save_row_bm25_index,
)

TABLE_ID = "tbl_" + "a" * 64
DOC_ID = "doc_" + "b" * 64


def _write_dummy_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    # 1. Write documents.parquet
    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo_1",
            "revision": "main",
            "relative_path": "reports/VNM_2020.txt",
            "company_code": "VNM",
            "report_year": 2020,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 1024,
            "encoding": "utf-8",
            "inventory_status": "active",
            "ruleset_version": "v1",
            "normalization_fingerprint": "d" * 64,
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA),
        release_dir / "documents.parquet",
    )

    # 2. Write tables.parquet
    tables = [
        {
            "table_id": TABLE_ID,
            "doc_id": DOC_ID,
            "source_ordinal": 0,
            "title_raw": "Báo cáo Kết quả Kinh doanh",
            "statement_type": "income_statement",
            "unit_raw": "triệu đồng",
            "unit_normalized": "VND_million",
            "line_start": 10,
            "line_end": 20,
            "row_count": 3,
            "column_count": 3,
            "quality_score": 1.0,
            "csv_path": "table_0.csv",
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA),
        release_dir / "tables.parquet",
    )

    # 3. Write cells.parquet
    cells = [
        # Row 0: Doanh thu bán hàng
        {
            "cell_id": "cell_1",
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu bán hàng",
            "row_label_canonical": "gross_revenue",
            "row_group_context_raw": "Doanh thu",
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "5000",
            "value_numeric": Decimal("5000"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 11,
            "source_line_end": 11,
            "extraction_confidence": 1.0,
        },
        # Row 1: Các khoản giảm trừ
        {
            "cell_id": "cell_2",
            "table_id": TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Các khoản giảm trừ",
            "row_label_canonical": None,
            "row_group_context_raw": "Doanh thu",
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 12,
            "source_line_end": 12,
            "extraction_confidence": 1.0,
        },
        # Row 2: Doanh thu thuần
        {
            "cell_id": "cell_3",
            "table_id": TABLE_ID,
            "row_idx": 2,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuần",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": "Doanh thu",
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "4900",
            "value_numeric": Decimal("4900"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 13,
            "source_line_end": 13,
            "extraction_confidence": 1.0,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA),
        release_dir / "cells.parquet",
    )

    return release_dir


def test_build_row_documents_extracts_correct_context(tmp_path: Path) -> None:
    release_dir = _write_dummy_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )

    assert len(row_docs) == 3

    # Check first row (row_idx=0)
    row0 = row_docs[0]
    assert row0.row_idx == 0
    assert row0.table_id == TABLE_ID
    assert row0.metadata.row_label_raw == "Doanh thu bán hàng"
    assert row0.metadata.row_label_canonical == "gross_revenue"
    assert row0.metadata.row_group_context_raw == "Doanh thu"
    assert row0.metadata.company_code == "VNM"
    assert row0.metadata.statement_type == "income_statement"

    # Check neighbor_rows context serialization
    # Row 0 should have row 1 and row 2 as succeeding neighbors
    assert "neighbor_rows: Các khoản giảm trừ | Doanh thu thuần" in row0.text

    # Row 1 should have row 0 as preceding neighbor and row 2 as succeeding neighbor
    row1 = row_docs[1]
    assert "neighbor_rows: Doanh thu bán hàng | Doanh thu thuần" in row1.text


def test_build_save_and_load_row_bm25_index(tmp_path: Path) -> None:
    release_dir = _write_dummy_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )

    # 1. Build index
    fingerprint = "e" * 64
    index = build_row_bm25_index(row_docs, dataset_fingerprint=fingerprint)
    assert index.manifest.document_count == 3
    assert index.manifest.dataset_fingerprint == fingerprint

    # 2. Save index
    index_dir = tmp_path / "row_index_dir"
    save_row_bm25_index(index, index_dir)
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "bm25s").exists()

    # 3. Load index
    loaded = load_row_bm25_index(index_dir)
    assert loaded.manifest.document_count == 3
    assert loaded.manifest.dataset_fingerprint == fingerprint
    assert len(loaded.documents) == 3
    assert loaded.documents[0].row_id == f"{TABLE_ID}|row_0"
