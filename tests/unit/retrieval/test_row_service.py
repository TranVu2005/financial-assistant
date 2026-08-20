from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_index import build_row_bm25_index
from financial_report_qa.retrieval.row_service import RowRetrievalService

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64
DOC_A = "doc_" + "a" * 64
DOC_B = "doc_" + "b" * 64


def _write_test_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    # 1. Write documents.parquet
    documents = [
        {
            "doc_id": DOC_A,
            "repo_id": "repo_1",
            "revision": "main",
            "relative_path": "reports/A.txt",
            "company_code": "VCB",
            "report_year": 2020,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 1024,
            "encoding": "utf-8",
            "inventory_status": "active",
            "ruleset_version": "v1",
            "normalization_fingerprint": "d" * 64,
        },
        {
            "doc_id": DOC_B,
            "repo_id": "repo_1",
            "revision": "main",
            "relative_path": "reports/B.txt",
            "company_code": "CTG",
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
            "table_id": TABLE_A,
            "doc_id": DOC_A,
            "source_ordinal": 0,
            "title_raw": "Báo cáo Kết quả Kinh doanh",
            "statement_type": "income_statement",
            "unit_raw": "triệu đồng",
            "unit_normalized": "VND_million",
            "line_start": 10,
            "line_end": 20,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": "table_a.csv",
        },
        {
            "table_id": TABLE_B,
            "doc_id": DOC_B,
            "source_ordinal": 0,
            "title_raw": "Báo cáo Cân đối Kế toán",
            "statement_type": "balance_sheet",
            "unit_raw": "triệu đồng",
            "unit_normalized": "VND_million",
            "line_start": 10,
            "line_end": 20,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": "table_b.csv",
        }
    ]
    pq.write_table(
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA),
        release_dir / "tables.parquet",
    )

    # 3. Write cells.parquet
    cells = [
        # Table A, Row 0: Doanh thu bán hàng
        {
            "cell_id": "cell_1",
            "table_id": TABLE_A,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu bán hàng và CCDV",
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
        # Table A, Row 1: Doanh thu tài chính
        {
            "cell_id": "cell_2",
            "table_id": TABLE_A,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Doanh thu hoạt động tài chính",
            "row_label_canonical": "financial_revenue",
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
        # Table B, Row 0: Tiền gửi tại NHNN
        {
            "cell_id": "cell_3",
            "table_id": TABLE_B,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Tiền gửi tại Ngân hàng Nhà nước",
            "row_label_canonical": "deposits_at_sbv",
            "row_group_context_raw": "Tài sản",
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "2000",
            "value_numeric": Decimal("2000"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 11,
            "source_line_end": 11,
            "extraction_confidence": 1.0,
        },
        # Table B, Row 1: Cho vay khách hàng
        {
            "cell_id": "cell_4",
            "table_id": TABLE_B,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Cho vay khách hàng",
            "row_label_canonical": "loans_to_customers",
            "row_group_context_raw": "Tài sản",
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "8000",
            "value_numeric": Decimal("8000"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 12,
            "source_line_end": 12,
            "extraction_confidence": 1.0,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA),
        release_dir / "cells.parquet",
    )

    return release_dir


def test_row_retrieval_filters_by_candidate_tables(tmp_path: Path) -> None:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    index = build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)
    service = RowRetrievalService(index)

    # 1. Search for "Doanh thu" only within TABLE_A candidates
    results = service.retrieve_rows("doanh thu tài chính", candidate_table_ids=(TABLE_A,))
    assert len(results) == 2
    assert {results[0].row_idx, results[1].row_idx} == {0, 1}
    assert results[0].table_id == TABLE_A
    assert results[1].table_id == TABLE_A

    # 2. Search for "Doanh thu" but restrict to TABLE_B (should return empty since TABLE_B has no "doanh thu" tokens)
    results = service.retrieve_rows("doanh thu", candidate_table_ids=(TABLE_B,))
    assert len(results) == 0

    # 3. Search for "Khách hàng" within TABLE_B candidates
    results = service.retrieve_rows("cho vay khách hàng", candidate_table_ids=(TABLE_B,))
    assert len(results) == 2
    assert {results[0].row_idx, results[1].row_idx} == {0, 1}
    assert results[0].table_id == TABLE_B
    assert results[1].table_id == TABLE_B


def test_row_retrieval_handles_empty_or_missing_tokens(tmp_path: Path) -> None:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    index = build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)
    service = RowRetrievalService(index)

    # Empty candidate tables list
    assert service.retrieve_rows("doanh thu", candidate_table_ids=()) == ()

    # Query with no vocabulary tokens
    assert service.retrieve_rows("xyzabc123", candidate_table_ids=(TABLE_A,)) == ()
