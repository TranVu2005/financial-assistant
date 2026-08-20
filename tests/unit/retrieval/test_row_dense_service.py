from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.row_dense_corpus import build_row_dense_corpus
from financial_report_qa.retrieval.row_dense_index import build_row_dense_index
from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService
from financial_report_qa.retrieval.row_documents import build_row_documents

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64
DOC_A = "doc_" + "a" * 64
DOC_B = "doc_" + "b" * 64


@dataclass
class Encoder:
    spec: DenseEncoderSpec
    query_calls: int = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        # Mock embeddings: Row 1 of Table A has score 1.0 for query, others have 0.0
        embeddings = []
        for text in texts:
            if "row_label: Doanh thu hoạt động tài chính" in text:
                embeddings.append([1.0, 0.0])
            elif "row_label: Cho vay khách hàng" in text:
                embeddings.append([0.0, 1.0])
            else:
                embeddings.append([np.sqrt(0.5), np.sqrt(0.5)])
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        if "tài chính" in text:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        elif "khách hàng" in text:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)


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


def test_row_dense_service_filters_by_candidate_tables(tmp_path: Path) -> None:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    corpus = build_row_dense_corpus(
        row_docs,
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    encoder = Encoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    index = build_row_dense_index(corpus, encoder)
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)
    service = RowDenseRetrievalService(index, encoder, cache)

    # 1. Search with TABLE_A candidates
    results = service.retrieve_rows("doanh thu hoạt động tài chính", candidate_table_ids=(TABLE_A,))
    assert len(results) == 2
    assert results[0].table_id == TABLE_A
    assert results[0].row_idx == 1  # Should be the tài chính row
    assert results[0].metadata.row_label_raw == "Doanh thu hoạt động tài chính"

    # 2. Search CTG row but restrict to CTG (TABLE_B) candidates
    results = service.retrieve_rows("cho vay khách hàng", candidate_table_ids=(TABLE_B,))
    assert len(results) == 2
    assert results[0].table_id == TABLE_B
    assert results[0].row_idx == 1  # Cho vay khách hàng row
    assert results[0].metadata.row_label_raw == "Cho vay khách hàng"


def test_row_dense_service_handles_empty_candidates(tmp_path: Path) -> None:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    corpus = build_row_dense_corpus(
        row_docs,
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    encoder = Encoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    index = build_row_dense_index(corpus, encoder)
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)
    service = RowDenseRetrievalService(index, encoder, cache)

    # Empty candidate tables list
    assert service.retrieve_rows("doanh thu tài chính", candidate_table_ids=()) == ()


def test_row_dense_service_rejects_mismatched_cache_spec(tmp_path: Path) -> None:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    corpus = build_row_dense_corpus(
        row_docs,
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    encoder = Encoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    index = build_row_dense_index(corpus, encoder)
    mismatched_cache = QueryEmbeddingCache(
        tmp_path,
        encoder.spec.model_copy(update={"batch_size": encoder.spec.batch_size + 1}),
    )

    with pytest.raises(DenseArtifactError, match="cache"):
        RowDenseRetrievalService(index, encoder, mismatched_cache)
