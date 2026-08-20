"""Unit tests for the fuzzy and alias-dictionary row retrieval branches."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_index import RowBM25Index, build_row_bm25_index
from financial_report_qa.retrieval.row_lexical import (
    RowAliasRetrievalService,
    RowFuzzyRetrievalService,
)
from financial_report_qa.retrieval.row_service import RowRetrievalService

TABLE_A = "tbl_" + "a" * 64
DOC_A = "doc_" + "a" * 64


def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    documents = [
        {
            "doc_id": DOC_A,
            "repo_id": "repo_1",
            "revision": "main",
            "relative_path": "reports/A.txt",
            "company_code": "ACB",
            "report_year": 2023,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 1024,
            "encoding": "utf-8",
            "inventory_status": "active",
            "ruleset_version": "v1",
            "normalization_fingerprint": "d" * 64,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA),
        release_dir / "documents.parquet",
    )

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
            "row_count": 3,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": "table_a.csv",
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA),
        release_dir / "tables.parquet",
    )

    cells = [
        {
            # OCR-noisy label: no diacritics, so it shares zero BM25 tokens
            # with a diacritic query -- only the fuzzy branch can find it.
            # No group context, so no other document line accidentally
            # shares a token with the query either.
            "cell_id": "cell_1",
            "table_id": TABLE_A,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Loi nhuan gop",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": "2023",
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2023",
            "unit": "VND_million",
            "source_line_start": 11,
            "source_line_end": 11,
            "extraction_confidence": 1.0,
        },
        {
            # Canonical-only match: raw label shares no tokens or characters
            # with the alias phrase, only the stored canonical does.
            "cell_id": "cell_2",
            "table_id": TABLE_A,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Chỉ tiêu 10",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": "Doanh thu",
            "column_label_raw": "Năm 2023",
            "column_label_canonical": "2023",
            "value_raw": "5000",
            "value_numeric": Decimal("5000"),
            "period": "2023",
            "unit": "VND_million",
            "source_line_start": 12,
            "source_line_end": 12,
            "extraction_confidence": 1.0,
        },
        {
            # Unrelated row: must never surface for either branch's query.
            "cell_id": "cell_3",
            "table_id": TABLE_A,
            "row_idx": 2,
            "col_idx": 1,
            "row_label_raw": "Chi phí quản lý doanh nghiệp",
            "row_label_canonical": "general_administration_expenses",
            "row_group_context_raw": "Chi phí",
            "column_label_raw": "Năm 2023",
            "column_label_canonical": "2023",
            "value_raw": "200",
            "value_numeric": Decimal("200"),
            "period": "2023",
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


@pytest.fixture()
def row_index(tmp_path: Path) -> RowBM25Index:
    release_dir = _write_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    return build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)


def test_fuzzy_finds_ocr_variant_bm25_misses(row_index: RowBM25Index) -> None:
    bm25 = RowRetrievalService(row_index)
    fuzzy = RowFuzzyRetrievalService(row_index)
    query = "Lợi nhuận gộp"

    bm25_hits = bm25.retrieve_rows(query, candidate_table_ids=(TABLE_A,))
    assert all(hit.row_idx != 0 for hit in bm25_hits)

    fuzzy_hits = fuzzy.retrieve_rows(query, candidate_table_ids=(TABLE_A,))
    assert fuzzy_hits
    assert fuzzy_hits[0].row_idx == 0
    assert fuzzy_hits[0].table_id == TABLE_A
    assert 0.0 < fuzzy_hits[0].score <= 1.0
    for i in range(len(fuzzy_hits) - 1):
        assert fuzzy_hits[i].score >= fuzzy_hits[i + 1].score


def test_fuzzy_scopes_to_candidate_tables(row_index: RowBM25Index) -> None:
    fuzzy = RowFuzzyRetrievalService(row_index)
    hits = fuzzy.retrieve_rows("bat ky cau hoi nao", candidate_table_ids=())
    assert hits == ()


def test_fuzzy_skips_rows_without_a_label(row_index: RowBM25Index) -> None:
    fuzzy = RowFuzzyRetrievalService(row_index)
    hits = fuzzy.retrieve_rows("x", candidate_table_ids=(TABLE_A,))
    # every scored hit must carry a real label -- no zero-length comparisons
    for hit in hits:
        assert hit.metadata.row_label_raw


def test_alias_finds_canonical_match_fuzzy_and_bm25_miss(row_index: RowBM25Index) -> None:
    alias = RowAliasRetrievalService(row_index)
    query = "Doanh thu thuần của ACB năm 2023 là bao nhiêu?"

    hits = alias.retrieve_rows(query, candidate_table_ids=(TABLE_A,))

    assert len(hits) == 1
    assert hits[0].row_idx == 1
    assert hits[0].table_id == TABLE_A
    assert hits[0].score == 1.0
    assert hits[0].metadata.row_label_canonical == "net_revenue"


def test_alias_no_match_returns_empty(row_index: RowBM25Index) -> None:
    alias = RowAliasRetrievalService(row_index)
    hits = alias.retrieve_rows(
        "Tổng tài sản của ACB năm 2023 là bao nhiêu?", candidate_table_ids=(TABLE_A,)
    )
    assert hits == ()


def test_alias_scopes_to_candidate_tables(row_index: RowBM25Index) -> None:
    alias = RowAliasRetrievalService(row_index)
    hits = alias.retrieve_rows(
        "Doanh thu thuần của ACB năm 2023 là bao nhiêu?", candidate_table_ids=()
    )
    assert hits == ()
