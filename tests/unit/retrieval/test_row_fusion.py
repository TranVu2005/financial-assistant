"""Unit tests for row-level weighted-RRF fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.row_dense_corpus import build_row_dense_corpus
from financial_report_qa.retrieval.row_dense_index import build_row_dense_index
from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_fusion import RowFusionService
from financial_report_qa.retrieval.row_fusion_contracts import RowFusionWeights
from financial_report_qa.retrieval.row_index import RowBM25Index, build_row_bm25_index
from financial_report_qa.retrieval.row_lexical import (
    RowAliasRetrievalService,
    RowFuzzyRetrievalService,
)
from financial_report_qa.retrieval.row_service import RowRetrievalService

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64
DOC_A = "doc_" + "a" * 64
DOC_B = "doc_" + "b" * 64


@dataclass
class _MockEncoder:
    spec: DenseEncoderSpec
    query_calls: int = 0

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
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
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA),
        release_dir / "tables.parquet",
    )

    cells = [
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


@pytest.fixture()
def services(tmp_path: Path) -> tuple[RowRetrievalService, RowDenseRetrievalService]:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    bm25_index = build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)
    bm25_service = RowRetrievalService(bm25_index)

    corpus = build_row_dense_corpus(
        row_docs,
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    encoder = _MockEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    dense_index = build_row_dense_index(corpus, encoder)
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)
    dense_service = RowDenseRetrievalService(dense_index, encoder, cache)

    return bm25_service, dense_service


def test_row_fusion_combines_bm25_and_dense(
    services: tuple[RowRetrievalService, RowDenseRetrievalService],
) -> None:
    bm25_service, dense_service = services
    weights = RowFusionWeights(bm25=1, dense=1)
    fusion = RowFusionService(bm25_service, dense_service, weights)

    trace = fusion.retrieve_rows(
        "doanh thu hoạt động tài chính",
        candidate_table_ids=(TABLE_A,),
    )

    assert trace.empty_reason is None
    assert trace.bm25_candidate_count > 0
    assert trace.dense_candidate_count > 0
    assert len(trace.results) > 0

    # Every result must have a positive fused score
    for candidate in trace.results:
        assert candidate.fused_score > 0
        assert candidate.table_id == TABLE_A

    # At least one candidate should have contributions from both branches
    both_branches = [
        c for c in trace.results if c.bm25_rank is not None and c.dense_rank is not None
    ]
    assert len(both_branches) > 0

    # Ranks must be consecutive starting from 1
    for i, candidate in enumerate(trace.results, start=1):
        assert candidate.rank == i

    # Results are sorted by descending fused_score
    for i in range(len(trace.results) - 1):
        assert trace.results[i].fused_score >= trace.results[i + 1].fused_score


def test_row_fusion_filters_zero_bm25_scores(
    services: tuple[RowRetrievalService, RowDenseRetrievalService],
) -> None:
    bm25_service, dense_service = services
    weights = RowFusionWeights(bm25=1, dense=1)
    fusion = RowFusionService(bm25_service, dense_service, weights)

    # Query that has no BM25 vocabulary match but dense can still find rows
    # "xyznonexistent" won't match any BM25 token, so BM25 returns empty
    trace = fusion.retrieve_rows(
        "xyznonexistent",
        candidate_table_ids=(TABLE_A,),
    )

    # BM25 should contribute zero candidates (no tokens match)
    assert trace.bm25_candidate_count == 0
    # Dense still contributes
    assert trace.dense_candidate_count > 0
    # All results come only from dense
    for candidate in trace.results:
        assert candidate.bm25_rank is None
        assert candidate.bm25_score is None
        assert candidate.dense_rank is not None


def test_row_fusion_zero_weight_branch(
    services: tuple[RowRetrievalService, RowDenseRetrievalService],
) -> None:
    bm25_service, dense_service = services

    # Dense weight = 0 → only BM25 contributes
    weights = RowFusionWeights(bm25=1, dense=0)
    fusion = RowFusionService(bm25_service, dense_service, weights)
    trace = fusion.retrieve_rows(
        "doanh thu hoạt động tài chính",
        candidate_table_ids=(TABLE_A,),
    )
    assert trace.dense_candidate_count == 0
    for candidate in trace.results:
        assert candidate.dense_rank is None
        assert candidate.dense_score is None
        assert candidate.bm25_rank is not None

    # BM25 weight = 0 → only dense contributes
    weights = RowFusionWeights(bm25=0, dense=1)
    fusion = RowFusionService(bm25_service, dense_service, weights)
    trace = fusion.retrieve_rows(
        "doanh thu hoạt động tài chính",
        candidate_table_ids=(TABLE_A,),
    )
    assert trace.bm25_candidate_count == 0
    for candidate in trace.results:
        assert candidate.bm25_rank is None
        assert candidate.bm25_score is None
        assert candidate.dense_rank is not None


def test_row_fusion_empty_candidates(
    services: tuple[RowRetrievalService, RowDenseRetrievalService],
) -> None:
    bm25_service, dense_service = services
    weights = RowFusionWeights(bm25=1, dense=1)
    fusion = RowFusionService(bm25_service, dense_service, weights)

    trace = fusion.retrieve_rows("doanh thu", candidate_table_ids=())

    assert trace.empty_reason == "no_candidates"
    assert trace.results == ()
    assert trace.bm25_candidate_count == 0
    assert trace.dense_candidate_count == 0
    assert trace.candidate_table_ids == ()


def test_row_fusion_dense_only(
    services: tuple[RowRetrievalService, RowDenseRetrievalService],
) -> None:
    bm25_service, dense_service = services
    weights = RowFusionWeights(bm25=0, dense=1)
    fusion = RowFusionService(bm25_service, dense_service, weights)

    trace = fusion.retrieve_rows(
        "cho vay khách hàng",
        candidate_table_ids=(TABLE_B,),
    )

    assert trace.bm25_candidate_count == 0
    assert trace.dense_candidate_count > 0
    assert len(trace.results) > 0

    # Top result should be the "Cho vay khách hàng" row (dense encoder returns [0,1])
    assert trace.results[0].table_id == TABLE_B
    assert trace.results[0].row_idx == 1
    assert trace.results[0].metadata.row_label_raw == "Cho vay khách hàng"

    # Fused scores come only from dense branch
    for candidate in trace.results:
        expected_score = weights.dense / (weights.rrf_k + candidate.dense_rank)
        assert abs(candidate.fused_score - expected_score) < 1e-9
        assert candidate.bm25_rank is None


@pytest.fixture()
def bm25_index(tmp_path: Path) -> RowBM25Index:
    release_dir = _write_test_release(tmp_path)
    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    return build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)


def test_row_fusion_fuzzy_branch_contributes(bm25_index: RowBM25Index) -> None:
    bm25_service = RowRetrievalService(bm25_index)
    fuzzy_service = RowFuzzyRetrievalService(bm25_index)
    weights = RowFusionWeights(bm25=1, dense=0, fuzzy=1)
    fusion = RowFusionService(bm25_service, dense=None, weights=weights, fuzzy=fuzzy_service)

    trace = fusion.retrieve_rows(
        "doanh thu hoạt động tài chính",
        candidate_table_ids=(TABLE_A,),
    )

    assert trace.fuzzy_candidate_count > 0
    fuzzy_hits = [c for c in trace.results if c.fuzzy_rank is not None]
    assert fuzzy_hits
    for candidate in fuzzy_hits:
        assert candidate.fuzzy_score is not None
        expected = 0.0
        if candidate.bm25_rank is not None:
            expected += weights.bm25 / (weights.rrf_k + candidate.bm25_rank)
        expected += weights.fuzzy / (weights.rrf_k + candidate.fuzzy_rank)
        assert abs(candidate.fused_score - expected) < 1e-9


def test_row_fusion_fuzzy_zero_weight_contributes_nothing(bm25_index: RowBM25Index) -> None:
    bm25_service = RowRetrievalService(bm25_index)
    fuzzy_service = RowFuzzyRetrievalService(bm25_index)
    weights = RowFusionWeights(bm25=1, dense=0, fuzzy=0)
    fusion = RowFusionService(bm25_service, dense=None, weights=weights, fuzzy=fuzzy_service)

    trace = fusion.retrieve_rows(
        "doanh thu hoạt động tài chính",
        candidate_table_ids=(TABLE_A,),
    )

    assert trace.fuzzy_candidate_count == 0
    for candidate in trace.results:
        assert candidate.fuzzy_rank is None
        assert candidate.fuzzy_score is None


@pytest.fixture()
def alias_only_bm25_index(tmp_path: Path) -> RowBM25Index:
    """A row whose raw label ("Chỉ tiêu 10") shares no tokens with the
    question, but whose stored canonical ("net_revenue") is exactly what
    "Doanh thu thuần" -- a real `METRIC_ALIASES` phrase -- resolves to.
    Only the alias branch can surface it."""
    release_dir = tmp_path / "alias_release"
    release_dir.mkdir(exist_ok=True)
    doc_id = "doc_" + "e" * 64

    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "doc_id": doc_id,
                    "repo_id": "repo_1",
                    "revision": "main",
                    "relative_path": "reports/E.txt",
                    "company_code": "VCB",
                    "report_year": 2020,
                    "statement_scope": "consolidated",
                    "sha256": "c" * 64,
                    "file_size_bytes": 1024,
                    "encoding": "utf-8",
                    "inventory_status": "active",
                    "ruleset_version": "v1",
                    "normalization_fingerprint": "d" * 64,
                }
            ],
            schema=DOCUMENT_SCHEMA,
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "table_id": TABLE_A,
                    "doc_id": doc_id,
                    "source_ordinal": 0,
                    "title_raw": "Báo cáo tài chính hợp nhất",
                    "statement_type": "income_statement",
                    "unit_raw": "triệu đồng",
                    "unit_normalized": "VND_million",
                    "line_start": 10,
                    "line_end": 20,
                    "row_count": 1,
                    "column_count": 2,
                    "quality_score": 1.0,
                    "csv_path": "table_a.csv",
                }
            ],
            schema=TABLE_SCHEMA,
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "cell_id": "cell_1",
                    "table_id": TABLE_A,
                    "row_idx": 0,
                    "col_idx": 1,
                    "row_label_raw": "Chỉ tiêu 10",
                    "row_label_canonical": "net_revenue",
                    "row_group_context_raw": None,
                    "column_label_raw": "Năm 2020",
                    "column_label_canonical": "2020",
                    "value_raw": "5000",
                    "value_numeric": Decimal("5000"),
                    "period": "2020",
                    "unit": "VND_million",
                    "source_line_start": 11,
                    "source_line_end": 11,
                    "extraction_confidence": 1.0,
                }
            ],
            schema=CELL_SCHEMA,
        ),
        release_dir / "cells.parquet",
    )

    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    return build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)


def test_row_fusion_alias_branch_finds_row_bm25_misses(alias_only_bm25_index: RowBM25Index) -> None:
    bm25_service = RowRetrievalService(alias_only_bm25_index)
    alias_service = RowAliasRetrievalService(alias_only_bm25_index)
    weights = RowFusionWeights(bm25=1, dense=0, alias=1)
    fusion = RowFusionService(
        bm25_service, dense=None, weights=weights, alias=alias_service
    )

    trace = fusion.retrieve_rows(
        "Doanh thu thuần là bao nhiêu?",
        candidate_table_ids=(TABLE_A,),
    )

    assert trace.bm25_candidate_count == 0
    assert trace.alias_candidate_count == 1
    assert len(trace.results) == 1
    candidate = trace.results[0]
    assert candidate.row_idx == 0
    assert candidate.bm25_rank is None
    assert candidate.alias_rank == 1
    assert candidate.alias_score == 1.0
    expected = weights.alias / (weights.rrf_k + 1)
    assert abs(candidate.fused_score - expected) < 1e-9


def test_row_fusion_alias_zero_weight_contributes_nothing(
    alias_only_bm25_index: RowBM25Index,
) -> None:
    bm25_service = RowRetrievalService(alias_only_bm25_index)
    alias_service = RowAliasRetrievalService(alias_only_bm25_index)
    weights = RowFusionWeights(bm25=1, dense=0, alias=0)
    fusion = RowFusionService(
        bm25_service, dense=None, weights=weights, alias=alias_service
    )

    trace = fusion.retrieve_rows(
        "Doanh thu thuần là bao nhiêu?",
        candidate_table_ids=(TABLE_A,),
    )

    assert trace.alias_candidate_count == 0
    assert trace.results == ()
