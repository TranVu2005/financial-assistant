"""Unit tests for plan.md §20 Row Recall@k / Table Recall@k measurement."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.row_documents import build_row_documents
from financial_report_qa.retrieval.row_fusion import RowFusionService
from financial_report_qa.retrieval.row_fusion_contracts import RowFusionWeights
from financial_report_qa.retrieval.row_index import build_row_bm25_index
from financial_report_qa.retrieval.row_recall_evaluation import (
    RowRecallQuestion,
    evaluate_row_recall,
    load_row_recall_gold,
)
from financial_report_qa.retrieval.row_service import RowRetrievalService
from financial_report_qa.retrieval.service import RetrievalService

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64
DOC_A = "doc_" + "a" * 64
DOC_B = "doc_" + "b" * 64


def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    documents = [
        {
            "doc_id": DOC_A,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "VCB/2020/VCB_report_extracted.txt",
            "company_code": "VCB",
            "report_year": 2020,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "v1",
            "normalization_fingerprint": "d" * 64,
        },
        {
            "doc_id": DOC_B,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "CTG/2020/CTG_report_extracted.txt",
            "company_code": "CTG",
            "report_year": 2020,
            "statement_scope": "consolidated",
            "sha256": "c" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "v1",
            "normalization_fingerprint": "d" * 64,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
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
            "line_start": 1,
            "line_end": 10,
            "row_count": 2,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": None,
        },
        {
            "table_id": TABLE_B,
            "doc_id": DOC_B,
            "source_ordinal": 0,
            "title_raw": "Báo cáo Cân đối Kế toán",
            "statement_type": "balance_sheet",
            "unit_raw": "triệu đồng",
            "unit_normalized": "VND_million",
            "line_start": 1,
            "line_end": 10,
            "row_count": 1,
            "column_count": 2,
            "quality_score": 1.0,
            "csv_path": None,
        },
    ]
    pq.write_table(
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )

    cells = [
        {
            "cell_id": "cell_" + "1" * 64,
            "table_id": TABLE_A,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuần",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "100",
            "value_numeric": Decimal("100"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 1.0,
        },
        {
            "cell_id": "cell_" + "2" * 64,
            "table_id": TABLE_A,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Giá vốn hàng bán",
            "row_label_canonical": "cost_of_goods_sold",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "50",
            "value_numeric": Decimal("50"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 6,
            "source_line_end": 6,
            "extraction_confidence": 1.0,
        },
        {
            "cell_id": "cell_" + "3" * 64,
            "table_id": TABLE_B,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Tổng tài sản",
            "row_label_canonical": "total_assets",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2020",
            "column_label_canonical": "2020",
            "value_raw": "5000",
            "value_numeric": Decimal("5000"),
            "period": "2020",
            "unit": "VND_million",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 1.0,
        },
    ]
    pq.write_table(pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet")
    return release_dir


def _services(release_dir: Path) -> tuple[RetrievalService, RowFusionService]:
    table_documents = (
        TableDocument(
            table_id=TABLE_A,
            doc_id=DOC_A,
            text="company_code: VCB\nperiod: 2020\nDoanh thu thuần | Giá vốn hàng bán",
            metadata=TableMetadata(
                table_id=TABLE_A,
                doc_id=DOC_A,
                company_code="VCB",
                periods=("2020",),
                statement_type="income_statement",
                source_path="a.txt",
                line_start=1,
                line_end=3,
            ),
            metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
        ),
        TableDocument(
            table_id=TABLE_B,
            doc_id=DOC_B,
            text="company_code: CTG\nperiod: 2020\nTổng tài sản",
            metadata=TableMetadata(
                table_id=TABLE_B,
                doc_id=DOC_B,
                company_code="CTG",
                periods=("2020",),
                statement_type="balance_sheet",
                source_path="b.txt",
                line_start=1,
                line_end=3,
            ),
            metric_labels=(MetricLabelObservation(canonical="total_assets", raw=None),),
        ),
    )
    table_service = RetrievalService(
        build_bm25_index(table_documents, dataset_fingerprint="f" * 64)
    )

    row_docs = build_row_documents(
        release_dir / "documents.parquet",
        release_dir / "tables.parquet",
        release_dir / "cells.parquet",
    )
    row_bm25_index = build_row_bm25_index(row_docs, dataset_fingerprint="f" * 64)
    row_fusion = RowFusionService(
        RowRetrievalService(row_bm25_index),
        dense=None,
        weights=RowFusionWeights(bm25=1, dense=0),
    )
    return table_service, row_fusion


def test_evaluate_row_recall_hits_and_ranks(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    table_service, row_fusion = _services(release_dir)

    questions = (
        RowRecallQuestion(
            question_id="q1",
            question="Doanh thu thuần của VCB năm 2020 là bao nhiêu?",
            gold_row_labels=("Doanh thu thuần",),
            gold_table_ids=(TABLE_A,),
        ),
        RowRecallQuestion(
            question_id="q2",
            question="Tổng tài sản của CTG năm 2020 là bao nhiêu?",
            gold_row_labels=("Tổng tài sản",),
            gold_table_ids=(TABLE_B,),
        ),
    )

    report = evaluate_row_recall(
        questions, table_service, row_fusion, dataset_fingerprint="f" * 64
    )

    assert report.question_count == 2
    assert report.table_recall_at_k == 1.0
    assert report.row_recall_at["1"] == 1.0
    for outcome in report.outcomes:
        assert outcome.table_recall_hit is True
        assert outcome.best_rank == 1


def test_evaluate_row_recall_table_miss_means_no_row_hit(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    table_service, row_fusion = _services(release_dir)

    # A gold table id that will never be retrieved by any real query.
    questions = (
        RowRecallQuestion(
            question_id="q1",
            question="Doanh thu thuần của VCB năm 2020 là bao nhiêu?",
            gold_row_labels=("Nhãn không tồn tại",),
            gold_table_ids=("tbl_" + "9" * 64,),
        ),
    )

    report = evaluate_row_recall(
        questions, table_service, row_fusion, dataset_fingerprint="f" * 64
    )

    assert report.table_recall_at_k == 0.0
    assert report.row_recall_at["1"] == 0.0
    assert report.outcomes[0].table_recall_hit is False
    assert report.outcomes[0].best_rank is None


def test_load_row_recall_gold_dedupes_labels_and_tables(tmp_path: Path) -> None:
    gold_path = tmp_path / "answer-gold-v1.jsonl"
    gold_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                {
                    "question_id": "q1",
                    "question": "So sánh doanh thu thuần 2020 và 2021.",
                    "evidence": [
                        {"table_id": TABLE_A, "row_label": "Doanh thu thuần", "period": 2020},
                        {"table_id": TABLE_A, "row_label": "Doanh thu thuần", "period": 2021},
                    ],
                },
                {
                    "question_id": "q2",
                    "question": "Không có evidence.",
                    "evidence": [],
                },
            )
        ),
        encoding="utf-8",
    )

    questions = load_row_recall_gold(gold_path)

    assert len(questions) == 1
    assert questions[0].question_id == "q1"
    assert questions[0].gold_row_labels == ("Doanh thu thuần",)
    assert questions[0].gold_table_ids == (TABLE_A,)


def test_evaluate_row_recall_empty_questions() -> None:
    report = evaluate_row_recall((), None, None, dataset_fingerprint="f" * 64)  # type: ignore[arg-type]
    assert report.question_count == 0
    assert report.table_recall_at_k == 0.0
    assert report.row_recall_at == {str(k): 0.0 for k in (1, 3, 5, 10)}
