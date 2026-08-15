"""Tests for the Day 20 verify-answers evaluation report (task 20.10)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, GoldTableEvidence
from financial_report_qa.verification.evaluation import (
    build_citation_lookup,
    evaluate_answer_packages_on_gold,
    load_answer_gold,
    write_answer_verification_report,
)

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
CELL_ID = "cell_" + "a" * 64


def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2023/report.txt",
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
            "title_raw": "Bang can doi ke toan",
            "statement_type": "balance_sheet",
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1,
            "line_end": 10,
            "row_count": 1,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    cells = [
        {
            "cell_id": CELL_ID,
            "table_id": TABLE_ID,
            "row_idx": 0,
            "col_idx": 1,
            "row_label_raw": "Tien mat",
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
        }
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
    return release_dir


def test_build_citation_lookup_resolves_provenance_fields(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    lookup = build_citation_lookup(release_dir, [CELL_ID])
    assert lookup[CELL_ID]["doc_relative_path"] == "ACB/2023/report.txt"
    assert lookup[CELL_ID]["source_line_start"] == 5
    assert lookup[CELL_ID]["table_title"] == "Bang can doi ke toan"


def test_evaluate_answer_packages_reports_verified_and_accuracy(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = GoldRetrievalQuestion(
        question_id="retq_" + "b" * 64,
        question="Tra cứu doanh thu thuần của ACB năm 2023.",
        intent="lookup",
        filters={},
        gold_table_ids=(TABLE_ID,),
        reviewed_by="tester",
        reviewed_at="2026-08-15T00:00:00+00:00",
        gold_evidence=(
            GoldTableEvidence(
                table_id=TABLE_ID,
                relative_path="ACB/2023/report.txt",
                line_start=5,
                line_end=5,
                verified=True,
            ),
        ),
        dataset_fingerprint="0" * 64,
    )
    settings = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))
    report = evaluate_answer_packages_on_gold(
        [question],
        release_dir,
        execution_settings=settings,
        answer_gold={"retq_" + "b" * 64: Decimal("100")},
    )
    assert report.plannable_count == 1
    assert report.answered_count == 1
    assert report.verified_count == 1
    assert report.rejected_count == 0
    assert report.accuracy_against_gold == 1.0


def test_write_answer_verification_report_creates_files(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = GoldRetrievalQuestion(
        question_id="retq_" + "c" * 64,
        question="Tra cứu doanh thu thuần của ACB năm 2023.",
        intent="lookup",
        filters={},
        gold_table_ids=(TABLE_ID,),
        reviewed_by="tester",
        reviewed_at="2026-08-15T00:00:00+00:00",
        gold_evidence=(
            GoldTableEvidence(
                table_id=TABLE_ID,
                relative_path="ACB/2023/report.txt",
                line_start=5,
                line_end=5,
                verified=True,
            ),
        ),
        dataset_fingerprint="0" * 64,
    )
    settings = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))
    report = evaluate_answer_packages_on_gold(
        [question], release_dir, execution_settings=settings, answer_gold=None
    )
    output_dir = tmp_path / "artifacts"
    json_path, markdown_path = write_answer_verification_report(report, output_dir)
    assert json_path.exists()
    assert markdown_path.exists()


def test_load_answer_gold_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "answer-gold-v1.jsonl"
    path.write_text(
        '{"question_id": "retq_' + "b" * 64 + '", "answer": "100"}\n'
        '{"question_id": "retq_' + "c" * 64 + '", "answer": "-56617982.0"}\n',
        encoding="utf-8",
    )
    gold = load_answer_gold(path)
    assert gold["retq_" + "b" * 64] == Decimal("100")
    assert gold["retq_" + "c" * 64] == Decimal("-56617982.0")
