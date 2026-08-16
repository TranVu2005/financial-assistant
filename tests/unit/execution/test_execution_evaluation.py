"""Tests for the Day 18 compile-plans evaluation harness."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.evaluation import evaluate_compiled_plans_on_gold
from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    GoldTableEvidence,
    RetrievalFilters,
)

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
FINGERPRINT = "0" * 64

_ALLOW_ALL = ExecutionSettings(
    timeout_seconds=5,
    max_rows=100000,
    allow_operations=(
        "lookup",
        "compare",
        "compare_companies",
        "difference",
        "growth_rate",
        "ratio",
        "average",
        "sum",
        "rank",
    ),
)


def _write_release(tmp_path: Path, cells: list[dict[str, object]]) -> Path:
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
            "row_count": 2,
            "column_count": 3,
            "quality_score": 0.9,
            "csv_path": None,
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


def _cell(cell_id: str, *, row: int, value_numeric: str, period: str) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": TABLE_ID,
        "row_idx": row,
        "col_idx": 1,
        "row_label_raw": "Doanh thu thuan",
        "row_label_canonical": "net_revenue",
        "row_group_context_raw": None,
        "column_label_raw": f"Năm {period}",
        "column_label_canonical": None,
        "value_raw": value_numeric,
        "value_numeric": Decimal(value_numeric),
        "period": period,
        "unit": "VND",
        "source_line_start": row + 1,
        "source_line_end": row + 1,
        "extraction_confidence": 0.9,
    }


def _gold_question(question_id: str, question: str) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion(
        question_id=question_id,
        dataset_fingerprint=FINGERPRINT,
        question=question,
        intent="lookup",
        filters=RetrievalFilters(company_codes=("ACB",), periods=("2023",)),
        gold_table_ids=(TABLE_ID,),
        reviewed_by="test-fixture",
        reviewed_at=datetime(2026, 8, 15, tzinfo=UTC),
        gold_evidence=(
            GoldTableEvidence(
                table_id=TABLE_ID,
                relative_path="ACB/2023/report.txt",
                line_start=1,
                line_end=10,
                verified=True,
            ),
        ),
    )


def test_evaluate_compiled_plans_counts_resolved_and_unresolved(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path, [_cell("cell_" + "a" * 64, row=1, value_numeric="500", period="2023")]
    )
    resolved_question = _gold_question(
        "retq_" + "1" * 64, "Doanh thu thuần của ACB năm 2023 là bao nhiêu?"
    )
    unresolved_question = _gold_question(
        "retq_" + "2" * 64, "Danh sách công ty con của ACB năm 2023 là gì?"
    )
    report = evaluate_compiled_plans_on_gold(
        (resolved_question, unresolved_question), release_dir, execution_settings=_ALLOW_ALL
    )
    assert report.question_count == 2
    assert report.plannable_count >= 1
    assert report.resolved_count >= 1
    assert 0.0 <= report.resolved_rate <= 1.0


def test_evaluate_compiled_plans_is_deterministic(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path, [_cell("cell_" + "a" * 64, row=1, value_numeric="500", period="2023")]
    )
    question = _gold_question("retq_" + "1" * 64, "Doanh thu thuần của ACB năm 2023 là bao nhiêu?")
    first = evaluate_compiled_plans_on_gold((question,), release_dir, execution_settings=_ALLOW_ALL)
    second = evaluate_compiled_plans_on_gold(
        (question,), release_dir, execution_settings=_ALLOW_ALL
    )
    assert first == second
