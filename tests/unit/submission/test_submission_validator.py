"""Tests for the Day 22 offline submission validator (plan.md §2.4 rule 9):
opens the packaged ZIP fresh (never trusts exporter in-memory state), checks
schema/id-set/CSV-replay, independent of `submission/exporter.py`."""

from __future__ import annotations

import zipfile
from decimal import Decimal
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.live_query import TableRetriever
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.contracts import (
    RawQuestion,
    SubmissionEvidence,
    SubmissionItem,
)
from financial_report_qa.submission.exporter import export_submission, write_submission_zip
from financial_report_qa.submission.validator import validate_submission_zip

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_ALLOW_LOOKUP = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))


def _write_release(tmp_path: Path) -> Path:
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
    # >= 2 numeric cells (submission compliance C1, and the Critical 1
    # backstop/evidence guard, 2026-08-21 final review): a single-cell
    # table's one CSV row would have `value` == `item.answer`.
    cells = [
        {
            "cell_id": CELL_ID,
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
    return release_dir


def _service() -> TableRetriever:
    document = TableDocument(
        table_id=TABLE_ID,
        doc_id=DOC_ID,
        text="company_code: ACB\nperiod: 2023\nDoanh thu thuần | 2023 | 100",
        metadata=TableMetadata(
            table_id=TABLE_ID,
            doc_id=DOC_ID,
            company_code="ACB",
            periods=("2023",),
            statement_type="income_statement",
            source_path="a.txt",
            line_start=1,
            line_end=3,
        ),
        metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
    )
    # cast: same known invariant-protocol mypy wart as retrieval/cli.py's sweep-k wiring.
    index = build_bm25_index((document,), dataset_fingerprint="f" * 64)
    return cast(TableRetriever, RetrievalService(index))


def _build_zip(tmp_path: Path) -> tuple[Path, Path]:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")
    _, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )
    zip_path = tmp_path / "submission.zip"
    write_submission_zip(items, csv_rows, zip_path)
    return zip_path, release_dir


def test_validate_submission_zip_accepts_a_clean_export(tmp_path: Path) -> None:
    zip_path, _ = _build_zip(tmp_path)
    report = validate_submission_zip(zip_path, expected_ids=[1])
    assert report.valid is True
    assert report.item_count == 1
    assert report.issues == ()


def test_validate_submission_zip_rejects_id_set_mismatch(tmp_path: Path) -> None:
    zip_path, _ = _build_zip(tmp_path)
    report = validate_submission_zip(zip_path, expected_ids=[1, 2])
    assert report.valid is False
    assert any(issue.code == "id_set_mismatch" for issue in report.issues)


def test_validate_submission_zip_rejects_tampered_answer(tmp_path: Path) -> None:
    zip_path, _ = _build_zip(tmp_path)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "submission.json":
                data = data.replace(b'"answer": 100.0', b'"answer": 999.0')
            dst.writestr(name, data)

    report = validate_submission_zip(tampered, expected_ids=[1])
    assert report.valid is False
    assert any(issue.code == "answer_mismatch" for issue in report.issues)


def test_validate_submission_zip_replays_a_numeric_looking_raw_label(tmp_path: Path) -> None:
    """Regression (Day 23 full-coverage strategy, found validating the real
    1.012-question backstop export): a `row_label_raw` that looks like a
    pure number (e.g. a footnote reference such as "2") round-trips through
    the packaged CSV as pandas' *inferred* int64 dtype, not string --
    `df1.row_label_raw == "2"` then compares an int column to a string and
    silently matches zero rows, crashing `.iloc[0]` on replay. Confirmed
    live: ids 468/486/513/546/564 all hit exactly this in the real export."""
    item = SubmissionItem.model_validate(
        {
            "id": 1,
            "question": "Câu hỏi bất kỳ.",
            "answer": 4.0,
            "relevant_docs": ("report",),
            "relevant_tables": ("report|5",),
            "evidence": (SubmissionEvidence(variable="df1", csv_path="data/q000001_df1.csv"),),
            "pandas_query": (
                'df1[(df1.company_code == "DLG") & (df1.row_label_raw == "2") '
                '& (df1.period == 2021)]["value"].iloc[0]'
            ),
        }
    )
    csv_rows = {
        "data/q000001_df1.csv": (
            {
                "company_code": "DLG",
                "row_label_canonical": None,
                "row_label_raw": "2",
                "period": 2021,
                "value": Decimal("4"),
            },
        )
    }
    zip_path = tmp_path / "submission.zip"
    write_submission_zip([item], csv_rows, zip_path)

    report = validate_submission_zip(zip_path, expected_ids=[1])

    assert report.valid is True, report.issues
    assert report.issues == ()


def test_validate_submission_zip_replays_a_column_refined_query(tmp_path: Path) -> None:
    """The packaged CSV and sandbox grammar must retain the same column
    dimension that the locator used to disambiguate the selected cell."""
    item = SubmissionItem.model_validate(
        {
            "id": 1,
            "question": "Thuế GTGT phải nộp cuối năm là bao nhiêu?",
            "answer": 200.0,
            "relevant_docs": ("report",),
            "relevant_tables": ("report|5",),
            "evidence": (SubmissionEvidence(variable="df1", csv_path="data/q000001_df1.csv"),),
            "pandas_query": (
                'df1[(df1.company_code == "PC1") '
                '& (df1.row_label_raw == "Thuế GTGT") '
                '& (df1.column_label == "Số phải nộp cuối năm") '
                '& (df1.period == 2025)]["value"].iloc[0]'
            ),
        }
    )
    csv_rows = {
        "data/q000001_df1.csv": (
            {
                "company_code": "PC1",
                "row_label_canonical": None,
                "row_label_raw": "Thuế GTGT",
                "column_label": "Số phải nộp đầu năm",
                "period": 2025,
                "value": Decimal("100"),
            },
            {
                "company_code": "PC1",
                "row_label_canonical": None,
                "row_label_raw": "Thuế GTGT",
                "column_label": "Số phải nộp cuối năm",
                "period": 2025,
                "value": Decimal("200"),
            },
        )
    }
    zip_path = tmp_path / "submission.zip"
    write_submission_zip([item], csv_rows, zip_path)

    report = validate_submission_zip(zip_path, expected_ids=[1])

    assert report.valid is True, report.issues
    assert report.issues == ()


def test_validate_submission_zip_rejects_orphan_csv(tmp_path: Path) -> None:
    zip_path, _ = _build_zip(tmp_path)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(zip_path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("data/orphan.csv", "company_code,value\nACB,1\n")

    report = validate_submission_zip(tampered, expected_ids=[1])
    assert report.valid is False
    assert any(issue.code == "orphan_csv" for issue in report.issues)
