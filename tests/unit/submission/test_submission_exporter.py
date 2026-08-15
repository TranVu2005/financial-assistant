"""Tests for the Day 22 submission exporter: live retrieval -> plan ->
execution -> verification -> `SubmissionItem` + CSV, for real, previously
unseen questions (no gold labels, no pre-computed rankings)."""

from __future__ import annotations

import json
import zipfile
from decimal import Decimal
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.config import ExecutionSettings, LLMSettings
from financial_report_qa.core.errors import SubmissionInputError
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.contracts import RawQuestion
from financial_report_qa.submission.exporter import (
    export_submission,
    load_raw_questions,
    write_submission_zip,
)

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_ALLOW_LOOKUP = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))
_LLM_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen3-4b-instruct-2507-q4_k_m",
    timeout_seconds=5.0,
    max_output_tokens=160,
    temperature=0.0,
    context_length=4096,
    json_schema_constrained=True,
)


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


def _service() -> RetrievalService:
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
    return RetrievalService(build_bm25_index((document,), dataset_fingerprint="f" * 64))


def test_export_submission_answers_a_real_unseen_question(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.question_count == 1
    assert report.answered_count == 1
    assert len(items) == 1
    item = items[0]
    assert item.id == 1
    assert item.answer == 100.0
    assert item.relevant_docs == ("ACB_financial_statements_2023_consolidated_extracted",)
    assert item.relevant_tables == ("ACB_financial_statements_2023_consolidated_extracted|5",)
    assert item.evidence[0].csv_path == "data/q000001_df1.csv"
    assert csv_rows["data/q000001_df1.csv"][0]["value"] == Decimal("100")


def test_export_submission_records_no_candidate_tables_as_abstained(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=2, question="Tra cứu doanh thu thuần của XYZCORP năm 2019.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.answered_count == 0
    assert items == ()
    assert csv_rows == {}


def test_export_submission_rule_abstain_without_llm_client_stays_abstained(
    tmp_path: Path,
) -> None:
    """Regression: `llm_client=None` (the default) must reproduce the exact
    pre-LLM-fallback behavior -- rule-planner abstain stays a `planning`
    abstain, nothing tries to reach a network endpoint."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=3, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.answered_count == 0
    assert items == ()
    assert report.outcomes[0].status == "abstained"
    assert report.outcomes[0].stage == "planning"


def test_export_submission_falls_back_to_llm_when_rule_planner_abstains(
    tmp_path: Path,
) -> None:
    """Day 22 coverage-improvement follow-up: when the rule planner abstains
    but an LLM client is supplied, `plan_router.route_plan` must be given a
    chance before the question is written off -- exercised here with a
    mocked httpx transport, never a live server (ADR 0006's existing test
    pattern in test_plan_router.py)."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=3, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")
    valid_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["ACB"],
            "periods": ["2023"],
            "metric": {"canonical": "net_revenue"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": valid_plan}}]})

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
    )

    assert report.answered_count == 1
    assert items[0].answer == 100.0
    assert report.outcomes[0].plan_source == "llm"


def test_export_submission_rule_success_never_calls_llm(tmp_path: Path) -> None:
    """Mirrors test_plan_router.py's own guarantee, at the exporter's own
    call site -- an LLM client being present must never override a rule plan
    that already succeeded."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM must not be called when the rule planner already succeeded")

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
    )

    assert report.answered_count == 1
    assert report.outcomes[0].plan_source == "rule"
    assert report.outcomes[0].status == "answered"


def test_load_raw_questions_parses_and_sorts_by_id(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": 2, "question": "b?"}\n{"id": 1, "question": "a?"}\n', encoding="utf-8")
    questions = load_raw_questions(path)
    assert [q.id for q in questions] == [1, 2]


def test_load_raw_questions_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": 1, "question": "a?"}\n{"id": 1, "question": "b?"}\n', encoding="utf-8")
    with pytest.raises(SubmissionInputError, match="duplicate"):
        load_raw_questions(path)


def test_write_submission_zip_is_deterministic_and_replayable(tmp_path: Path) -> None:
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

    sha1 = write_submission_zip(items, csv_rows, tmp_path / "out1.zip")
    sha2 = write_submission_zip(items, csv_rows, tmp_path / "out2.zip")
    assert sha1 == sha2

    with zipfile.ZipFile(tmp_path / "out1.zip") as archive:
        names = archive.namelist()
        assert names[0] == "submission.json"
        assert names[1:] == sorted(names[1:])
        payload = json.loads(archive.read("submission.json"))
        assert payload[0]["id"] == 1
        assert "data/q000001_df1.csv" in names
