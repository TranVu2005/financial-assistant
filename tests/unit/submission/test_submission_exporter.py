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
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
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
    _bare_year_periods,
    _render_csv_bytes,
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
    placements = [
        {
            "table_id": TABLE_ID,
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
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


def test_export_submission_evidence_csv_contains_the_full_extracted_table(
    tmp_path: Path,
) -> None:
    """Regression: the packaged evidence CSV must be the real extracted
    table the compiler actually searched (every numeric cell of the source
    table(s), via `execution.cell_frame.build_cell_frame`), not a single
    synthesized row for just the one cell the answer used."""
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
            "cell_id": "cell_" + "b" * 64,
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
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.answered_count == 1
    rows = csv_rows["data/q000001_df1.csv"]
    assert len(rows) == 2
    labels = {row["row_label_raw"] for row in rows}
    assert labels == {"Doanh thu thuan", "Gia von hang ban"}
    assert all(row.get("column_label") == "Năm 2023" for row in rows)


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
        apply_backstop=False,
    )

    assert report.answered_count == 0
    assert items == ()
    assert csv_rows == {}


def test_export_submission_rule_abstain_without_llm_client_stays_abstained(
    tmp_path: Path,
) -> None:
    """Regression: `llm_client=None` (the default) must reproduce the exact
    pre-LLM-fallback behavior -- rule-planner abstain stays a `planning`
    abstain, nothing tries to reach a network endpoint. `apply_backstop=False`
    isolates this from the Day 23 backstop tier, a separate concern."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=3, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        apply_backstop=False,
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


def test_export_submission_grounds_raw_metric_when_rule_planner_would_abstain(
    tmp_path: Path,
) -> None:
    """Day 23 plan §1.1/Step 1: a question naming a metric that only exists
    as a `row_label_raw` (no canonical alias) must not die with
    `metric_unknown` when that exact label is the unambiguous match among
    this question's own retrieved candidate tables."""
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2020/ACB_financial_statements_2020_consolidated_extracted.txt",
            "company_code": "ACB",
            "report_year": 2020,
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
            "title_raw": "Thuyet minh",
            "statement_type": None,
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
            "row_label_raw": "Lãi tiền gửi",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2020",
            "column_label_canonical": None,
            "value_raw": "70",
            "value_numeric": Decimal("70"),
            "period": "2020",
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

    document = TableDocument(
        table_id=TABLE_ID,
        doc_id=DOC_ID,
        text="company_code: ACB\nperiod: 2020\nLãi tiền gửi | 2020 | 70",
        metadata=TableMetadata(
            table_id=TABLE_ID,
            doc_id=DOC_ID,
            company_code="ACB",
            periods=("2020",),
            statement_type=None,
            source_path="a.txt",
            line_start=1,
            line_end=3,
        ),
        metric_labels=(),
    )
    service = RetrievalService(build_bm25_index((document,), dataset_fingerprint="f" * 64))
    question = RawQuestion(id=1, question="Lãi tiền gửi của ACB năm 2020 là bao nhiêu?")

    report, items, csv_rows = export_submission(
        [question],
        service,
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.answered_count == 1
    assert items[0].answer == 70.0
    assert report.outcomes[0].plan_source == "rule_raw_grounded"


def test_export_submission_falls_to_grounded_llm_when_typed_llm_abstains(
    tmp_path: Path,
) -> None:
    """Day 23 full-coverage strategy tier 3: when the typed, vocabulary-free
    LLM planner (ADR 0006 B1) can't produce a valid plan, a second attempt
    shown the real candidate-table content should succeed by copying the
    real row label verbatim."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=3, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")

    invalid_typed_plan = json.dumps({"operation": "not_a_real_operation"})
    grounded_valid_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["ACB"],
            "periods": ["2023"],
            "metric": {"raw_text": "Doanh thu thuan"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_message = body["messages"][1]["content"]
        content = (
            grounded_valid_plan if "Nội dung bảng ứng viên" in user_message else invalid_typed_plan
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

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
    assert report.outcomes[0].plan_source == "llm_grounded"
    assert items[0].answer == 100.0


def test_export_submission_backstop_fills_when_every_tier_fails(tmp_path: Path) -> None:
    """Day 23 full-coverage strategy tier 4: plan.md §2.4 rule 1 fails the
    *entire* ZIP if even one id is missing, so a question no reasoning tier
    can answer must still produce a contract-valid item."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=5, question="Câu hỏi hoàn toàn không xác định được công ty nào.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
    )

    assert report.question_count == 1
    assert report.answered_count == 0
    assert report.backstopped_count == 1
    assert len(items) == 1
    assert items[0].id == 5
    assert report.outcomes[0].status == "backstopped"


def test_export_submission_backstop_disabled_keeps_old_behavior(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=5, question="Câu hỏi hoàn toàn không xác định được công ty nào.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        apply_backstop=False,
    )

    assert report.backstopped_count == 0
    assert len(items) == 0
    assert report.outcomes[0].status == "abstained"


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


def test_export_submission_with_row_fusion(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from financial_report_qa.retrieval.row_fusion import RowFusionService
    from financial_report_qa.retrieval.row_fusion_contracts import RowFusionTrace, RowFusionWeights

    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    # 1. Test when row_fusion is None (backward compatibility check)
    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        row_fusion=None,
    )
    assert report.answered_count == 1

    # 2. Test when row_fusion is a mock service. It should be called during planning fallback.
    # To force fallback to LLM planners (where row_fusion results are actually used),
    # we use a question that the rule planner is guaranteed to abstain on (e.g. unknown metric).
    unsupported_question = RawQuestion(
        id=2, question="Tra cứu chỉ số không tồn tại của ACB năm 2023."
    )

    mock_fusion = MagicMock(spec=RowFusionService)
    # Set up mock to return an empty trace
    mock_fusion.retrieve_rows.return_value = RowFusionTrace(
        query=unsupported_question.question,
        weights=RowFusionWeights(bm25=1, dense=1),
        candidate_table_ids=(TABLE_ID,),
        bm25_candidate_count=0,
        dense_candidate_count=0,
        results=(),
    )

    # Note: LLM client is needed because LLM fallback only executes when llm_client is not None.
    # We mock LLM client to return a dummy json choice.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choice": 0})

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    export_submission(
        [unsupported_question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
        row_fusion=mock_fusion,
    )

    # Verify retrieve_rows was called exactly once on our mock
    mock_fusion.retrieve_rows.assert_called_once_with(
        unsupported_question.question, candidate_table_ids=(TABLE_ID,)
    )


def test_export_submission_semantic_grounding_recovery(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    from financial_report_qa.retrieval.row_documents import RowMetadata
    from financial_report_qa.retrieval.row_fusion import RowFusionService
    from financial_report_qa.retrieval.row_fusion_contracts import (
        RowFusedCandidate,
        RowFusionTrace,
        RowFusionWeights,
    )

    release_dir = _write_release(tmp_path)

    # We want rule planner to abstain (by passing a question that metric parser cannot canonically map)
    # so that LLM cell grounding gets invoked.
    unsupported_question = RawQuestion(
        id=11, question="Tra cứu chỉ số không rõ ràng của ACB năm 2023."
    )

    # Top candidates from row fusion:
    # 1. "Doanh thu ao" (which will fail compile with metric_not_found)
    # 2. "Doanh thu thuan" (which exists in release cells and will succeed compile)
    candidates = (
        RowFusedCandidate(
            row_id=f"{TABLE_ID}|row_99",
            table_id=TABLE_ID,
            row_idx=99,
            rank=1,
            fused_score=0.9,
            metadata=RowMetadata(table_id=TABLE_ID, row_idx=99, row_label_raw="Doanh thu ao"),
            snippet="dummy",
        ),
        RowFusedCandidate(
            row_id=f"{TABLE_ID}|row_0",
            table_id=TABLE_ID,
            row_idx=0,
            rank=2,
            fused_score=0.8,
            metadata=RowMetadata(table_id=TABLE_ID, row_idx=0, row_label_raw="Doanh thu thuan"),
            snippet="dummy",
        ),
    )

    mock_fusion = MagicMock(spec=RowFusionService)
    mock_fusion.retrieve_rows.return_value = RowFusionTrace(
        query=unsupported_question.question,
        weights=RowFusionWeights(bm25=1, dense=0),
        candidate_table_ids=(TABLE_ID,),
        bm25_candidate_count=2,
        dense_candidate_count=0,
        results=candidates,
    )

    # Mock LLM to choose option 1: "Doanh thu ao" (index 1)
    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"choices": [{"message": {"content": '{"choice": 1}'}}]}
        return httpx.Response(200, json=payload)

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, items, csv_rows = export_submission(
        [unsupported_question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
        row_fusion=mock_fusion,
    )

    # It must have successfully recovered using candidate switching to option 2 "Doanh thu thuan"
    assert report.answered_count == 1
    assert report.outcomes[0].status == "answered"
    assert report.outcomes[0].plan_source == "llm_cell_grounded_recovered"
    # plan.md §9: the accepted row's own fused_score, not the rejected
    # "Doanh thu ao" candidate's.
    assert report.outcomes[0].grounding_score == 0.8
    assert len(items) == 1
    assert items[0].answer == Decimal("100")


def _duplicate_label_release(tmp_path: Path) -> Path:
    """The release shape that produced the dev benchmark's dominant error:
    two identically-labelled rows holding different figures (plan.md §19)."""
    release_dir = _write_release(tmp_path)
    cells = [
        {
            "cell_id": "cell_" + character * 64,
            "table_id": TABLE_ID,
            "row_idx": row_idx,
            "col_idx": 1,
            "row_label_raw": "Doanh thu thuan",
            "row_label_canonical": "net_revenue",
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2023",
            "column_label_canonical": None,
            "value_raw": value,
            "value_numeric": Decimal(value),
            "period": "2023",
            "unit": "VND",
            "source_line_start": 5,
            "source_line_end": 5,
            "extraction_confidence": 0.9,
        }
        for character, row_idx, value in (("b", 0, "100"), ("c", 1, "900"))
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "table_id": TABLE_ID,
                    "row_idx": cell["row_idx"],
                    "col_idx": cell["col_idx"],
                    "cell_id": cell["cell_id"],
                }
                for cell in cells
            ],
            schema=PLACEMENT_SCHEMA,
        ),
        release_dir / "placements.parquet",
    )
    return release_dir


def test_export_submission_grounds_the_rule_plan_to_the_retrieved_row_position(
    tmp_path: Path,
) -> None:
    """plan.md §9/§14 on the primary path: the rule planner answers most
    questions without ever entering grounding recovery, so position binding
    has to apply there too -- otherwise the wrong-row answers the dev
    benchmark measured are never touched, because they are answers."""
    from unittest.mock import MagicMock

    from financial_report_qa.retrieval.row_documents import RowMetadata
    from financial_report_qa.retrieval.row_fusion import RowFusionService
    from financial_report_qa.retrieval.row_fusion_contracts import (
        RowFusedCandidate,
        RowFusionTrace,
        RowFusionWeights,
    )

    release_dir = _duplicate_label_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    row_fusion = MagicMock(spec=RowFusionService)
    row_fusion.retrieve_rows.return_value = RowFusionTrace(
        query=question.question,
        weights=RowFusionWeights(bm25=1, dense=1),
        candidate_table_ids=(TABLE_ID,),
        bm25_candidate_count=1,
        dense_candidate_count=0,
        results=(
            RowFusedCandidate(
                row_id=f"{TABLE_ID}|row_1",
                table_id=TABLE_ID,
                row_idx=1,
                rank=1,
                fused_score=0.9,
                snippet="Doanh thu thuan | 900",
                metadata=RowMetadata(
                    table_id=TABLE_ID,
                    row_idx=1,
                    company_code="ACB",
                    row_label_raw="Doanh thu thuan",
                    row_label_canonical="net_revenue",
                ),
            ),
        ),
    )

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        row_fusion=row_fusion,
    )

    assert report.answered_count == 1
    assert items[0].answer == 900.0
    assert "df1.loc[" in items[0].pandas_query
    assert "row_label" not in items[0].pandas_query


def _evidence_planner_fusion(row_idx: int = 0) -> object:
    from unittest.mock import MagicMock

    from financial_report_qa.retrieval.row_documents import RowMetadata
    from financial_report_qa.retrieval.row_fusion import RowFusionService
    from financial_report_qa.retrieval.row_fusion_contracts import (
        RowFusedCandidate,
        RowFusionTrace,
        RowFusionWeights,
    )

    row_fusion = MagicMock(spec=RowFusionService)
    row_fusion.retrieve_rows.return_value = RowFusionTrace(
        query="q",
        weights=RowFusionWeights(bm25=1, dense=1),
        candidate_table_ids=(TABLE_ID,),
        bm25_candidate_count=1,
        dense_candidate_count=0,
        results=(
            RowFusedCandidate(
                row_id=f"{TABLE_ID}|row_{row_idx}",
                table_id=TABLE_ID,
                row_idx=row_idx,
                rank=1,
                fused_score=0.9,
                snippet="Doanh thu thuan | 100",
                metadata=RowMetadata(
                    table_id=TABLE_ID,
                    row_idx=row_idx,
                    company_code="ACB",
                    row_label_raw="Doanh thu thuan",
                    row_label_canonical="net_revenue",
                ),
            ),
        ),
    )
    return row_fusion


def test_export_submission_uses_the_evidence_aware_planner_when_rules_abstain(
    tmp_path: Path,
) -> None:
    """plan.md §12: with grounded facts in hand the planner only has to name
    `{operation, operands}` -- it is never asked to invent a table, a row
    locator, a column locator or a metric name."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=7, question="Tra cứu chỉ tiêu không rõ ràng của ACB năm 2023.")

    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"operation": "lookup", "operands": ["F1"]})
                        }
                    }
                ]
            },
        )

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
        row_fusion=_evidence_planner_fusion(),
    )

    assert report.answered_count == 1
    assert items[0].answer == 100.0
    assert report.outcomes[0].plan_source == "llm_evidence_planner"
    # §9/§14: the plan it produced is position-bound, so execution is a
    # deterministic positional read rather than another label match.
    assert "df1.loc[" in items[0].pandas_query
    # The very first model call is the evidence-planner one: the facts are in
    # the prompt and no locator field is in the schema.
    first_prompt = "\n".join(str(m["content"]) for m in seen[0]["messages"])  # type: ignore[index,union-attr]
    assert "F1:" in first_prompt
    assert "Doanh thu thuan" in first_prompt


def test_export_submission_evidence_planner_falls_through_when_the_model_declines(
    tmp_path: Path,
) -> None:
    """The tier is additive: a planner that cannot name an operation must
    leave the existing LLM planner chain exactly as it was."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=8, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")
    typed_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["ACB"],
            "periods": ["2023"],
            "metric": {"canonical": "net_revenue"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = "\n".join(str(m["content"]) for m in body["messages"])
        # Decline only the evidence-planner call; answer the typed-plan one.
        content = "không rõ" if "Các số liệu đã trích sẵn" in prompt else typed_plan
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
        row_fusion=_evidence_planner_fusion(),
    )

    assert report.answered_count == 1
    assert items[0].answer == 100.0
    assert report.outcomes[0].plan_source == "llm"


def test_export_submission_evidence_planner_is_skipped_without_row_fusion(
    tmp_path: Path,
) -> None:
    """No row retrieval means no grounded facts, and §12 explicitly refuses to
    let the planner work without them -- so the tier must not fire at all."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=9, question="Tra cứu tổng lợi thế cạnh tranh của ACB năm 2023.")
    typed_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["ACB"],
            "periods": ["2023"],
            "metric": {"canonical": "net_revenue"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = "\n".join(str(m["content"]) for m in body["messages"])
        assert "Các số liệu đã trích sẵn" not in prompt
        return httpx.Response(200, json={"choices": [{"message": {"content": typed_plan}}]})

    llm_client = LLMClient(_LLM_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)

    report, _, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=llm_client,
        row_fusion=None,
    )
    assert report.outcomes[0].plan_source == "llm"


def test_bare_year_periods_keeps_bare_four_digit_years() -> None:
    assert _bare_year_periods(("2022", "2023")) == (2022, 2023)


def test_bare_year_periods_drops_a_full_iso_date() -> None:
    r"""A live full-export run crashed on question text naming an explicit
    date ("ngày 31/12/2015") rather than a bare year: `entity_parser`
    resolves such a question's `entities.periods` to an ISO date string
    ("2015-12-31"), which the §12 wiring passed straight into `int(...)` and
    blew up with an uncaught `ValueError` -- taking down the whole export,
    not just this one question. `rule_planner.py` already guards this exact
    case (only ever building a plan when every period matches the bare-year
    pattern, `_PERIOD_PATTERN = re.compile(r"^\d{4}$")`); the evidence-planner
    wiring must apply the same filter instead of assuming every parsed
    period is a bare year."""
    assert _bare_year_periods(("2015-12-31",)) == ()


def test_bare_year_periods_drops_only_the_non_year_entries() -> None:
    assert _bare_year_periods(("2023", "2015-12-31", "2022")) == (2023, 2022)


def test_bare_year_periods_of_no_periods_is_empty() -> None:
    assert _bare_year_periods(()) == ()


def test_render_csv_bytes_includes_position_columns() -> None:
    """84 query dùng df1.table_id/df1.row_idx; CSV phải mang theo hai cột đó."""
    rows = [
        {
            "table_id": "tbl_abc",
            "row_idx": 19,
            "col_idx": 2,
            "company_code": "VNM",
            "row_label_canonical": None,
            "row_label_raw": "Doanh thu thuần",
            "column_label": "2023",
            "period": 2023,
            "value": 1200.0,
        }
    ]
    header = _render_csv_bytes(rows).decode("utf-8").splitlines()[0]
    assert header.split(",") == [
        "table_id",
        "row_idx",
        "col_idx",
        "company_code",
        "row_label_canonical",
        "row_label_raw",
        "column_label",
        "period",
        "value",
    ]
