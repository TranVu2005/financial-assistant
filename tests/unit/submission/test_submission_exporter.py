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
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.row_fusion import DEFAULT_ROW_CANDIDATE_COUNT
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


def _write_release(
    tmp_path: Path, *, values: tuple[Decimal, Decimal] = (Decimal("100"), Decimal("60"))
) -> Path:
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
    # >= 2 numeric cells (plan.md submission compliance C1, and the Critical
    # 1 backstop/evidence guard added in the 2026-08-21 final review both
    # require it): a table with a single numeric cell reads as a hardcoded
    # answer -- its one CSV row's `value` equals `item.answer`.
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
            "value_raw": str(values[0]),
            "value_numeric": values[0],
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
            "value_raw": str(values[1]),
            "value_numeric": values[1],
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


@pytest.fixture
def release_dir(tmp_path: Path) -> Path:
    """Release with two distinct table_ids (each with a numeric, dated
    cell) so retrieval-rank-order tests can pick a pair whose rank order
    disagrees with alphabetical order."""
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    table_id_low = "tbl_" + "1" * 64
    table_id_high = "tbl_" + "9" * 64
    cell_id_low = "cell_" + "1" * 64
    cell_id_high = "cell_" + "9" * 64
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
            "table_id": table_id,
            "doc_id": DOC_ID,
            "source_ordinal": ordinal,
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
        for ordinal, table_id in enumerate((table_id_low, table_id_high))
    ]
    cells = [
        {
            "cell_id": cell_id,
            "table_id": table_id,
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
        for table_id, cell_id in (
            (table_id_low, cell_id_low),
            (table_id_high, cell_id_high),
        )
    ]
    # Distinct source_line_start per table so their reported "doc|line"
    # strings differ -- otherwise the two tables' output tokens would
    # collide and order could never be observed.
    cells[0]["source_line_start"] = 5
    cells[0]["source_line_end"] = 5
    cells[1]["source_line_start"] = 50
    cells[1]["source_line_end"] = 50
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
            "table_id": cell["table_id"],
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


@pytest.fixture
def two_ranked_table_ids(release_dir: Path) -> tuple[str, str]:
    """Two real table_ids in the release, deliberately picked so the first
    element sorts AFTER the second alphabetically -- if code mistakenly uses
    build_cell_frame (ORDER BY table_id) as the order source, this test
    should catch the reversal."""
    import duckdb

    connection = duckdb.connect(":memory:")
    frame = connection.execute(
        "SELECT DISTINCT table_id FROM read_parquet(?) "
        "WHERE value_numeric IS NOT NULL AND period IS NOT NULL "
        "ORDER BY table_id DESC LIMIT 2",
        [str(release_dir / "cells.parquet")],
    ).fetchdf()
    connection.close()
    ids = frame["table_id"].tolist()
    return ids[0], ids[1]  # ids[0] > ids[1] alphabetically (ORDER BY DESC)


def test_relevant_tables_come_from_retrieval_not_from_evidence(release_dir: Path) -> None:
    """Retrieval score (50%) is graded independently of answering success,
    so the reported table list must reflect retrieval, not the tables the
    answer happened to use."""
    import inspect

    from financial_report_qa.submission.exporter import _relevant_docs_and_tables

    signature = inspect.signature(_relevant_docs_and_tables)
    assert "compiled" not in signature.parameters, (
        "this function must not depend on the execution result"
    )
    assert "retrieved_table_ids" in signature.parameters


def test_relevant_tables_preserve_retrieval_rank_order(
    release_dir: Path, two_ranked_table_ids: tuple[str, str]
) -> None:
    """MRR5 (dashboard column TABLES MRR5) scores rank, not set membership.

    The rank-1 table in the input must be the first element of the output --
    even when it does not come first alphabetically by table_id
    (build_cell_frame's ORDER BY table_id would sort it wrong if used as the
    order source)."""
    from financial_report_qa.submission.exporter import _relevant_docs_and_tables

    rank1_table_id, rank2_table_id = two_ranked_table_ids
    # Deliberately choose a pair where rank1 sorts AFTER rank2 alphabetically,
    # so the test cannot pass by coincidence with build_cell_frame's
    # alphabetical ORDER BY table_id.
    assert rank1_table_id > rank2_table_id, (
        "fixture must produce rank1 after rank2 alphabetically for this test to be meaningful"
    )

    _docs_a, tables_rank1_first = _relevant_docs_and_tables(
        (rank1_table_id, rank2_table_id), release_dir
    )
    _docs_b, tables_rank2_first = _relevant_docs_and_tables(
        (rank2_table_id, rank1_table_id), release_dir
    )

    # The two tables must map to distinct output tokens (otherwise order
    # could never be observed), and swapping the input rank order must swap
    # the output order. An implementation that (re)sources its order from
    # build_cell_frame's alphabetical ORDER BY table_id would produce the
    # SAME output regardless of input order -- this assertion catches that.
    assert tables_rank1_first[0] != tables_rank1_first[1]
    assert tables_rank1_first == (tables_rank2_first[1], tables_rank2_first[0]), (
        "output order must track retrieval-rank input order, not table_id alphabetical order"
    )


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
            "row_count": 2,
            "column_count": 2,
            "quality_score": 0.9,
            "csv_path": None,
        }
    ]
    # >= 2 numeric cells: see the `_write_release` comment above -- a
    # single-cell table now fails the Critical 1 backstop/evidence guard.
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
        },
        {
            "cell_id": "cell_" + "e" * 64,
            "table_id": TABLE_ID,
            "row_idx": 1,
            "col_idx": 1,
            "row_label_raw": "Lãi tiền vay",
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": "Năm 2020",
            "column_label_canonical": None,
            "value_raw": "30",
            "value_numeric": Decimal("30"),
            "period": "2020",
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
        unsupported_question.question,
        candidate_table_ids=(TABLE_ID,),
        k=DEFAULT_ROW_CANDIDATE_COUNT,
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

    # We want rule planner to abstain (by passing a question that metric
    # parser cannot canonically map) so that LLM cell grounding gets invoked.
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
    # Spec 2026-08-21 §5.2: position binding no longer strips the semantic
    # label from the predicate -- the label names the metric, `row_idx` only
    # breaks the tie between these two identically-labelled rows (0: 100,
    # 1: 900).
    assert 'df1.row_label_canonical == "net_revenue"' in items[0].pandas_query


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


def test_export_submission_offline_row_decisions_work_without_a_live_llm_client(
    tmp_path: Path,
) -> None:
    """Critical-1 regression: an offline `--row-choice-decisions` file must
    make grounding recovery's Attempt 0 row choice reachable even with
    `llm_client=None` -- that is the entire point of the offline decisions
    path (plan.md §7.1). Before the fix, `ground_with_recovery`'s Attempt 0
    only checked `fusion_rows` truthiness, not whether `row_decisions` was
    actually supplied, so this scenario worked "by accident" for any
    non-empty fusion_rows regardless of whether a decisions file was passed
    -- and, worse, the *reverse* case (no decisions file, but a live
    llm_client) silently skipped the live-LLM `choose_row_label` path
    entirely. This test pins the offline path: no llm_client, a real
    row_decisions mapping, non-empty fusion_rows -> `plan_source` must be
    `"llm_row_choice"`."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=7, question="Tra cứu chỉ tiêu không rõ ràng của ACB năm 2023.")

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        llm_client=None,
        row_fusion=_evidence_planner_fusion(),
        row_decisions={7: 0},
    )

    assert report.answered_count == 1
    assert items[0].answer == 100.0
    assert report.outcomes[0].plan_source == "llm_row_choice"
    assert "df1.loc[" in items[0].pandas_query


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


def test_answered_path_never_emits_synthesized_single_row(tmp_path: Path) -> None:
    """Bất biến BI-1: evidence CSV luôn là lát cắt bảng thật.

    Khi `_real_table_evidence_rows` không thể replay bảng thật ra đúng đáp
    án (trả về `None`), câu hỏi phải thất bại execution với
    `evidence_frame_replay_mismatch` -- không còn nhánh dựng ngược một dòng
    CSV từ đáp án (`_replay_rows_to_csv_rows`, đã bị xoá) để rơi vào.
    """
    from financial_report_qa.submission import exporter

    assert not hasattr(exporter, "_replay_rows_to_csv_rows"), (
        "_replay_rows_to_csv_rows là nhánh sinh CSV một dòng -- phải bị xoá"
    )

    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    calls: list[str] = []

    def _fake_real_rows(compiled, release_dir, *, timeout_seconds):
        calls.append("called")
        return None

    original = exporter._real_table_evidence_rows
    exporter._real_table_evidence_rows = _fake_real_rows
    try:
        report, items, csv_rows = exporter.export_submission(
            [question],
            _service(),
            release_dir,
            execution_settings=_ALLOW_LOOKUP,
            dataset_fingerprint="0" * 64,
            k=10,
            apply_backstop=False,
        )
    finally:
        exporter._real_table_evidence_rows = original

    assert calls == ["called"]
    assert report.answered_count == 0
    assert items == ()
    assert csv_rows == {}
    outcome = report.outcomes[0]
    assert outcome.status == "error"
    assert outcome.stage == "execution"
    assert outcome.code == "evidence_frame_replay_mismatch"


def _lookup_plan_in(unit: str) -> FinancialQueryPlan:
    return FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="net_revenue"),
        expected_unit=unit,
    )


def test_unit_converted_query_replays_against_the_real_unscaled_table(tmp_path: Path) -> None:
    """Regression: a "triệu đồng" answer must survive the evidence gate.

    `compile_plan` used to apply a monetary unit conversion by scaling the
    *replay frame* and leaving `pandas_query` alone. Its own replay then
    agreed, but `_real_table_evidence_rows` -- and `validate_submission_zip`
    after it -- replay that same query against the real, unscaled corpus
    slice, so the query returned raw VND while `answer` held the converted
    figure. Every question asking for triệu/tỷ đồng was thrown away as
    `evidence_frame_replay_mismatch` (173 questions on the 2026-08-22 full
    export, 124 of them "tỷ đồng"). The conversion has to live in the query.
    """
    from financial_report_qa.execution.compiler import compile_plan
    from financial_report_qa.submission import exporter

    raw = Decimal("208253201298")
    release_dir = _write_release(tmp_path, values=(raw, Decimal("60000000000")))

    compiled = compile_plan(
        _lookup_plan_in("VND_million"), release_dir, execution_settings=_ALLOW_LOOKUP
    )
    assert compiled.status == "answered", compiled.error_message
    assert compiled.answer == raw / Decimal(1_000_000)
    # Division, not multiplication: the query grammar allows Add/Sub/Div but
    # not Mult, so a `* factor` form is rejected by the sandbox outright.
    assert "/ 1000000" in compiled.pandas_query

    rows = exporter._real_table_evidence_rows(compiled, release_dir, timeout_seconds=5)
    assert rows is not None, "converted answer must survive the real-table replay gate"
    # The packaged CSV stays the real corpus slice -- raw VND, never rescaled.
    assert any(row["value"] == float(raw) for row in rows)


def test_evidence_gate_still_rejects_a_genuinely_wrong_replay(tmp_path: Path) -> None:
    """The tolerance added for float64 round-trip must not blunt the gate.

    A query whose replay disagrees by a real margin (here a wrong period, so
    the frame replays a different cell) must still be refused, otherwise the
    BI-1 invariant that the CSV genuinely reproduces the answer is gone.
    """
    from financial_report_qa.execution.compiler import compile_plan
    from financial_report_qa.submission import exporter

    release_dir = _write_release(tmp_path, values=(Decimal("100"), Decimal("60")))
    compiled = compile_plan(_lookup_plan_in("VND"), release_dir, execution_settings=_ALLOW_LOOKUP)
    assert compiled.status == "answered", compiled.error_message

    tampered = compiled.model_copy(update={"answer": compiled.answer + Decimal("5")})
    assert exporter._real_table_evidence_rows(tampered, release_dir, timeout_seconds=5) is None
