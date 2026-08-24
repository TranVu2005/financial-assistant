"""Tests for the submission exporter: live retrieval -> numbered cell
candidates -> offline masked-PAL decision -> `SubmissionItem` + CSV, for
real, previously unseen questions (no gold labels, no pre-computed
rankings). Spec 2026-08-24 §4.3: the masked-PAL program is the ONLY
answering path; anything it cannot answer flows to the backstop tier."""

from __future__ import annotations

import json
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, call

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import SubmissionInputError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.execution.program_contracts import ProgramDecision
from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.live_query import TableRetriever
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion import DEFAULT_ROW_CANDIDATE_COUNT, RowFusionService
from financial_report_qa.retrieval.row_fusion_contracts import (
    RowFusedCandidate,
    RowFusionTrace,
    RowFusionWeights,
)
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.contracts import RawQuestion
from financial_report_qa.submission.exporter import (
    _render_csv_bytes,
    export_submission,
    load_raw_questions,
    write_submission_zip,
)

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_ALLOW_LOOKUP = ExecutionSettings(timeout_seconds=5, max_rows=20000)


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
    # cast: RetrievalTrace structurally satisfies TableRetriever but mypy
    # cannot prove it against the _RankedResult protocol (same known pattern
    # as retrieval/cli.py's sweep-k wiring).
    index = build_bm25_index((document,), dataset_fingerprint="f" * 64)
    return cast(TableRetriever, RetrievalService(index))


def _fused_row(*, row_idx: int, rank: int = 1, label: str = "Doanh thu thuan") -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{TABLE_ID}|row_{row_idx}",
        table_id=TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=0.9 - rank / 10,
        snippet=f"{label} | 2023",
        metadata=RowMetadata(
            table_id=TABLE_ID,
            row_idx=row_idx,
            company_code="ACB",
            row_label_raw=label,
        ),
    )


def _row_fusion(*results: RowFusedCandidate) -> MagicMock:
    """A `RowFusionService` test double returning exactly `results`."""
    fusion = MagicMock(spec=RowFusionService)

    def _trace(query: str, **_: object) -> RowFusionTrace:
        return RowFusionTrace(
            query=query,
            weights=RowFusionWeights(bm25=1, dense=0),
            candidate_table_ids=(TABLE_ID,),
            bm25_candidate_count=len(results),
            dense_candidate_count=0,
            results=tuple(results),
        )

    fusion.retrieve_rows.side_effect = _trace
    return fusion


def _decisions(per_question: dict[int, tuple[int, ...]]) -> dict[int, ProgramDecision]:
    return {
        question_id: ProgramDecision(
            question_id=question_id,
            cells=cells,
            program="[NUM_0]",
            scale="none",
        )
        for question_id, cells in per_question.items()
    }


_LOOKUP_DECISIONS = {1: (0,)}


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
        row_fusion=_row_fusion(_fused_row(row_idx=0)),
        program_decisions=_decisions(_LOOKUP_DECISIONS),
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


def test_export_submission_evidence_csv_contains_the_full_extracted_table(
    tmp_path: Path,
) -> None:
    """Regression: the packaged evidence CSV must be the real extracted
    table the answer was bound into (every numeric cell of the source
    table(s), via `execution.cell_frame.build_cell_frame`), not a single
    synthesized row for just the one cell the answer used."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    report, items, csv_rows = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        row_fusion=_row_fusion(_fused_row(row_idx=0)),
        program_decisions=_decisions(_LOOKUP_DECISIONS),
    )

    assert report.answered_count == 1
    rows = csv_rows["data/q000001_df1.csv"]
    assert len(rows) == 2
    labels = {row["row_label_raw"] for row in rows}
    assert labels == {"Doanh thu thuan", "Gia von hang ban"}
    assert all(row.get("column_label") == "Năm 2023" for row in rows)


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
        program_decisions={},
    )

    assert report.answered_count == 0
    assert items == ()
    assert csv_rows == {}
    assert report.outcomes[0].status == "abstained"
    assert report.outcomes[0].stage == "retrieval"


def test_export_submission_without_row_candidates_fails_at_execution(tmp_path: Path) -> None:
    """A question whose fusion returns no ranked row has an empty numbered
    candidate list -- the answering path fails with exactly one execution
    code and nothing reaches packaging (`apply_backstop=False` isolates this
    from the backstop tier)."""
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
        row_fusion=_row_fusion(),
        program_decisions={},
    )

    assert report.answered_count == 0
    assert items == ()
    outcome = report.outcomes[0]
    assert outcome.status == "error"
    assert outcome.stage == "execution"
    assert outcome.code == "no_cell_candidates"


def test_a_missing_decision_for_answerable_candidates_is_a_failure(tmp_path: Path) -> None:
    """Every question with candidates needs a decision entry; a missing one
    fails the question (and flows to backstop) instead of silently picking a
    default cell."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=9, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        row_fusion=_row_fusion(_fused_row(row_idx=0)),
        apply_backstop=False,
        program_decisions={},
    )

    outcome = report.outcomes[0]
    assert outcome.status == "error"
    assert outcome.stage == "execution"
    assert outcome.code is not None and outcome.code != ""


def test_duplicate_labels_are_told_apart_by_the_ranked_position(tmp_path: Path) -> None:
    """plan.md §9/§14 carried over: two identically-labelled rows holding
    different figures are distinguished by which row the ranking put first;
    the decision's cell index points into that ranked list."""
    release_dir = _write_release(tmp_path)
    # Overwrite cells: two identically-labelled rows, values 100 and 900.
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

    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")
    report, items, _ = export_submission(
        [question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        # Rank 1 is row_1 (the 900 row): its cell gets index 0.
        row_fusion=_row_fusion(_fused_row(row_idx=1)),
        program_decisions=_decisions({1: (0,)}),
    )

    assert report.answered_count == 1
    assert items[0].answer == 900.0
    # The rendered predicate is semantic first (the canonical label here),
    # and `row_idx` breaks the tie between the two identically-labelled rows.
    assert 'df1.row_label_canonical == "net_revenue"' in items[0].pandas_query
    assert "df1.row_idx == 1" in items[0].pandas_query


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
        program_decisions={},
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
        program_decisions={},
    )

    assert report.backstopped_count == 0
    assert len(items) == 0
    assert report.outcomes[0].status != "answered"


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
        row_fusion=_row_fusion(_fused_row(row_idx=0)),
        program_decisions=_decisions(_LOOKUP_DECISIONS),
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


def test_row_fusion_sees_the_un_narrowed_retrieved_set_once_per_path(tmp_path: Path) -> None:
    """The answering path fuses over exactly the retrieval result (never a
    scope-narrowed subset -- that would shift every numbered cell index
    against the offline payloads), and the backstop hint may fuse again over
    the narrowed set only after the answering path already failed."""
    release_dir = _write_release(tmp_path)
    unsupported_question = RawQuestion(
        id=2, question="Tra cứu chỉ số không tồn tại của ACB năm 2023."
    )

    mock_fusion = MagicMock(spec=RowFusionService)
    mock_fusion.retrieve_rows.return_value = RowFusionTrace(
        query=unsupported_question.question,
        weights=RowFusionWeights(bm25=1, dense=1),
        candidate_table_ids=(TABLE_ID,),
        bm25_candidate_count=0,
        dense_candidate_count=0,
        results=(),
    )

    export_submission(
        [unsupported_question],
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=10,
        row_fusion=mock_fusion,
        program_decisions={},
    )

    # Once on the answering path over the retrieved set, once more inside the
    # backstop's row-hint lookup. Both carry the same default candidate count
    # `submission row-batches --rows-per-question` uses.
    expected_call = call(
        unsupported_question.question,
        candidate_table_ids=(TABLE_ID,),
        k=DEFAULT_ROW_CANDIDATE_COUNT,
    )
    assert mock_fusion.retrieve_rows.call_args_list == [expected_call, expected_call]


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


def test_a_single_cell_table_cannot_stand_as_evidence(tmp_path: Path) -> None:
    """BI-1/Critical 1: a one-row CSV would read `result = df["answer"]
    .iloc[0]` -- the hardcode shape compliance C1+C2 exist to catch. When the
    evidence gate refuses the real-table slice, the question fails execution
    with `evidence_frame_replay_mismatch`; no synthesized replacement row is
    ever packaged."""
    from financial_report_qa.submission import exporter

    assert not hasattr(exporter, "_replay_rows_to_csv_rows"), (
        "_replay_rows_to_csv_rows là nhánh sinh CSV một dòng -- phải bị xoá"
    )

    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")

    calls: list[str] = []

    def _fake_program_rows(executed, release_dir_arg, *, timeout_seconds):
        calls.append("called")
        return None

    original = exporter._program_evidence_rows
    exporter._program_evidence_rows = _fake_program_rows
    try:
        report, items, csv_rows = exporter.export_submission(
            [question],
            _service(),
            release_dir,
            execution_settings=_ALLOW_LOOKUP,
            dataset_fingerprint="0" * 64,
            k=10,
            row_fusion=_row_fusion(_fused_row(row_idx=0)),
            apply_backstop=False,
            program_decisions=_decisions(_LOOKUP_DECISIONS),
        )
    finally:
        exporter._program_evidence_rows = original

    assert calls == ["called"]
    assert report.answered_count == 0
    assert items == ()
    assert csv_rows == {}
    outcome = report.outcomes[0]
    assert outcome.status == "error"
    assert outcome.stage == "execution"
    assert outcome.code == "evidence_frame_replay_mismatch"


def test_export_reports_only_the_two_allowed_plan_sources(tmp_path: Path) -> None:
    """Tiêu chí thành công §12.5: chỉ còn `llm_decision` và `backstop`."""
    release_dir = _write_release(tmp_path)
    questions = [RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")]
    report, _items, _rows = export_submission(
        questions,
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=5,
        row_fusion=_row_fusion(_fused_row(row_idx=0)),
        program_decisions=_decisions(_LOOKUP_DECISIONS),
    )
    assert {outcome.plan_source for outcome in report.outcomes} <= {"llm_decision", "backstop"}


def test_exporter_no_longer_imports_any_deleted_tier() -> None:
    """Ghim N6 ở mức module: các tầng đã bỏ không được quay lại qua import."""
    import financial_report_qa.submission.exporter as exporter_module

    source = Path(exporter_module.__file__).read_text(encoding="utf-8")
    for gone in (
        "plan_router",
        "llm_planner",
        "llm_evidence_planner",
        "evidence_planner",
        "column_refinement",
        "raw_metric_grounding",
        "llm_cell_grounding",
        "rule_planner",
        "ground_question",
        "compile_plan",
        "FinancialQueryPlan",
        "CompiledQuery",
        "load_decisions",
        "build_batch_payload",
    ):
        assert gone not in source, f"{gone} thuộc thang tầng đã bỏ"


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


def test_relevant_tables_preserve_retrieval_rank_order(release_dir: Path) -> None:
    """MRR5 (dashboard column TABLES MRR5) scores rank, not set membership.

    The rank-1 table in the input must be the first element of the output --
    even when it does not come first alphabetically by table_id
    (build_cell_frame's ORDER BY table_id would sort it wrong if used as the
    order source)."""
    import duckdb

    from financial_report_qa.submission.exporter import _relevant_docs_and_tables

    connection = duckdb.connect(":memory:")
    frame = connection.execute(
        "SELECT DISTINCT table_id FROM read_parquet(?) "
        "WHERE value_numeric IS NOT NULL AND period IS NOT NULL "
        "ORDER BY table_id DESC LIMIT 2",
        [str(release_dir / "cells.parquet")],
    ).fetchdf()
    connection.close()
    rank1_table_id, rank2_table_id = frame["table_id"].tolist()
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

    assert tables_rank1_first[0] != tables_rank1_first[1]
    assert tables_rank1_first == (tables_rank2_first[1], tables_rank2_first[0]), (
        "output order must track retrieval-rank input order, not table_id alphabetical order"
    )
