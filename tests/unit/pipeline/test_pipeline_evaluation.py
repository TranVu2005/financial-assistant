"""Tests for the Day 21 E2E pipeline evaluation (ADR 0010 decision E1).

Unlike `verification/evaluation.py::evaluate_answer_packages_on_gold` (Day
20), which feeds `gold_table_ids` into both plan construction and
`retrieved_table_ids`, these tests always pass a SEPARATE `rankings` mapping
as the only source of candidate tables -- exercising the real retrieval seam
that Day 20's harness bypassed (Day 21 plan §1.1).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.pipeline.evaluation import evaluate_scope_policies, run_e2e_pipeline
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, GoldTableEvidence

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_SEPARATE = "tbl_" + "2" * 64
DOC_ID = "doc_" + "a" * 64
DOC_ID_SEPARATE = "doc_" + "b" * 64
CELL_ID = "cell_" + "a" * 64
CELL_ID_SEPARATE = "cell_" + "b" * 64

_ALLOW_LOOKUP = ExecutionSettings(timeout_seconds=5, max_rows=20000, allow_operations=("lookup",))


def _document(doc_id: str, *, scope: str = "consolidated") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": "ACB/2023/report.txt",
        "company_code": "ACB",
        "report_year": 2023,
        "statement_scope": scope,
        "sha256": "0" * 64,
        "file_size_bytes": 10,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }


def _table(table_id: str, doc_id: str) -> dict[str, object]:
    return {
        "table_id": table_id,
        "doc_id": doc_id,
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


def _cell(cell_id: str, table_id: str, *, value: str) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": 0,
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


def _write_release(
    tmp_path: Path,
    documents: list[dict[str, object]],
    tables: list[dict[str, object]],
    cells: list[dict[str, object]],
) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
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


def _question(
    question_id: str, question: str, *, gold_table_ids: tuple[str, ...]
) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion(
        question_id=question_id,
        question=question,
        intent="lookup",
        filters={},
        gold_table_ids=gold_table_ids,
        reviewed_by="tester",
        reviewed_at="2026-08-15T00:00:00+00:00",
        gold_evidence=(
            GoldTableEvidence(
                table_id=gold_table_ids[0],
                relative_path="ACB/2023/report.txt",
                line_start=5,
                line_end=5,
                verified=True,
            ),
        ),
        dataset_fingerprint="0" * 64,
    )


def test_pipeline_succeeds_and_scores_against_gold_when_retrieval_finds_the_table(
    tmp_path: Path,
) -> None:
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "1" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
        answer_gold={qid: Decimal("100")},
    )
    assert report.question_count == 1
    assert report.verified_count == 1
    assert report.results[0].stage is None
    assert report.results[0].answer == Decimal("100")
    assert report.results[0].gold_in_retrieved is True
    assert report.scored_against_gold_count == 1
    assert report.correct_count == 1
    assert report.overconfident_wrong_count == 0
    assert report.accuracy_against_gold == 1.0


def test_pipeline_records_no_candidate_tables_as_retrieval_stage(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "2" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {},  # retrieval found nothing for this question
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
    )
    result = report.results[0]
    assert result.stage == "retrieval"
    assert result.code == "no_candidate_tables"
    assert result.gold_in_retrieved is False


def test_pipeline_records_abstain_as_planning_stage(tmp_path: Path) -> None:
    """Day 21 plan §1.9: abstains must be a first-class result, not swallowed
    (the Day 20 harness's `continue` hid 19/70 gold70 abstains)."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "3" * 64
    # No company named -> entity_parser flags company_missing -> planner abstains.
    question = _question(qid, "Doanh thu thuần năm 2023 là bao nhiêu?", gold_table_ids=(TABLE_ID,))
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
    )
    result = report.results[0]
    assert result.stage == "planning"
    assert result.code == "entity_ambiguous"
    assert report.stage_counts.get("planning") == 1


def test_pipeline_grounds_raw_metric_when_rule_planner_would_abstain(tmp_path: Path) -> None:
    """Day 23 plan Step 1: `run_e2e_pipeline` must apply the same raw-metric
    grounding fallback as the submission exporter, so gold accuracy measured
    here reflects the fallback's real effect, not a stale rule-only path."""
    cell = _cell(CELL_ID, TABLE_ID, value="70")
    cell["row_label_raw"] = "Lãi tiền gửi"
    cell["row_label_canonical"] = None
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [cell],
    )
    qid = "retq_" + "9" * 64
    question = _question(
        qid, "Lãi tiền gửi của ACB năm 2023 là bao nhiêu?", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
        answer_gold={qid: Decimal("70")},
    )
    result = report.results[0]
    assert result.stage is None
    assert result.answer == Decimal("70")
    assert report.correct_count == 1


def test_pipeline_classifies_cell_ambiguous_as_planning_when_gold_is_retrieved(
    tmp_path: Path,
) -> None:
    """Day 21 plan §1.2/ADR 0010 decision E1: 22/24 real `cell_ambiguous`
    failures had gold IN the candidate set -- classifying this as a
    retrieval failure would be wrong. It must be `planning` (the plan is
    missing the statement_scope dimension that would disambiguate)."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID, scope="consolidated"), _document(DOC_ID_SEPARATE, scope="separate")],
        [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_SEPARATE, DOC_ID_SEPARATE)],
        [
            _cell(CELL_ID, TABLE_ID, value="100"),
            _cell(CELL_ID_SEPARATE, TABLE_ID_SEPARATE, value="200"),
        ],
    )
    qid = "retq_" + "4" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID, TABLE_ID_SEPARATE)},  # gold (TABLE_ID) IS in the retrieved set
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
    )
    result = report.results[0]
    assert result.gold_in_retrieved is True
    assert result.code == "cell_ambiguous"
    assert result.stage == "planning"


def test_pipeline_classifies_cell_ambiguous_as_retrieval_when_gold_is_missing(
    tmp_path: Path,
) -> None:
    """Mirror of the previous test: when gold is NOT in the retrieved set,
    the same `cell_ambiguous` code means retrieval, not planning."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID, scope="consolidated"), _document(DOC_ID_SEPARATE, scope="separate")],
        [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_SEPARATE, DOC_ID_SEPARATE)],
        [
            _cell(CELL_ID, TABLE_ID, value="100"),
            _cell(CELL_ID_SEPARATE, TABLE_ID_SEPARATE, value="200"),
        ],
    )
    qid = "retq_" + "5" * 64
    other_table = "tbl_" + "9" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(other_table,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID, TABLE_ID_SEPARATE)},  # gold NOT in the retrieved set
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
    )
    result = report.results[0]
    assert result.gold_in_retrieved is False
    assert result.code == "cell_ambiguous"
    assert result.stage == "retrieval"


def test_pipeline_answered_but_wrong_is_overconfident_wrong(tmp_path: Path) -> None:
    """A `verified` package with the WRONG answer relative to hand-labeled
    gold is exactly the "confident but wrong" case the project's hard
    constraint (< 5%) is about -- must be counted, not silently averaged
    away."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "6" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
        answer_gold={qid: Decimal("999")},  # deliberately wrong vs the computed 100
    )
    assert report.results[0].stage is None
    assert report.scored_against_gold_count == 1
    assert report.correct_count == 0
    assert report.overconfident_wrong_count == 1


def test_pipeline_report_records_rankings_provenance(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "7" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="artifacts/evaluations/day14/v2/retrieval-v2-bm25-v4-422df141c935.json",
        rankings_sha256="cafebabe",
    )
    assert report.rankings_source.endswith("bm25-v4-422df141c935.json")
    assert report.rankings_sha256 == "cafebabe"
    assert report.dataset_fingerprint == "0" * 64


def test_pipeline_abstain_when_scope_unstated_flag_blocks_unfiltered_answer(
    tmp_path: Path,
) -> None:
    """Day 21 plan §1.5 policy sweep: `abstain_when_scope_unstated=True` must
    refuse to compile a plan whose `statement_scope` is None, rather than
    falling through to the unfiltered ambiguous-candidate behavior."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID, scope="consolidated"), _document(DOC_ID_SEPARATE, scope="separate")],
        [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_SEPARATE, DOC_ID_SEPARATE)],
        [
            _cell(CELL_ID, TABLE_ID, value="100"),
            _cell(CELL_ID_SEPARATE, TABLE_ID_SEPARATE, value="200"),
        ],
    )
    qid = "retq_" + "8" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID, TABLE_ID_SEPARATE)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
        abstain_when_scope_unstated=True,
    )
    result = report.results[0]
    assert result.stage == "planning"
    assert result.code == "scope_unstated"


def test_pipeline_abstain_when_scope_unstated_flag_defaults_to_false(tmp_path: Path) -> None:
    """Existing callers (e.g. the base `run-e2e` CLI path) must be
    unaffected -- the flag is opt-in for the policy-comparison report."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID)],
        [_table(TABLE_ID, DOC_ID)],
        [_cell(CELL_ID, TABLE_ID, value="100")],
    )
    qid = "retq_" + "9" * 64
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = run_e2e_pipeline(
        [question],
        {qid: (TABLE_ID,)},
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
    )
    assert report.results[0].stage is None


def test_evaluate_scope_policies_reports_all_three_and_matches_measured_tradeoff(
    tmp_path: Path,
) -> None:
    """Day 21 plan §1.5: reproduces the exact tradeoff shape measured by hand
    -- `default_consolidated` must not simply dominate `none` (it trades
    accuracy for coverage), and `abstain_when_unstated` must never answer a
    question whose scope was never determined."""
    release_dir = _write_release(
        tmp_path,
        [_document(DOC_ID, scope="consolidated"), _document(DOC_ID_SEPARATE, scope="separate")],
        [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_SEPARATE, DOC_ID_SEPARATE)],
        [
            _cell(CELL_ID, TABLE_ID, value="100"),
            _cell(CELL_ID_SEPARATE, TABLE_ID_SEPARATE, value="200"),
        ],
    )
    qid = "retq_" + "a" * 64
    # Scope-unstated question with two real candidate values (100
    # consolidated, 200 separate) -- exactly the shape that forces the three
    # policies to disagree.
    question = _question(
        qid, "Tra cứu doanh thu thuần của ACB năm 2023.", gold_table_ids=(TABLE_ID,)
    )
    report = evaluate_scope_policies(
        [question],
        {qid: (TABLE_ID, TABLE_ID_SEPARATE)},
        release_dir,
        base_execution_settings=_ALLOW_LOOKUP,
        rankings_source="test-ranking.json",
        rankings_sha256="deadbeef",
        answer_gold={qid: Decimal("100")},
    )
    by_policy = {result.policy: result for result in report.policies}
    assert set(by_policy) == {"none", "default_consolidated", "abstain_when_unstated"}

    # none: unfiltered candidates conflict -> not verified.
    assert by_policy["none"].answered_count == 0

    # default_consolidated: infers consolidated, answers, and happens to
    # match gold here.
    assert by_policy["default_consolidated"].answered_count == 1
    assert by_policy["default_consolidated"].correct_count == 1

    # abstain_when_unstated: never answers an unstated-scope question.
    assert by_policy["abstain_when_unstated"].answered_count == 0
