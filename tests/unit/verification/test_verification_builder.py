"""Tests for the Day 20 `build_answer_package` orchestrator (ADR 0009).

Ties `checks.py` + `templates.py` together the way `execution/compiler.py`
ties `locator`/`operations`/`pandas_query` together for Day 18: one function,
never a guessed or half-verified package.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.verification.builder import build_answer_package

TABLE_ID = "tbl_" + "1" * 64
QUESTION_ID = "retq_" + "a" * 64
CELL_ID = "cell_" + "a" * 64

_CITATION_LOOKUP: dict[str, dict[str, object]] = {
    CELL_ID: {
        "doc_relative_path": "ACB/2023/report.txt",
        "source_line_start": 12,
        "source_line_end": 12,
        "table_title": "Bang can doi ke toan",
    }
}


def _cell(**overrides: object) -> CellMatch:
    defaults: dict[str, object] = {
        "table_id": TABLE_ID,
        "cell_ids": ("cell_" + "a" * 64,),
        "value": Decimal("100"),
        "unit": "VND",
        "period": 2023,
        "period_inferred": False,
    }
    defaults.update(overrides)
    return CellMatch.model_validate(defaults)


def _plan(**overrides: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("ACB",),
        "periods": ("2023",),
        "candidate_table_ids": (TABLE_ID,),
        "metric": MetricSelector(canonical="cash_and_cash_equivalents"),
    }
    defaults.update(overrides)
    return FinancialQueryPlan(**defaults)  # type: ignore[arg-type]


def _compiled(**overrides: object) -> CompiledQuery:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "status": "answered",
        "answer": Decimal("100"),
        "unit": "VND",
        "evidence": (_cell(),),
        "pandas_query": 'df1[(df1.period == 2023)]["value"].iloc[0]',
        "error_code": None,
        "error_message": None,
        "replay_rows": (
            {
                "company_code": "ACB",
                "row_label_canonical": "cash_and_cash_equivalents",
                "row_label_raw": None,
                "period": 2023,
                "value": Decimal("100"),
            },
        ),
    }
    defaults.update(overrides)
    return CompiledQuery.model_validate(defaults)


def test_build_answer_package_verified_on_clean_input() -> None:
    plan = _plan()
    compiled = _compiled()
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "verified"
    assert package.verification_issues == ()
    assert package.answer == Decimal("100")
    assert package.evidence[0].table_id == TABLE_ID
    assert "ACB" in package.answer_text
    assert package.display in package.answer_text


def test_build_answer_package_rejected_when_evidence_outside_retrieval() -> None:
    plan = _plan()
    other_table = "tbl_" + "9" * 64
    compiled = _compiled(evidence=(_cell(table_id=other_table),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "evidence_outside_retrieval" in codes


def test_build_answer_package_rejected_on_recompute_mismatch() -> None:
    plan = _plan()
    compiled = _compiled(answer=Decimal("999"))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "recompute_mismatch" in codes


def test_build_answer_package_carries_period_inferred_warning_but_stays_verified() -> None:
    """Day 20 plan Sec 1.5: 6/30 gold70 answers rely on an inferred period --
    this must warn, not block."""
    plan = _plan()
    compiled = _compiled(evidence=(_cell(period_inferred=True),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "verified"
    assert package.period_inferred is True
    codes = {issue.code for issue in package.verification_issues}
    assert "period_inferred_warning" in codes


def test_build_answer_package_rejected_when_scope_inferred() -> None:
    """Day 21 plan §1.5/ADR 0010 decision B1: unlike a merely-inferred period,
    a `CompiledQuery.scope_inferred=True` result must block, not just warn --
    the compiler resolved a real value conflict using a default the plan
    never stated."""
    plan = _plan()
    compiled = _compiled(scope_inferred=True)
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "scope_inferred" in codes


def test_build_answer_package_can_downgrade_scope_inferred_to_a_warning() -> None:
    """Day 24: under the organizers' scoring (correct / TOTAL questions) a
    scope-guessed answer costs exactly what an abstention costs -- 0 -- so a
    submission run may opt to ship it. Off by default: the blocking behavior
    stays correct for internal quality measurement (ADR 0010 B1)."""
    plan = _plan()
    compiled = _compiled(scope_inferred=True)
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
        allow_inferred_scope=True,
    )
    assert package.verification_status == "verified"
    codes = {issue.code for issue in package.verification_issues}
    assert "scope_inferred" in codes, "the issue must still be recorded, just not blocking"


def test_allow_inferred_scope_does_not_unblock_a_real_correctness_failure() -> None:
    """The escape hatch is scoped to `scope_inferred` only -- a genuine
    correctness block (e.g. recompute mismatch) must still reject."""
    plan = _plan()
    compiled = _compiled(scope_inferred=True, answer=Decimal("999999"))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
        allow_inferred_scope=True,
    )
    assert package.verification_status == "rejected"


def test_build_answer_package_raises_on_non_answered_compiled_query() -> None:
    plan = _plan()
    compiled = CompiledQuery.model_validate(
        {
            "operation": "lookup",
            "status": "error",
            "answer": None,
            "unit": None,
            "evidence": (),
            "pandas_query": "<plan rejected before rendering>",
            "error_code": "metric_not_found",
            "error_message": "no match",
        }
    )
    with pytest.raises(ValueError):
        build_answer_package(
            question_id=QUESTION_ID,
            question="Tra cứu tiền mặt của ACB năm 2023.",
            plan=plan,
            compiled=compiled,
            retrieved_table_ids=frozenset({TABLE_ID}),
        )


def test_build_answer_package_citation_fields_populated_from_evidence() -> None:
    plan = _plan()
    compiled = _compiled()
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup={
            "cell_" + "a" * 64: {
                "doc_relative_path": "ACB/2023/report.txt",
                "source_line_start": 12,
                "source_line_end": 12,
                "table_title": "Bang can doi ke toan",
            }
        },
    )
    citation = package.evidence[0]
    assert citation.doc_relative_path == "ACB/2023/report.txt"
    assert citation.source_line_start == 12


def _write_release_for_fact_checks(tmp_path):
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA

    doc_id = "doc_" + "a" * 64
    release_dir = Path(tmp_path) / "release"
    release_dir.mkdir(exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "doc_id": doc_id,
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
            ],
            schema=DOCUMENT_SCHEMA,
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "table_id": TABLE_ID,
                    "doc_id": doc_id,
                    "source_ordinal": 0,
                    "title_raw": "Bao cao",
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
            ],
            schema=TABLE_SCHEMA,
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [
                {
                    "cell_id": CELL_ID,
                    "table_id": TABLE_ID,
                    "row_idx": 14,
                    "col_idx": 1,
                    "row_label_raw": "Tien mat",
                    "row_label_canonical": "cash_and_cash_equivalents",
                    "row_group_context_raw": None,
                    "column_label_raw": None,
                    "column_label_canonical": None,
                    "value_raw": "100",
                    "value_numeric": Decimal("100"),
                    "period": "2023",
                    "unit": "VND",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "extraction_confidence": 0.9,
                }
            ],
            schema=CELL_SCHEMA,
        ),
        release_dir / "cells.parquet",
    )
    return release_dir


def test_build_answer_package_runs_per_fact_checks_when_release_dir_is_given(tmp_path) -> None:
    """plan.md §15: with a release to re-check against, each fact behind the
    answer must be independently re-located -- not just the whole formula."""
    release_dir = _write_release_for_fact_checks(tmp_path)
    plan = _plan()
    compiled = _compiled(evidence=(_cell(row_index=14),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
        release_dir=release_dir,
    )
    assert package.verification_status == "verified"
    assert package.verification_issues == ()


def test_build_answer_package_rejects_a_fact_that_no_longer_re_locates(tmp_path) -> None:
    release_dir = _write_release_for_fact_checks(tmp_path)
    plan = _plan()
    # `row_index=99` names a position no cell in the release actually holds.
    compiled = _compiled(evidence=(_cell(row_index=99),))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
        release_dir=release_dir,
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert "fact_not_found" in codes


def test_build_answer_package_skips_per_fact_checks_without_a_release_dir() -> None:
    """Backward compatible: existing callers that never pass `release_dir`
    (and evidence with no `row_index` at all, e.g. every pre-§9 fixture)
    see no new behavior."""
    plan = _plan()
    compiled = _compiled()
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu tiền mặt của ACB năm 2023.",
        plan=plan,
        compiled=compiled,
        retrieved_table_ids=frozenset({TABLE_ID}),
        citation_lookup=_CITATION_LOOKUP,
    )
    assert package.verification_status == "verified"
    codes = {issue.code for issue in package.verification_issues}
    assert "fact_not_found" not in codes
    assert "fact_value_mismatch" not in codes
