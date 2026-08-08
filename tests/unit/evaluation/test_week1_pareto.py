"""Unit tests for Pareto error distribution computation."""

from financial_report_qa.evaluation.week1_contracts import (
    ExpectedTable,
    FailureEvent,
    TableAssessment,
    stable_annotation_id,
)
from financial_report_qa.evaluation.week1_pareto import compute_pareto_analysis


def test_compute_pareto_analysis_empty() -> None:
    assert compute_pareto_analysis(()) == ()


def test_compute_pareto_analysis_distribution() -> None:
    ann_id = stable_annotation_id("doc_1", 10, 20, "balance_sheet")
    exp = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id="doc_1",
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type="balance_sheet",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=3,
        unit_normalized="VND",
        expected_periods=("2024",),
        notes="",
    )

    ta1 = TableAssessment(
        annotation=exp,
        table_id=None,
        overlap_numerator=0,
        overlap_denominator=10,
        failures=(
            FailureEvent(code="missing_table", doc_id="doc_1"),
            FailureEvent(code="missing_table", doc_id="doc_1"),
        ),
        usable=False,
    )

    ta2 = TableAssessment(
        annotation=exp,
        table_id="tbl_1",
        overlap_numerator=5,
        overlap_denominator=10,
        failures=(FailureEvent(code="shape_mismatch", doc_id="doc_1"),),
        usable=False,
    )

    rows = compute_pareto_analysis((ta1, ta2))
    assert len(rows) == 2
    assert rows[0].rank == 1
    assert rows[0].code == "missing_table"
    assert rows[0].count == 2
    assert rows[0].share == "66.67%"
    assert rows[0].cumulative_share == "66.67%"

    assert rows[1].rank == 2
    assert rows[1].code == "shape_mismatch"
    assert rows[1].count == 1
    assert rows[1].share == "33.33%"
    assert rows[1].cumulative_share == "100.00%"
