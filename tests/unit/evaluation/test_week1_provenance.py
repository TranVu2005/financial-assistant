"""Unit tests for cell provenance auditing and table usability evaluation."""

from decimal import Decimal

from financial_report_qa.evaluation.week1_contracts import ExpectedTable, TableAssessment
from financial_report_qa.evaluation.week1_provenance import (
    audit_cell_provenance,
    evaluate_table_usability,
)
from financial_report_qa.schemas import CellRecord, TableRecord, stable_table_id


def test_audit_cell_provenance_success() -> None:
    doc_lines = ("Line 1", "Line 2: Revenue 1000", "Line 3")
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)
    cell = CellRecord(
        cell_id="cell_1",
        table_id=tbl_id,
        row_idx=0,
        col_idx=0,
        row_label_raw="Revenue",
        row_label_canonical="Revenue",
        column_label_raw="2024",
        column_label_canonical="2024",
        value_raw="1000",
        value_numeric=Decimal("1000"),
        period="2024",
        unit="VND",
        source_line_start=2,
        source_line_end=2,
        extraction_confidence=1.0,
    )

    verified, excerpt, failures = audit_cell_provenance(cell, doc_lines)
    assert verified is True
    assert excerpt == "Line 2: Revenue 1000"
    assert len(failures) == 0


def test_audit_cell_provenance_invalid_span() -> None:
    doc_lines = ("Line 1", "Line 2")
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)
    cell = CellRecord(
        cell_id="cell_1",
        table_id=tbl_id,
        row_idx=0,
        col_idx=0,
        row_label_raw="Revenue",
        row_label_canonical="Revenue",
        column_label_raw="2024",
        column_label_canonical="2024",
        value_raw="1000",
        value_numeric=Decimal("1000"),
        period="2024",
        unit="VND",
        source_line_start=1,
        source_line_end=10,  # Out of bounds
        extraction_confidence=1.0,
    )

    verified, excerpt, failures = audit_cell_provenance(cell, doc_lines)
    assert verified is False
    assert failures == ["invalid_provenance"]


def test_evaluate_table_usability_shape_mismatch() -> None:
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)

    extracted_tbl = TableRecord(
        table_id=tbl_id,
        doc_id=doc_id,
        title_raw="Balance Sheet",
        statement_type="balance_sheet",
        unit_raw="VND",
        unit_normalized="VND",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=2,  # Differs from expected column_count=3
        quality_score=1.0,
        csv_path=None,
    )

    exp = ExpectedTable(
        annotation_schema_version="1",
        annotation_id="exp_001",
        doc_id=doc_id,
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

    initial_ta = TableAssessment(
        annotation=exp,
        table_id=tbl_id,
        overlap_numerator=11,
        overlap_denominator=11,
        failures=(),
        usable=True,
    )

    matched = {"exp_001": extracted_tbl}
    final_tas = evaluate_table_usability((initial_ta,), matched, ())

    assert len(final_tas) == 1
    assert final_tas[0].usable is False
    assert len(final_tas[0].failures) == 1
    assert final_tas[0].failures[0].code == "shape_mismatch"
