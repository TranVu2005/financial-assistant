"""Unit tests for table matching logic."""

from financial_report_qa.evaluation.week1_contracts import ExpectedTable
from financial_report_qa.evaluation.week1_matching import (
    assess_table_matching,
    derive_table_match_key,
)
from financial_report_qa.schemas import TableRecord, stable_table_id


def test_derive_table_match_key_deterministic() -> None:
    key1 = derive_table_match_key("doc_1", "balance_sheet", 10, 20)
    key2 = derive_table_match_key("doc_1", "balance_sheet", 10, 20)
    assert key1 == key2
    assert key1.startswith("tblmatch_")


def test_assess_table_matching_exact_and_missing() -> None:
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
        column_count=3,
        quality_score=1.0,
        csv_path=None,
    )

    exp1 = ExpectedTable(
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

    exp2 = ExpectedTable(
        annotation_schema_version="1",
        annotation_id="exp_002",
        doc_id=doc_id,
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type="income_statement",
        line_start=50,
        line_end=70,
        row_count=10,
        column_count=3,
        unit_normalized="VND",
        expected_periods=("2024",),
        notes="",
    )

    assessments, matched = assess_table_matching((exp1, exp2), (extracted_tbl,))

    assert len(assessments) == 2
    assert assessments[0].annotation.annotation_id == "exp_001"
    assert assessments[0].table_id == tbl_id
    assert assessments[0].usable is True

    assert assessments[1].annotation.annotation_id == "exp_002"
    assert assessments[1].table_id is None
    assert assessments[1].usable is False

    assert matched == {"exp_001": extracted_tbl}
