"""Unit tests for table matching logic."""

from financial_report_qa.evaluation.week1_contracts import ExpectedTable, stable_annotation_id
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

    exp1_id = stable_annotation_id(doc_id, 10, 20, "balance_sheet")
    exp1 = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=exp1_id,
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

    exp2_id = stable_annotation_id(doc_id, 50, 70, "income_statement")
    exp2 = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=exp2_id,
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
    assert assessments[0].annotation.annotation_id == exp1_id
    assert assessments[0].table_id == tbl_id
    assert assessments[0].usable is True

    assert assessments[1].annotation.annotation_id == exp2_id
    assert assessments[1].table_id is None
    assert assessments[1].usable is False

    assert matched == {exp1_id: extracted_tbl}


def _table(
    table_id: str,
    doc_id: str,
    line_start: int,
    line_end: int,
    statement_type: str = "balance_sheet",
) -> TableRecord:
    return TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw="Table",
        statement_type=statement_type,
        unit_raw="VND",
        unit_normalized="VND",
        line_start=line_start,
        line_end=line_end,
        row_count=5,
        column_count=3,
        quality_score=1.0,
        csv_path=None,
    )


def _annotation(
    annotation_id: str,
    doc_id: str,
    line_start: int,
    line_end: int,
    statement_type: str = "balance_sheet",
) -> ExpectedTable:
    ann_id = stable_annotation_id(doc_id, line_start, line_end, statement_type)  # type: ignore[arg-type]
    return ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id=doc_id,
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type=statement_type,  # type: ignore[arg-type]
        line_start=line_start,
        line_end=line_end,
        row_count=5,
        column_count=3,
        unit_normalized="VND",
        expected_periods=("2024",),
        notes="",
    )


def test_matcher_prefers_best_span_even_when_candidates_overlap() -> None:
    doc_id = "doc_" + "a" * 64
    ann = _annotation("ann_1", doc_id, 10, 19)
    weaker = _table(stable_table_id(doc_id, 10, 17), doc_id, 10, 17)
    exact = _table(stable_table_id(doc_id, 10, 19), doc_id, 10, 19)
    assessments, matched = assess_table_matching((ann,), (weaker, exact))
    assert assessments[0].table_id == exact.table_id


def test_matcher_never_reuses_one_observed_table() -> None:
    doc_id = "doc_" + "a" * 64
    ann1 = _annotation("ann_a", doc_id, 10, 20)
    ann2 = _annotation("ann_b", doc_id, 12, 22)
    tbl_only = _table(stable_table_id(doc_id, 10, 20), doc_id, 10, 20)
    assessments, matched = assess_table_matching((ann1, ann2), (tbl_only,))
    assigned_count = sum(1 for a in assessments if a.table_id is not None)
    assert assigned_count == 1
    assert len(matched) == 1


def test_matcher_uses_global_optimum_not_greedy_pair_order() -> None:
    doc_id = "doc_" + "a" * 64
    ann_a = _annotation("ann_a", doc_id, 10, 19)
    ann_b = _annotation("ann_b", doc_id, 15, 24)
    tbl_a = _table(stable_table_id(doc_id, 10, 19), doc_id, 10, 19)
    tbl_b = _table(stable_table_id(doc_id, 15, 24), doc_id, 15, 24)

    assessments, _ = assess_table_matching((ann_a, ann_b), (tbl_b, tbl_a))

    assert {a.annotation.annotation_id: a.table_id for a in assessments} == {
        ann_a.annotation_id: tbl_a.table_id,
        ann_b.annotation_id: tbl_b.table_id,
    }


def test_matcher_accepts_partial_span_overlap_at_eighty_percent() -> None:
    doc_id = "doc_" + "a" * 64
    ann = _annotation("ann_1", doc_id, 10, 19)
    tbl = _table(stable_table_id(doc_id, 10, 17), doc_id, 10, 17)

    assessments, _ = assess_table_matching((ann,), (tbl,))

    assert assessments[0].table_id == tbl.table_id
    assert assessments[0].overlap_numerator == 8
    assert assessments[0].overlap_denominator == 10
    assert assessments[0].usable is True
    assert assessments[0].failures == ()
