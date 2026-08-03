"""Unit tests for cell provenance auditing and table usability evaluation."""

from decimal import Decimal
from pathlib import Path

from test_week1_dataset import _write_release

from financial_report_qa.evaluation.week1_contracts import ExpectedTable, TableAssessment
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.evaluation.week1_provenance import (
    audit_cell_provenance,
    evaluate_table_usability,
    generate_cell_audits,
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


def test_generate_cell_audits_rejects_noncanonical_cell_id(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)
    dataset = load_gate_dataset(manifest_path, release_path)
    dataset.cells_by_table_id[table.table_id] = (cell.model_copy(update={"cell_id": "bad"}),)
    corpus_dir = tmp_path / "corpus"
    doc_path = corpus_dir / document.relative_path
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text("Header\n" + "Asset 100\n" * 15, encoding="utf-8")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id="ann_001",
        doc_id=document.doc_id,
        relative_path=document.relative_path,
        statement_type="balance_sheet",
        line_start=table.line_start,
        line_end=table.line_end,
        row_count=table.row_count,
        column_count=table.column_count,
        unit_normalized="VND",
        expected_periods=("2024",),
    )

    audits = generate_cell_audits(dataset, corpus_dir, (expected,), {"ann_001": table})

    assert audits[0].verified is False


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


def test_sample_is_deterministic_stratified_and_table_capped() -> None:
    from collections import Counter

    from financial_report_qa.evaluation.week1_contracts import SAMPLING_VERSION, CellAudit
    from financial_report_qa.evaluation.week1_sampling import select_audit_cells

    doc_id = "doc_" + "a" * 64
    tbl_1 = stable_table_id(doc_id, 10, 20)
    tbl_2 = stable_table_id(doc_id, 30, 40)

    candidates: list[CellAudit] = []
    for i in range(20):
        tbl = tbl_1 if i < 10 else tbl_2
        candidates.append(
            CellAudit(
                annotation_schema_version="1",
                sampling_version=SAMPLING_VERSION,
                cell_id=f"cell_{i}",
                doc_id=doc_id,
                relative_path="VCB/2024/report.txt",
                company_code="VCB",
                report_year=2024,
                annotation_id="exp_1",
                statement_type="balance_sheet",
                table_id=tbl,
                row_idx=i % 5,
                col_idx=i // 5,
                row_label_raw="Label",
                column_label_raw="Col",
                value_raw="100",
                value_numeric=100.0,
                period="2024",
                unit="VND",
                source_line_start=1,
                source_line_end=2,
                source_excerpt="Excerpt",
                verified=True,
                review_notes="",
            )
        )

    cand_tuple = tuple(candidates)
    selected = select_audit_cells(cand_tuple, sample_size=4, max_per_table=2)
    reversed_selected = select_audit_cells(
        tuple(reversed(cand_tuple)), sample_size=4, max_per_table=2
    )

    assert selected == reversed_selected
    assert len({item.cell_id for item in selected}) == 4
    assert max(Counter(item.table_id for item in selected).values()) <= 2
