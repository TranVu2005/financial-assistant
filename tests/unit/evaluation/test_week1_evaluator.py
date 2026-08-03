"""Unit tests for gate evaluation and publication workflow."""

from pathlib import Path

from test_week1_dataset import _write_release

from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.evaluation.week1_evaluator import (
    evaluate_week1_gate,
    publish_gate_artifacts,
)
from financial_report_qa.evaluation.week1_sampling import prepare_pilot


def test_evaluate_and_publish_week1_gate(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)
    dataset = load_gate_dataset(manifest_path, release_path)

    annotation_dir = tmp_path / "annotations"
    prepare_pilot(dataset, annotation_dir, company_count=1, documents_per_company=1)

    # Populate an expected table matching extracted table
    exp_table_row = {
        "annotation_schema_version": "1",
        "annotation_id": "exp_001",
        "doc_id": document.doc_id,
        "relative_path": document.relative_path,
        "statement_type": "balance_sheet",
        "line_start": "10",
        "line_end": "20",
        "row_count": "2",
        "column_count": "2",
        "unit_normalized": "VND",
        "expected_periods": "2024",
        "notes": "Test match",
    }
    write_csv_rows(
        annotation_dir / "expected-tables.csv",
        EXPECTED_TABLE_COLUMNS,
        (exp_table_row,),
        allow_identical=True,
    )

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc_file = corpus_dir / document.relative_path
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("Header\n" + "Asset 100\n" * 15, encoding="utf-8")

    result, assessments, cell_audits = evaluate_week1_gate(dataset, corpus_dir, annotation_dir)

    assert result.document_count == 1
    assert result.annotated_table_count == 1
    assert result.matched_table_count == 1
    assert result.usable_table_count == 1
    assert result.passed is True

    output_dir = tmp_path / "output"
    publish_gate_artifacts(result, cell_audits, output_dir)

    assert (output_dir / "gate-result.json").is_file()
    assert (output_dir / "cell-audit.csv").is_file()
    assert (output_dir / "pareto-errors.csv").is_file()
    assert (output_dir / "evaluation_report.md").is_file()
