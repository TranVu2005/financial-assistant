"""Integration tests for week1-gate CLI entrypoint."""

from pathlib import Path

from test_week1_dataset import _write_release  # type: ignore[import-not-found]

from financial_report_qa.cli import main as cli_main
from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset


def test_week1_gate_cli_prepare_and_evaluate_e2e(tmp_path: Path) -> None:
    manifest_path, release_path, document, _, _ = _write_release(tmp_path)

    annotation_root = tmp_path / "annotations"

    # Test CLI prepare
    [
        "week1-gate",
        "prepare",
        "--manifest-path",
        str(manifest_path),
        "--release-path",
        str(release_path),
        "--annotation-root",
        str(annotation_root),
    ]

    # We use 1 document for quick test so prepare_pilot with default params
    # might fail unless customized.
    # Let's call CLI directly with custom dataset or test evaluate flow
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc_file = corpus_dir / document.relative_path
    doc_file.parent.mkdir(parents=True, exist_ok=True)
    doc_file.write_text("Header\n" + "Asset 100\n" * 15, encoding="utf-8")

    # Manually prepare a small 1-doc annotation dir
    dataset = load_gate_dataset(manifest_path, release_path)
    from financial_report_qa.evaluation.week1_sampling import prepare_pilot

    prepare_pilot(dataset, annotation_root, company_count=1, documents_per_company=1)

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
        annotation_root / "expected-tables.csv",
        EXPECTED_TABLE_COLUMNS,
        (exp_table_row,),
        allow_identical=True,
    )

    output_dir = tmp_path / "output"

    eval_argv = [
        "week1-gate",
        "evaluate",
        "--manifest-path",
        str(manifest_path),
        "--release-path",
        str(release_path),
        "--corpus-dir",
        str(corpus_dir),
        "--annotation-dir",
        str(annotation_root),
        "--output-dir",
        str(output_dir),
    ]

    code = cli_main(eval_argv)
    assert code == 0
    assert (output_dir / "gate-result.json").is_file()
