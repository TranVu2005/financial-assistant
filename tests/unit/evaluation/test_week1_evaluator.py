import json
from pathlib import Path

import pytest
from test_week1_dataset import _write_release

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    CELL_AUDIT_COLUMNS,
    EXPECTED_TABLE_COLUMNS,
    GateCheck,
    GateResult,
    percentage_passes,
    write_canonical_json,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.evaluation.week1_evaluator import (
    evaluate_week1_gate,
    publish_gate_artifacts,
)
from financial_report_qa.evaluation.week1_sampling import prepare_pilot


def test_evaluate_rejects_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)
    dataset = load_gate_dataset(manifest_path, release_path)

    annotation_dir = tmp_path / "annotations"
    prepare_pilot(dataset, annotation_dir, company_count=1, documents_per_company=1)

    metadata_path = annotation_dir / "pilot-metadata.json"
    meta_data = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta_data["dataset_fingerprint"] = "0" * 64
    write_canonical_json(metadata_path, meta_data)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    with pytest.raises(Week1GateInputError, match="dataset fingerprint mismatch"):
        evaluate_week1_gate(dataset, corpus_dir, annotation_dir)


def test_evaluate_rejects_source_manifest_fingerprint_mismatch(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)
    dataset = load_gate_dataset(manifest_path, release_path)

    annotation_dir = tmp_path / "annotations"
    prepare_pilot(dataset, annotation_dir, company_count=1, documents_per_company=1)

    metadata_path = annotation_dir / "pilot-metadata.json"
    meta_data = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta_data["source_manifest_sha256"] = "f" * 64
    write_canonical_json(metadata_path, meta_data)

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    with pytest.raises(Week1GateInputError, match="source manifest fingerprint mismatch"):
        evaluate_week1_gate(dataset, corpus_dir, annotation_dir)


@pytest.mark.parametrize(
    ("numerator", "denominator", "threshold", "passed"),
    [
        (85, 100, 85, True),
        (84, 100, 85, False),
        (7, 10, 70, True),
        (6, 10, 70, False),
        (0, 0, 85, False),
    ],
)
def test_percentage_checks_use_integer_arithmetic(
    numerator: int, denominator: int, threshold: int, passed: bool
) -> None:
    assert percentage_passes(numerator, denominator, threshold) is passed


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

    write_csv_rows(annotation_dir / "cell-audit.csv", CELL_AUDIT_COLUMNS, ())

    with pytest.raises(Week1GateInputError, match="Not enough eligible cells"):
        evaluate_week1_gate(dataset, corpus_dir, annotation_dir)


def test_publish_gate_artifacts_emits_exact_output_set(tmp_path: Path) -> None:
    result = GateResult(
        sampling_version="week1-pilot-v1",
        annotation_schema_version="1",
        dataset_fingerprint="f" * 64,
        source_manifest_sha256="a" * 64,
        pilot_documents_sha256="b" * 64,
        expected_tables_sha256="c" * 64,
        cell_audit_sha256="d" * 64,
        evaluation_inputs_sha256="e" * 64,
        document_count=60,
        annotated_table_count=90,
        matched_table_count=90,
        usable_table_count=90,
        checks=(
            GateCheck(
                name="pilot_document_count",
                passed=True,
                numerator=60,
                denominator=60,
                threshold_percent=100,
            ),
        ),
        statement_metrics={},
        stratum_metrics={},
        pareto_rows=(),
        passed=True,
    )
    output_dir = tmp_path / "output"
    publish_gate_artifacts(result, (), output_dir)

    assert sorted(p.name for p in output_dir.iterdir()) == [
        "gate-report.md",
        "gate-result.json",
        "pareto-errors.csv",
    ]


def test_publish_gate_artifacts_rejects_unexpected_existing_files(tmp_path: Path) -> None:
    result = GateResult(
        sampling_version="week1-pilot-v1",
        annotation_schema_version="1",
        dataset_fingerprint="f" * 64,
        source_manifest_sha256="a" * 64,
        pilot_documents_sha256="b" * 64,
        expected_tables_sha256="c" * 64,
        cell_audit_sha256="d" * 64,
        evaluation_inputs_sha256="e" * 64,
        document_count=60,
        annotated_table_count=90,
        matched_table_count=90,
        usable_table_count=90,
        checks=(),
        statement_metrics={},
        stratum_metrics={},
        pareto_rows=(),
        passed=True,
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "evaluation_report.md").write_text("legacy\n", encoding="utf-8")

    with pytest.raises(Week1GateInputError, match="unexpected artifacts"):
        publish_gate_artifacts(result, (), output_dir)
