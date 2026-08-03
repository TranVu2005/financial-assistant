"""Evaluator orchestrator and report publication for Week 1 Quality Gate."""

import hashlib
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    ANNOTATION_SCHEMA_VERSION,
    CELL_AUDIT_COLUMNS,
    EXPECTED_TABLE_COLUMNS,
    PARETO_CSV_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    SAMPLING_VERSION,
    CellAudit,
    ExpectedTable,
    GateCheck,
    GateResult,
    PilotDocument,
    PilotMetadata,
    TableAssessment,
    read_csv_rows,
    write_canonical_json,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset
from financial_report_qa.evaluation.week1_matching import assess_table_matching
from financial_report_qa.evaluation.week1_pareto import compute_pareto_analysis
from financial_report_qa.evaluation.week1_provenance import (
    evaluate_table_usability,
    generate_cell_audits,
)


def evaluate_week1_gate(
    dataset: GateDataset,
    corpus_dir: Path,
    annotation_dir: Path,
) -> tuple[GateResult, tuple[TableAssessment, ...], tuple[CellAudit, ...]]:
    """Run full Week 1 quality gate evaluation against canonical dataset and annotations."""
    metadata_path = annotation_dir / "pilot-metadata.json"
    if not metadata_path.is_file():
        raise Week1GateInputError(f"Missing pilot-metadata.json in {annotation_dir}")

    meta = PilotMetadata.model_validate_json(metadata_path.read_bytes())
    if meta.sampling_version != SAMPLING_VERSION:
        raise Week1GateInputError(f"Unsupported sampling version: {meta.sampling_version}")
    if meta.annotation_schema_version != ANNOTATION_SCHEMA_VERSION:
        raise Week1GateInputError(f"Unsupported schema version: {meta.annotation_schema_version}")

    docs_csv_path = annotation_dir / "pilot-documents.csv"
    doc_rows = read_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS)
    docs_sha256 = hashlib.sha256(docs_csv_path.read_bytes()).hexdigest()
    if docs_sha256 != meta.pilot_documents_sha256:
        raise Week1GateInputError("pilot-documents.csv content hash mismatch with metadata")

    pilot_docs = tuple(PilotDocument.model_validate(row) for row in doc_rows)

    exp_csv_path = annotation_dir / "expected-tables.csv"
    exp_rows = read_csv_rows(exp_csv_path, EXPECTED_TABLE_COLUMNS)
    expected_tables_sha256 = hashlib.sha256(exp_csv_path.read_bytes()).hexdigest()

    expected_tables: list[ExpectedTable] = []
    for r in exp_rows:
        periods_tuple = tuple(p.strip() for p in r["expected_periods"].split(";") if p.strip())
        expected_tables.append(
            ExpectedTable(
                annotation_schema_version="1",
                annotation_id=r["annotation_id"],
                doc_id=r["doc_id"],
                relative_path=r["relative_path"],
                statement_type=r["statement_type"],  # type: ignore[arg-type]
                line_start=int(r["line_start"]),
                line_end=int(r["line_end"]),
                row_count=int(r["row_count"]),
                column_count=int(r["column_count"]),
                unit_normalized=r["unit_normalized"],
                expected_periods=periods_tuple,
                notes=r.get("notes", ""),
            )
        )

    expected_tables_tuple = tuple(expected_tables)

    # Filter dataset extracted tables to pilot documents
    pilot_doc_ids = {doc.doc_id for doc in pilot_docs}
    pilot_extracted_tables = tuple(
        tbl for tbl in dataset.tables_by_id.values() if tbl.doc_id in pilot_doc_ids
    )

    # Matching phase
    initial_assessments, matched_tables = assess_table_matching(
        expected_tables_tuple, pilot_extracted_tables
    )

    # Provenance auditing phase
    cell_audits = generate_cell_audits(dataset, corpus_dir, expected_tables_tuple, matched_tables)

    # Final usability evaluation
    final_assessments = evaluate_table_usability(initial_assessments, matched_tables, cell_audits)

    # Metrics & Gate Checks calculation
    total_exp = len(expected_tables_tuple)
    total_matched = len(matched_tables)
    total_usable = sum(1 for ta in final_assessments if ta.usable)

    matched_pct = int(
        (
            Decimal(total_matched) * Decimal(100) / Decimal(total_exp if total_exp > 0 else 1)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
    usable_pct = int(
        (
            Decimal(total_usable) * Decimal(100) / Decimal(total_exp if total_exp > 0 else 1)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )

    cell_audit_sha256 = hashlib.sha256(
        "\n".join(ca.cell_id for ca in cell_audits).encode("utf-8")
    ).hexdigest()

    # Provenance check
    prov_failures = sum(
        1 for ta in final_assessments for f in ta.failures if f.code == "invalid_provenance"
    )

    checks = (
        GateCheck(
            name="table_matching_rate",
            passed=matched_pct >= 90,
            numerator=total_matched,
            denominator=total_exp,
            threshold_percent=90,
        ),
        GateCheck(
            name="table_usability_rate",
            passed=usable_pct >= 85,
            numerator=total_usable,
            denominator=total_exp,
            threshold_percent=85,
        ),
        GateCheck(
            name="provenance_validity_rate",
            passed=prov_failures == 0,
            numerator=total_exp - prov_failures,
            denominator=total_exp,
            threshold_percent=100,
        ),
    )

    gate_passed = all(c.passed for c in checks)
    pareto_rows = compute_pareto_analysis(final_assessments)

    # Statement & stratum breakdowns
    statement_metrics: dict[str, dict[str, Any]] = {}
    for st in ("balance_sheet", "income_statement", "cash_flow_statement"):
        st_exp = [ta for ta in final_assessments if ta.annotation.statement_type == st]
        st_matched = [ta for ta in st_exp if ta.table_id is not None]
        st_usable = [ta for ta in st_exp if ta.usable]
        statement_metrics[st] = {
            "expected_count": len(st_exp),
            "matched_count": len(st_matched),
            "usable_count": len(st_usable),
        }

    stratum_metrics: dict[str, dict[str, Any]] = {}

    result = GateResult(
        sampling_version=SAMPLING_VERSION,
        annotation_schema_version="1",
        dataset_fingerprint=dataset.dataset_fingerprint,
        source_manifest_sha256=dataset.source_manifest_sha256,
        pilot_documents_sha256=docs_sha256,
        expected_tables_sha256=expected_tables_sha256,
        cell_audit_sha256=cell_audit_sha256,
        document_count=len(pilot_docs),
        annotated_table_count=total_exp,
        matched_table_count=total_matched,
        usable_table_count=total_usable,
        checks=checks,
        statement_metrics=statement_metrics,
        stratum_metrics=stratum_metrics,
        pareto_rows=pareto_rows,
        passed=gate_passed,
    )

    return result, final_assessments, cell_audits


def publish_gate_artifacts(
    result: GateResult,
    cell_audits: tuple[CellAudit, ...],
    output_dir: Path,
) -> None:
    """Publish canonical gate evaluation reports and audit artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write gate-result.json
    write_canonical_json(output_dir / "gate-result.json", result.model_dump(mode="json"))

    # Write cell-audit.csv
    audit_rows = [ca.model_dump(mode="json") for ca in cell_audits]
    write_csv_rows(output_dir / "cell-audit.csv", CELL_AUDIT_COLUMNS, audit_rows)

    # Write pareto-errors.csv
    pareto_rows = [p.model_dump(mode="json") for p in result.pareto_rows]
    write_csv_rows(output_dir / "pareto-errors.csv", PARETO_CSV_COLUMNS, pareto_rows)

    # Write human-readable markdown summary
    md_content = f"""# Week 1 Quality Gate Evaluation Report

- **Status:** {"PASSED" if result.passed else "FAILED"}
- **Sampling Version:** {result.sampling_version}
- **Dataset Fingerprint:** `{result.dataset_fingerprint}`
- **Pilot Documents:** {result.document_count}
- **Annotated Tables:** {result.annotated_table_count}
- **Matched Tables:** {result.matched_table_count}
- **Usable Tables:** {result.usable_table_count}

## Quality Gate Checks
"""
    for check in result.checks:
        status_str = "PASS" if check.passed else "FAIL"
        md_content += (
            f"- **{check.name}:** {status_str} "
            f"({check.numerator}/{check.denominator}, "
            f"threshold >= {check.threshold_percent}%)\n"
        )

    md_content += "\n## Pareto Error Analysis\n"
    if not result.pareto_rows:
        md_content += "No failure errors recorded.\n"
    else:
        md_content += (
            "| Rank | Code | Count | Share | Cumulative Share |\n| --- | --- | --- | --- | --- |\n"
        )
        for p in result.pareto_rows:
            md_content += (
                f"| {p.rank} | {p.code} | {p.count} | {p.share} | {p.cumulative_share} |\n"
            )

    (output_dir / "evaluation_report.md").write_text(md_content, encoding="utf-8")
