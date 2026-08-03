"""Evaluator orchestrator and report publication for Week 1 Quality Gate."""

import hashlib
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
    percentage_passes,
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
from financial_report_qa.evaluation.week1_sampling import select_audit_cells


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

    if meta.dataset_fingerprint != dataset.dataset_fingerprint:
        raise Week1GateInputError("dataset fingerprint mismatch")
    if meta.source_manifest_sha256 != dataset.source_manifest_sha256:
        raise Week1GateInputError("source manifest fingerprint mismatch")

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

    # Automated provenance auditing phase & deterministic cell sampling
    all_cell_audits = generate_cell_audits(
        dataset, corpus_dir, expected_tables_tuple, matched_tables
    )
    expected_sample = select_audit_cells(all_cell_audits, sample_size=30)

    cell_audit_csv_path = annotation_dir / "cell-audit.csv"
    if not cell_audit_csv_path.is_file():
        raise Week1GateInputError(f"Missing cell-audit.csv in {annotation_dir}")

    audit_rows = read_csv_rows(cell_audit_csv_path, CELL_AUDIT_COLUMNS)
    if len(audit_rows) != len(expected_sample):
        raise Week1GateInputError(
            f"cell-audit.csv row count mismatch: found {len(audit_rows)}, "
            f"expected {len(expected_sample)}"
        )

    expected_sample_by_id = {ca.cell_id: ca for ca in expected_sample}
    user_audits: list[CellAudit] = []
    seen_cell_ids: set[str] = set()

    for r in audit_rows:
        cell_id = r["cell_id"]
        if cell_id in seen_cell_ids:
            raise Week1GateInputError(f"Duplicate cell_id '{cell_id}' in cell-audit.csv")
        seen_cell_ids.add(cell_id)
        if cell_id not in expected_sample_by_id:
            raise Week1GateInputError(f"Unexpected cell_id '{cell_id}' in cell-audit.csv")

        exp_ca = expected_sample_by_id[cell_id]
        expected_row = {
            key: "" if value is None else str(value)
            for key, value in exp_ca.model_dump(mode="json").items()
            if key not in {"verified", "review_notes"}
        }
        actual_row = {key: r[key] for key in expected_row}
        if actual_row != expected_row:
            raise Week1GateInputError(
                f"Immutable audit fields changed for cell_id '{cell_id}' in cell-audit.csv"
            )

        v_str = (r.get("verified") or "").strip()
        if v_str == "true":
            v_bool: bool | None = True
        elif v_str == "false":
            v_bool = False
        else:
            raise Week1GateInputError(
                f"Unverified or invalid verified status for cell_id '{cell_id}' "
                f"in cell-audit.csv: '{r.get('verified')}'"
            )

        user_audits.append(
            exp_ca.model_copy(
                update={
                    "verified": v_bool,
                    "review_notes": r.get("review_notes", ""),
                }
            )
        )

    cell_audits = tuple(user_audits)

    # Final usability evaluation
    final_assessments = evaluate_table_usability(
        initial_assessments, matched_tables, all_cell_audits
    )

    # Metrics & Gate Checks calculation
    total_exp = len(expected_tables_tuple)
    total_matched = len(matched_tables)
    total_usable = sum(1 for ta in final_assessments if ta.usable)

    cell_audit_sha256 = hashlib.sha256(cell_audit_csv_path.read_bytes()).hexdigest()

    eval_input_lines = sorted(
        [
            docs_sha256,
            expected_tables_sha256,
            cell_audit_sha256,
            dataset.dataset_fingerprint,
            dataset.source_manifest_sha256,
        ]
    )
    evaluation_inputs_sha256 = hashlib.sha256(
        "\n".join(eval_input_lines).encode("utf-8")
    ).hexdigest()

    sampled_cell_count = len(cell_audits)
    verified_cell_count = sum(1 for ca in cell_audits if ca.verified is True)
    accepted_cell_count = len(all_cell_audits)
    provenance_valid_cell_count = sum(1 for ca in all_cell_audits if ca.verified is True)

    statement_metrics: dict[str, dict[str, Any]] = {}
    min_statement_count = 30
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
    eligible_strata = 0
    passing_strata = 0
    by_stratum: dict[tuple[str, int, str], list[TableAssessment]] = {}
    pilot_doc_by_id = {doc.doc_id: doc for doc in pilot_docs}
    for ta in final_assessments:
        doc = pilot_doc_by_id.get(ta.annotation.doc_id)
        if doc is None:
            continue
        by_stratum.setdefault(
            (doc.company_code, doc.report_year, ta.annotation.statement_type), []
        ).append(ta)

    for key in sorted(by_stratum):
        rows = by_stratum[key]
        annotated = len(rows)
        usable = sum(1 for ta in rows if ta.usable)
        included = annotated >= 10
        passed = percentage_passes(usable, annotated, 70) if included else None
        stratum_key = f"{key[0]}:{key[1]}:{key[2]}"
        stratum_metrics[stratum_key] = {
            "annotated_count": annotated,
            "usable_count": usable,
            "included": included,
            "passed": passed,
        }
        if included:
            eligible_strata += 1
            if passed:
                passing_strata += 1

    checks = (
        GateCheck(
            name="pilot_document_count",
            passed=len(pilot_docs) == 60,
            numerator=len(pilot_docs),
            denominator=60,
            threshold_percent=100,
        ),
        GateCheck(
            name="statement_type_coverage",
            passed=all(
                metrics["expected_count"] >= min_statement_count
                for metrics in statement_metrics.values()
            ),
            numerator=min(metrics["expected_count"] for metrics in statement_metrics.values()),
            denominator=min_statement_count,
            threshold_percent=100,
        ),
        GateCheck(
            name="overall_table_usability",
            passed=percentage_passes(total_usable, total_exp, 85),
            numerator=total_usable,
            denominator=total_exp,
            threshold_percent=85,
        ),
        GateCheck(
            name="accepted_cell_provenance",
            passed=accepted_cell_count > 0 and provenance_valid_cell_count == accepted_cell_count,
            numerator=provenance_valid_cell_count,
            denominator=accepted_cell_count,
            threshold_percent=100,
        ),
        GateCheck(
            name="manual_cell_audit",
            passed=sampled_cell_count == 30 and verified_cell_count == 30,
            numerator=verified_cell_count,
            denominator=30,
            threshold_percent=100,
        ),
        GateCheck(
            name="eligible_strata_usability",
            passed=eligible_strata == 0 or passing_strata == eligible_strata,
            numerator=passing_strata,
            denominator=eligible_strata,
            threshold_percent=70,
        ),
    )

    gate_passed = all(c.passed for c in checks)
    pareto_rows = compute_pareto_analysis(final_assessments)

    result = GateResult(
        sampling_version=SAMPLING_VERSION,
        annotation_schema_version="1",
        dataset_fingerprint=dataset.dataset_fingerprint,
        source_manifest_sha256=dataset.source_manifest_sha256,
        pilot_documents_sha256=docs_sha256,
        expected_tables_sha256=expected_tables_sha256,
        cell_audit_sha256=cell_audit_sha256,
        evaluation_inputs_sha256=evaluation_inputs_sha256,
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
    """Publish canonical gate evaluation reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = cell_audits
    expected_artifacts = {"gate-result.json", "gate-report.md", "pareto-errors.csv"}
    existing_artifacts = {path.name for path in output_dir.iterdir() if path.is_file()}
    extra_artifacts = sorted(existing_artifacts - expected_artifacts)
    if extra_artifacts:
        raise Week1GateInputError(
            "Refusing to publish into report directory with unexpected artifacts: "
            + ", ".join(extra_artifacts)
        )

    # Write gate-result.json
    write_canonical_json(output_dir / "gate-result.json", result.model_dump(mode="json"))

    # Write pareto-errors.csv
    pareto_rows = [p.model_dump(mode="json") for p in result.pareto_rows]
    write_csv_rows(
        output_dir / "pareto-errors.csv",
        PARETO_CSV_COLUMNS,
        pareto_rows,
        allow_identical=True,
    )

    status_str = "PASSED" if result.passed else "FAILED"
    md_lines = [
        "# Week 1 Quality Gate Evaluation Report",
        "",
        f"**Status: {status_str}**",
        "",
        "## Summary",
        f"- **Sampling Version:** {result.sampling_version}",
        f"- **Pilot Documents:** {result.document_count}",
        f"- **Annotated Tables:** {result.annotated_table_count}",
        f"- **Matched Tables:** {result.matched_table_count}",
        f"- **Usable Tables:** {result.usable_table_count}",
        "",
        "## Quality Gate Checks",
        "| Check | Value | Threshold | Status |",
        "| --- | --- | --- | --- |",
    ]
    for check in result.checks:
        c_status = "PASS" if check.passed else "FAIL"
        md_lines.append(
            f"| {check.name} | {check.numerator}/{check.denominator} | "
            f">= {check.threshold_percent}% | {c_status} |"
        )

    md_lines.extend(["", "## Pareto Error Analysis"])
    if not result.pareto_rows:
        md_lines.append("No failure errors recorded.")
    else:
        md_lines.extend(
            [
                "| Rank | Code | Count | Share | Cumulative Share |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for p in result.pareto_rows:
            md_lines.append(
                f"| {p.rank} | {p.code} | {p.count} | {p.share} | {p.cumulative_share} |"
            )

    md_lines.extend(
        [
            "",
            "## Release Metadata",
            f"- **Dataset Fingerprint:** `{result.dataset_fingerprint}`",
            f"- **Source Manifest SHA256:** `{result.source_manifest_sha256}`",
            f"- **Evaluation Inputs SHA256:** `{result.evaluation_inputs_sha256}`",
            "",
        ]
    )

    content = "\n".join(md_lines)
    (output_dir / "gate-report.md").write_text(content, encoding="utf-8")
