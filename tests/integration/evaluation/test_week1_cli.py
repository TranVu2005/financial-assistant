"""Integration tests for the real week1-gate CLI lifecycle."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.cli import main as cli_main
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.week1_contracts import (
    CELL_AUDIT_COLUMNS,
    EXPECTED_TABLE_COLUMNS,
    read_csv_rows,
    stable_annotation_id,
    write_canonical_json,
    write_csv_rows,
)
from financial_report_qa.ingestion.provenance import stable_cell_id
from financial_report_qa.schemas import stable_document_id, stable_table_id

STATEMENT_TYPES = ("balance_sheet", "income_statement", "cash_flow_statement")


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


def _write_full_release(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, Any]]]:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "documents.jsonl"
    release_path = tmp_path / "release"
    release_path.mkdir()
    corpus_dir = tmp_path / "corpus"

    document_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    manifest_lines: list[str] = []

    for doc_index in range(60):
        company_code = f"C{doc_index // 3:02d}"
        sha256 = _sha(doc_index)
        doc_id = stable_document_id(sha256)
        relative_path = f"{company_code}/2024/consolidated/report_{doc_index:02d}.txt"
        doc_row = {
            "doc_id": doc_id,
            "repo_id": "test_repo",
            "revision": "main",
            "relative_path": relative_path,
            "company_code": company_code,
            "report_year": 2024,
            "statement_scope": "consolidated",
            "sha256": sha256,
            "file_size_bytes": 1000,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "1",
            "normalization_fingerprint": "b" * 64,
        }
        document_rows.append(doc_row)
        manifest_doc = dict(doc_row)
        manifest_doc["record_type"] = "document"
        manifest_doc.pop("ruleset_version")
        manifest_doc.pop("normalization_fingerprint")
        manifest_doc["notes"] = []
        manifest_lines.append(json.dumps(manifest_doc, sort_keys=True))

        source_lines = [f"Line {line}" for line in range(1, 50)]
        table_specs = [(STATEMENT_TYPES[doc_index % 3], 10, 20)]
        if doc_index < 10:
            table_specs.append(("balance_sheet", 30, 40))
        elif doc_index < 20:
            table_specs.append(("income_statement", 30, 40))
        elif doc_index < 30:
            table_specs.append(("cash_flow_statement", 30, 40))

        for table_number, (statement_type, line_start, line_end) in enumerate(table_specs):
            table_id = stable_table_id(doc_id, line_start, line_end, table_number)
            table_rows.append(
                {
                    "table_id": table_id,
                    "doc_id": doc_id,
                    "source_ordinal": table_number,
                    "title_raw": statement_type,
                    "statement_type": statement_type,
                    "unit_raw": "VND",
                    "unit_normalized": "VND",
                    "line_start": line_start,
                    "line_end": line_end,
                    "row_count": 1,
                    "column_count": 1,
                    "quality_score": 1.0,
                    "csv_path": None,
                }
            )
            value = str(1000 + doc_index * 10 + table_number)
            source_lines[line_start] = f"{statement_type} value {value}"
            cell_rows.append(
                {
                    "cell_id": stable_cell_id(table_id, 0, 0),
                    "table_id": table_id,
                    "row_idx": 0,
                    "col_idx": 0,
                    "row_label_raw": "value",
                    "row_label_canonical": "value",
                    "column_label_raw": "2024",
                    "column_label_canonical": "2024",
                    "value_raw": value,
                    "value_numeric": Decimal(value),
                    "period": "2024",
                    "unit": "VND",
                    "source_line_start": line_start + 1,
                    "source_line_end": line_start + 1,
                    "extraction_confidence": 1.0,
                }
            )
            placement_rows.append(
                {
                    "table_id": table_id,
                    "row_idx": 0,
                    "col_idx": 0,
                    "cell_id": stable_cell_id(table_id, 0, 0),
                }
            )

        doc_path = corpus_dir / relative_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    write_table = cast(Any, pq.write_table)
    write_table(
        pa.Table.from_pylist(document_rows, schema=DOCUMENT_SCHEMA),
        release_path / "documents.parquet",
    )
    write_table(
        pa.Table.from_pylist(table_rows, schema=TABLE_SCHEMA),
        release_path / "tables.parquet",
    )
    write_table(
        pa.Table.from_pylist(cell_rows, schema=CELL_SCHEMA),
        release_path / "cells.parquet",
    )
    write_table(
        pa.Table.from_pylist(placement_rows, schema=PLACEMENT_SCHEMA),
        release_path / "placements.parquet",
    )
    write_table(
        pa.Table.from_pylist([], schema=ISSUE_SCHEMA),
        release_path / "issues.parquet",
    )

    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": len(document_rows),
        "table_count": len(table_rows),
        "cell_count": len(cell_rows),
        "placement_count": len(placement_rows),
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(
        json.dumps(release_manifest, sort_keys=True), encoding="utf-8"
    )
    return manifest_path, release_path, corpus_dir, table_rows


def _common_args(
    manifest_path: Path, release_path: Path, corpus_dir: Path, annotation_dir: Path
) -> list[str]:
    return [
        "--manifest-path",
        str(manifest_path),
        "--release-path",
        str(release_path),
        "--corpus-dir",
        str(corpus_dir),
        "--annotation-dir",
        str(annotation_dir),
    ]


def _fill_expected_tables(annotation_dir: Path, table_rows: list[dict[str, Any]]) -> None:
    selected_doc_ids = {
        row["doc_id"]
        for row in read_csv_rows(
            annotation_dir / "pilot-documents.csv",
            (
                "annotation_schema_version",
                "dataset_fingerprint",
                "source_manifest_sha256",
                "doc_id",
                "relative_path",
                "company_code",
                "report_year",
                "statement_scope",
            ),
        )
    }
    rows = []
    for index, table in enumerate(t for t in table_rows if t["doc_id"] in selected_doc_ids):
        rows.append(
            {
                "annotation_schema_version": "1",
                "annotation_id": stable_annotation_id(
                    table["doc_id"],
                    table["line_start"],
                    table["line_end"],
                    table["statement_type"],
                ),
                "doc_id": table["doc_id"],
                "relative_path": next(
                    row["relative_path"]
                    for row in read_csv_rows(
                        annotation_dir / "pilot-documents.csv",
                        (
                            "annotation_schema_version",
                            "dataset_fingerprint",
                            "source_manifest_sha256",
                            "doc_id",
                            "relative_path",
                            "company_code",
                            "report_year",
                            "statement_scope",
                        ),
                    )
                    if row["doc_id"] == table["doc_id"]
                ),
                "statement_type": table["statement_type"],
                "line_start": str(table["line_start"]),
                "line_end": str(table["line_end"]),
                "row_count": str(table["row_count"]),
                "column_count": str(table["column_count"]),
                "unit_normalized": str(table["unit_normalized"]),
                "expected_periods": "2024",
                "notes": "",
            }
        )
    write_csv_rows(
        annotation_dir / "expected-tables.csv",
        EXPECTED_TABLE_COLUMNS,
        rows,
        allow_identical=True,
    )


def _mark_all_audits_verified(annotation_dir: Path) -> None:
    rows = [
        dict(row) for row in read_csv_rows(annotation_dir / "cell-audit.csv", CELL_AUDIT_COLUMNS)
    ]
    for row in rows:
        row["verified"] = "true"
    write_csv_rows(
        annotation_dir / "cell-audit.csv",
        CELL_AUDIT_COLUMNS,
        rows,
        allow_identical=True,
    )


def test_week1_gate_cli_full_3stage_lifecycle_is_idempotent(tmp_path: Path) -> None:
    manifest_path, release_path, corpus_dir, table_rows = _write_full_release(tmp_path)
    annotation_dir = tmp_path / "annotations"
    report_root = tmp_path / "reports"

    assert (
        cli_main(
            [
                "week1-gate",
                "prepare",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--annotation-root",
                str(annotation_dir),
            ]
        )
        == 0
    )
    _fill_expected_tables(annotation_dir, table_rows)
    assert (
        cli_main(
            [
                "week1-gate",
                "sample-cells",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
            ]
        )
        == 0
    )

    audit_rows = read_csv_rows(annotation_dir / "cell-audit.csv", CELL_AUDIT_COLUMNS)
    assert len(audit_rows) == 30
    assert {row["verified"] for row in audit_rows} == {""}

    _mark_all_audits_verified(annotation_dir)
    assert (
        cli_main(
            [
                "week1-gate",
                "evaluate",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
                "--report-root",
                str(report_root),
            ]
        )
        == 0
    )
    first_bytes = {p.name: p.read_bytes() for p in report_root.iterdir()}
    assert sorted(first_bytes) == ["gate-report.md", "gate-result.json", "pareto-errors.csv"]

    assert (
        cli_main(
            [
                "week1-gate",
                "evaluate",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
                "--report-root",
                str(report_root),
            ]
        )
        == 0
    )
    second_bytes = {p.name: p.read_bytes() for p in report_root.iterdir()}
    assert second_bytes == first_bytes


def _mark_one_audit_failed(annotation_dir: Path) -> None:
    rows = [
        dict(row) for row in read_csv_rows(annotation_dir / "cell-audit.csv", CELL_AUDIT_COLUMNS)
    ]
    for idx, row in enumerate(rows):
        row["verified"] = "false" if idx == 0 else "true"
    write_csv_rows(
        annotation_dir / "cell-audit.csv",
        CELL_AUDIT_COLUMNS,
        rows,
        allow_identical=True,
    )


def test_week1_gate_cli_valid_gate_failure_returns_one(tmp_path: Path) -> None:
    manifest_path, release_path, corpus_dir, table_rows = _write_full_release(tmp_path)
    annotation_dir = tmp_path / "annotations"
    report_root = tmp_path / "reports"
    assert (
        cli_main(
            [
                "week1-gate",
                "prepare",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--annotation-root",
                str(annotation_dir),
            ]
        )
        == 0
    )
    _fill_expected_tables(annotation_dir, table_rows)
    assert (
        cli_main(
            [
                "week1-gate",
                "sample-cells",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
            ]
        )
        == 0
    )
    _mark_one_audit_failed(annotation_dir)

    assert (
        cli_main(
            [
                "week1-gate",
                "evaluate",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
                "--output-dir",
                str(report_root),
            ]
        )
        == 1
    )


def test_week1_gate_cli_input_errors_return_two_without_report_mutation(tmp_path: Path) -> None:
    manifest_path, release_path, corpus_dir, table_rows = _write_full_release(tmp_path)
    annotation_dir = tmp_path / "annotations"
    report_root = tmp_path / "reports"
    report_root.mkdir()
    sentinel = report_root / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")

    assert (
        cli_main(
            [
                "week1-gate",
                "prepare",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--annotation-root",
                str(annotation_dir),
            ]
        )
        == 0
    )
    _fill_expected_tables(annotation_dir, table_rows)

    metadata_path = annotation_dir / "pilot-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_fingerprint"] = "0" * 64
    write_canonical_json(metadata_path, metadata)

    assert (
        cli_main(
            [
                "week1-gate",
                "evaluate",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
                "--output-dir",
                str(report_root),
            ]
        )
        == 2
    )
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_week1_gate_cli_full_review_lifecycle(tmp_path: Path) -> None:
    manifest_path, release_path, corpus_dir, table_rows = _write_full_release(tmp_path)
    annotation_dir = tmp_path / "annotations"
    report_root = tmp_path / "reports"
    review_path = tmp_path / "table-review.csv"

    # 1. prepare
    assert (
        cli_main(
            [
                "week1-gate",
                "prepare",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--annotation-root",
                str(annotation_dir),
            ]
        )
        == 0
    )

    # 2. prepare-review
    assert (
        cli_main(
            [
                "week1-gate",
                "prepare-review",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--corpus-dir",
                str(corpus_dir),
                "--annotation-dir",
                str(annotation_dir),
                "--output-path",
                str(review_path),
            ]
        )
        == 0
    )
    assert review_path.is_file()

    # Read review CSV
    import csv

    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        review_rows = list(reader)

    # Pre-populate human inputs: include=true for at least 90 tables to satisfy count validation
    for r in review_rows:
        r["include"] = "true"
        r["unit_normalized"] = "VND"
        r["expected_periods"] = "2024"

    # Write back review CSV with LF line endings
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)

    # 3. finalize-tables
    assert (
        cli_main(
            [
                "week1-gate",
                "finalize-tables",
                "--manifest-path",
                str(manifest_path),
                "--release-path",
                str(release_path),
                "--annotation-dir",
                str(annotation_dir),
                "--review-path",
                str(review_path),
            ]
        )
        == 0
    )

    expected_tables_csv = annotation_dir / "expected-tables.csv"
    assert expected_tables_csv.is_file()
    final_rows = read_csv_rows(expected_tables_csv, EXPECTED_TABLE_COLUMNS)
    assert len(final_rows) == 90

    # 4. sample-cells
    assert (
        cli_main(
            [
                "week1-gate",
                "sample-cells",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
            ]
        )
        == 0
    )

    # 5. evaluate
    _mark_all_audits_verified(annotation_dir)
    assert (
        cli_main(
            [
                "week1-gate",
                "evaluate",
                *_common_args(manifest_path, release_path, corpus_dir, annotation_dir),
                "--report-root",
                str(report_root),
            ]
        )
        == 0
    )
    assert (report_root / "gate-result.json").is_file()
