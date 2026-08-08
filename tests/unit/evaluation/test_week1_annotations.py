import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.week1_annotations import (
    load_annotation_bundle,
    generate_review_worksheet,
    finalize_review_worksheet,
)
from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    read_csv_rows,
    stable_annotation_id,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.schemas import stable_document_id, stable_table_id
from financial_report_qa.ingestion.provenance import stable_cell_id


def _sha(index: int) -> str:
    return f"{index + 1:064x}"


def complete_annotation_fixture(tmp_path: Path) -> tuple[any, Path]:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True, parents=True)
    manifest_path = manifest_dir / "documents.jsonl"
    release_path = tmp_path / "release"
    release_path.mkdir(exist_ok=True, parents=True)
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir(exist_ok=True, parents=True)

    document_rows = []
    manifest_lines = []
    pilot_doc_rows = []
    expected_table_rows = []

    # 60 documents: 20 companies x 3 documents
    # statement scopes: consolidated, separate, consolidated
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

        pilot_doc_rows.append(
            {
                "annotation_schema_version": "1",
                "dataset_fingerprint": "f" * 64,
                "source_manifest_sha256": "placeholder",  # Will fill below
                "doc_id": doc_id,
                "relative_path": relative_path,
                "company_code": company_code,
                "report_year": "2024",
                "statement_scope": "consolidated",
            }
        )

        # Generate exactly 90 annotations: 30 BS, 30 IS, 30 CF
        # C00..C09 (30 docs): each gets 1 BS, 1 IS (60 total)
        # C10..C19 (30 docs): each gets 1 CF (30 total)
        comp_idx = doc_index // 3
        if comp_idx < 10:
            # BS annotation
            bs_id = stable_annotation_id(doc_id, 10, 20, "balance_sheet")
            expected_table_rows.append(
                {
                    "annotation_schema_version": "1",
                    "annotation_id": bs_id,
                    "doc_id": doc_id,
                    "relative_path": relative_path,
                    "statement_type": "balance_sheet",
                    "line_start": "10",
                    "line_end": "20",
                    "row_count": "5",
                    "column_count": "3",
                    "unit_normalized": "VND",
                    "expected_periods": "2024",
                    "notes": "",
                }
            )
            # IS annotation
            is_id = stable_annotation_id(doc_id, 30, 40, "income_statement")
            expected_table_rows.append(
                {
                    "annotation_schema_version": "1",
                    "annotation_id": is_id,
                    "doc_id": doc_id,
                    "relative_path": relative_path,
                    "statement_type": "income_statement",
                    "line_start": "30",
                    "line_end": "40",
                    "row_count": "5",
                    "column_count": "3",
                    "unit_normalized": "VND",
                    "expected_periods": "2024",
                    "notes": "",
                }
            )
        else:
            # CF annotation
            cf_id = stable_annotation_id(doc_id, 10, 20, "cash_flow_statement")
            expected_table_rows.append(
                {
                    "annotation_schema_version": "1",
                    "annotation_id": cf_id,
                    "doc_id": doc_id,
                    "relative_path": relative_path,
                    "statement_type": "cash_flow_statement",
                    "line_start": "10",
                    "line_end": "20",
                    "row_count": "5",
                    "column_count": "3",
                    "unit_normalized": "VND",
                    "expected_periods": "2024",
                    "notes": "",
                }
            )

    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    for doc in pilot_doc_rows:
        doc["source_manifest_sha256"] = source_manifest_sha256

    # Write pilot documents
    docs_csv_path = annotation_dir / "pilot-documents.csv"
    write_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS, pilot_doc_rows)
    docs_sha256 = hashlib.sha256(docs_csv_path.read_bytes()).hexdigest()

    # Write expected tables
    expected_csv_path = annotation_dir / "expected-tables.csv"
    write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, expected_table_rows)

    # Write metadata
    metadata = {
        "annotation_schema_version": "1",
        "sampling_version": "week1-pilot-v1",
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": 60,
        "company_count": 20,
        "pilot_documents_sha256": docs_sha256,
    }
    (annotation_dir / "pilot-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Write release parquet files
    write_table = pq.write_table
    write_table(
        pa.Table.from_pylist(document_rows, schema=DOCUMENT_SCHEMA),
        release_path / "documents.parquet",
    )
    write_table(
        pa.Table.from_pylist([], schema=TABLE_SCHEMA),
        release_path / "tables.parquet",
    )
    write_table(
        pa.Table.from_pylist([], schema=CELL_SCHEMA),
        release_path / "cells.parquet",
    )
    write_table(
        pa.Table.from_pylist([], schema=PLACEMENT_SCHEMA),
        release_path / "placements.parquet",
    )
    write_table(
        pa.Table.from_pylist([], schema=ISSUE_SCHEMA),
        release_path / "issues.parquet",
    )

    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": 60,
        "table_count": 0,
        "cell_count": 0,
        "placement_count": 0,
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(
        json.dumps(release_manifest, sort_keys=True), encoding="utf-8"
    )

    dataset = load_gate_dataset(manifest_path, release_path)
    return dataset, annotation_dir


def mutate_annotation_fixture(annotation_dir: Path, mutation: str) -> None:
    docs_csv_path = annotation_dir / "pilot-documents.csv"
    doc_rows = list(read_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS))
    expected_csv_path = annotation_dir / "expected-tables.csv"
    expected_rows = list(read_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS))
    metadata_path = annotation_dir / "pilot-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if mutation == "duplicate_pilot_doc":
        # Add duplicate row to pilot docs CSV
        doc_rows.append(doc_rows[-1])
        write_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS, doc_rows, allow_identical=True)
        # Update metadata sha
        metadata["pilot_documents_sha256"] = hashlib.sha256(
            docs_csv_path.read_bytes()
        ).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    elif mutation == "wrong_company_count":
        # Change company code of the last 3 documents to a company that already exists
        comp_to_use = doc_rows[0]["company_code"]
        for row in doc_rows[-3:]:
            row["company_code"] = comp_to_use
        write_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS, doc_rows, allow_identical=True)
        metadata["pilot_documents_sha256"] = hashlib.sha256(
            docs_csv_path.read_bytes()
        ).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    elif mutation == "wrong_documents_per_company":
        # Change one company code of a document, causing unequal distribution
        doc_rows[-1]["company_code"] = "COMP_EXTRA"
        write_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS, doc_rows, allow_identical=True)
        metadata["pilot_documents_sha256"] = hashlib.sha256(
            docs_csv_path.read_bytes()
        ).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    elif mutation == "expected_doc_outside_pilot":
        # Reference a doc_id that is not in the pilot
        expected_rows[0]["doc_id"] = "doc_outside_pilot"
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, expected_rows, allow_identical=True)

    elif mutation == "expected_path_mismatch":
        # Mismatch the relative path for a document
        expected_rows[0]["relative_path"] = "different/path/report.txt"
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, expected_rows, allow_identical=True)

    elif mutation == "overlapping_same_statement_annotations":
        # Create an overlapping statement in the first doc
        doc_id = expected_rows[0]["doc_id"]
        rel_path = expected_rows[0]["relative_path"]
        statement_type = expected_rows[0]["statement_type"]
        new_id = stable_annotation_id(doc_id, 15, 25, statement_type)
        expected_rows.append(
            {
                "annotation_schema_version": "1",
                "annotation_id": new_id,
                "doc_id": doc_id,
                "relative_path": rel_path,
                "statement_type": statement_type,
                "line_start": "15",
                "line_end": "25",
                "row_count": "5",
                "column_count": "3",
                "unit_normalized": "VND",
                "expected_periods": "2024",
                "notes": "",
            }
        )
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, expected_rows, allow_identical=True)

    elif mutation == "fewer_than_30_balance_sheets":
        # Filter out all balance sheet annotations
        filtered = [r for r in expected_rows if r["statement_type"] != "balance_sheet"]
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, filtered, allow_identical=True)

    elif mutation == "fewer_than_30_income_statements":
        # Filter out all income statement annotations
        filtered = [r for r in expected_rows if r["statement_type"] != "income_statement"]
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, filtered, allow_identical=True)

    elif mutation == "fewer_than_30_cash_flows":
        # Filter out all cash flow statement annotations
        filtered = [r for r in expected_rows if r["statement_type"] != "cash_flow_statement"]
        write_csv_rows(expected_csv_path, EXPECTED_TABLE_COLUMNS, filtered, allow_identical=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_pilot_doc",
        "wrong_company_count",
        "wrong_documents_per_company",
        "expected_doc_outside_pilot",
        "expected_path_mismatch",
        "overlapping_same_statement_annotations",
        "fewer_than_30_balance_sheets",
        "fewer_than_30_income_statements",
        "fewer_than_30_cash_flows",
    ],
)
def test_load_annotation_bundle_rejects_invalid_gate_input(
    tmp_path: Path, mutation: str
) -> None:
    dataset, annotation_dir = complete_annotation_fixture(tmp_path)
    mutate_annotation_fixture(annotation_dir, mutation)
    with pytest.raises(Week1GateInputError):
        load_annotation_bundle(dataset, annotation_dir, require_expected_tables=True)


def test_load_annotation_bundle_success(tmp_path: Path) -> None:
    dataset, annotation_dir = complete_annotation_fixture(tmp_path)
    bundle = load_annotation_bundle(dataset, annotation_dir, require_expected_tables=True)
    assert len(bundle.pilot_documents) == 60
    assert len(bundle.expected_tables) == 90


def test_review_worksheet_generation_and_finalization(tmp_path: Path) -> None:
    dataset, annotation_dir = complete_annotation_fixture(tmp_path)
    manifest_path = tmp_path / "manifests" / "documents.jsonl"
    release_path = tmp_path / "release"
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True, parents=True)

    # 1. Write mock files for the pilot documents in the corpus
    for doc in dataset.documents_by_id.values():
        doc_path = corpus_dir / doc.relative_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        # 50 lines to satisfy line range constraints in ExpectedTable (e.g. 10 to 40)
        lines = [f"Line {i}" for i in range(1, 51)]
        doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 2. Write mock tables and cells to the release parquet files
    from decimal import Decimal
    table_rows = []
    cell_rows = []
    placement_rows = []
    for doc_index, doc in enumerate(dataset.documents_by_id.values()):
        for stmt_idx, stmt in enumerate(["balance_sheet", "income_statement", "cash_flow_statement"]):
            table_id = stable_table_id(doc.doc_id, 10, 20, stmt_idx)
            table_rows.append({
                "table_id": table_id,
                "doc_id": doc.doc_id,
                "source_ordinal": stmt_idx,
                "title_raw": stmt.replace("_", " ").title(),
                "statement_type": stmt,
                "unit_raw": "VND",
                "unit_normalized": "VND",
                "line_start": 10,
                "line_end": 20,
                "row_count": 2,
                "column_count": 2,
                "quality_score": 1.0,
                "csv_path": None,
            })
            cell_id = stable_cell_id(table_id, 0, 0)
            cell_rows.append({
                "cell_id": cell_id,
                "table_id": table_id,
                "row_idx": 0,
                "col_idx": 0,
                "row_label_raw": "Asset",
                "row_label_canonical": "Asset",
                "column_label_raw": "2024",
                "column_label_canonical": "2024",
                "value_raw": "100",
                "value_numeric": Decimal("100"),
                "period": "2024",
                "unit": "VND",
                "source_line_start": 11,
                "source_line_end": 11,
                "extraction_confidence": 1.0,
            })
            placement_rows.append({
                "table_id": table_id,
                "row_idx": 0,
                "col_idx": 0,
                "cell_id": cell_id,
            })

    # Write release parquet files
    write_table = pq.write_table
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

    # Rewrite release manifest.json
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": 60,
        "table_count": len(table_rows),
        "cell_count": len(cell_rows),
        "placement_count": len(placement_rows),
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(
        json.dumps(release_manifest, sort_keys=True), encoding="utf-8"
    )

    # Reload dataset
    dataset = load_gate_dataset(manifest_path, release_path)

    review_path = tmp_path / "table-review.csv"

    # 1. Generate worksheet
    generate_review_worksheet(dataset, corpus_dir, annotation_dir, review_path)
    assert review_path.is_file()

    # Read review worksheet
    import csv
    with review_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        review_rows = list(reader)

    # Suggestions must include only pilot documents and remain stably sorted
    assert len(review_rows) > 0
    # verify columns
    expected_cols = {
        "include", "doc_id", "relative_path", "company_code", "report_year",
        "table_id", "line_start", "line_end", "title_raw", "source_excerpt",
        "statement_type", "row_count", "column_count", "unit_normalized",
        "expected_periods", "notes"
    }
    assert set(review_rows[0].keys()) == expected_cols

    # Verify sort order
    for idx in range(len(review_rows) - 1):
        r1, r2 = review_rows[idx], review_rows[idx + 1]
        assert (r1["doc_id"], int(r1["line_start"]), int(r1["line_end"])) <= \
               (r2["doc_id"], int(r2["line_start"]), int(r2["line_end"]))

    # 2. Finalize review worksheet
    # If the target file expected-tables.csv is non-empty, it should fail
    with pytest.raises(Week1GateInputError, match="expected-tables.csv already contains annotated data"):
        finalize_review_worksheet(dataset, annotation_dir, review_path)

    # Empty expected-tables.csv to simulate template
    expected_tables_csv = annotation_dir / "expected-tables.csv"
    expected_tables_csv.write_bytes(",".join(EXPECTED_TABLE_COLUMNS).encode("utf-8") + b"\n")

    # If review worksheet contains no includes, finalized should be empty
    finalize_review_worksheet(dataset, annotation_dir, review_path)
    final_rows = read_csv_rows(expected_tables_csv, EXPECTED_TABLE_COLUMNS)
    assert len(final_rows) == 0

    # Let's set include = true for 90 rows (enough to pass count validation)
    # We will modify the CSV
    for row in review_rows[:90]:
        row["include"] = "true"
        row["unit_normalized"] = "VND"
        row["expected_periods"] = "2024"

    # Write review worksheet back
    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(review_rows)

    # finalization should succeed now
    expected_tables_csv.write_bytes(",".join(EXPECTED_TABLE_COLUMNS).encode("utf-8") + b"\n")
    finalize_review_worksheet(dataset, annotation_dir, review_path)
    final_rows = read_csv_rows(expected_tables_csv, EXPECTED_TABLE_COLUMNS)
    assert len(final_rows) == 90

    # Sort order in expected-tables.csv
    for idx in range(len(final_rows) - 1):
        r1, r2 = final_rows[idx], final_rows[idx + 1]
        assert (r1["doc_id"], int(r1["line_start"]), int(r1["line_end"]), r1["statement_type"]) <= \
               (r2["doc_id"], int(r2["line_start"]), int(r2["line_end"]), r2["statement_type"])

    # Test invalid values fail
    for bad_val, field in [
        ("bad_doc", "doc_id"),
        ("0", "line_start"),
        ("invalid_type", "statement_type"),
        ("invalid_unit", "unit_normalized"),
        ("2024|2023", "expected_periods")
    ]:
        bad_rows = [dict(r) for r in review_rows[:90]]
        for r in bad_rows:
            r["include"] = "true"
        bad_rows[0][field] = bad_val

        bad_review_path = tmp_path / f"bad-{field}.csv"
        with bad_review_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(bad_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(bad_rows)

        expected_tables_csv.write_bytes(",".join(EXPECTED_TABLE_COLUMNS).encode("utf-8") + b"\n")
        with pytest.raises(Week1GateInputError):
            finalize_review_worksheet(dataset, annotation_dir, bad_review_path)

