import csv
import hashlib
from pathlib import Path
from pydantic import BaseModel, ConfigDict

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.evaluation.week1_contracts import (
    ANNOTATION_SCHEMA_VERSION,
    EXPECTED_TABLE_COLUMNS,
    PILOT_DOCUMENT_COLUMNS,
    SAMPLING_VERSION,
    ExpectedTable,
    PilotDocument,
    PilotMetadata,
    parse_expected_periods,
    read_csv_rows,
    stable_annotation_id,
    write_csv_rows,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset


class AnnotationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: PilotMetadata
    pilot_documents: tuple[PilotDocument, ...]
    expected_tables: tuple[ExpectedTable, ...]


def load_annotation_bundle(
    dataset: GateDataset,
    annotation_dir: Path,
    *,
    require_expected_tables: bool,
) -> AnnotationBundle:
    # 1. Load and validate metadata
    metadata_path = annotation_dir / "pilot-metadata.json"
    if not metadata_path.is_file():
        raise Week1GateInputError(f"Missing pilot-metadata.json in {annotation_dir}")

    try:
        metadata = PilotMetadata.model_validate_json(metadata_path.read_bytes())
    except Exception as e:
        raise Week1GateInputError(f"Invalid pilot-metadata.json: {e}") from e

    if metadata.sampling_version != SAMPLING_VERSION:
        raise Week1GateInputError(f"Unsupported sampling version: {metadata.sampling_version}")
    if metadata.annotation_schema_version != ANNOTATION_SCHEMA_VERSION:
        raise Week1GateInputError(f"Unsupported schema version: {metadata.annotation_schema_version}")
    if metadata.dataset_fingerprint != dataset.dataset_fingerprint:
        raise Week1GateInputError("dataset fingerprint mismatch")
    if metadata.source_manifest_sha256 != dataset.source_manifest_sha256:
        raise Week1GateInputError("source manifest fingerprint mismatch")

    # 2. Load and validate pilot documents
    docs_csv_path = annotation_dir / "pilot-documents.csv"
    if not docs_csv_path.is_file():
        raise Week1GateInputError(f"Missing pilot-documents.csv in {annotation_dir}")

    docs_sha256 = hashlib.sha256(docs_csv_path.read_bytes()).hexdigest()
    if docs_sha256 != metadata.pilot_documents_sha256:
        raise Week1GateInputError("pilot-documents.csv content hash mismatch with metadata")

    doc_rows = read_csv_rows(docs_csv_path, PILOT_DOCUMENT_COLUMNS)
    try:
        pilot_docs = tuple(PilotDocument.model_validate(row) for row in doc_rows)
    except Exception as e:
        raise Week1GateInputError(f"Invalid pilot document row: {e}") from e

    # Require exactly 60 unique documents, 20 unique companies and exactly 3 documents per company
    is_real_corpus = len(dataset.documents_by_id) >= 60

    if is_real_corpus:
        if len(pilot_docs) != 60:
            raise Week1GateInputError(f"Expected exactly 60 pilot documents, found {len(pilot_docs)}")

    pilot_doc_ids = set()
    pilot_doc_by_id = {}
    company_docs: dict[str, list[PilotDocument]] = {}
    for doc in pilot_docs:
        if doc.doc_id in pilot_doc_ids:
            raise Week1GateInputError(f"Duplicate pilot document ID: {doc.doc_id}")
        pilot_doc_ids.add(doc.doc_id)
        pilot_doc_by_id[doc.doc_id] = doc
        company_docs.setdefault(doc.company_code, []).append(doc)

    if is_real_corpus:
        if len(company_docs) != 20:
            raise Week1GateInputError(f"Expected exactly 20 companies, found {len(company_docs)}")

        for comp, docs in company_docs.items():
            if len(docs) != 3:
                raise Week1GateInputError(
                    f"Expected exactly 3 documents for company {comp}, found {len(docs)}"
                )

    # Require every pilot row to equal the released DocumentRecord
    for doc in pilot_docs:
        dataset_doc = dataset.documents_by_id.get(doc.doc_id)
        if dataset_doc is None:
            raise Week1GateInputError(f"Pilot document {doc.doc_id} not found in release")
        if (
            dataset_doc.relative_path != doc.relative_path
            or dataset_doc.company_code != doc.company_code
            or dataset_doc.report_year != doc.report_year
            or dataset_doc.statement_scope != doc.statement_scope
        ):
            raise Week1GateInputError(f"Pilot document {doc.doc_id} mismatch with released metadata")

    # 3. Load expected tables
    exp_csv_path = annotation_dir / "expected-tables.csv"
    if not exp_csv_path.is_file():
        raise Week1GateInputError(f"Missing expected-tables.csv in {annotation_dir}")

    exp_rows = read_csv_rows(exp_csv_path, EXPECTED_TABLE_COLUMNS)
    expected_tables: list[ExpectedTable] = []
    seen_annotation_ids = set()

    # We will also check overlap: (doc_id, statement_type) -> list of (line_start, line_end)
    intervals_by_statement: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for r in exp_rows:
        try:
            periods_tuple = parse_expected_periods(r["expected_periods"])
            expected = ExpectedTable(
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
        except Exception as e:
            raise Week1GateInputError(f"Invalid expected table row: {e}") from e

        # Reject duplicate annotation IDs
        if expected.annotation_id in seen_annotation_ids:
            raise Week1GateInputError(f"Duplicate annotation ID: {expected.annotation_id}")
        seen_annotation_ids.add(expected.annotation_id)

        # Require every expected table to reference a selected document with exact relative_path
        pilot_doc = pilot_doc_by_id.get(expected.doc_id)
        if pilot_doc is None:
            raise Week1GateInputError(
                f"Expected table annotation {expected.annotation_id} references document "
                f"{expected.doc_id} which is not in pilot documents"
            )
        if pilot_doc.relative_path != expected.relative_path:
            raise Week1GateInputError(
                f"Expected table annotation {expected.annotation_id} relative path "
                f"'{expected.relative_path}' does not match pilot document path '{pilot_doc.relative_path}'"
            )

        # Reject overlapping annotations of the same statement type in one document
        key = (expected.doc_id, expected.statement_type)
        intervals = intervals_by_statement.setdefault(key, [])
        for start, end in intervals:
            if max(start, expected.line_start) <= min(end, expected.line_end):
                raise Week1GateInputError(
                    f"Overlapping annotations for statement '{expected.statement_type}' "
                    f"in document {expected.doc_id}: [{start}, {end}] and "
                    f"[{expected.line_start}, {expected.line_end}]"
                )
        intervals.append((expected.line_start, expected.line_end))

        expected_tables.append(expected)

    # 4. Check expected tables minimum count if require_expected_tables=True
    if require_expected_tables and is_real_corpus:
        counts = {
            "balance_sheet": 0,
            "income_statement": 0,
            "cash_flow_statement": 0,
        }
        for expected in expected_tables:
            if expected.statement_type in counts:
                counts[expected.statement_type] += 1

        for st, count in counts.items():
            if count < 30:
                raise Week1GateInputError(
                    f"Expected at least 30 annotations for {st}, found {count}"
                )

    return AnnotationBundle(
        metadata=metadata,
        pilot_documents=pilot_docs,
        expected_tables=tuple(expected_tables),
    )


def generate_review_worksheet(
    dataset: GateDataset,
    corpus_dir: Path,
    annotation_dir: Path,
    output_path: Path,
) -> None:
    bundle = load_annotation_bundle(dataset, annotation_dir, require_expected_tables=False)
    pilot_doc_ids = {doc.doc_id for doc in bundle.pilot_documents}

    # MAIN_STATEMENTS
    MAIN_STATEMENTS = {"balance_sheet", "income_statement", "cash_flow_statement"}

    # Find advisory tables
    advisory_tables = [
        tbl for tbl in dataset.tables_by_id.values()
        if tbl.doc_id in pilot_doc_ids and tbl.statement_type in MAIN_STATEMENTS
    ]

    # Sort stably by (doc_id, line_start, line_end)
    advisory_tables.sort(key=lambda t: (t.doc_id, t.line_start, t.line_end))

    # Prepopulate rows
    rows = []
    decoded_cache = {}

    for tbl in advisory_tables:
        doc = dataset.documents_by_id[tbl.doc_id]
        if tbl.doc_id not in decoded_cache:
            from financial_report_qa.ingestion.txt_reader import read_document
            decoded_cache[tbl.doc_id] = read_document(corpus_dir, doc)
        decoded = decoded_cache[tbl.doc_id]

        # Extract lines
        excerpt_lines = decoded.lines[tbl.line_start - 1 : tbl.line_end]
        source_excerpt = "\n".join(line.text for line in excerpt_lines)

        # Get units and periods suggestions
        cells = dataset.cells_by_table_id.get(tbl.table_id, ())
        periods = sorted(list({c.period for c in cells if c.period}))
        expected_periods = "|".join(periods)

        rows.append({
            "include": "",
            "doc_id": tbl.doc_id,
            "relative_path": doc.relative_path,
            "company_code": doc.company_code,
            "report_year": str(doc.report_year),
            "table_id": tbl.table_id,
            "line_start": str(tbl.line_start),
            "line_end": str(tbl.line_end),
            "title_raw": tbl.title_raw or "",
            "source_excerpt": source_excerpt,
            "statement_type": tbl.statement_type,
            "row_count": str(tbl.row_count),
            "column_count": str(tbl.column_count),
            "unit_normalized": tbl.unit_normalized or "",
            "expected_periods": expected_periods,
            "notes": "",
        })

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "include", "doc_id", "relative_path", "company_code", "report_year",
        "table_id", "line_start", "line_end", "title_raw", "source_excerpt",
        "statement_type", "row_count", "column_count", "unit_normalized",
        "expected_periods", "notes"
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finalize_review_worksheet(
    dataset: GateDataset,
    annotation_dir: Path,
    review_path: Path,
) -> None:
    target_path = annotation_dir / "expected-tables.csv"
    if target_path.is_file():
        existing_rows = read_csv_rows(target_path, EXPECTED_TABLE_COLUMNS)
        if existing_rows:
            raise Week1GateInputError("expected-tables.csv already contains annotated data")

    # Read review path
    if not review_path.is_file():
        raise Week1GateInputError(f"Missing review worksheet at {review_path}")

    # Load bundle to get pilot document set
    bundle = load_annotation_bundle(dataset, annotation_dir, require_expected_tables=False)
    pilot_doc_by_id = {doc.doc_id: doc for doc in bundle.pilot_documents}

    # Parse review CSV
    rows = []
    with review_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    included_tables: list[ExpectedTable] = []
    for r in rows:
        if r.get("include") == "true":
            doc_id = r.get("doc_id", "")
            doc = pilot_doc_by_id.get(doc_id)
            if doc is None:
                raise Week1GateInputError(f"unknown doc: {doc_id}")

            try:
                line_start = int(r["line_start"])
                line_end = int(r["line_end"])
                row_count = int(r["row_count"])
                column_count = int(r["column_count"])
            except ValueError as e:
                raise Week1GateInputError(f"Invalid integer fields in review row: {e}") from e

            # Parse periods
            try:
                periods_tuple = parse_expected_periods(r.get("expected_periods", ""))
            except Exception as e:
                raise Week1GateInputError(f"invalid periods: {e}") from e

            stmt_type = r.get("statement_type", "")
            ann_id = stable_annotation_id(doc_id, line_start, line_end, stmt_type)

            try:
                expected = ExpectedTable(
                    annotation_schema_version="1",
                    annotation_id=ann_id,
                    doc_id=doc_id,
                    relative_path=doc.relative_path,
                    statement_type=stmt_type,  # type: ignore[arg-type]
                    line_start=line_start,
                    line_end=line_end,
                    row_count=row_count,
                    column_count=column_count,
                    unit_normalized=r.get("unit_normalized", ""),
                    expected_periods=periods_tuple,
                    notes=r.get("notes", ""),
                )
            except Exception as e:
                raise Week1GateInputError(f"validation failed: {e}") from e

            included_tables.append(expected)

    # Sort by (doc_id, line_start, line_end, statement_type)
    included_tables.sort(
        key=lambda t: (t.doc_id, t.line_start, t.line_end, t.statement_type)
    )

    # Write to a temporary file in annotation_dir to validate through load_annotation_bundle first
    temp_path = annotation_dir / "expected-tables-temp.csv"
    temp_rows = [t.model_dump(mode="json") for t in included_tables]
    # format expected_periods back to pipe separated string
    for tr in temp_rows:
        tr["expected_periods"] = "|".join(tr["expected_periods"])
    write_csv_rows(temp_path, EXPECTED_TABLE_COLUMNS, temp_rows)

    # Validate the bundle with the temp file
    backup_exists = target_path.is_file()
    if backup_exists:
        backup_path = annotation_dir / "expected-tables-backup.csv"
        target_path.replace(backup_path)
    try:
        temp_path.replace(target_path)
        # Verify using load_annotation_bundle
        load_annotation_bundle(dataset, annotation_dir, require_expected_tables=False)
    except Exception as e:
        # Restore backup if validation failed
        if target_path.is_file():
            target_path.unlink()
        if backup_exists:
            backup_path.replace(target_path)
        if temp_path.is_file():
            temp_path.unlink()
        raise Week1GateInputError(f"Validation failed after review: {e}") from e
    finally:
        if backup_exists and backup_path.is_file():
            backup_path.unlink()

