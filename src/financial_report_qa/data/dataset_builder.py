import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.core.errors import DatasetBuildError
from financial_report_qa.data.manifests import read_manifest
from financial_report_qa.ingestion import (
    detect_table_candidates,
    extract_candidates,
    read_document,
)
from financial_report_qa.ingestion.provenance import (
    DecodedDocument,
    DetectionResult,
    ExtractionResult,
)
from financial_report_qa.normalization import normalize_extraction
from financial_report_qa.normalization._shared import issue_sort_key
from financial_report_qa.schemas.documents import DocumentRecord
from financial_report_qa.schemas.normalization import NormalizedDocument

DOCUMENT_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("repo_id", pa.string()),
        ("revision", pa.string()),
        ("relative_path", pa.string()),
        ("company_code", pa.string()),
        ("report_year", pa.int32()),
        ("statement_scope", pa.string()),
        ("sha256", pa.string()),
        ("file_size_bytes", pa.int64()),
        ("encoding", pa.string()),
        ("inventory_status", pa.string()),
        ("ruleset_version", pa.string()),
        ("normalization_fingerprint", pa.string()),
    ]
)

TABLE_SCHEMA = pa.schema(
    [
        ("table_id", pa.string()),
        ("doc_id", pa.string()),
        ("title_raw", pa.string()),
        ("statement_type", pa.string()),
        ("unit_raw", pa.string()),
        ("unit_normalized", pa.string()),
        ("line_start", pa.int32()),
        ("line_end", pa.int32()),
        ("row_count", pa.int32()),
        ("column_count", pa.int32()),
        ("quality_score", pa.float64()),
        ("csv_path", pa.string()),
    ]
)

CELL_SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("table_id", pa.string()),
        ("row_idx", pa.int32()),
        ("col_idx", pa.int32()),
        ("row_label_raw", pa.string()),
        ("row_label_canonical", pa.string()),
        ("column_label_raw", pa.string()),
        ("column_label_canonical", pa.string()),
        pa.field("value_raw", pa.string(), nullable=False),
        ("value_numeric", pa.decimal128(38, 10)),
        ("period", pa.string()),
        ("unit", pa.string()),
        pa.field("source_line_start", pa.int32(), nullable=False),
        pa.field("source_line_end", pa.int32(), nullable=False),
        pa.field("extraction_confidence", pa.float64(), nullable=False),
    ]
)

ISSUE_SCHEMA = pa.schema(
    [
        ("code", pa.string()),
        ("doc_id", pa.string()),
        ("table_id", pa.string()),
        ("cell_id", pa.string()),
        ("field", pa.string()),
        ("raw_value", pa.string()),
    ]
)

SOURCE_TABLE_OCCURRENCE_SCHEMA = pa.schema(
    [
        pa.field("source_table_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("relative_path", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("ordinal", pa.int32(), nullable=False),
        pa.field("line_start", pa.int32(), nullable=False),
        pa.field("line_end", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("canonical_table_id", pa.string()),
        pa.field("rejection_code", pa.string()),
        pa.field("duplicate_of_relative_path", pa.string()),
    ]
)


class DatasetBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_root: Path
    manifest_path: Path
    processed_root: Path
    schema_version: str = Field(default="1", min_length=1)


class DatasetBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_path: Path
    dataset_fingerprint: str
    source_manifest_sha256: str
    document_count: int
    table_count: int
    cell_count: int
    issue_count: int
    issue_counts_by_code: dict[str, int]


@dataclass(frozen=True)
class FlattenedDataset:
    documents: tuple[dict[str, object], ...]
    tables: tuple[dict[str, object], ...]
    cells: tuple[dict[str, object], ...]
    issues: tuple[dict[str, object], ...]
    source_table_occurrences: tuple[dict[str, object], ...] = ()


def _source_table_id(
    *,
    relative_path: str,
    source_sha256: str,
    ordinal: int,
    line_start: int,
    line_end: int,
) -> str:
    payload = (
        f"{relative_path}|{source_sha256}|{ordinal}|{line_start}|{line_end}".encode()
    )
    return hashlib.sha256(payload).hexdigest()


def build_source_table_occurrences(
    document: DecodedDocument,
    detection: DetectionResult,
    extraction: ExtractionResult,
    canonical_table_ids: dict[tuple[int, int, int], str],
) -> list[dict[str, object]]:
    rejected_by_key = {
        (item.ordinal, item.line_start, item.line_end): item
        for item in extraction.rejected
        if item.kind == "html"
    }
    rows: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()

    html_items = sorted(
        (
            *(
                item
                for item in detection.candidates
                if item.kind == "html"
            ),
            *(
                item
                for item in detection.rejected
                if item.kind == "html"
            ),
        ),
        key=lambda item: (item.ordinal, item.line_start, item.line_end),
    )

    for item in html_items:
        key = (item.ordinal, item.line_start, item.line_end)
        canonical_table_id = canonical_table_ids.get(key)
        rejected = rejected_by_key.get(key)
        if canonical_table_id is None and rejected is None:
            raise ValueError(
                "every html candidate must resolve to a canonical or rejected outcome"
            )

        source_table_id = _source_table_id(
            relative_path=document.document.relative_path,
            source_sha256=document.document.sha256,
            ordinal=item.ordinal,
            line_start=item.line_start,
            line_end=item.line_end,
        )
        if source_table_id in seen_source_ids:
            raise ValueError("duplicate source table occurrence ID")
        seen_source_ids.add(source_table_id)

        rows.append(
            {
                "source_table_id": source_table_id,
                "doc_id": document.document.doc_id,
                "relative_path": document.document.relative_path,
                "source_sha256": document.document.sha256,
                "ordinal": item.ordinal,
                "line_start": item.line_start,
                "line_end": item.line_end,
                "status": "canonical" if canonical_table_id is not None else "rejected",
                "canonical_table_id": canonical_table_id,
                "rejection_code": None if canonical_table_id is not None else rejected.reason,
                "duplicate_of_relative_path": None,
            }
        )

    return rows


def build_duplicate_source_table_occurrences(
    duplicate_document: DocumentRecord,
    primary_rows: Sequence[dict[str, object]],
    duplicate_of_relative_path: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    for primary_row in primary_rows:
        ordinal = int(primary_row["ordinal"])
        line_start = int(primary_row["line_start"])
        line_end = int(primary_row["line_end"])
        source_table_id = _source_table_id(
            relative_path=duplicate_document.relative_path,
            source_sha256=duplicate_document.sha256,
            ordinal=ordinal,
            line_start=line_start,
            line_end=line_end,
        )
        if source_table_id in seen_source_ids:
            raise ValueError("duplicate source table occurrence ID")
        seen_source_ids.add(source_table_id)
        rows.append(
            {
                "source_table_id": source_table_id,
                "doc_id": duplicate_document.doc_id,
                "relative_path": duplicate_document.relative_path,
                "source_sha256": duplicate_document.sha256,
                "ordinal": ordinal,
                "line_start": line_start,
                "line_end": line_end,
                "status": "duplicate",
                "canonical_table_id": None,
                "rejection_code": None,
                "duplicate_of_relative_path": duplicate_of_relative_path,
            }
        )
    return rows


def flatten_normalized_documents(
    normalized_docs: tuple[NormalizedDocument, ...]
) -> FlattenedDataset:
    doc_rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    cell_rows: list[dict[str, object]] = []
    issue_rows: list[dict[str, object]] = []

    sorted_docs = sorted(
        normalized_docs,
        key=lambda n: (
            n.document.relative_path.casefold(),
            n.document.relative_path,
            n.document.doc_id,
        ),
    )

    for norm in sorted_docs:
        doc = norm.document
        doc_rows.append(
            {
                "doc_id": doc.doc_id,
                "repo_id": doc.repo_id,
                "revision": doc.revision,
                "relative_path": doc.relative_path,
                "company_code": doc.company_code,
                "report_year": doc.report_year,
                "statement_scope": doc.statement_scope,
                "sha256": doc.sha256,
                "file_size_bytes": doc.file_size_bytes,
                "encoding": doc.encoding,
                "inventory_status": doc.inventory_status,
                "ruleset_version": norm.ruleset_version,
                "normalization_fingerprint": norm.normalization_fingerprint,
            }
        )

        for tbl in norm.extraction.tables:
            t_rec = tbl.table
            table_rows.append(
                {
                    "table_id": t_rec.table_id,
                    "doc_id": t_rec.doc_id,
                    "title_raw": t_rec.title_raw,
                    "statement_type": t_rec.statement_type,
                    "unit_raw": t_rec.unit_raw,
                    "unit_normalized": t_rec.unit_normalized,
                    "line_start": t_rec.line_start,
                    "line_end": t_rec.line_end,
                    "row_count": t_rec.row_count,
                    "column_count": t_rec.column_count,
                    "quality_score": t_rec.quality_score,
                    "csv_path": t_rec.csv_path,
                }
            )

            for c in tbl.cells:
                cell_rows.append(
                    {
                        "cell_id": c.cell_id,
                        "table_id": c.table_id,
                        "row_idx": c.row_idx,
                        "col_idx": c.col_idx,
                        "row_label_raw": c.row_label_raw,
                        "row_label_canonical": c.row_label_canonical,
                        "column_label_raw": c.column_label_raw,
                        "column_label_canonical": c.column_label_canonical,
                        "value_raw": c.value_raw,
                        "value_numeric": c.value_numeric,
                        "period": c.period,
                        "unit": c.unit,
                        "source_line_start": c.source_line_start,
                        "source_line_end": c.source_line_end,
                        "extraction_confidence": c.extraction_confidence,
                    }
                )

        sorted_issues = sorted(norm.issues, key=issue_sort_key)
        for iss in sorted_issues:
            issue_rows.append(
                {
                    "code": iss.code,
                    "doc_id": iss.doc_id,
                    "table_id": iss.table_id,
                    "cell_id": iss.cell_id,
                    "field": iss.field,
                    "raw_value": iss.raw_value,
                }
            )

    table_rows.sort(key=lambda r: (str(r["doc_id"]), str(r["table_id"])))
    cell_rows.sort(
        key=lambda r: (
            str(r["table_id"]),
            int(str(r["row_idx"])),
            int(str(r["col_idx"])),
            str(r["cell_id"]),
        )
    )

    return FlattenedDataset(
        documents=tuple(doc_rows),
        tables=tuple(table_rows),
        cells=tuple(cell_rows),
        issues=tuple(issue_rows),
    )


def _build_canonical_source_table_ids(
    detection: DetectionResult,
    extraction: ExtractionResult,
) -> dict[tuple[int, int, int], str]:
    rejected_keys = {
        (item.ordinal, item.line_start, item.line_end)
        for item in extraction.rejected
        if item.kind == "html"
    }
    canonical_table_ids: dict[tuple[int, int, int], str] = {}
    for candidate in detection.candidates:
        if candidate.kind != "html":
            continue
        key = (candidate.ordinal, candidate.line_start, candidate.line_end)
        if key in rejected_keys:
            continue
        matching_tables = [
            table.table.table_id
            for table in extraction.tables
            if table.table.line_start <= candidate.line_start
            and table.table.line_end >= candidate.line_end
        ]
        if len(matching_tables) != 1:
            raise DatasetBuildError(
                "html candidate must map to exactly one canonical table"
            )
        canonical_table_ids[key] = matching_tables[0]
    return canonical_table_ids


def _source_occurrence_sort_key(
    row: dict[str, object],
) -> tuple[str, str, int, int, int, str]:
    relative_path = str(row["relative_path"])
    return (
        relative_path.casefold(),
        relative_path,
        int(row["ordinal"]),
        int(row["line_start"]),
        int(row["line_end"]),
        str(row["source_table_id"]),
    )


def _validate_source_table_occurrences(
    rows: Sequence[dict[str, object]],
) -> dict[str, int]:
    counts = {"canonical": 0, "rejected": 0, "duplicate": 0}
    seen_source_ids: set[str] = set()
    for row in rows:
        source_table_id = str(row["source_table_id"])
        if source_table_id in seen_source_ids:
            raise DatasetBuildError("source table occurrence IDs must be globally unique")
        seen_source_ids.add(source_table_id)

        status = str(row["status"])
        if status not in counts:
            raise DatasetBuildError(f"unknown source table occurrence status: {status}")
        counts[status] += 1

        canonical_table_id = row["canonical_table_id"]
        rejection_code = row["rejection_code"]
        duplicate_of_relative_path = row["duplicate_of_relative_path"]
        if status == "canonical":
            if canonical_table_id is None:
                raise DatasetBuildError(
                    "canonical source table occurrences require canonical_table_id"
                )
            if rejection_code is not None or duplicate_of_relative_path is not None:
                raise DatasetBuildError(
                    "canonical source table occurrences cannot carry"
                    " rejection or duplicate metadata"
                )
        elif status == "rejected":
            if canonical_table_id is not None or rejection_code is None:
                raise DatasetBuildError(
                    "rejected source table occurrences must have only a rejection code"
                )
            if duplicate_of_relative_path is not None:
                raise DatasetBuildError(
                    "rejected source table occurrences cannot carry duplicate metadata"
                )
        else:
            if canonical_table_id is not None or rejection_code is not None:
                raise DatasetBuildError(
                    "duplicate source table occurrences cannot carry canonical or rejection data"
                )
            if duplicate_of_relative_path is None:
                raise DatasetBuildError(
                    "duplicate source table occurrences require duplicate_of_relative_path"
                )

    if sum(counts.values()) != len(rows):
        raise DatasetBuildError("source table occurrence counts must sum to total rows")
    return counts


def _duplicate_primary_relative_path(document: DocumentRecord) -> str:
    duplicate_notes = [
        note.removeprefix("duplicate_of=")
        for note in document.notes
        if note.startswith("duplicate_of=")
    ]
    if len(duplicate_notes) != 1 or not duplicate_notes[0]:
        raise DatasetBuildError(
            "duplicate document must declare exactly one"
            f" duplicate_of note: {document.relative_path}"
        )
    return duplicate_notes[0]


def build_dataset(config: DatasetBuildConfig) -> DatasetBuildResult:
    manifest_snapshot = read_manifest(config.manifest_path)

    ready_documents = [
        doc
        for doc in manifest_snapshot.inventory.documents
        if doc.inventory_status == "ready"
    ]
    duplicate_documents = [
        doc
        for doc in manifest_snapshot.inventory.documents
        if doc.inventory_status == "duplicate"
    ]

    normalized_docs: list[NormalizedDocument] = []
    ready_occurrence_rows_by_path: dict[str, list[dict[str, object]]] = {}
    ready_documents_by_path = {doc.relative_path: doc for doc in ready_documents}
    source_table_occurrence_rows: list[dict[str, object]] = []
    for doc in ready_documents:
        file_path = config.snapshot_root / doc.relative_path
        if not file_path.is_file():
            raise DatasetBuildError(
                f"document file missing from snapshot: {file_path}"
            )

        decoded_document = read_document(config.snapshot_root, doc)
        detection_result = detect_table_candidates(decoded_document)
        extraction_result = extract_candidates(decoded_document, detection_result)
        norm_doc = normalize_extraction(doc, extraction_result)
        normalized_docs.append(norm_doc)
        occurrence_rows = build_source_table_occurrences(
            decoded_document,
            detection_result,
            extraction_result,
            _build_canonical_source_table_ids(detection_result, extraction_result),
        )
        ready_occurrence_rows_by_path[doc.relative_path] = occurrence_rows
        source_table_occurrence_rows.extend(occurrence_rows)

    for doc in duplicate_documents:
        duplicate_of_relative_path = _duplicate_primary_relative_path(doc)
        primary_document = ready_documents_by_path.get(duplicate_of_relative_path)
        if primary_document is None:
            raise DatasetBuildError(
                f"duplicate document primary layout missing: {doc.relative_path}"
            )
        if doc.sha256 != primary_document.sha256:
            raise DatasetBuildError(
                f"duplicate document sha256 mismatch: {doc.relative_path}"
            )
        primary_rows = ready_occurrence_rows_by_path.get(duplicate_of_relative_path)
        if primary_rows is None:
            raise DatasetBuildError(
                f"duplicate document primary occurrence layout missing: {doc.relative_path}"
            )
        source_table_occurrence_rows.extend(
            build_duplicate_source_table_occurrences(
                doc,
                primary_rows,
                duplicate_of_relative_path,
            )
        )

    flattened = flatten_normalized_documents(tuple(normalized_docs))
    source_table_occurrences = tuple(
        sorted(source_table_occurrence_rows, key=_source_occurrence_sort_key)
    )
    flattened = FlattenedDataset(
        documents=flattened.documents,
        tables=flattened.tables,
        cells=flattened.cells,
        issues=flattened.issues,
        source_table_occurrences=source_table_occurrences,
    )
    source_table_occurrence_counts = _validate_source_table_occurrences(
        flattened.source_table_occurrences
    )

    # Calculate dataset payload fingerprint
    def _json_safe(obj: object) -> object:
        """Convert non-JSON-serializable types for deterministic fingerprinting."""
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        return obj

    payload = _json_safe({
        "schema_version": config.schema_version,
        "source_manifest_sha256": manifest_snapshot.sha256,
        "documents": flattened.documents,
        "tables": flattened.tables,
        "cells": flattened.cells,
        "issues": flattened.issues,
        "source_table_occurrences": flattened.source_table_occurrences,
    })
    dataset_fingerprint = hashlib.sha256(
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()

    release_dir = (
        config.processed_root
        / f"release_v{config.schema_version}_{dataset_fingerprint[:12]}"
    )
    config.processed_root.mkdir(parents=True, exist_ok=True)

    # Write tables to temporary directory first
    temp_dir = Path(tempfile.mkdtemp(dir=config.processed_root, prefix=".tmp_release_"))
    try:
        doc_table = pa.Table.from_pylist(list(flattened.documents), schema=DOCUMENT_SCHEMA)
        table_table = pa.Table.from_pylist(list(flattened.tables), schema=TABLE_SCHEMA)
        cell_table = pa.Table.from_pylist(list(flattened.cells), schema=CELL_SCHEMA)
        issue_table = pa.Table.from_pylist(list(flattened.issues), schema=ISSUE_SCHEMA)
        source_table_occurrence_table = pa.Table.from_pylist(
            list(flattened.source_table_occurrences),
            schema=SOURCE_TABLE_OCCURRENCE_SCHEMA,
        )

        pq.write_table(
            doc_table, temp_dir / "documents.parquet", compression="snappy"
        )  # type: ignore[no-untyped-call]
        pq.write_table(
            table_table, temp_dir / "tables.parquet", compression="snappy"
        )  # type: ignore[no-untyped-call]
        pq.write_table(
            cell_table, temp_dir / "cells.parquet", compression="snappy"
        )  # type: ignore[no-untyped-call]
        pq.write_table(
            issue_table, temp_dir / "issues.parquet", compression="snappy"
        )  # type: ignore[no-untyped-call]
        pq.write_table(
            source_table_occurrence_table,
            temp_dir / "source_table_occurrences.parquet",
            compression="snappy",
        )  # type: ignore[no-untyped-call]


        issue_counts: dict[str, int] = {}
        for iss in flattened.issues:
            code = str(iss["code"])
            issue_counts[code] = issue_counts.get(code, 0) + 1

        release_manifest = {
            "schema_version": config.schema_version,
            "dataset_fingerprint": dataset_fingerprint,
            "source_manifest_sha256": manifest_snapshot.sha256,
            "document_count": len(flattened.documents),
            "table_count": len(flattened.tables),
            "cell_count": len(flattened.cells),
            "issue_count": len(flattened.issues),
            "issue_counts_by_code": issue_counts,
            "source_table_occurrence_counts": {
                "total": len(flattened.source_table_occurrences),
                **source_table_occurrence_counts,
            },
        }
        (temp_dir / "manifest.json").write_bytes(
            json.dumps(
                release_manifest, ensure_ascii=False, indent=2, sort_keys=True
            ).encode("utf-8")
        )

        if release_dir.exists():
            import shutil

            shutil.rmtree(release_dir)
        temp_dir.replace(release_dir)
    except Exception:
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return DatasetBuildResult(
        release_path=release_dir,
        dataset_fingerprint=dataset_fingerprint,
        source_manifest_sha256=manifest_snapshot.sha256,
        document_count=len(flattened.documents),
        table_count=len(flattened.tables),
        cell_count=len(flattened.cells),
        issue_count=len(flattened.issues),
        issue_counts_by_code=issue_counts,
    )
