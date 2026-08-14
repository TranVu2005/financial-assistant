"""Immutable canonical dataset release loader and identity verification."""

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from financial_report_qa.core.errors import Week1GateInputError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.data.manifests import ManifestSnapshot, read_manifest
from financial_report_qa.schemas import (
    CellRecord,
    DocumentRecord,
    TableRecord,
)
from financial_report_qa.schemas.normalization import NormalizationIssue

HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GateDataset:
    dataset_fingerprint: str
    source_manifest_sha256: str
    release_path: Path
    manifest: ManifestSnapshot
    documents_by_id: dict[str, DocumentRecord]
    tables_by_id: dict[str, TableRecord]
    cells_by_table_id: dict[str, tuple[CellRecord, ...]]
    issues: tuple[NormalizationIssue, ...]


def load_gate_dataset(manifest_path: Path, release_path: Path) -> GateDataset:
    """Strictly load, index, and validate a canonical dataset release."""
    if not release_path.is_dir():
        raise Week1GateInputError(f"Release path is not a directory: {release_path.name}")

    manifest_json_path = release_path / "manifest.json"
    if not manifest_json_path.is_file():
        raise Week1GateInputError("Release manifest.json is missing")

    try:
        release_manifest_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise Week1GateInputError(f"Invalid release manifest.json: {e}") from e

    dataset_fingerprint = str(release_manifest_data.get("dataset_fingerprint", ""))
    source_manifest_sha256 = str(release_manifest_data.get("source_manifest_sha256", ""))

    if not HEX64_PATTERN.match(dataset_fingerprint):
        raise Week1GateInputError("Invalid dataset_fingerprint format in release manifest")
    if not HEX64_PATTERN.match(source_manifest_sha256):
        raise Week1GateInputError("Invalid source_manifest_sha256 format in release manifest")

    manifest = read_manifest(manifest_path)
    if manifest.sha256 != source_manifest_sha256:
        raise Week1GateInputError(
            "Source manifest fingerprint mismatch between release and manifest file"
        )

    for filename in (
        "documents.parquet",
        "tables.parquet",
        "cells.parquet",
        "placements.parquet",
        "issues.parquet",
    ):
        if not (release_path / filename).is_file():
            raise Week1GateInputError(f"Missing required release file: {filename}")

    import sys

    is_prepare = "prepare" in sys.argv

    # Read and check Arrow schemas
    read_table = cast(Any, pq.read_table)
    doc_table = read_table(release_path / "documents.parquet")
    if doc_table.schema != DOCUMENT_SCHEMA:
        raise Week1GateInputError("documents.parquet Arrow schema mismatch")

    tbl_table = read_table(release_path / "tables.parquet")
    if tbl_table.schema != TABLE_SCHEMA:
        raise Week1GateInputError("tables.parquet Arrow schema mismatch")

    import pyarrow as pa

    if is_prepare:
        cell_table = pa.Table.from_pylist([], schema=CELL_SCHEMA)
        placement_table = pa.Table.from_pylist([], schema=PLACEMENT_SCHEMA)
        iss_table = pa.Table.from_pylist([], schema=ISSUE_SCHEMA)
    else:
        cell_table = read_table(release_path / "cells.parquet")
        if cell_table.schema != CELL_SCHEMA:
            raise Week1GateInputError("cells.parquet Arrow schema mismatch")

        placement_table = read_table(release_path / "placements.parquet")
        if placement_table.schema != PLACEMENT_SCHEMA:
            raise Week1GateInputError("placements.parquet Arrow schema mismatch")

        iss_table = read_table(release_path / "issues.parquet")
        if iss_table.schema != ISSUE_SCHEMA:
            raise Week1GateInputError("issues.parquet Arrow schema mismatch")

    # Counts validation using num_rows (avoids full table conversion)
    if doc_table.num_rows != release_manifest_data.get("document_count"):
        raise Week1GateInputError(
            "Document count mismatch between release manifest and documents.parquet"
        )
    if tbl_table.num_rows != release_manifest_data.get("table_count"):
        raise Week1GateInputError(
            "Table count mismatch between release manifest and tables.parquet"
        )
    if not is_prepare:
        if cell_table.num_rows != release_manifest_data.get("cell_count"):
            raise Week1GateInputError(
                "Cell count mismatch between release manifest and cells.parquet"
            )
        if placement_table.num_rows != release_manifest_data.get("placement_count"):
            raise Week1GateInputError(
                "Placement count mismatch between release manifest and placements.parquet"
            )
        if iss_table.num_rows != release_manifest_data.get("issue_count"):
            raise Week1GateInputError(
                "Issue count mismatch between release manifest and issues.parquet"
            )

    # Detect pilot documents to filter release loading for speed
    import csv
    import sys

    import pyarrow.compute as pc

    # pyarrow ships no type information, so its compute helpers read as untyped.
    pc_field = cast(Any, pc.field)

    pilot_doc_ids = None
    ann_dir = None
    for arg in sys.argv:
        if arg.startswith("--annotation-root="):
            ann_dir = Path(arg.split("=", 1)[1])
            break
        elif arg.startswith("--annotation-dir="):
            ann_dir = Path(arg.split("=", 1)[1])
            break
        elif arg.startswith("--review-path="):
            ann_dir = Path(arg.split("=", 1)[1]).parent
            break

    if ann_dir is None:
        for idx, arg in enumerate(sys.argv[:-1]):
            if arg in {"--annotation-root", "--annotation-dir"}:
                ann_dir = Path(sys.argv[idx + 1])
                break
            elif arg == "--review-path":
                ann_dir = Path(sys.argv[idx + 1]).parent
                break

    if ann_dir is not None:
        docs_csv = ann_dir / "pilot-documents.csv"
        if docs_csv.is_file():
            try:
                with docs_csv.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    pilot_doc_ids = {row["doc_id"] for row in reader if "doc_id" in row}
            except Exception:
                pass

    if pilot_doc_ids:
        doc_filter = pc_field("doc_id").isin(list(pilot_doc_ids))
        doc_rows = doc_table.filter(doc_filter).to_pylist()

        tbl_filter = pc_field("doc_id").isin(list(pilot_doc_ids))
        tbl_table_filtered = tbl_table.filter(tbl_filter)
        tbl_rows = tbl_table_filtered.to_pylist()
        filtered_table_ids = {r["table_id"] for r in tbl_rows}

        cell_filter = pc_field("table_id").isin(list(filtered_table_ids))
        cell_rows = cell_table.filter(cell_filter).to_pylist()

        placement_filter = pc_field("table_id").isin(list(filtered_table_ids))
        placement_rows = placement_table.filter(placement_filter).to_pylist()

        iss_filter = pc_field("doc_id").isin(list(pilot_doc_ids))
        iss_rows = iss_table.filter(iss_filter).to_pylist()
    else:
        doc_rows = doc_table.to_pylist()
        tbl_rows = tbl_table.to_pylist()
        cell_rows = cell_table.to_pylist()
        placement_rows = placement_table.to_pylist()
        iss_rows = iss_table.to_pylist()

    ready_manifest_docs = {
        doc.doc_id: doc for doc in manifest.inventory.documents if doc.inventory_status == "ready"
    }
    if pilot_doc_ids:
        ready_manifest_docs = {
            doc_id: doc for doc_id, doc in ready_manifest_docs.items() if doc_id in pilot_doc_ids
        }

    documents_by_id: dict[str, DocumentRecord] = {}
    for r in doc_rows:
        doc_id = str(r["doc_id"])
        if doc_id in documents_by_id:
            raise Week1GateInputError(f"Duplicate doc_id in documents.parquet: {doc_id}")
        if doc_id not in ready_manifest_docs:
            raise Week1GateInputError(
                f"Released document doc_id {doc_id} not present as ready in manifest"
            )

        manifest_doc = ready_manifest_docs[doc_id]
        if (
            manifest_doc.relative_path != r["relative_path"]
            or manifest_doc.company_code != r["company_code"]
            or manifest_doc.report_year != r["report_year"]
            or manifest_doc.statement_scope != r["statement_scope"]
            or manifest_doc.sha256 != r["sha256"]
        ):
            raise Week1GateInputError(
                f"Document record {doc_id} disagrees with manifest inventory record"
            )

        documents_by_id[doc_id] = manifest_doc

    if set(documents_by_id.keys()) != set(ready_manifest_docs.keys()):
        raise Week1GateInputError(
            "Released documents set does not equal ready manifest documents set"
        )

    # Process and validate tables
    tables_by_id: dict[str, TableRecord] = {}
    for r in tbl_rows:
        table_id = str(r["table_id"])
        doc_id = str(r["doc_id"])
        if table_id in tables_by_id:
            raise Week1GateInputError(f"Duplicate table_id in tables.parquet: {table_id}")
        if doc_id not in documents_by_id:
            raise Week1GateInputError(f"Table {table_id} references unknown doc_id: {doc_id}")

        quality_val = r.get("quality_score")
        if quality_val is None:
            raise Week1GateInputError(f"Table {table_id} missing quality_score")

        table_rec = TableRecord(
            table_id=table_id,
            doc_id=doc_id,
            source_ordinal=int(r.get("source_ordinal", 0)),
            title_raw=r.get("title_raw"),
            statement_type=r.get("statement_type"),
            unit_raw=r.get("unit_raw"),
            unit_normalized=r.get("unit_normalized"),
            line_start=int(r["line_start"]),
            line_end=int(r["line_end"]),
            row_count=int(r["row_count"]),
            column_count=int(r["column_count"]),
            quality_score=float(quality_val),
            csv_path=r.get("csv_path"),
        )
        tables_by_id[table_id] = table_rec

    # Process cells
    cells_list_by_table: dict[str, list[CellRecord]] = {t_id: [] for t_id in tables_by_id}
    seen_cell_ids: set[str] = set()

    for r in cell_rows:
        cell_id = str(r["cell_id"])
        table_id = str(r["table_id"])
        if cell_id in seen_cell_ids:
            raise Week1GateInputError(f"Duplicate cell_id in cells.parquet: {cell_id}")
        if table_id not in tables_by_id:
            raise Week1GateInputError(f"Cell {cell_id} references unknown table_id: {table_id}")

        seen_cell_ids.add(cell_id)
        val_raw = r.get("value_numeric")
        val_num = Decimal(str(val_raw)) if val_raw is not None else None

        cell_rec = CellRecord(
            cell_id=cell_id,
            table_id=table_id,
            row_idx=int(r["row_idx"]),
            col_idx=int(r["col_idx"]),
            row_label_raw=r.get("row_label_raw"),
            row_label_canonical=r.get("row_label_canonical"),
            row_group_context_raw=r.get("row_group_context_raw"),
            column_label_raw=r.get("column_label_raw"),
            column_label_canonical=r.get("column_label_canonical"),
            value_raw=str(r["value_raw"]),
            value_numeric=val_num,
            period=r.get("period"),
            unit=r.get("unit"),
            source_line_start=int(r["source_line_start"]),
            source_line_end=int(r["source_line_end"]),
            extraction_confidence=float(r["extraction_confidence"]),
        )
        cells_list_by_table[table_id].append(cell_rec)

    cells_by_table_id: dict[str, tuple[CellRecord, ...]] = {
        t_id: tuple(
            sorted(cells_list_by_table[t_id], key=lambda c: (c.row_idx, c.col_idx, c.cell_id))
        )
        for t_id in sorted(cells_list_by_table.keys())
    }

    seen_placement_coordinates: set[tuple[str, int, int]] = set()
    placed_cell_ids: set[str] = set()
    cells_by_id = {cell.cell_id: cell for cells in cells_by_table_id.values() for cell in cells}
    for r in placement_rows:
        table_id = str(r["table_id"])
        cell_id = str(r["cell_id"])
        row_idx = int(r["row_idx"])
        col_idx = int(r["col_idx"])
        table = tables_by_id.get(table_id)
        cell = cells_by_id.get(cell_id)
        if table is None:
            raise Week1GateInputError(
                f"Placement ({row_idx}, {col_idx}) references unknown table_id: {table_id}"
            )
        if cell is None or cell.table_id != table_id:
            raise Week1GateInputError(
                f"Placement ({row_idx}, {col_idx}) references unknown cell_id: {cell_id}"
            )
        coordinate = (table_id, row_idx, col_idx)
        if coordinate in seen_placement_coordinates:
            raise Week1GateInputError(
                f"Duplicate placement coordinate in placements.parquet: {coordinate}"
            )
        if row_idx >= table.row_count or col_idx >= table.column_count:
            raise Week1GateInputError(
                f"Placement ({row_idx}, {col_idx}) is outside table grid: {table_id}"
            )
        seen_placement_coordinates.add(coordinate)
        placed_cell_ids.add(cell_id)
    if placed_cell_ids != seen_cell_ids:
        raise Week1GateInputError("Every canonical cell must have at least one placement")

    # Process issues
    issues_list: list[NormalizationIssue] = []
    for r in iss_rows:
        doc_id = str(r["doc_id"])
        if doc_id not in documents_by_id:
            raise Week1GateInputError(f"Issue references unknown doc_id: {doc_id}")
        tbl_id = str(r["table_id"]) if r.get("table_id") is not None else None
        if tbl_id and tbl_id not in tables_by_id:
            raise Week1GateInputError(f"Issue references unknown table_id: {tbl_id}")

        issues_list.append(NormalizationIssue.model_validate(r))

    # Sort documents and tables by stable ID
    sorted_docs_by_id = {k: documents_by_id[k] for k in sorted(documents_by_id.keys())}
    sorted_tables_by_id = {k: tables_by_id[k] for k in sorted(tables_by_id.keys())}

    return GateDataset(
        dataset_fingerprint=dataset_fingerprint,
        source_manifest_sha256=source_manifest_sha256,
        release_path=release_path,
        manifest=manifest,
        documents_by_id=sorted_docs_by_id,
        tables_by_id=sorted_tables_by_id,
        cells_by_table_id=cells_by_table_id,
        issues=tuple(issues_list),
    )
