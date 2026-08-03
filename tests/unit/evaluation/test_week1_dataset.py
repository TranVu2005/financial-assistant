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
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.week1_dataset import load_gate_dataset
from financial_report_qa.ingestion.provenance import stable_cell_id
from financial_report_qa.schemas import (
    CellRecord,
    DocumentRecord,
    TableRecord,
    stable_document_id,
    stable_table_id,
)


def _write_release(
    tmp_path: Path,
) -> tuple[Path, Path, DocumentRecord, TableRecord, CellRecord]:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "documents.jsonl"
    doc_raw = {
        "record_type": "document",
        "doc_id": stable_document_id("a" * 64),
        "repo_id": "test_repo",
        "revision": "main",
        "relative_path": "VCB/2024/Consolidated/report.txt",
        "company_code": "VCB",
        "report_year": 2024,
        "statement_scope": "consolidated",
        "sha256": "a" * 64,
        "file_size_bytes": 1000,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "notes": [],
    }
    manifest_path.write_text(json.dumps(doc_raw) + "\n", encoding="utf-8")
    doc_id = stable_document_id("a" * 64)

    release_path = tmp_path / "release"
    release_path.mkdir()

    document = DocumentRecord(
        doc_id=doc_id,
        repo_id="test_repo",
        revision="main",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256="a" * 64,
        file_size_bytes=1000,
        encoding="utf-8",
        inventory_status="ready",
    )
    table_id = stable_table_id(doc_id, 10, 20)
    table = TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw="Balance Sheet",
        statement_type="balance_sheet",
        unit_raw="VND",
        unit_normalized="VND",
        line_start=10,
        line_end=20,
        row_count=2,
        column_count=2,
        quality_score=1.0,
        csv_path=None,
    )
    cell_id = stable_cell_id(table_id, 0, 0)
    from decimal import Decimal
    from typing import Any, cast

    write_table = cast(Any, pq.write_table)

    cell = CellRecord(
        cell_id=cell_id,
        table_id=table_id,
        row_idx=0,
        col_idx=0,
        row_label_raw="Asset",
        row_label_canonical="Asset",
        column_label_raw="2024",
        column_label_canonical="2024",
        value_raw="100",
        value_numeric=Decimal("100"),
        period="2024",
        unit="VND",
        source_line_start=11,
        source_line_end=11,
        extraction_confidence=1.0,
    )

    doc_table = pa.Table.from_pylist(
        [
            {
                "doc_id": doc_id,
                "repo_id": "test_repo",
                "revision": "main",
                "relative_path": "VCB/2024/Consolidated/report.txt",
                "company_code": "VCB",
                "report_year": 2024,
                "statement_scope": "consolidated",
                "sha256": "a" * 64,
                "file_size_bytes": 1000,
                "encoding": "utf-8",
                "inventory_status": "ready",
                "ruleset_version": "1",
                "normalization_fingerprint": "b" * 64,
            }
        ],
        schema=DOCUMENT_SCHEMA,
    )
    write_table(doc_table, release_path / "documents.parquet")

    tbl_table = pa.Table.from_pylist(
        [
            {
                "table_id": table_id,
                "doc_id": doc_id,
                "title_raw": "Balance Sheet",
                "statement_type": "balance_sheet",
                "unit_raw": "VND",
                "unit_normalized": "VND",
                "line_start": 10,
                "line_end": 20,
                "row_count": 2,
                "column_count": 2,
                "quality_score": 1.0,
                "csv_path": None,
            }
        ],
        schema=TABLE_SCHEMA,
    )
    write_table(tbl_table, release_path / "tables.parquet")

    cell_table = pa.Table.from_pylist(
        [
            {
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
            }
        ],
        schema=CELL_SCHEMA,
    )
    write_table(cell_table, release_path / "cells.parquet")

    issue_table = pa.Table.from_pylist([], schema=ISSUE_SCHEMA)
    write_table(issue_table, release_path / "issues.parquet")

    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_hash,
        "document_count": 1,
        "table_count": 1,
        "cell_count": 1,
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(json.dumps(release_manifest), encoding="utf-8")

    return manifest_path, release_path, document, table, cell


def _mutate_release(release_path: Path, mutation: str) -> None:
    if mutation == "missing_cells":
        (release_path / "cells.parquet").unlink()
    elif mutation == "source_hash":
        manifest_data = json.loads((release_path / "manifest.json").read_text(encoding="utf-8"))
        manifest_data["source_manifest_sha256"] = "0" * 64
        (release_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    elif mutation == "table_count":
        manifest_data = json.loads((release_path / "manifest.json").read_text(encoding="utf-8"))
        manifest_data["table_count"] = 999
        (release_path / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    elif mutation == "dangling_cell":
        from decimal import Decimal
        from typing import Any, cast

        import pyarrow as pa
        import pyarrow.parquet as pq

        from financial_report_qa.data.dataset_builder import CELL_SCHEMA

        bad_cell_table = pa.Table.from_pylist(
            [
                {
                    "cell_id": "cell_bad",
                    "table_id": "tbl_unknown",
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
                }
            ],
            schema=CELL_SCHEMA,
        )
        cast(Any, pq.write_table)(bad_cell_table, release_path / "cells.parquet")


def test_load_gate_dataset_indexes_verified_release(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)

    dataset = load_gate_dataset(manifest_path, release_path)

    assert dataset.dataset_fingerprint == "f" * 64
    assert dataset.source_manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert dataset.documents_by_id == {document.doc_id: document}
    assert dataset.tables_by_id == {table.table_id: table}
    assert dataset.cells_by_table_id == {table.table_id: (cell,)}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_cells", "Missing required release file: cells.parquet"),
        ("source_hash", "Source manifest fingerprint mismatch"),
        ("table_count", "Table count mismatch"),
        ("dangling_cell", "unknown table_id"),
    ],
)
def test_load_gate_dataset_fails_closed_on_release_corruption(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path, release_path, _, _, _ = _write_release(tmp_path)
    _mutate_release(release_path, mutation)
    with pytest.raises(Week1GateInputError, match=message):
        load_gate_dataset(manifest_path, release_path)
