import hashlib
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DatasetBuildConfig,
    build_dataset,
    flatten_normalized_documents,
)
from financial_report_qa.data.inventory import InventoryResult
from financial_report_qa.data.manifests import write_manifest
from financial_report_qa.ingestion.provenance import ExtractionResult
from financial_report_qa.normalization._shared import RULESET_VERSION
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.normalization import NormalizedDocument


def test_cell_schema_uses_fixed_decimal_and_no_absolute_paths() -> None:
    assert CELL_SCHEMA.field("value_numeric").type == pa.decimal128(38, 10)
    assert CELL_SCHEMA.field("source_line_start").type == pa.int32()
    assert "absolute_path" not in CELL_SCHEMA.names


def test_flattened_rows_have_stable_order() -> None:
    def normalized_document(suffix: Literal["a", "b"]) -> NormalizedDocument:
        digest = suffix * 64
        document = DocumentRecord(
            doc_id=stable_document_id(digest),
            repo_id="org/vifinqa",
            revision="rev-1",
            relative_path=f"VCB/2024/Consolidated/{suffix}.txt",
            company_code="VCB",
            report_year=2024,
            statement_scope="consolidated",
            sha256=digest,
            file_size_bytes=1,
            encoding="utf-8",
            inventory_status="ready",
        )
        extraction = ExtractionResult(
            doc_id=document.doc_id, blocks=(), tables=(), rejected=()
        )
        return NormalizedDocument(
            document=document,
            extraction=extraction,
            issues=(),
            ruleset_version=RULESET_VERSION,
            normalization_fingerprint=digest,
        )

    rows = flatten_normalized_documents(
        (normalized_document("b"), normalized_document("a"))
    )
    paths = [str(row["relative_path"]) for row in rows.documents]
    assert paths == sorted(paths)

    assert rows.cells == tuple(
        sorted(
            rows.cells,
            key=lambda row: (
                str(row["table_id"]),
                int(str(row["row_idx"])),
                int(str(row["col_idx"])),
                str(row["cell_id"]),
            ),
        )
    )


def test_build_dataset_creates_atomic_parquet_release(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    doc_path = snapshot_root / "VCB/2024/Consolidated/report.txt"
    doc_path.parent.mkdir(parents=True)
    content = (
        "<table><tr><td>Doanh thu bán hàng và cung cấp dịch vụ</td>"
        "<td>1.500</td></tr></table>"
    )
    doc_path.write_text(content, encoding="utf-8")
    raw_bytes = doc_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    manifest_path = tmp_path / "documents.jsonl"
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="ready",
    )

    write_manifest(InventoryResult(documents=(document,), issues=()), manifest_path)

    config = DatasetBuildConfig(
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
        processed_root=tmp_path / "processed",
    )
    result = build_dataset(config)

    assert result.release_path.exists()
    assert (result.release_path / "cells.parquet").exists()
    assert (result.release_path / "manifest.json").exists()
    cell_table = pq.read_table(
        result.release_path / "cells.parquet"
    )  # type: ignore[no-untyped-call]
    assert CELL_SCHEMA.equals(cell_table.schema)


