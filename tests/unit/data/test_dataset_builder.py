import hashlib
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DatasetBuildConfig,
    SOURCE_TABLE_OCCURRENCE_SCHEMA,
    build_dataset,
    build_source_table_occurrences,
    flatten_normalized_documents,
)
from financial_report_qa.data.inventory import InventoryResult
from financial_report_qa.data.manifests import write_manifest
from financial_report_qa.ingestion.provenance import (
    DecodedDocument,
    DetectionResult,
    ExtractionResult,
    RejectedCandidate,
    SourceLine,
    TableCandidate,
    TextBlock,
)
from financial_report_qa.normalization._shared import RULESET_VERSION
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.normalization import NormalizedDocument


def test_cell_schema_uses_fixed_decimal_and_no_absolute_paths() -> None:
    assert CELL_SCHEMA.field("value_numeric").type == pa.decimal128(38, 10)
    assert CELL_SCHEMA.field("source_line_start").type == pa.int32()
    assert "absolute_path" not in CELL_SCHEMA.names


def test_source_table_occurrence_schema_matches_task_1_contract() -> None:
    assert SOURCE_TABLE_OCCURRENCE_SCHEMA == pa.schema(
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


def test_build_source_table_occurrences_tracks_rejection_and_continuation() -> None:
    digest = "a" * 64
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=120,
        encoding="utf-8",
        inventory_status="ready",
    )
    decoded_document = DecodedDocument(
        document=document,
        text="\n".join(
            [
                "<table><tr><td>A</td></tr></table>",
                "<table><tr><td>B</td></tr></table>",
                "<table><div>bad</div></table>",
            ]
        )
        + "\n",
        lines=tuple(
            SourceLine(number=index, text=line, line_ending="\n")
            for index, line in enumerate(
                (
                    "<table><tr><td>A</td></tr></table>",
                    "<table><tr><td>B</td></tr></table>",
                    "<table><div>bad</div></table>",
                ),
                start=1,
            )
        ),
        blocks=(
            TextBlock(
                kind="table",
                line_start=1,
                line_end=3,
                text=(
                    "<table><tr><td>A</td></tr></table>\n"
                    "<table><tr><td>B</td></tr></table>\n"
                    "<table><div>bad</div></table>"
                ),
            ),
        ),
    )
    detection = DetectionResult(
        candidates=(
            TableCandidate(
                ordinal=0,
                kind="html",
                raw_source="<table><tr><td>A</td></tr></table>\n",
                line_start=1,
                line_end=1,
                confidence=1.0,
                evidence=("html_table_marker",),
            ),
            TableCandidate(
                ordinal=1,
                kind="html",
                raw_source="<table><tr><td>B</td></tr></table>\n",
                line_start=2,
                line_end=2,
                confidence=1.0,
                evidence=("html_table_marker",),
            ),
            TableCandidate(
                ordinal=2,
                kind="structured_text",
                raw_source="skip me",
                line_start=10,
                line_end=12,
                confidence=0.8,
                evidence=("numeric_density",),
            ),
            TableCandidate(
                ordinal=3,
                kind="html",
                raw_source="<table><div>bad</div></table>\n",
                line_start=3,
                line_end=3,
                confidence=1.0,
                evidence=("html_table_marker",),
            ),
        ),
        rejected=(),
        blocks=decoded_document.blocks,
    )
    extraction = ExtractionResult(
        doc_id=document.doc_id,
        blocks=decoded_document.blocks,
        tables=(),
        rejected=(
            RejectedCandidate(
                ordinal=3,
                kind="html",
                raw_source="<table><div>bad</div></table>\n",
                line_start=3,
                line_end=3,
                reason="unsupported_html_structure",
            ),
        ),
    )

    rows = build_source_table_occurrences(
        decoded_document,
        detection,
        extraction,
        {
            (0, 1, 1): "table-1",
            (1, 2, 2): "table-1",
        },
    )

    assert [row["status"] for row in rows] == ["canonical", "canonical", "rejected"]
    assert [row["canonical_table_id"] for row in rows] == ["table-1", "table-1", None]
    assert rows[2]["rejection_code"] == "unsupported_html_structure"
    assert all(row["duplicate_of_relative_path"] is None for row in rows)
    assert len({str(row["source_table_id"]) for row in rows}) == 3


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

