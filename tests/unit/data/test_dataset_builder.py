import hashlib
import json
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    SOURCE_TABLE_OCCURRENCE_SCHEMA,
    DatasetBuildConfig,
    build_dataset,
    build_duplicate_source_table_occurrences,
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


def test_duplicate_rows_reuse_layout_but_not_canonical_data() -> None:
    digest = "b" * 64
    duplicate_doc = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="SSH/2024/duplicate.txt",
        company_code="SSH",
        report_year=2024,
        statement_scope="separate",
        sha256=digest,
        file_size_bytes=120,
        encoding="utf-8",
        inventory_status="duplicate",
        notes=("duplicate_of=SSH/2024/primary.txt",),
    )
    primary_rows = [
        {
            "source_table_id": "primary-source-1",
            "doc_id": stable_document_id(digest),
            "relative_path": "SSH/2024/primary.txt",
            "source_sha256": digest,
            "ordinal": 0,
            "line_start": 10,
            "line_end": 20,
            "status": "canonical",
            "canonical_table_id": "table-1",
            "rejection_code": None,
            "duplicate_of_relative_path": None,
        },
        {
            "source_table_id": "primary-source-2",
            "doc_id": stable_document_id(digest),
            "relative_path": "SSH/2024/primary.txt",
            "source_sha256": digest,
            "ordinal": 1,
            "line_start": 21,
            "line_end": 30,
            "status": "rejected",
            "canonical_table_id": None,
            "rejection_code": "unsupported_html_structure",
            "duplicate_of_relative_path": None,
        },
    ]

    rows = build_duplicate_source_table_occurrences(
        duplicate_doc,
        primary_rows,
        "SSH/2024/primary.txt",
    )

    assert {row["status"] for row in rows} == {"duplicate"}
    assert all(row["canonical_table_id"] is None for row in rows)
    assert all(row["rejection_code"] is None for row in rows)
    assert {row["duplicate_of_relative_path"] for row in rows} == {"SSH/2024/primary.txt"}
    assert all(row["doc_id"] == duplicate_doc.doc_id for row in rows)
    assert all(row["relative_path"] == duplicate_doc.relative_path for row in rows)
    assert all(row["source_sha256"] == duplicate_doc.sha256 for row in rows)
    assert [(row["ordinal"], row["line_start"], row["line_end"]) for row in rows] == [
        (0, 10, 20),
        (1, 21, 30),
    ]
    assert len({str(row["source_table_id"]) for row in rows}) == 2
    assert {str(row["source_table_id"]) for row in rows}.isdisjoint(
        {str(row["source_table_id"]) for row in primary_rows}
    )


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
        extraction = ExtractionResult(doc_id=document.doc_id, blocks=(), tables=(), rejected=())
        return NormalizedDocument(
            document=document,
            extraction=extraction,
            issues=(),
            ruleset_version=RULESET_VERSION,
            normalization_fingerprint=digest,
        )

    rows = flatten_normalized_documents((normalized_document("b"), normalized_document("a")))
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
        "<table><tr><td>Doanh thu bán hàng và cung cấp dịch vụ</td><td>1.500</td></tr></table>"
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
    cell_table = pq.read_table(result.release_path / "cells.parquet")  # type: ignore[no-untyped-call]
    assert CELL_SCHEMA.equals(cell_table.schema)


def test_build_dataset_emits_source_occurrence_artifact_with_duplicate_rows(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    primary_rel = "VCB/2024/Consolidated/primary.txt"
    duplicate_rel = "VCB/2024/Consolidated/duplicate.txt"
    content = "<table><tr><td>Doanh thu</td><td>1.500</td></tr></table>\n"

    primary_path = snapshot_root / primary_rel
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text(content, encoding="utf-8")

    duplicate_path = snapshot_root / duplicate_rel
    duplicate_path.write_text(content, encoding="utf-8")

    raw_bytes = primary_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    primary_doc = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=primary_rel,
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="ready",
    )
    duplicate_doc = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=duplicate_rel,
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="duplicate",
        notes=(f"duplicate_of={primary_rel}",),
    )

    manifest_path = tmp_path / "documents.jsonl"
    write_manifest(
        InventoryResult(documents=(primary_doc, duplicate_doc), issues=()),
        manifest_path,
    )

    result = build_dataset(
        DatasetBuildConfig(
            snapshot_root=snapshot_root,
            manifest_path=manifest_path,
            processed_root=tmp_path / "processed",
        )
    )

    occurrence_table = pq.read_table(result.release_path / "source_table_occurrences.parquet")  # type: ignore[no-untyped-call]
    assert SOURCE_TABLE_OCCURRENCE_SCHEMA.equals(occurrence_table.schema)
    assert occurrence_table.num_rows == 2
    assert occurrence_table.column("status").to_pylist() == ["duplicate", "canonical"]
    assert occurrence_table.column("canonical_table_id").to_pylist().count(None) == 1
    assert occurrence_table.column("duplicate_of_relative_path").to_pylist() == [
        primary_rel,
        None,
    ]
    assert len(set(occurrence_table.column("source_table_id").to_pylist())) == 2

    manifest = json.loads((result.release_path / "manifest.json").read_text("utf-8"))
    assert manifest["source_table_occurrence_counts"] == {
        "total": 2,
        "canonical": 1,
        "rejected": 0,
        "duplicate": 1,
    }


def test_build_source_table_occurrences_raises_on_html_candidate_without_outcome() -> None:
    """ValueError when an HTML candidate maps to neither canonical nor rejected."""
    digest = "d" * 64
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="XX/2024/report.txt",
        company_code="XX",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=10,
        encoding="utf-8",
        inventory_status="ready",
    )
    decoded_document = DecodedDocument(
        document=document,
        text="<table><tr><td>A</td></tr></table>\n",
        lines=(SourceLine(number=1, text="<table><tr><td>A</td></tr></table>", line_ending="\n"),),
        blocks=(
            TextBlock(
                kind="table",
                line_start=1,
                line_end=1,
                text="<table><tr><td>A</td></tr></table>",
            ),
        ),
    )
    detection = DetectionResult(
        candidates=(
            TableCandidate(
                ordinal=0,
                kind="html",
                raw_source="<table><tr><td>A</td></tr></table>",
                line_start=1,
                line_end=1,
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
        rejected=(),  # no rejection either!
    )

    import pytest

    with pytest.raises(ValueError, match="canonical or rejected outcome"):
        build_source_table_occurrences(decoded_document, detection, extraction, {})


def test_build_dataset_rejects_malformed_duplicate_note(tmp_path: Path) -> None:
    """DatasetBuildError when duplicate_of note is missing or malformed."""
    from financial_report_qa.core.errors import DatasetBuildError

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    content = "<table><tr><td>A</td></tr></table>\n"
    doc_path = snapshot_root / "XX/2024/report.txt"
    doc_path.parent.mkdir(parents=True)
    doc_path.write_text(content, encoding="utf-8")
    raw_bytes = doc_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    # Duplicate with no duplicate_of= note
    doc_ready = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="XX/2024/report.txt",
        company_code="XX",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="ready",
    )
    doc_bad_dup = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="XX/2024/dup.txt",
        company_code="XX",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="duplicate",
        notes=(),  # missing duplicate_of=
    )
    dup_path = snapshot_root / "XX/2024/dup.txt"
    dup_path.write_text(content, encoding="utf-8")

    manifest_path = tmp_path / "documents.jsonl"
    write_manifest(InventoryResult(documents=(doc_ready, doc_bad_dup), issues=()), manifest_path)

    import pytest

    with pytest.raises(DatasetBuildError, match="duplicate_of note"):
        build_dataset(
            DatasetBuildConfig(
                snapshot_root=snapshot_root,
                manifest_path=manifest_path,
                processed_root=tmp_path / "processed",
            )
        )


def test_build_dataset_rejects_duplicate_sha_mismatch(tmp_path: Path) -> None:
    """DatasetBuildError when duplicate SHA doesn't match primary."""
    from financial_report_qa.core.errors import DatasetBuildError

    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()

    content = "<table><tr><td>A</td></tr></table>\n"
    primary_path = snapshot_root / "XX/2024/primary.txt"
    primary_path.parent.mkdir(parents=True)
    primary_path.write_text(content, encoding="utf-8")
    raw_bytes = primary_path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    different_digest = "f" * 64  # deliberately wrong

    doc_ready = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="XX/2024/primary.txt",
        company_code="XX",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="ready",
    )
    doc_dup = DocumentRecord(
        doc_id=stable_document_id(different_digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="XX/2024/dup.txt",
        company_code="XX",
        report_year=2024,
        statement_scope="consolidated",
        sha256=different_digest,  # different from primary
        file_size_bytes=len(raw_bytes),
        encoding="utf-8",
        inventory_status="duplicate",
        notes=("duplicate_of=XX/2024/primary.txt",),
    )
    dup_path = snapshot_root / "XX/2024/dup.txt"
    dup_path.write_text(content, encoding="utf-8")

    manifest_path = tmp_path / "documents.jsonl"
    write_manifest(InventoryResult(documents=(doc_ready, doc_dup), issues=()), manifest_path)

    import pytest

    with pytest.raises(DatasetBuildError, match="sha256 mismatch"):
        build_dataset(
            DatasetBuildConfig(
                snapshot_root=snapshot_root,
                manifest_path=manifest_path,
                processed_root=tmp_path / "processed",
            )
        )


def test_validate_source_table_occurrences_rejects_unknown_status() -> None:
    """DatasetBuildError when occurrence row has an invalid status."""
    import pytest

    from financial_report_qa.core.errors import DatasetBuildError
    from financial_report_qa.data.dataset_builder import _validate_source_table_occurrences

    rows: list[dict[str, object]] = [
        {
            "source_table_id": "id1",
            "status": "invalid_status",
            "canonical_table_id": None,
            "rejection_code": None,
            "duplicate_of_relative_path": None,
        }
    ]
    with pytest.raises(DatasetBuildError, match="unknown source table occurrence status"):
        _validate_source_table_occurrences(rows)


def test_validate_source_table_occurrences_rejects_canonical_without_table_id() -> None:
    """DatasetBuildError when a canonical row is missing canonical_table_id."""
    import pytest

    from financial_report_qa.core.errors import DatasetBuildError
    from financial_report_qa.data.dataset_builder import _validate_source_table_occurrences

    rows: list[dict[str, object]] = [
        {
            "source_table_id": "id1",
            "status": "canonical",
            "canonical_table_id": None,  # missing!
            "rejection_code": None,
            "duplicate_of_relative_path": None,
        }
    ]
    with pytest.raises(DatasetBuildError, match="require canonical_table_id"):
        _validate_source_table_occurrences(rows)


def test_validate_source_table_occurrences_rejects_duplicate_without_path() -> None:
    """DatasetBuildError when a duplicate row is missing duplicate_of_relative_path."""
    import pytest

    from financial_report_qa.core.errors import DatasetBuildError
    from financial_report_qa.data.dataset_builder import _validate_source_table_occurrences

    rows: list[dict[str, object]] = [
        {
            "source_table_id": "id1",
            "status": "duplicate",
            "canonical_table_id": None,
            "rejection_code": None,
            "duplicate_of_relative_path": None,  # missing!
        }
    ]
    with pytest.raises(DatasetBuildError, match="require duplicate_of_relative_path"):
        _validate_source_table_occurrences(rows)


def test_validate_source_table_occurrences_rejects_globally_non_unique_ids() -> None:
    """DatasetBuildError when two occurrence rows share the same source_table_id."""
    import pytest

    from financial_report_qa.core.errors import DatasetBuildError
    from financial_report_qa.data.dataset_builder import _validate_source_table_occurrences

    rows: list[dict[str, object]] = [
        {
            "source_table_id": "duplicate-id",
            "status": "canonical",
            "canonical_table_id": "tbl-1",
            "rejection_code": None,
            "duplicate_of_relative_path": None,
        },
        {
            "source_table_id": "duplicate-id",  # same as above
            "status": "canonical",
            "canonical_table_id": "tbl-2",
            "rejection_code": None,
            "duplicate_of_relative_path": None,
        },
    ]
    with pytest.raises(DatasetBuildError, match="globally unique"):
        _validate_source_table_occurrences(rows)
