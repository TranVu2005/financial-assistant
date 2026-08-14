"""Unit tests for cell provenance auditing and table usability evaluation."""

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import Week1GateSourceError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    ISSUE_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.evaluation.week1_contracts import (
    SAMPLING_VERSION,
    CellAudit,
    ExpectedTable,
    TableAssessment,
    stable_annotation_id,
)
from financial_report_qa.evaluation.week1_dataset import GateDataset, load_gate_dataset
from financial_report_qa.evaluation.week1_provenance import (
    audit_cell_provenance,
    evaluate_table_usability,
    generate_cell_audits,
)
from financial_report_qa.ingestion.provenance import stable_cell_id
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas import (
    CellRecord,
    DocumentRecord,
    TableRecord,
    stable_document_id,
    stable_table_id,
)

# pyarrow ships no type information, so its writer reads as an untyped call.
_write_table = cast(Any, pq.write_table)


def test_audit_cell_provenance_success() -> None:
    doc_lines = ("Line 1", "Line 2: Revenue 1000", "Line 3")
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)
    cell = CellRecord(
        cell_id="cell_1",
        table_id=tbl_id,
        row_idx=0,
        col_idx=0,
        row_label_raw="Revenue",
        row_label_canonical="Revenue",
        column_label_raw="2024",
        column_label_canonical="2024",
        value_raw="1000",
        value_numeric=Decimal("1000"),
        period="2024",
        unit="VND",
        source_line_start=2,
        source_line_end=2,
        extraction_confidence=1.0,
    )

    verified, excerpt, failures = audit_cell_provenance(cell, doc_lines)
    assert verified is True
    assert excerpt == "Line 2: Revenue 1000"
    assert len(failures) == 0


def test_audit_cell_provenance_invalid_span() -> None:
    doc_lines = ("Line 1", "Line 2")
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)
    cell = CellRecord(
        cell_id="cell_1",
        table_id=tbl_id,
        row_idx=0,
        col_idx=0,
        row_label_raw="Revenue",
        row_label_canonical="Revenue",
        column_label_raw="2024",
        column_label_canonical="2024",
        value_raw="1000",
        value_numeric=Decimal("1000"),
        period="2024",
        unit="VND",
        source_line_start=1,
        source_line_end=10,  # Out of bounds
        extraction_confidence=1.0,
    )

    verified, excerpt, failures = audit_cell_provenance(cell, doc_lines)
    assert verified is False
    assert failures == ["invalid_provenance"]


def test_audit_cell_provenance_accepts_decoded_entity_and_br() -> None:
    doc_id = "doc_" + "a" * 64
    cell = CellRecord(
        cell_id="cell_1",
        table_id=stable_table_id(doc_id, 1, 1),
        row_idx=0,
        col_idx=0,
        row_label_raw=None,
        row_label_canonical=None,
        column_label_raw=None,
        column_label_canonical=None,
        value_raw="Lợi nhuận & thu nhập\nkhác",
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=1,
        source_line_end=1,
        extraction_confidence=1.0,
    )

    verified, _, failures = audit_cell_provenance(
        cell,
        ("<td>Lợi nhuận &amp; thu nhập<br>khác</td>",),
    )

    assert verified is True
    assert failures == []


def _usable_fixture() -> tuple[TableAssessment, TableRecord]:
    doc_id = "doc_" + "b" * 64
    table_id = stable_table_id(doc_id, 10, 20)
    ann_id = stable_annotation_id(doc_id, 10, 20, "balance_sheet")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id=doc_id,
        relative_path="VCB/2024/report.txt",
        statement_type="balance_sheet",
        line_start=10,
        line_end=20,
        row_count=2,
        column_count=2,
        unit_normalized="VND",
        expected_periods=("2024",),
    )
    assessment = TableAssessment(
        annotation=expected,
        table_id=table_id,
        overlap_numerator=11,
        overlap_denominator=11,
        failures=(),
        usable=True,
    )
    table = TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw=None,
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
    return assessment, table


def _audit(
    *, value_raw: str, value_numeric: float | None, period: str, row_label: str
) -> CellAudit:
    doc_id = "doc_" + "b" * 64
    ann_id = stable_annotation_id(doc_id, 10, 20, "balance_sheet")
    return CellAudit(
        annotation_schema_version="1",
        sampling_version=SAMPLING_VERSION,
        cell_id=f"cell_{value_raw}",
        doc_id=doc_id,
        relative_path="VCB/2024/report.txt",
        company_code="VCB",
        report_year=2024,
        annotation_id=ann_id,
        statement_type="balance_sheet",
        table_id=stable_table_id(doc_id, 10, 20),
        row_idx=1,
        col_idx=1,
        row_label_raw=row_label,
        column_label_raw="2024",
        value_raw=value_raw,
        value_numeric=value_numeric,
        period=period,
        unit="VND",
        source_line_start=10,
        source_line_end=10,
        source_excerpt=value_raw,
        verified=True,
    )


def test_usability_ignores_header_and_metric_label_cells_for_numeric_gate() -> None:
    audits = (
        _audit(value_raw="Chỉ tiêu", value_numeric=None, period="", row_label=""),
        _audit(value_raw="Doanh thu", value_numeric=None, period="", row_label="Doanh thu"),
    )
    assessment, table = _usable_fixture()

    result = evaluate_table_usability(
        (assessment,), {assessment.annotation.annotation_id: table}, audits
    )[0]

    assert result.usable is True
    assert result.failures == ()


def test_usability_flags_numeric_looking_value_that_failed_normalization() -> None:
    audits = (_audit(value_raw="100", value_numeric=None, period="2024", row_label="Doanh thu"),)
    assessment, table = _usable_fixture()

    result = evaluate_table_usability(
        (assessment,), {assessment.annotation.annotation_id: table}, audits
    )[0]

    assert [failure.code for failure in result.failures] == ["no_numeric_value"]


def test_usability_flags_numeric_value_with_missing_period() -> None:
    audits = (_audit(value_raw="100", value_numeric=100.0, period="", row_label="Doanh thu"),)
    assessment, table = _usable_fixture()

    result = evaluate_table_usability(
        (assessment,), {assessment.annotation.annotation_id: table}, audits
    )[0]

    assert [failure.code for failure in result.failures] == ["period_mismatch"]


def test_generate_cell_audits_rejects_noncanonical_cell_id(tmp_path: Path) -> None:
    doc_text = (
        "Header line 1\n"
        "Header line 2\n"
        "Header line 3\n"
        "Header line 4\n"
        "Header line 5\n"
        "Header line 6\n"
        "Header line 7\n"
        "Header line 8\n"
        "Đơn vị tính: VND\n"
        "<table><tr><th></th><th>2024</th></tr>\n"
        "<tr><td>Total assets</td><td>100</td></tr>\n"
        "</table>"
    )
    actual_bytes = doc_text.encode("utf-8")
    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    actual_size = len(actual_bytes)
    doc_id = stable_document_id(actual_hash)
    relative_path = "VCB/2024/Consolidated/report.txt"

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True, parents=True)
    doc_path = corpus_dir / relative_path
    doc_path.parent.mkdir(exist_ok=True, parents=True)
    doc_path.write_bytes(actual_bytes)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True, parents=True)
    manifest_path = manifest_dir / "documents.jsonl"

    doc_raw = {
        "record_type": "document",
        "doc_id": doc_id,
        "repo_id": "test_repo",
        "revision": "main",
        "relative_path": relative_path,
        "company_code": "VCB",
        "report_year": 2024,
        "statement_scope": "consolidated",
        "sha256": actual_hash,
        "file_size_bytes": actual_size,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "notes": [],
    }
    manifest_path.write_text(json.dumps(doc_raw) + "\n", encoding="utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    release_path = tmp_path / "release"
    release_path.mkdir(exist_ok=True, parents=True)

    table_id = stable_table_id(doc_id, 10, 12)
    cell_id = "bad"

    # Parquet documents
    doc_table = pa.Table.from_pylist(
        [
            {
                "doc_id": doc_id,
                "repo_id": "test_repo",
                "revision": "main",
                "relative_path": relative_path,
                "company_code": "VCB",
                "report_year": 2024,
                "statement_scope": "consolidated",
                "sha256": actual_hash,
                "file_size_bytes": actual_size,
                "encoding": "utf-8",
                "inventory_status": "ready",
                "ruleset_version": "1",
                "normalization_fingerprint": "b" * 64,
            }
        ],
        schema=DOCUMENT_SCHEMA,
    )
    _write_table(doc_table, release_path / "documents.parquet")

    # Parquet tables
    tbl_table = pa.Table.from_pylist(
        [
            {
                "table_id": table_id,
                "doc_id": doc_id,
                "source_ordinal": 0,
                "title_raw": "Balance Sheet",
                "statement_type": "balance_sheet",
                "unit_raw": "VND",
                "unit_normalized": "VND",
                "line_start": 10,
                "line_end": 12,
                "row_count": 2,
                "column_count": 2,
                "quality_score": 1.0,
                "csv_path": None,
            }
        ],
        schema=TABLE_SCHEMA,
    )
    _write_table(tbl_table, release_path / "tables.parquet")

    # Parquet cells
    from decimal import Decimal

    cell_table = pa.Table.from_pylist(
        [
            {
                "cell_id": cell_id,
                "table_id": table_id,
                "row_idx": 1,
                "col_idx": 1,
                "row_label_raw": "Total assets",
                "row_label_canonical": "total_assets",
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
    _write_table(cell_table, release_path / "cells.parquet")

    # Parquet placements
    placement_table = pa.Table.from_pylist(
        [{"table_id": table_id, "row_idx": 1, "col_idx": 1, "cell_id": cell_id}],
        schema=PLACEMENT_SCHEMA,
    )
    _write_table(placement_table, release_path / "placements.parquet")

    # Parquet issues
    issue_table = pa.Table.from_pylist([], schema=ISSUE_SCHEMA)
    _write_table(issue_table, release_path / "issues.parquet")

    # Rewrite manifest.json
    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": 1,
        "table_count": 1,
        "cell_count": 1,
        "placement_count": 1,
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(json.dumps(release_manifest), encoding="utf-8")

    table = TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw="Balance Sheet",
        statement_type="balance_sheet",
        unit_raw="VND",
        unit_normalized="VND",
        line_start=10,
        line_end=12,
        row_count=2,
        column_count=2,
        quality_score=1.0,
        csv_path=None,
    )

    dataset = load_gate_dataset(manifest_path, release_path)

    ann_id = stable_annotation_id(doc_id, 10, 12, "balance_sheet")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id=doc_id,
        relative_path=relative_path,
        statement_type="balance_sheet",
        line_start=10,
        line_end=12,
        row_count=2,
        column_count=2,
        unit_normalized="VND",
        expected_periods=("2024",),
    )

    audits = generate_cell_audits(dataset, corpus_dir, (expected,), {ann_id: table})
    assert audits[0].verified is False


def test_evaluate_table_usability_shape_mismatch() -> None:
    doc_id = "doc_" + "a" * 64
    tbl_id = stable_table_id(doc_id, 10, 20)

    extracted_tbl = TableRecord(
        table_id=tbl_id,
        doc_id=doc_id,
        title_raw="Balance Sheet",
        statement_type="balance_sheet",
        unit_raw="VND",
        unit_normalized="VND",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=2,  # Differs from expected column_count=3
        quality_score=1.0,
        csv_path=None,
    )

    ann_id = stable_annotation_id(doc_id, 10, 20, "balance_sheet")
    exp = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id=doc_id,
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type="balance_sheet",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=3,
        unit_normalized="VND",
        expected_periods=("2024",),
        notes="",
    )

    initial_ta = TableAssessment(
        annotation=exp,
        table_id=tbl_id,
        overlap_numerator=11,
        overlap_denominator=11,
        failures=(),
        usable=True,
    )

    matched = {ann_id: extracted_tbl}
    final_tas = evaluate_table_usability((initial_ta,), matched, ())

    assert len(final_tas) == 1
    assert final_tas[0].usable is False
    assert len(final_tas[0].failures) == 1
    assert final_tas[0].failures[0].code == "shape_mismatch"


def test_sample_is_deterministic_stratified_and_table_capped() -> None:
    from collections import Counter

    from financial_report_qa.evaluation.week1_contracts import SAMPLING_VERSION, CellAudit
    from financial_report_qa.evaluation.week1_sampling import select_audit_cells

    doc_id = "doc_" + "a" * 64
    tbl_1 = stable_table_id(doc_id, 10, 20)
    tbl_2 = stable_table_id(doc_id, 30, 40)

    candidates: list[CellAudit] = []
    for i in range(20):
        tbl = tbl_1 if i < 10 else tbl_2
        line_start = 10 if i < 10 else 30
        line_end = 20 if i < 10 else 40
        ann_id = stable_annotation_id(doc_id, line_start, line_end, "balance_sheet")
        candidates.append(
            CellAudit(
                annotation_schema_version="1",
                sampling_version=SAMPLING_VERSION,
                cell_id=f"cell_{i}",
                doc_id=doc_id,
                relative_path="VCB/2024/report.txt",
                company_code="VCB",
                report_year=2024,
                annotation_id=ann_id,
                statement_type="balance_sheet",
                table_id=tbl,
                row_idx=i % 5,
                col_idx=i // 5,
                row_label_raw="Label",
                column_label_raw="Col",
                value_raw="100",
                value_numeric=100.0,
                period="2024",
                unit="VND",
                source_line_start=1,
                source_line_end=2,
                source_excerpt="Excerpt",
                verified=True,
                review_notes="",
            )
        )

    cand_tuple = tuple(candidates)
    selected = select_audit_cells(cand_tuple, sample_size=4, max_per_table=2)
    reversed_selected = select_audit_cells(
        tuple(reversed(cand_tuple)), sample_size=4, max_per_table=2
    )

    assert selected == reversed_selected
    assert len({item.cell_id for item in selected}) == 4
    assert max(Counter(item.table_id for item in selected).values()) <= 2


@dataclass(frozen=True)
class _ProvenanceCase:
    dataset: GateDataset
    corpus_dir: Path
    expected_tables: tuple[ExpectedTable, ...]
    matched_tables: dict[str, TableRecord]
    document: DocumentRecord


def _write_provenance_case(
    tmp_path: Path,
    *,
    source_bytes: bytes | None = None,
    encoding: str = "utf-8",
    source_value_raw: str = "100",
    released_value_raw: str = "100",
) -> _ProvenanceCase:
    if source_bytes is not None:
        actual_bytes = source_bytes
    else:
        lines = [
            "Header line 1",
            "Header line 2",
            "Header line 3",
            "Header line 4",
            "Header line 5",
            "Header line 6",
            "Header line 7",
            "Header line 8",
            "Đơn vị tính: VND",  # Line 9
            "<table><tr><th></th><th>2024</th></tr>",  # Line 10
            f"<tr><td>Total assets</td><td>{source_value_raw}</td></tr>",  # Line 11
            "</table>",  # Line 12
        ]
        text = "\n".join(lines)
        actual_bytes = text.encode(encoding)

    actual_hash = hashlib.sha256(actual_bytes).hexdigest()
    actual_size = len(actual_bytes)
    doc_id = stable_document_id(actual_hash)
    relative_path = "VCB/2024/Consolidated/report.txt"

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True, parents=True)
    doc_path = corpus_dir / relative_path
    doc_path.parent.mkdir(exist_ok=True, parents=True)
    doc_path.write_bytes(actual_bytes)

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(exist_ok=True, parents=True)
    manifest_path = manifest_dir / "documents.jsonl"

    doc_raw = {
        "record_type": "document",
        "doc_id": doc_id,
        "repo_id": "test_repo",
        "revision": "main",
        "relative_path": relative_path,
        "company_code": "VCB",
        "report_year": 2024,
        "statement_scope": "consolidated",
        "sha256": actual_hash,
        "file_size_bytes": actual_size,
        "encoding": encoding,
        "inventory_status": "ready",
        "notes": [],
    }
    manifest_path.write_text(json.dumps(doc_raw) + "\n", encoding="utf-8")
    source_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    release_path = tmp_path / "release"
    release_path.mkdir(exist_ok=True, parents=True)

    table_id = stable_table_id(doc_id, 10, 12)
    table = TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw="Balance Sheet",
        statement_type="balance_sheet",
        unit_raw="VND",
        unit_normalized="VND",
        line_start=10,
        line_end=12,
        row_count=2,
        column_count=2,
        quality_score=1.0,
        csv_path=None,
    )

    cell_id = stable_cell_id(table_id, 1, 1)

    doc_table = pa.Table.from_pylist(
        [
            {
                "doc_id": doc_id,
                "repo_id": "test_repo",
                "revision": "main",
                "relative_path": relative_path,
                "company_code": "VCB",
                "report_year": 2024,
                "statement_scope": "consolidated",
                "sha256": actual_hash,
                "file_size_bytes": actual_size,
                "encoding": encoding,
                "inventory_status": "ready",
                "ruleset_version": "1",
                "normalization_fingerprint": "b" * 64,
            }
        ],
        schema=DOCUMENT_SCHEMA,
    )
    _write_table(doc_table, release_path / "documents.parquet")

    tbl_table = pa.Table.from_pylist(
        [
            {
                "table_id": table_id,
                "doc_id": doc_id,
                "source_ordinal": 0,
                "title_raw": "Balance Sheet",
                "statement_type": "balance_sheet",
                "unit_raw": "VND",
                "unit_normalized": "VND",
                "line_start": 10,
                "line_end": 12,
                "row_count": 2,
                "column_count": 2,
                "quality_score": 1.0,
                "csv_path": None,
            }
        ],
        schema=TABLE_SCHEMA,
    )
    _write_table(tbl_table, release_path / "tables.parquet")

    cell_table = pa.Table.from_pylist(
        [
            {
                "cell_id": cell_id,
                "table_id": table_id,
                "row_idx": 1,
                "col_idx": 1,
                "row_label_raw": "Total assets",
                "row_label_canonical": "total_assets",
                "column_label_raw": "2024",
                "column_label_canonical": "2024",
                "value_raw": released_value_raw,
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
    _write_table(cell_table, release_path / "cells.parquet")

    placement_table = pa.Table.from_pylist(
        [{"table_id": table_id, "row_idx": 1, "col_idx": 1, "cell_id": cell_id}],
        schema=PLACEMENT_SCHEMA,
    )
    _write_table(placement_table, release_path / "placements.parquet")

    issue_table = pa.Table.from_pylist([], schema=ISSUE_SCHEMA)
    _write_table(issue_table, release_path / "issues.parquet")

    release_manifest = {
        "dataset_fingerprint": "f" * 64,
        "source_manifest_sha256": source_manifest_sha256,
        "document_count": 1,
        "table_count": 1,
        "cell_count": 1,
        "placement_count": 1,
        "issue_count": 0,
    }
    (release_path / "manifest.json").write_text(json.dumps(release_manifest), encoding="utf-8")

    dataset = load_gate_dataset(manifest_path, release_path)

    ann_id = stable_annotation_id(doc_id, 10, 12, "balance_sheet")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=ann_id,
        doc_id=doc_id,
        relative_path=relative_path,
        statement_type="balance_sheet",
        line_start=10,
        line_end=12,
        row_count=2,
        column_count=2,
        unit_normalized="VND",
        expected_periods=("2024",),
    )

    matched_tables = {ann_id: table}
    updated_doc = dataset.documents_by_id[doc_id]

    return _ProvenanceCase(dataset, corpus_dir, (expected,), matched_tables, updated_doc)


def test_generate_cell_audits_fails_on_source_hash_mismatch(tmp_path: Path) -> None:
    case = _write_provenance_case(tmp_path)
    source_path = case.corpus_dir / case.document.relative_path
    source_path.write_bytes(source_path.read_bytes() + b"tampered")
    with pytest.raises(Week1GateSourceError):
        generate_cell_audits(
            case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
        )


def test_generate_cell_audits_fails_on_invalid_utf8_instead_of_replacing(
    tmp_path: Path,
) -> None:
    case = _write_provenance_case(tmp_path, source_bytes=b"\xff", encoding="utf-8")
    with pytest.raises(Week1GateSourceError):
        generate_cell_audits(
            case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
        )


def test_generate_cell_audits_marks_canonical_cell_drift_invalid(tmp_path: Path) -> None:
    case = _write_provenance_case(tmp_path, source_value_raw="100", released_value_raw="999")
    audits = generate_cell_audits(
        case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
    )
    assert audits
    assert all(audit.verified is False for audit in audits)


def test_generate_cell_audits_accepts_exact_reextraction(tmp_path: Path) -> None:
    case = _write_provenance_case(tmp_path)
    audits = generate_cell_audits(
        case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
    )
    assert audits
    assert all(audit.verified is True for audit in audits)


def test_generate_cell_audits_caching_only_reads_once(tmp_path: Path) -> None:
    from unittest.mock import patch

    import financial_report_qa.evaluation.week1_provenance as prov_mod

    case = _write_provenance_case(tmp_path)
    second_ann = case.expected_tables[0].model_copy(
        update={
            "annotation_id": stable_annotation_id(case.document.doc_id, 10, 12, "income_statement"),
            "statement_type": "income_statement",
        }
    )
    expected_tables = (case.expected_tables[0], second_ann)
    matched_tables = {
        case.expected_tables[0].annotation_id: case.matched_tables[
            case.expected_tables[0].annotation_id
        ],
        second_ann.annotation_id: case.matched_tables[
            case.expected_tables[0].annotation_id
        ].model_copy(update={"statement_type": "income_statement"}),
    }

    with patch.object(prov_mod, "read_document", wraps=read_document) as mock_read:
        audits = generate_cell_audits(
            case.dataset, case.corpus_dir, expected_tables, matched_tables
        )
        assert len(audits) > 0
        assert mock_read.call_count == 1
