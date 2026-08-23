"""Unit tests for the synced-text rewrite of source TXT documents."""

from __future__ import annotations

import codecs
import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import ExportError, SourceReadError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.export.csv_export import CsvExportManifest, export_normalized_csvs
from financial_report_qa.export.synced_text import (
    TableExportEntry,
    build_synced_text,
    export_synced_text,
)
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id

# The reader accepts fixture documents whose sha256 is "a"*64 or starts with
# "0000", so two mock documents can coexist with distinct content-addressed IDs.
MOCK_SHA256 = "a" * 64
ZERO_SHA256 = "0" * 64
DOC_ID_A = stable_document_id(MOCK_SHA256)
DOC_ID_B = stable_document_id(ZERO_SHA256)
RELATIVE_PATH_A = "reports/ACB/ACB_fs_extracted.txt"
RELATIVE_PATH_B = "reports/VCB/VCB_fs_extracted.txt"

TABLE_IDS_A = ["tbl_" + digit * 64 for digit in ("1", "2", "3")]
TABLE_ID_B = "tbl_" + "4" * 64

_LINK_RE = re.compile(r"\[TABLE: (?P<table_id>\S+) -> (?P<link>[^\]]+)\]")


def _document_a_text() -> str:
    """Page markers, prose, and three table blocks (one written with CRLF)."""
    return (
        "===== PAGE 1 =====\n"
        "Bảng cân đối kế toán hợp nhất\n"
        "\n"
        "<table>\n"
        "<tr><td>Tiền và tương đương tiền</td><td>100</td></tr>\n"
        "</table>\n"
        "\n"
        "Thuyết minh V.1 nêu chi tiết tiền gửi.\n"
        "\n"
        "===== PAGE 2 =====\n"
        "\n"
        "<table>\r\n"
        "<tr><td>Doanh thu</td><td>500</td></tr>\r\n"
        "</table>\r\n"
        "\n"
        "Kết thúc phần bảng cân đối.\n"
        "\n"
        "<table>\n"
        "<tr><td>Lợi nhuận sau thuế</td><td>120</td></tr>\n"
        "</table>\n"
        "\n"
    )


def _document_b_text() -> str:
    return (
        "===== PAGE 1 =====\n"
        "Báo cáo kết quả kinh doanh\n"
        "\n"
        "<table>\n"
        "<tr><td>Doanh thu thuần</td><td>1.000</td></tr>\n"
        "</table>\n"
        "\n"
        "Ghi chú đơn vị: triệu đồng.\n"
        "\n"
    )


def _spans(text: str) -> list[tuple[int, int]]:
    """Inclusive 1-based [line_start, line_end] of each <table>...</table>."""
    lines = text.splitlines()
    opens = [number for number, line in enumerate(lines, start=1) if line == "<table>"]
    closes = [number for number, line in enumerate(lines, start=1) if line == "</table>"]
    return list(zip(opens, closes))


def _block(text: str, span: tuple[int, int]) -> str:
    """Exact original substring covered by a 1-based inclusive line span."""
    lines = text.splitlines(keepends=True)
    return "".join(lines[span[0] - 1 : span[1]])


def _expected_synced(
    original: str,
    spans: list[tuple[int, int]],
    table_ids: list[str],
    links: dict[str, str],
) -> str:
    expected = original
    for span, table_id in zip(spans, table_ids):
        expected = expected.replace(
            _block(original, span),
            f"[TABLE: {table_id} -> {links[table_id]}]\n",
            1,
        )
    return expected


DOCUMENT_A_TEXT = _document_a_text()
DOCUMENT_B_TEXT = _document_b_text()
SPANS_A = _spans(DOCUMENT_A_TEXT)
SPANS_B = _spans(DOCUMENT_B_TEXT)
PAYLOAD_A = DOCUMENT_A_TEXT.encode("utf-8")
PAYLOAD_B = codecs.BOM_UTF8 + DOCUMENT_B_TEXT.encode("utf-8")


# ---------------------------------------------------------------------------
# Local helpers (self-contained; not imported from any shared test package).
# ---------------------------------------------------------------------------


def _snapshot_record(
    relative_path: str,
    payload: bytes,
    *,
    encoding: str = "utf-8",
    sha256: str = MOCK_SHA256,
) -> DocumentRecord:
    """DocumentRecord matching a snapshot payload under the reader's mock rule."""
    return DocumentRecord(
        doc_id=stable_document_id(sha256),
        repo_id="repo",
        revision="1",
        relative_path=relative_path,
        company_code="ACB",
        report_year=2023,
        statement_scope="consolidated",
        sha256=sha256,
        file_size_bytes=len(payload),
        encoding=encoding,
        inventory_status="ready",
    )


def _write_snapshot(root: Path, relative_path: str, payload: bytes) -> None:
    target = root.joinpath(*PurePosixPath(relative_path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _document_row(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "doc_id": DOC_ID_A,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": RELATIVE_PATH_A,
        "company_code": "ACB",
        "report_year": 2023,
        "statement_scope": "consolidated",
        "sha256": MOCK_SHA256,
        "file_size_bytes": len(PAYLOAD_A),
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }
    record.update(overrides)
    return record


def _table_record(
    table_id: str,
    doc_id: str,
    *,
    line_start: int,
    line_end: int,
    statement_type: str | None = "balance_sheet",
    unit_raw: str | None = "VND",
    unit_normalized: str | None = "vnd",
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "source_ordinal": 0,
        "title_raw": "Bang bao cao",
        "statement_type": statement_type,
        "unit_raw": unit_raw,
        "unit_normalized": unit_normalized,
        "line_start": line_start,
        "line_end": line_end,
        "row_count": 2,
        "column_count": 2,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _parquet_cell(
    cell_id: str, table_id: str, row_idx: int, col_idx: int, value_raw: str
) -> dict[str, Any]:
    is_header = row_idx == 0
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_label_raw": None if is_header else value_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": None,
        "column_label_canonical": None,
        "value_raw": value_raw,
        "value_numeric": None,
        "period": None,
        "unit": None,
        "source_line_start": 5,
        "source_line_end": 5,
        "extraction_confidence": 0.9,
    }


def _write_release(
    root: Path,
    *,
    documents: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    cells: list[dict[str, Any]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), root / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), root / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), root / "cells.parquet"
    )
    placements = [
        {
            "table_id": cell["table_id"],
            "row_idx": cell["row_idx"],
            "col_idx": cell["col_idx"],
            "cell_id": cell["cell_id"],
        }
        for cell in cells
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA), root / "placements.parquet"
    )
    release_manifest = {"dataset_fingerprint": "fp", "table_count": len(tables)}
    (root / "manifest.json").write_text(json.dumps(release_manifest), encoding="utf-8")
    return root


def _world(tmp_path: Path) -> tuple[Path, Path, CsvExportManifest]:
    """Release + snapshot + Task-1 CSV manifest covering both mock documents."""
    release_dir = tmp_path / "release"
    snapshot_root = tmp_path / "snapshot"
    csv_output = tmp_path / "csv_out"

    _write_snapshot(snapshot_root, RELATIVE_PATH_A, PAYLOAD_A)
    _write_snapshot(snapshot_root, RELATIVE_PATH_B, PAYLOAD_B)

    documents = [
        _document_row(),
        _document_row(
            doc_id=DOC_ID_B,
            relative_path=RELATIVE_PATH_B,
            company_code="VCB",
            report_year=2022,
            sha256=ZERO_SHA256,
            file_size_bytes=len(PAYLOAD_B),
            encoding="utf-8-sig",
        ),
    ]
    tables = [
        _table_record(TABLE_IDS_A[0], DOC_ID_A, line_start=SPANS_A[0][0], line_end=SPANS_A[0][1]),
        _table_record(TABLE_IDS_A[1], DOC_ID_A, line_start=SPANS_A[1][0], line_end=SPANS_A[1][1]),
        _table_record(TABLE_IDS_A[2], DOC_ID_A, line_start=SPANS_A[2][0], line_end=SPANS_A[2][1]),
        _table_record(TABLE_ID_B, DOC_ID_B, line_start=SPANS_B[0][0], line_end=SPANS_B[0][1]),
    ]
    cells: list[dict[str, Any]] = []
    for index, (table_id, spans) in enumerate(
        [(TABLE_IDS_A[0], SPANS_A), (TABLE_IDS_A[1], SPANS_A),
         (TABLE_IDS_A[2], SPANS_A), (TABLE_ID_B, SPANS_B)]
    ):
        prefix = f"t{index}"
        cells.append(_parquet_cell(f"{prefix}_h00", table_id, 0, 0, "Chỉ tiêu"))
        cells.append(_parquet_cell(f"{prefix}_h01", table_id, 0, 1, "31/12/2023"))
        cells.append(_parquet_cell(f"{prefix}_c10", table_id, 1, 0, "Tiền"))
        cells.append(_parquet_cell(f"{prefix}_c11", table_id, 1, 1, "100"))

    release_path = _write_release(release_dir, documents=documents, tables=tables, cells=cells)
    csv_manifest = export_normalized_csvs(release_path, csv_output)
    return release_path, snapshot_root, csv_manifest


# ---------------------------------------------------------------------------
# build_synced_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_build_replaces_single_span_and_preserves_rest_byte_for_byte(
    tmp_path: Path, encoding: str
) -> None:
    snapshot_root = tmp_path / "snapshot"
    payload = (codecs.BOM_UTF8 if encoding == "utf-8-sig" else b"") + DOCUMENT_A_TEXT.encode(
        "utf-8"
    )
    _write_snapshot(snapshot_root, RELATIVE_PATH_A, payload)
    record = _snapshot_record(RELATIVE_PATH_A, payload, encoding=encoding)
    assert read_document(snapshot_root, record).text == DOCUMENT_A_TEXT

    span = SPANS_A[1]  # the CRLF table block in the middle of the document
    entry = TableExportEntry(
        line_start=span[0], line_end=span[1], table_id=TABLE_IDS_A[0], csv_relpath="out/x.csv"
    )

    synced = build_synced_text(snapshot_root, record, [entry])

    replacement = f"[TABLE: {TABLE_IDS_A[0]} -> out/x.csv]\n"
    assert synced == DOCUMENT_A_TEXT.replace(_block(DOCUMENT_A_TEXT, span), replacement, 1)


def test_build_without_entries_returns_original_text(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_snapshot(snapshot_root, RELATIVE_PATH_A, PAYLOAD_A)
    record = _snapshot_record(RELATIVE_PATH_A, PAYLOAD_A)

    assert build_synced_text(snapshot_root, record, []) == DOCUMENT_A_TEXT


def test_build_replaces_many_tables_from_scrambled_input_order(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_snapshot(snapshot_root, RELATIVE_PATH_A, PAYLOAD_A)
    record = _snapshot_record(RELATIVE_PATH_A, PAYLOAD_A)

    # Deliberately scrambled order (middle, first, last): the function must
    # sort by descending line_end itself so no offset drifts.
    links = {table_id: f"out/{table_id[-1]}.csv" for table_id in TABLE_IDS_A}
    entries = [
        TableExportEntry(
            line_start=SPANS_A[index][0],
            line_end=SPANS_A[index][1],
            table_id=TABLE_IDS_A[index],
            csv_relpath=links[TABLE_IDS_A[index]],
        )
        for index in (1, 0, 2)
    ]

    synced = build_synced_text(snapshot_root, record, entries)

    expected = _expected_synced(DOCUMENT_A_TEXT, SPANS_A, TABLE_IDS_A, links)
    assert synced == expected
    assert synced.count("[TABLE:") == 3
    assert synced.count("</table>") == 0


@pytest.mark.parametrize(
    "spans",
    [
        [(0, 3)],  # line_start below 1
        [(20, 22)],  # line_end past the last line (document A has 21 lines)
        [(6, 4)],  # line_start greater than line_end
        [(4, 6), (5, 8)],  # directly overlapping spans
        [(4, 8), (6, 7)],  # contained span overlaps its container
        [(4, 6), (4, 6)],  # identical duplicate spans overlap
    ],
)
def test_build_rejects_out_of_range_or_overlapping_spans(
    tmp_path: Path, spans: list[tuple[int, int]]
) -> None:
    snapshot_root = tmp_path / "snapshot"
    _write_snapshot(snapshot_root, RELATIVE_PATH_A, PAYLOAD_A)
    record = _snapshot_record(RELATIVE_PATH_A, PAYLOAD_A)
    entries = [
        TableExportEntry(line_start=start, line_end=end, table_id=f"t{n}", csv_relpath="o.csv")
        for n, (start, end) in enumerate(spans)
    ]

    with pytest.raises(ExportError):
        build_synced_text(snapshot_root, record, entries)


def test_build_propagates_missing_source_read_error(tmp_path: Path) -> None:
    record = _snapshot_record(RELATIVE_PATH_A, PAYLOAD_A)  # never written to disk

    with pytest.raises(SourceReadError):
        build_synced_text(tmp_path / "snapshot", record, [])


# ---------------------------------------------------------------------------
# export_synced_text on a small release fixture built by the Task-1 exporter
# ---------------------------------------------------------------------------


def test_export_writes_mirrored_texts_with_links_to_task_one_csvs(tmp_path: Path) -> None:
    release, snapshot_root, csv_manifest = _world(tmp_path)
    synced_output = tmp_path / "synced"

    result = export_synced_text(release, snapshot_root, csv_manifest, synced_output)

    links = {
        entry.table_id: (csv_manifest.output_dir / entry.csv_path).as_posix()
        for entry in csv_manifest.entries
    }
    mirror_a = synced_output / PurePosixPath(RELATIVE_PATH_A)
    mirror_b = synced_output / PurePosixPath(RELATIVE_PATH_B)

    assert result.output_dir == synced_output
    assert result.document_count == 2
    assert result.table_count == 4
    assert mirror_a.read_text(encoding="utf-8") == _expected_synced(
        DOCUMENT_A_TEXT, SPANS_A, TABLE_IDS_A, links
    )
    assert mirror_b.read_text(encoding="utf-8") == _expected_synced(
        DOCUMENT_B_TEXT, SPANS_B, [TABLE_ID_B], links
    )
    # Prose and page markers survive verbatim; every link target exists on disk.
    synced_a = mirror_a.read_text(encoding="utf-8")
    assert synced_a.splitlines()[0] == "===== PAGE 1 ====="
    assert "Thuyết minh V.1 nêu chi tiết tiền gửi." in synced_a

    found = dict(_LINK_RE.findall(synced_a))
    assert set(found) == set(TABLE_IDS_A)
    for link in found.values():
        assert Path(link).is_file()

    # Atomic writes leave no temporary siblings behind.
    unexpected = [
        path.name for path in synced_output.rglob("*") if path.is_file() and path.suffix == ".tmp"
    ]
    assert unexpected == []


def test_export_skips_documents_without_mapped_tables(tmp_path: Path) -> None:
    release, snapshot_root, csv_manifest = _world(tmp_path)
    kept_entry = csv_manifest.entries[0]
    partial_path = tmp_path / "partial_manifest.jsonl"
    partial_path.write_text(json.dumps(asdict(kept_entry)) + "\n", encoding="utf-8")
    partial_manifest = CsvExportManifest(
        output_dir=csv_manifest.output_dir,
        manifest_path=partial_path,
        table_count=1,
        entries=(kept_entry,),
    )
    synced_output = tmp_path / "synced_partial"

    result = export_synced_text(release, snapshot_root, partial_manifest, synced_output)

    assert result.document_count == 1
    assert result.table_count == 1
    synced_a = (synced_output / PurePosixPath(RELATIVE_PATH_A)).read_text(encoding="utf-8")
    assert synced_a.count("[TABLE:") == 1
    assert synced_a.count("</table>") == 2  # unmapped blocks stay untouched
    assert not (synced_output / PurePosixPath(RELATIVE_PATH_B).parent).exists()


def test_export_defaults_to_data_interim_synced_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, snapshot_root, csv_manifest = _world(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = export_synced_text(release, snapshot_root, csv_manifest)

    assert result.output_dir == Path("data/interim/synced_text")
    assert result.document_count == 2
    default_mirror = tmp_path / "data/interim/synced_text" / PurePosixPath(RELATIVE_PATH_B)
    assert default_mirror.is_file()


def test_export_fails_hard_when_snapshot_file_is_missing(tmp_path: Path) -> None:
    release, snapshot_root, csv_manifest = _world(tmp_path)
    (snapshot_root / PurePosixPath(RELATIVE_PATH_B)).unlink()

    with pytest.raises(SourceReadError):
        export_synced_text(release, snapshot_root, csv_manifest, tmp_path / "synced")


def test_export_surfaces_missing_release_as_export_error(tmp_path: Path) -> None:
    """A nonexistent release dir raises ExportError (not a raw duckdb error)."""
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"table_id": TABLE_IDS_A[0], "csv_path": "ACB__table_1.csv"}) + "\n",
        encoding="utf-8",
    )
    csv_manifest = CsvExportManifest(
        output_dir=tmp_path / "csv_out",
        manifest_path=manifest_path,
        table_count=1,
        entries=(),
    )
    missing_release = tmp_path / "missing"

    with pytest.raises(ExportError, match="cannot read release parquet") as exc_info:
        export_synced_text(missing_release, tmp_path / "snapshot", csv_manifest, tmp_path / "out")

    assert str(missing_release) in str(exc_info.value)
