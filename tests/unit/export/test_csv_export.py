"""Unit tests for the normalized-table CSV/metadata export layer."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import ExportError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.export.csv_export import (
    CellRow,
    PlacementRow,
    TableExportMetadata,
    build_normalized_table,
    detect_header_row_count,
    export_normalized_csvs,
    flatten_header,
)

TABLE_ID = "tbl_" + "1" * 64
DOC_ID_A = "doc_" + "a" * 64
DOC_ID_B = "doc_" + "b" * 64


# ---------------------------------------------------------------------------
# Local helpers (self-contained so later tasks can copy them verbatim).
# ---------------------------------------------------------------------------


def _cell(
    cell_id: str,
    row_idx: int,
    col_idx: int,
    *,
    value_raw: str = "",
    value_numeric: Decimal | None = None,
    row_label_raw: str | None = None,
    row_group_context_raw: str | None = None,
    column_label_raw: str | None = None,
    table_id: str = TABLE_ID,
) -> CellRow:
    """Build one CellRow; unlabeled cells default to header-cell shape."""
    return CellRow(
        cell_id=cell_id,
        table_id=table_id,
        row_idx=row_idx,
        col_idx=col_idx,
        value_raw=value_raw,
        value_numeric=value_numeric,
        row_label_raw=row_label_raw,
        row_group_context_raw=row_group_context_raw,
        column_label_raw=column_label_raw,
    )


def _header_cell(cell_id: str, row_idx: int, col_idx: int, text: str) -> CellRow:
    return _cell(cell_id, row_idx, col_idx, value_raw=text)


def _placements_for(cells: list[CellRow]) -> list[PlacementRow]:
    """Anchor placement per cell; tests append extra span placements."""
    return [
        PlacementRow(
            table_id=cell.table_id,
            row_idx=cell.row_idx,
            col_idx=cell.col_idx,
            cell_id=cell.cell_id,
        )
        for cell in cells
    ]


def _document(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "doc_id": DOC_ID_A,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": "reports/ACB/ACB_fs_extracted.txt",
        "company_code": "ACB",
        "report_year": 2023,
        "statement_scope": "consolidated",
        "sha256": "0" * 64,
        "file_size_bytes": 128,
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
    source_ordinal: int = 0,
    statement_type: str | None = "income_statement",
    unit_raw: str | None = "VND",
    unit_normalized: str | None = "vnd",
) -> dict[str, Any]:
    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "source_ordinal": source_ordinal,
        "title_raw": "Bang bao cao",
        "statement_type": statement_type,
        "unit_raw": unit_raw,
        "unit_normalized": unit_normalized,
        "line_start": line_start,
        "line_end": line_start + 9,
        "row_count": 2,
        "column_count": 2,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _parquet_cell(
    cell_id: str,
    table_id: str,
    row_idx: int,
    col_idx: int,
    *,
    value_raw: str = "",
    value_numeric: Decimal | None = None,
    row_label_raw: str | None = None,
    row_group_context_raw: str | None = None,
    column_label_raw: str | None = None,
) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": row_group_context_raw,
        "column_label_raw": column_label_raw,
        "column_label_canonical": None,
        "value_raw": value_raw,
        "value_numeric": value_numeric,
        "period": None,
        "unit": None,
        "source_line_start": 1,
        "source_line_end": 1,
        "extraction_confidence": 0.9,
    }


def _write_release(
    root: Path,
    *,
    documents: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    cells: list[dict[str, Any]],
    placements: list[dict[str, Any]] | None = None,
    table_count: int | None = None,
    write_manifest: bool = True,
) -> Path:
    """Materialize a tiny but realistic release directory (parquet + manifest)."""
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
    if placements is None:
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
    if write_manifest:
        count = len(tables) if table_count is None else table_count
        release_manifest = {
            "dataset_fingerprint": "fp",
            "document_count": len(documents),
            "table_count": count,
        }
        (root / "manifest.json").write_text(json.dumps(release_manifest), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# flatten_header
# ---------------------------------------------------------------------------


def test_flatten_header_single_level() -> None:
    assert flatten_header(["Tổng cộng"]) == "Tổng_cộng"


def test_flatten_header_multiple_levels() -> None:
    assert flatten_header(["Tổng cộng", "31/12/2022"]) == "Tổng_cộng_31/12/2022"


def test_flatten_header_drops_empty_and_whitespace_levels() -> None:
    assert flatten_header(["", "   ", "Số tiền", "\t"]) == "Số_tiền"


def test_flatten_header_collapses_whitespace_runs_and_keeps_slashes() -> None:
    assert flatten_header(["Số   cuối\n năm\t31/12/2022"]) == "Số_cuối_năm_31/12/2022"


def test_flatten_header_empty_list() -> None:
    assert flatten_header([]) == ""


# ---------------------------------------------------------------------------
# detect_header_row_count
# ---------------------------------------------------------------------------


def test_detect_header_rows_zero_when_first_row_carries_labels() -> None:
    cells = [_cell("c0", 0, 0, value_raw="Chi tieu", row_label_raw="Chi tieu")]
    assert detect_header_row_count(cells, _placements_for(cells)) == 0


def test_detect_header_rows_stop_at_fully_blank_row() -> None:
    cells = [
        _header_cell("h", 0, 0, "Chỉ tiêu"),
        _cell("b", 1, 0, value_raw="   "),
        _cell("d", 2, 0, value_raw="Tiền", row_label_raw="Tiền"),
    ]
    assert detect_header_row_count(cells, _placements_for(cells)) == 1


def test_detect_header_rows_multi_row_header_via_rowspan() -> None:
    top_left = _header_cell("h00", 0, 0, "Chỉ tiêu")
    top_right = _header_cell("h01", 0, 1, "31/12/2022")
    data = _cell("d20", 2, 0, value_raw="Tiền gửi", row_label_raw="Tiền gửi")
    cells = [top_left, top_right, data]
    placements = _placements_for(cells) + [
        PlacementRow(table_id=TABLE_ID, row_idx=1, col_idx=0, cell_id=top_left.cell_id),
        PlacementRow(table_id=TABLE_ID, row_idx=1, col_idx=1, cell_id=top_right.cell_id),
    ]
    assert detect_header_row_count(cells, placements) == 2


def test_detect_header_rows_empty_table() -> None:
    assert detect_header_row_count([], []) == 0


# ---------------------------------------------------------------------------
# build_normalized_table
# ---------------------------------------------------------------------------


def test_build_rowspan_row_label_repeats_across_grid_rows() -> None:
    label = _cell(
        "lbl",
        1,
        0,
        value_raw="Tiền và tương đương tiền",
        row_label_raw="Tiền và tương đương tiền",
    )
    first = _cell("v11", 1, 1, value_raw="100", value_numeric=Decimal("100.0000000000"))
    second = _cell("v21", 2, 1, value_raw="90.5", value_numeric=None)
    cells = [
        _header_cell("h00", 0, 0, "Chỉ tiêu"),
        _header_cell("h01", 0, 1, "2023"),
        label,
        first,
        second,
    ]
    placements = _placements_for(cells) + [
        PlacementRow(table_id=TABLE_ID, row_idx=2, col_idx=0, cell_id=label.cell_id),
    ]

    table = build_normalized_table(cells, placements, header_rows=1)

    assert table.headers == ("Chỉ_tiêu", "2023")
    assert table.rows == (
        ("Tiền và tương đương tiền", "100"),
        ("Tiền và tương đương tiền", "90.5"),
    )


def test_build_colspan_header_dedupes_consecutive_duplicate_levels() -> None:
    spanning = _header_cell("h01", 0, 1, "Số tiền")
    repeated = _header_cell("h11", 1, 1, "Số tiền")
    debt = _header_cell("h12", 1, 2, "Nợ")
    data = _cell("d10", 2, 0, value_raw="Tiền", row_label_raw="Tiền")
    cells = [_header_cell("h00", 0, 0, "Chỉ tiêu"), spanning, repeated, debt, data]
    placements = _placements_for(cells) + [
        PlacementRow(table_id=TABLE_ID, row_idx=0, col_idx=2, cell_id=spanning.cell_id),
    ]

    table = build_normalized_table(cells, placements, header_rows=2)

    # Col 1 repeats "Số tiền" across both header rows (deduped once); col 2
    # carries the colspan repetition plus a distinct second level.
    assert table.headers == ("Chỉ_tiêu", "Số_tiền", "Số_tiền_Nợ")


def test_build_keeps_a_leading_auxiliary_column_when_the_metric_column_is_second() -> None:
    """The metric column is not always column 0 (an STT/ordinal column can lead
    it, exactly as `ingestion/table_extractor.py::_metric_column_index` was
    written to handle) -- column 0's own value must survive the export."""
    stt_header = _header_cell("h00", 0, 0, "STT")
    metric_header = _header_cell("h01", 0, 1, "Chỉ tiêu")
    year_header = _header_cell("h02", 0, 2, "2023")
    stt_value = _cell("d00", 1, 0, value_raw="1", row_label_raw="Doanh thu")
    metric_value = _cell(
        "d01", 1, 1, value_raw="Doanh thu", row_label_raw="Doanh thu", column_label_raw="Chỉ tiêu"
    )
    amount = _cell(
        "d02", 1, 2, value_raw="100", value_numeric=Decimal("100.0000000000"),
        row_label_raw="Doanh thu", column_label_raw="2023",
    )
    cells = [stt_header, metric_header, year_header, stt_value, metric_value, amount]

    table = build_normalized_table(cells, _placements_for(cells), header_rows=1)

    assert table.headers == ("STT", "Chỉ_tiêu", "2023")
    assert table.rows == (("1", "Doanh thu", "100"),)


def test_build_group_context_nested_two_levels() -> None:
    labeled = _cell(
        "c10",
        1,
        0,
        value_raw="Tiền gửi",
        row_label_raw="Tiền gửi",
        row_group_context_raw="Vi mô > Ngân hàng",
    )
    value = _cell("c11", 1, 1, value_raw="15", value_numeric=Decimal("15.0000000000"))
    cells = [
        _header_cell("h00", 0, 0, "Chỉ tiêu"),
        _header_cell("h01", 0, 1, "2023"),
        labeled,
        value,
    ]

    table = build_normalized_table(cells, _placements_for(cells), header_rows=1)

    assert table.rows == (("Vi mô > Ngân hàng > Tiền gửi", "15"),)


def test_build_without_group_context_keeps_plain_row_label() -> None:
    labeled = _cell("c10", 1, 0, value_raw="Doanh thu", row_label_raw="Doanh thu")
    value = _cell("c11", 1, 1, value_raw="500", value_numeric=Decimal("500.0000000000"))
    cells = [_header_cell("h00", 0, 0, "Item"), _header_cell("h01", 0, 1, "Value"), labeled, value]

    table = build_normalized_table(cells, _placements_for(cells), header_rows=1)

    assert table.rows == (("Doanh thu", "500"),)


def test_build_with_zero_header_rows_treats_every_row_as_data() -> None:
    labeled = _cell("c00", 0, 0, value_raw="Doanh thu", row_label_raw="Doanh thu")
    value = _cell("c01", 0, 1, value_raw="500000", value_numeric=Decimal("500000.0000000000"))
    cells = [labeled, value]

    table = build_normalized_table(cells, _placements_for(cells), header_rows=0)

    assert table.headers == ("", "")
    assert table.rows == (("Doanh thu", "500000"),)


def test_build_formats_numerics_without_trailing_zeros() -> None:
    big = _cell("v11", 1, 1, value_raw="raw-ignored", value_numeric=Decimal("100000.0000000000"))
    negative = _cell("v12", 1, 2, value_raw="raw-ignored", value_numeric=Decimal("-12.5000000000"))
    labeled = _cell("c10", 1, 0, value_raw="Tiền", row_label_raw="Tiền")
    cells = [_header_cell("h00", 0, 0, "A"), _header_cell("h01", 0, 1, "B"), labeled, big, negative]

    table = build_normalized_table(cells, _placements_for(cells), header_rows=1)

    assert table.headers == ("A", "B", "")
    assert table.rows == (("Tiền", "100000", "-12.5"),)


def test_build_numeric_none_falls_back_to_value_raw_and_blank_without_placement() -> None:
    padded = _cell("v11", 1, 1, value_raw="  80.000  ", value_numeric=None)
    labeled = _cell("c10", 1, 0, value_raw="Chi phí", row_label_raw="Chi phí")
    filler = _cell("v02", 2, 2, value_raw="x", value_numeric=None)
    other = _cell("c20", 2, 0, value_raw="Khác", row_label_raw="Khác")
    cells = [_header_cell("h00", 0, 0, "A"), labeled, padded, other, filler]
    # No placement at (1, 2): the grid still has 3 columns via row 2.
    placements = [
        PlacementRow(table_id=TABLE_ID, row_idx=0, col_idx=0, cell_id="h00"),
        PlacementRow(table_id=TABLE_ID, row_idx=1, col_idx=0, cell_id=labeled.cell_id),
        PlacementRow(table_id=TABLE_ID, row_idx=1, col_idx=1, cell_id=padded.cell_id),
        PlacementRow(table_id=TABLE_ID, row_idx=2, col_idx=0, cell_id=other.cell_id),
        PlacementRow(table_id=TABLE_ID, row_idx=2, col_idx=2, cell_id=filler.cell_id),
    ]

    table = build_normalized_table(cells, placements, header_rows=1)

    assert table.rows == (
        ("Chi phí", "80.000", ""),
        ("Khác", "", "x"),
    )


# ---------------------------------------------------------------------------
# export_normalized_csvs on a small parquet release fixture
# ---------------------------------------------------------------------------


def _happy_release(root: Path) -> tuple[Path, str, str, str]:
    """Two documents, three tables (incl. rowspan/colspan and null statement/unit)."""
    table_a1 = "tbl_" + "1" * 64
    table_a2 = "tbl_" + "2" * 64
    table_b1 = "tbl_" + "3" * 64
    documents = [
        _document(),
        _document(
            doc_id=DOC_ID_B,
            relative_path="reports/VCB/VCB_fs_extracted.txt",
            company_code="VCB",
            report_year=2022,
        ),
    ]
    tables = [
        _table_record(table_a1, DOC_ID_A, line_start=10),
        _table_record(table_a2, DOC_ID_A, line_start=20),
        _table_record(
            table_b1,
            DOC_ID_B,
            line_start=5,
            statement_type=None,
            unit_raw=None,
            unit_normalized=None,
        ),
    ]
    cells = [
        # Table A1: colspan header + group-context rows (line_start 10).
        _parquet_cell("a1_h00", table_a1, 0, 0, value_raw="Chỉ tiêu"),
        _parquet_cell("a1_h01", table_a1, 0, 1, value_raw="31/12/2022"),
        _parquet_cell("a1_h01_span", table_a1, 0, 2, value_raw="31/12/2022"),
        _parquet_cell(
            "a1_c10",
            table_a1,
            1,
            0,
            value_raw="Tiền và các khoản tương đương tiền",
            row_label_raw="Tiền và các khoản tương đương tiền",
        ),
        _parquet_cell(
            "a1_c11",
            table_a1,
            1,
            1,
            value_raw="100000",
            value_numeric=Decimal("100000.0000000000"),
        ),
        _parquet_cell("a1_c12", table_a1, 1, 2, value_raw="80.000"),
        _parquet_cell(
            "a1_c20",
            table_a1,
            2,
            0,
            value_raw="Tiền gửi",
            row_label_raw="Tiền gửi",
            row_group_context_raw="Vi mô > Ngân hàng",
        ),
        _parquet_cell("a1_c21", table_a1, 2, 1, value_raw=" 15 "),
        # Table A2: plain two-column table with one numeric (line_start 20).
        _parquet_cell("a2_h00", table_a2, 0, 0, value_raw="Item"),
        _parquet_cell("a2_h01", table_a2, 0, 1, value_raw="Value"),
        _parquet_cell("a2_c10", table_a2, 1, 0, value_raw="Total", row_label_raw="Total"),
        _parquet_cell(
            "a2_c11", table_a2, 1, 1, value_raw="-12.5", value_numeric=Decimal("-12.5000000000")
        ),
        # Table B1: minimal table with no statement/unit (line_start 5).
        _parquet_cell("b1_h00", table_b1, 0, 0, value_raw="Khoản mục"),
        _parquet_cell("b1_h01", table_b1, 0, 1, value_raw="Giá trị"),
        _parquet_cell("b1_c10", table_b1, 1, 0, value_raw="Doanh thu", row_label_raw="Doanh thu"),
        _parquet_cell("b1_c11", table_b1, 1, 1, value_raw="x"),
    ]
    release = _write_release(root, documents=documents, tables=tables, cells=cells)
    return release, table_a1, table_a2, table_b1


def test_export_writes_numbered_csvs_and_sorted_manifest(tmp_path: Path) -> None:
    release, table_a1, table_a2, table_b1 = _happy_release(tmp_path / "release")
    output_dir = tmp_path / "out"

    manifest = export_normalized_csvs(release, output_dir)

    assert manifest.output_dir == output_dir
    assert manifest.manifest_path == output_dir / "manifest.jsonl"
    assert manifest.table_count == 3
    # File numbering restarts per document and follows line_start order; the
    # output directory holds exactly the four expected artifacts (no temp files).
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "ACB__table_1.csv",
        "ACB__table_2.csv",
        "VCB__table_1.csv",
        "manifest.jsonl",
    ]
    assert [entry.table_id for entry in manifest.entries] == [table_a1, table_a2, table_b1]

    raw = (output_dir / "ACB__table_1.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel/Vietnamese text
    assert b"\r" not in raw  # LF line endings only
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    assert rows == [
        ["Chỉ_tiêu", "31/12/2022", "31/12/2022"],
        ["Tiền và các khoản tương đương tiền", "100000", "80.000"],
        ["Vi mô > Ngân hàng > Tiền gửi", "15", ""],
    ]

    second_csv = (output_dir / "ACB__table_2.csv").read_bytes().decode("utf-8-sig")
    second_rows = list(csv.reader(second_csv.splitlines()))
    assert second_rows == [["Item", "Value"], ["Total", "-12.5"]]

    lines = (output_dir / "manifest.jsonl").read_bytes().decode("utf-8").splitlines()
    assert len(lines) == 3
    expected_fields = {
        "table_id",
        "company",
        "year",
        "report_type",
        "statement",
        "unit",
        "csv_path",
    }
    first_line = json.loads(lines[0])
    assert set(first_line) == expected_fields
    assert first_line == {
        "table_id": table_a1,
        "company": "ACB",
        "year": 2023,
        "report_type": "consolidated",
        "statement": "income_statement",
        "unit": "vnd",
        "csv_path": "ACB__table_1.csv",
    }
    third_line = json.loads(lines[2])
    assert third_line["statement"] is None
    assert third_line["unit"] is None
    assert third_line["csv_path"] == "VCB__table_1.csv"

    assert manifest.entries[0] == TableExportMetadata(
        table_id=table_a1,
        company="ACB",
        year=2023,
        report_type="consolidated",
        statement="income_statement",
        unit="vnd",
        csv_path="ACB__table_1.csv",
    )


def test_export_rejects_release_manifest_count_mismatch(tmp_path: Path) -> None:
    documents = [_document()]
    tables = [_table_record(TABLE_ID, DOC_ID_A, line_start=10)]
    cells = [_parquet_cell("h00", TABLE_ID, 0, 0, value_raw="Item")]
    release = _write_release(
        tmp_path / "release",
        documents=documents,
        tables=tables,
        cells=cells,
        table_count=2,
    )

    with pytest.raises(ExportError, match="differs"):
        export_normalized_csvs(release, tmp_path / "out")


def test_export_requires_release_manifest(tmp_path: Path) -> None:
    documents = [_document()]
    tables = [_table_record(TABLE_ID, DOC_ID_A, line_start=10)]
    cells = [_parquet_cell("h00", TABLE_ID, 0, 0, value_raw="Item")]
    release = _write_release(
        tmp_path / "release",
        documents=documents,
        tables=tables,
        cells=cells,
        write_manifest=False,
    )

    with pytest.raises(ExportError, match="manifest"):
        export_normalized_csvs(release, tmp_path / "out")


def test_export_surfaces_missing_release_as_export_error(tmp_path: Path) -> None:
    """A nonexistent release dir raises ExportError (not a raw duckdb error)."""
    missing_release = tmp_path / "missing"

    with pytest.raises(ExportError, match="cannot read release parquet") as exc_info:
        export_normalized_csvs(missing_release, tmp_path / "out")

    assert str(missing_release) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, duckdb.Error)


@pytest.mark.parametrize(
    "relative_path",
    ["ACB_fs_extracted.txt", "reports/../ACB_fs_extracted.txt"],
)
def test_export_rejects_unsafe_doc_base_name(tmp_path: Path, relative_path: str) -> None:
    documents = [_document(relative_path=relative_path)]
    tables = [_table_record(TABLE_ID, DOC_ID_A, line_start=10)]
    cells = [_parquet_cell("h00", TABLE_ID, 0, 0, value_raw="Item")]
    release = _write_release(tmp_path / "release", documents=documents, tables=tables, cells=cells)

    with pytest.raises(ExportError, match="safe POSIX base name"):
        export_normalized_csvs(release, tmp_path / "out")


def test_export_rejects_duplicate_line_start_within_document(tmp_path: Path) -> None:
    other_table = "tbl_" + "2" * 64
    documents = [_document()]
    tables = [
        _table_record(TABLE_ID, DOC_ID_A, line_start=10, source_ordinal=0),
        _table_record(other_table, DOC_ID_A, line_start=10, source_ordinal=0),
    ]
    cells = [
        _parquet_cell("t1_h00", TABLE_ID, 0, 0, value_raw="Item"),
        _parquet_cell("t2_h00", other_table, 0, 0, value_raw="Item"),
    ]
    release = _write_release(tmp_path / "release", documents=documents, tables=tables, cells=cells)

    with pytest.raises(ExportError, match="duplicate \\(line_start, source_ordinal\\)"):
        export_normalized_csvs(release, tmp_path / "out")


def test_export_rejects_duplicate_file_name_across_documents(tmp_path: Path) -> None:
    other_doc = "doc_" + "c" * 64
    other_table = "tbl_" + "2" * 64
    documents = [
        _document(relative_path="group_one/ACB/doc.txt"),
        _document(doc_id=other_doc, relative_path="group_two/ACB/doc.txt"),
    ]
    tables = [
        _table_record(TABLE_ID, DOC_ID_A, line_start=10),
        _table_record(other_table, other_doc, line_start=10),
    ]
    cells = [
        _parquet_cell("t1_h00", TABLE_ID, 0, 0, value_raw="Item"),
        _parquet_cell("t2_h00", other_table, 0, 0, value_raw="Item"),
    ]
    release = _write_release(tmp_path / "release", documents=documents, tables=tables, cells=cells)

    with pytest.raises(ExportError, match="duplicate file name"):
        export_normalized_csvs(release, tmp_path / "out")
