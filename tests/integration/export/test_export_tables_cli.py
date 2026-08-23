"""End-to-end integration tests for the ``export-tables`` CLI.

Builds a tiny release directly with pyarrow (two documents, three tables --
one without statement/unit metadata, headers with rowspan/colspan gaps and
repeated group context), writes matching snapshot TXT files, then drives
``financial_report_qa.export.cli.main`` in-process and asserts the exported
CSV grid, the JSONL manifest, the mirrored synced text, and the failure path
when a snapshot file disappears between runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.export.cli import main as export_main

DOC_A_PATH = "ABC/2023/report.txt"
DOC_B_PATH = "XYZ/2024/report.txt"

TABLE_A1_ID = "tbl_" + "1" * 64
TABLE_A2_ID = "tbl_" + "2" * 64
TABLE_B1_ID = "tbl_" + "3" * 64

CSV_A1_NAME = "2023__table_1.csv"
CSV_A2_NAME = "2023__table_2.csv"
CSV_B1_NAME = "2024__table_1.csv"

# Physical TXT lines; table spans are 1-based inclusive index ranges into these.
DOC_A_LINES = (
    "===== PAGE 1 =====",
    "BÁO CÁO TÀI CHÍNH HỢP NHẤT NĂM 2023",
    "<table>",
    "Tài sản | 31/12/2023 | 31/12/2022",
    "(trống) | Cuối năm | Đầu năm",
    "Tiền và tương đương tiền | 1.000,50 | 950 | 800,250",
    "Các khoản phải thu khách hàng | 1.234 | 2.500.000 |",
    "Tổng cộng | N/A",
    "</table>",
    "Ghi chú: Bảng trên trình bày tài sản ngắn hạn theo giá gốc.",
    "===== PAGE 2 =====",
    "<table>",
    "Kết quả kinh doanh | 2023",
    "Doanh thu thuần | 12.000 | -42,75",
    "Chi phí | (không có)",
    "</table>",
    "Hết báo cáo.",
)
DOC_B_LINES = (
    "===== PAGE 1 =====",
    "BÁO CÁO LƯU CHUYỂN TIỀN TỆ NĂM 2024",
    "<table>",
    "Dòng tiền | 2024",
    "(trống) | Quý 4",
    "Hoạt động kinh doanh | Lợi nhuận trước thuế | 7.000.000 | 7",
    "Khấu hao | n/a | 0,10",
    "</table>",
    "Lập bởi bộ phận kế toán.",
)

EXPECTED_CSV_ROWS: dict[str, list[list[str]]] = {
    CSV_A1_NAME: [
        ["Tài_sản", "31/12/2023_Cuối_năm", "31/12/2022_Đầu_năm"],
        ["TÀI SẢN > Tiền và tương đương tiền", "950", "800.25"],
        ["TÀI SẢN > Các khoản phải thu khách hàng", "2500000", ""],
        ["Tổng cộng", "", ""],
    ],
    CSV_A2_NAME: [
        ["Kết_quả_kinh_doanh", "2023"],
        ["Doanh thu thuần", "-42.75"],
        ["Chi phí", ""],
    ],
    CSV_B1_NAME: [
        ["Dòng_tiền", "2024_Quý_4"],
        ["HOẠT ĐỘNG KINH DOANH > Lợi nhuận trước thuế", "7"],
        ["HOẠT ĐỘNG KINH DOANH > Khấu hao", "0.1"],
    ],
}

EXPECTED_MANIFEST_ENTRIES = [
    {
        "company": "ABC",
        "csv_path": CSV_A1_NAME,
        "report_type": "consolidated",
        "statement": "balance_sheet",
        "table_id": TABLE_A1_ID,
        "unit": "nghìn đồng",
        "year": 2023,
    },
    {
        "company": "ABC",
        "csv_path": CSV_A2_NAME,
        "report_type": "consolidated",
        "statement": None,
        "table_id": TABLE_A2_ID,
        "unit": None,
        "year": 2023,
    },
    {
        "company": "XYZ",
        "csv_path": CSV_B1_NAME,
        "report_type": "separate",
        "statement": "cash_flow_statement",
        "table_id": TABLE_B1_ID,
        "unit": "triệu VND",
        "year": 2024,
    },
]


def _cell(
    table_id: str,
    row_idx: int,
    col_idx: int,
    *,
    value_raw: str = "",
    value_numeric: Decimal | None = None,
    row_label: str | None = None,
    group_context: str | None = None,
    column_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one cells.parquet row plus its placements.parquet row."""
    cell_id = f"{table_id}-c{row_idx}-{col_idx}"
    cell_row = {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_label_raw": row_label,
        "row_label_canonical": None,
        "row_group_context_raw": group_context,
        "column_label_raw": column_label,
        "column_label_canonical": None,
        "value_raw": value_raw,
        "value_numeric": value_numeric,
        "period": None,
        "unit": None,
        "source_line_start": 1,
        "source_line_end": 1,
        "extraction_confidence": 1.0,
    }
    placement_row = {
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "cell_id": cell_id,
    }
    return cell_row, placement_row


def _header_cell(table_id: str, row_idx: int, col_idx: int, text: str) -> (
    tuple[dict[str, Any], dict[str, Any]]
):
    return _cell(table_id, row_idx, col_idx, value_raw=text)


def _register_table(
    table_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    *,
    table_id: str,
    doc_id: str,
    source_ordinal: int,
    title: str,
    statement_type: str | None,
    unit_raw: str | None,
    unit_normalized: str | None,
    line_start: int,
    line_end: int,
    row_count: int,
    column_count: int,
    grid: list[tuple[dict[str, Any], dict[str, Any]]],
) -> None:
    table_rows.append(
        {
            "table_id": table_id,
            "doc_id": doc_id,
            "source_ordinal": source_ordinal,
            "title_raw": title,
            "statement_type": statement_type,
            "unit_raw": unit_raw,
            "unit_normalized": unit_normalized,
            "line_start": line_start,
            "line_end": line_end,
            "row_count": row_count,
            "column_count": column_count,
            "quality_score": 1.0,
            "csv_path": None,
        }
    )
    for cell_row, placement_row in grid:
        cell_rows.append(cell_row)
        placement_rows.append(placement_row)


def _document_row(sha256: str, file_size_bytes: int, relative_path: str, company_code: str,
                  report_year: int, statement_scope: str) -> dict[str, Any]:
    return {
        "doc_id": f"doc_{sha256}",
        "repo_id": "integration-repo",
        "revision": "r1",
        "relative_path": relative_path,
        "company_code": company_code,
        "report_year": report_year,
        "statement_scope": statement_scope,
        "sha256": sha256,
        "file_size_bytes": file_size_bytes,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "b" * 64,
    }


def _write_snapshot(
    snapshot_root: Path, relative_path: str, lines: tuple[str, ...]
) -> tuple[str, int]:
    payload = "".join(line + "\n" for line in lines).encode("utf-8")
    target = snapshot_root.joinpath(*PurePosixPath(relative_path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return digest, len(payload)


def _build_release(tmp_path: Path) -> tuple[Path, Path]:
    """Write one release (parquet + manifest.json) and its snapshot TXT root."""
    release_dir = tmp_path / "release"
    snapshot_root = tmp_path / "snapshot"
    release_dir.mkdir()

    sha_a, size_a = _write_snapshot(snapshot_root, DOC_A_PATH, DOC_A_LINES)
    sha_b, size_b = _write_snapshot(snapshot_root, DOC_B_PATH, DOC_B_LINES)
    doc_a_id = f"doc_{sha_a}"
    doc_b_id = f"doc_{sha_b}"

    document_rows = [
        _document_row(sha_a, size_a, DOC_A_PATH, "ABC", 2023, "consolidated"),
        _document_row(sha_b, size_b, DOC_B_PATH, "XYZ", 2024, "separate"),
    ]

    table_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []

    # Document A, table 1: two-level header with a rowspan-style gap at (1, 0),
    # repeated group context, a missing trailing placement, formatted raw
    # numbers next to their normalized decimals, and a non-numeric fallback.
    a1_grid = [
        _header_cell(TABLE_A1_ID, 0, 0, "Tài sản"),
        _header_cell(TABLE_A1_ID, 0, 1, "31/12/2023"),
        _header_cell(TABLE_A1_ID, 0, 2, "31/12/2022"),
        _header_cell(TABLE_A1_ID, 1, 1, "Cuối năm"),
        _header_cell(TABLE_A1_ID, 1, 2, "Đầu năm"),
        _cell(
            TABLE_A1_ID,
            2,
            0,
            value_raw="1.000,50",
            value_numeric=Decimal("1000.50"),
            row_label="Tiền và tương đương tiền",
            group_context="TÀI SẢN",
        ),
        _cell(
            TABLE_A1_ID,
            2,
            1,
            value_raw="950",
            value_numeric=Decimal("950"),
            column_label="31/12/2023",
        ),
        _cell(
            TABLE_A1_ID,
            2,
            2,
            value_raw="800,250",
            value_numeric=Decimal("800.250"),
            column_label="31/12/2022",
        ),
        _cell(
            TABLE_A1_ID,
            3,
            0,
            value_raw="1.234",
            value_numeric=Decimal("1234.50"),
            row_label="Các khoản phải thu khách hàng",
            group_context="TÀI SẢN",
        ),
        _cell(TABLE_A1_ID, 3, 1, value_raw="2.500.000", value_numeric=Decimal("2500000")),
        _cell(TABLE_A1_ID, 4, 0, value_raw="N/A", row_label="Tổng cộng"),
    ]
    _register_table(
        table_rows, cell_rows, placement_rows,
        table_id=TABLE_A1_ID,
        doc_id=doc_a_id,
        source_ordinal=0,
        title="Bảng cân đối kế toán",
        statement_type="balance_sheet",
        unit_raw="Nghìn đồng",
        unit_normalized="nghìn đồng",
        line_start=3,
        line_end=9,
        row_count=5,
        column_count=3,
        grid=a1_grid,
    )

    # Document A, table 2: single header row, negative decimal, blank cell and
    # no statement/unit metadata anywhere.
    a2_grid = [
        _header_cell(TABLE_A2_ID, 0, 0, "Kết quả kinh doanh"),
        _header_cell(TABLE_A2_ID, 0, 1, "2023"),
        _cell(
            TABLE_A2_ID,
            1,
            0,
            value_raw="12.000",
            value_numeric=Decimal("12000"),
            row_label="Doanh thu thuần",
        ),
        _cell(TABLE_A2_ID, 1, 1, value_raw="-42,75", value_numeric=Decimal("-42.75")),
        _cell(TABLE_A2_ID, 2, 0, value_raw="(không có)", row_label="Chi phí"),
    ]
    _register_table(
        table_rows, cell_rows, placement_rows,
        table_id=TABLE_A2_ID,
        doc_id=doc_a_id,
        source_ordinal=1,
        title="Kết quả kinh doanh",
        statement_type=None,
        unit_raw=None,
        unit_normalized=None,
        line_start=12,
        line_end=16,
        row_count=3,
        column_count=2,
        grid=a2_grid,
    )

    # Document B, table 1: colspan-style gap at (1, 0) and repeated group
    # context across consecutive data rows.
    b1_grid = [
        _header_cell(TABLE_B1_ID, 0, 0, "Dòng tiền"),
        _header_cell(TABLE_B1_ID, 0, 1, "2024"),
        _header_cell(TABLE_B1_ID, 1, 1, "Quý 4"),
        _cell(
            TABLE_B1_ID,
            2,
            0,
            value_raw="7.000.000",
            value_numeric=Decimal("7000000"),
            row_label="Lợi nhuận trước thuế",
            group_context="HOẠT ĐỘNG KINH DOANH",
        ),
        _cell(TABLE_B1_ID, 2, 1, value_raw="7", value_numeric=Decimal("7")),
        _cell(
            TABLE_B1_ID,
            3,
            0,
            value_raw="n/a",
            row_label="Khấu hao",
            group_context="HOẠT ĐỘNG KINH DOANH",
        ),
        _cell(TABLE_B1_ID, 3, 1, value_raw="0,10", value_numeric=Decimal("0.10")),
    ]
    _register_table(
        table_rows, cell_rows, placement_rows,
        table_id=TABLE_B1_ID,
        doc_id=doc_b_id,
        source_ordinal=0,
        title="Báo cáo lưu chuyển tiền tệ",
        statement_type="cash_flow_statement",
        unit_raw="Triệu đồng",
        unit_normalized="triệu VND",
        line_start=3,
        line_end=8,
        row_count=4,
        column_count=2,
        grid=b1_grid,
    )

    write_table = cast(Any, pq.write_table)
    write_table(
        pa.Table.from_pylist(document_rows, schema=DOCUMENT_SCHEMA),
        release_dir / "documents.parquet",
    )
    write_table(
        pa.Table.from_pylist(table_rows, schema=TABLE_SCHEMA),
        release_dir / "tables.parquet",
    )
    write_table(
        pa.Table.from_pylist(cell_rows, schema=CELL_SCHEMA),
        release_dir / "cells.parquet",
    )
    write_table(
        pa.Table.from_pylist(placement_rows, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
    )
    (release_dir / "manifest.json").write_text(
        json.dumps({"document_count": 2, "table_count": 3}), encoding="utf-8"
    )
    return release_dir, snapshot_root


def _run_export(
    release_dir: Path, snapshot_root: Path, csv_dir: Path, text_dir: Path
) -> int:
    return export_main(
        [
            "--release-dir",
            str(release_dir),
            "--snapshot-root",
            str(snapshot_root),
            "--csv-output-dir",
            str(csv_dir),
            "--text-output-dir",
            str(text_dir),
        ]
    )


def _mirror_path(output_dir: Path, relative_path: str) -> Path:
    return output_dir.joinpath(*PurePosixPath(relative_path).parts)


def _read_csv_rows(path: Path) -> list[list[str]]:
    assert path.read_bytes().startswith(b"\xef\xbb\xbf"), f"missing UTF-8 BOM: {path}"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def test_export_tables_cli_end_to_end_writes_csvs_and_synced_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release_dir, snapshot_root = _build_release(tmp_path)
    csv_dir = tmp_path / "csv-out"
    text_dir = tmp_path / "text-out"

    assert _run_export(release_dir, snapshot_root, csv_dir, text_dir) == 0
    assert capsys.readouterr().out.strip() == "exported 3 tables; synced 2 documents"

    manifest_lines = (csv_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in manifest_lines]
    assert entries == EXPECTED_MANIFEST_ENTRIES
    for entry in entries:
        assert isinstance(entry, dict)
        assert set(entry) == {
            "table_id",
            "company",
            "year",
            "report_type",
            "statement",
            "unit",
            "csv_path",
        }
        assert (csv_dir / entry["csv_path"]).is_file()

    for name, expected_rows in EXPECTED_CSV_ROWS.items():
        assert _read_csv_rows(csv_dir / name) == expected_rows, name

    link_a1 = f"[TABLE: {TABLE_A1_ID} -> {(csv_dir / CSV_A1_NAME).as_posix()}]"
    link_a2 = f"[TABLE: {TABLE_A2_ID} -> {(csv_dir / CSV_A2_NAME).as_posix()}]"
    expected_a_text = "".join(
        line + "\n"
        for line in (
            DOC_A_LINES[0],
            DOC_A_LINES[1],
            link_a1,
            DOC_A_LINES[9],
            DOC_A_LINES[10],
            link_a2,
            DOC_A_LINES[16],
        )
    )
    mirrored_a = _mirror_path(text_dir, DOC_A_PATH).read_text(encoding="utf-8")
    assert mirrored_a == expected_a_text

    link_b1 = f"[TABLE: {TABLE_B1_ID} -> {(csv_dir / CSV_B1_NAME).as_posix()}]"
    expected_b_text = "".join(
        line + "\n"
        for line in (DOC_B_LINES[0], DOC_B_LINES[1], link_b1, DOC_B_LINES[8])
    )
    mirrored_b = _mirror_path(text_dir, DOC_B_PATH).read_text(encoding="utf-8")
    assert mirrored_b == expected_b_text

    # Untouched prose and page markers survive byte-for-byte around the links.
    for snippet in ("===== PAGE 1 =====", "===== PAGE 2 =====", DOC_A_LINES[9]):
        assert snippet in mirrored_a
    for snippet in ("===== PAGE 1 =====", DOC_B_LINES[8]):
        assert snippet in mirrored_b


def test_export_tables_cli_missing_snapshot_file_returns_one_with_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release_dir, snapshot_root = _build_release(tmp_path)
    csv_dir = tmp_path / "csv-out"
    text_dir = tmp_path / "text-out"

    assert _run_export(release_dir, snapshot_root, csv_dir, text_dir) == 0
    capsys.readouterr()

    _mirror_path(snapshot_root, DOC_A_PATH).unlink()
    assert _run_export(release_dir, snapshot_root, csv_dir, text_dir) == 1

    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert DOC_A_PATH in captured.err
    assert "exported" not in captured.out
