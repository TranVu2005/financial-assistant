from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import TableFrameError
from financial_report_qa.data.dataset_builder import (
    CELL_SCHEMA,
    DOCUMENT_SCHEMA,
    PLACEMENT_SCHEMA,
    TABLE_SCHEMA,
)
from financial_report_qa.data.table_frame import (
    build_grid,
    build_values_frame,
    export_table_csvs,
    iter_table_frames,
    load_table_frame,
)

DOC_ID = "doc_" + "a" * 64
TABLE_ID_A = "tbl_" + "1" * 64
TABLE_ID_B = "tbl_" + "2" * 64


def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir()

    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2020/report.txt",
            "company_code": "ACB",
            "report_year": 2020,
            "statement_scope": "consolidated",
            "sha256": "0" * 64,
            "file_size_bytes": 10,
            "encoding": "utf-8",
            "inventory_status": "ready",
            "ruleset_version": "1",
            "normalization_fingerprint": "0" * 64,
        }
    ]

    tables = [
        {
            "table_id": TABLE_ID_A,
            "doc_id": DOC_ID,
            "source_ordinal": 0,
            "title_raw": "Bang can doi ke toan",
            "statement_type": "balance_sheet",
            "unit_raw": "VND",
            "unit_normalized": "vnd",
            "line_start": 1,
            "line_end": 10,
            "row_count": 2,
            "column_count": 3,
            "quality_score": 0.9,
            "csv_path": None,
        },
        {
            "table_id": TABLE_ID_B,
            "doc_id": DOC_ID,
            "source_ordinal": 1,
            "title_raw": None,
            "statement_type": None,
            "unit_raw": None,
            "unit_normalized": None,
            "line_start": 20,
            "line_end": 25,
            "row_count": 1,
            "column_count": 1,
            "quality_score": 0.5,
            "csv_path": None,
        },
    ]

    # Table A: a 2x3 grid where the header cell (0,0)-(0,1) spans two columns
    # via duplicated placements (matching how colspan is flattened at extraction).
    def cell(
        row: int,
        col: int,
        *,
        value: str,
        row_label: str | None = None,
        group_context: str | None = None,
    ) -> dict[str, object]:
        return {
            "cell_id": f"cell_{TABLE_ID_A}_{row}_{col}",
            "table_id": TABLE_ID_A,
            "row_idx": row,
            "col_idx": col,
            "row_label_raw": row_label,
            "row_label_canonical": None,
            "row_group_context_raw": group_context,
            "column_label_raw": None,
            "column_label_canonical": None,
            "value_raw": value,
            "value_numeric": None,
            "period": None,
            "unit": None,
            "source_line_start": row + 1,
            "source_line_end": row + 1,
            "extraction_confidence": 0.9,
        }

    cells = [
        cell(0, 0, value="Header span"),
        cell(0, 2, value="2020"),
        cell(1, 0, value="Tien mat", row_label="Tien mat", group_context="A. TAI SAN"),
        cell(1, 1, value="100"),
        cell(1, 2, value="200"),
        {
            "cell_id": f"cell_{TABLE_ID_B}_0_0",
            "table_id": TABLE_ID_B,
            "row_idx": 0,
            "col_idx": 0,
            "row_label_raw": None,
            "row_label_canonical": None,
            "row_group_context_raw": None,
            "column_label_raw": None,
            "column_label_canonical": None,
            "value_raw": "Solo",
            "value_numeric": None,
            "period": None,
            "unit": None,
            "source_line_start": 20,
            "source_line_end": 20,
            "extraction_confidence": 0.5,
        },
    ]

    placements = [
        {"table_id": TABLE_ID_A, "row_idx": 0, "col_idx": 0, "cell_id": f"cell_{TABLE_ID_A}_0_0"},
        # Colspan: header token at (0,0) also placed at (0,1).
        {"table_id": TABLE_ID_A, "row_idx": 0, "col_idx": 1, "cell_id": f"cell_{TABLE_ID_A}_0_0"},
        {"table_id": TABLE_ID_A, "row_idx": 0, "col_idx": 2, "cell_id": f"cell_{TABLE_ID_A}_0_2"},
        {"table_id": TABLE_ID_A, "row_idx": 1, "col_idx": 0, "cell_id": f"cell_{TABLE_ID_A}_1_0"},
        {"table_id": TABLE_ID_A, "row_idx": 1, "col_idx": 1, "cell_id": f"cell_{TABLE_ID_A}_1_1"},
        {"table_id": TABLE_ID_A, "row_idx": 1, "col_idx": 2, "cell_id": f"cell_{TABLE_ID_A}_1_2"},
        {"table_id": TABLE_ID_B, "row_idx": 0, "col_idx": 0, "cell_id": f"cell_{TABLE_ID_B}_0_0"},
    ]

    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(placements, schema=PLACEMENT_SCHEMA),
        release_dir / "placements.parquet",
    )
    return release_dir


def test_build_grid_pivots_positional_layout_including_colspan_duplication() -> None:
    rows = pd.DataFrame(
        [
            {"row_idx": 0, "col_idx": 0, "value_raw": "Header span"},
            {"row_idx": 0, "col_idx": 1, "value_raw": "Header span"},
            {"row_idx": 0, "col_idx": 2, "value_raw": "2020"},
            {"row_idx": 1, "col_idx": 0, "value_raw": "Tien mat"},
            {"row_idx": 1, "col_idx": 1, "value_raw": "100"},
            {"row_idx": 1, "col_idx": 2, "value_raw": "200"},
        ]
    )
    grid = build_grid(rows)
    assert grid.shape == (2, 3)
    assert grid.iloc[0].tolist() == ["Header span", "Header span", "2020"]
    assert grid.iloc[1].tolist() == ["Tien mat", "100", "200"]


def test_build_grid_returns_empty_frame_for_no_rows() -> None:
    assert build_grid(pd.DataFrame(columns=["row_idx", "col_idx", "value_raw"])).empty


def test_build_values_frame_orders_by_position_and_requires_columns() -> None:
    rows = pd.DataFrame(
        [
            {
                "cell_id": "c2",
                "row_idx": 1,
                "col_idx": 0,
                "row_label_raw": "Tien mat",
                "row_label_canonical": "cash",
                "row_group_context_raw": "A. TAI SAN",
                "column_label_raw": None,
                "column_label_canonical": None,
                "value_raw": "100",
                "value_numeric": 100,
                "period": "2020",
                "unit": "vnd",
                "source_line_start": 2,
                "source_line_end": 2,
                "extraction_confidence": 0.9,
            },
            {
                "cell_id": "c1",
                "row_idx": 0,
                "col_idx": 0,
                "row_label_raw": None,
                "row_label_canonical": None,
                "row_group_context_raw": None,
                "column_label_raw": None,
                "column_label_canonical": None,
                "value_raw": "Header",
                "value_numeric": None,
                "period": None,
                "unit": None,
                "source_line_start": 1,
                "source_line_end": 1,
                "extraction_confidence": 0.9,
            },
        ]
    )
    values = build_values_frame(rows)
    assert values["cell_id"].tolist() == ["c1", "c2"]
    assert values.loc[1, "row_group_context_raw"] == "A. TAI SAN"

    with pytest.raises(TableFrameError):
        build_values_frame(rows.drop(columns=["period"]))


def test_load_table_frame_reconstructs_grid_and_values(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    table_frame = load_table_frame(release_dir, TABLE_ID_A)

    assert table_frame.doc_id == DOC_ID
    assert table_frame.relative_path == "ACB/2020/report.txt"
    assert table_frame.title_raw == "Bang can doi ke toan"
    assert table_frame.statement_type == "balance_sheet"
    assert table_frame.unit_normalized == "vnd"

    assert table_frame.grid.shape == (2, 3)
    assert table_frame.grid.iloc[0].tolist() == ["Header span", "Header span", "2020"]
    assert table_frame.grid.iloc[1].tolist() == ["Tien mat", "100", "200"]

    # One row per grid position (placement), so the colspanned header cell_id
    # appears twice: once at (0, 0) and once at (0, 1).
    assert len(table_frame.values) == 6
    assert table_frame.values["cell_id"].duplicated().sum() == 1
    tien_mat_row = table_frame.values[table_frame.values["row_label_raw"] == "Tien mat"].iloc[0]
    assert tien_mat_row["row_group_context_raw"] == "A. TAI SAN"


def test_load_table_frame_handles_null_metadata(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    table_frame = load_table_frame(release_dir, TABLE_ID_B)
    assert table_frame.title_raw is None
    assert table_frame.statement_type is None
    assert table_frame.unit_normalized is None
    assert table_frame.grid.iloc[0].tolist() == ["Solo"]


def test_load_table_frame_raises_for_unknown_table_id(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    with pytest.raises(TableFrameError):
        load_table_frame(release_dir, "tbl_" + "9" * 64)


def test_iter_table_frames_yields_every_table_in_order(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    frames = list(iter_table_frames(release_dir))
    assert [f.table_id for f in frames] == [TABLE_ID_A, TABLE_ID_B]
    assert frames[0].grid.shape == (2, 3)
    assert frames[1].grid.shape == (1, 1)


def test_export_table_csvs_writes_files_and_manifest(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    output_dir = tmp_path / "csv_out"
    result = export_table_csvs(release_dir, output_dir)

    assert result.table_count == 2
    assert (output_dir / f"{TABLE_ID_A}.csv").is_file()
    assert (output_dir / f"{TABLE_ID_B}.csv").is_file()
    assert result.manifest_path.is_file()

    written = (output_dir / f"{TABLE_ID_A}.csv").read_text(encoding="utf-8-sig").splitlines()
    assert written[0] == "Header span,Header span,2020"
    assert written[1] == "Tien mat,100,200"


def test_export_table_csvs_writes_a_utf8_bom_for_excel_compatibility(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    output_dir = tmp_path / "csv_out"
    export_table_csvs(release_dir, output_dir)

    raw = (output_dir / f"{TABLE_ID_A}.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")


def test_export_table_csvs_filters_by_table_ids(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    output_dir = tmp_path / "csv_out"
    result = export_table_csvs(release_dir, output_dir, table_ids=[TABLE_ID_B])

    assert result.table_count == 1
    assert not (output_dir / f"{TABLE_ID_A}.csv").exists()
    assert (output_dir / f"{TABLE_ID_B}.csv").is_file()
