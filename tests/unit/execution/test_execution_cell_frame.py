"""Tests for the Day 18 long-format cell projection (ADR 0007 decision B1/C2)."""

from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import ExecutionInputError
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.cell_frame import build_cell_frame

DOC_ID = "doc_" + "a" * 64
TABLE_ID = "tbl_" + "1" * 64


def _write_release(
    tmp_path: Path,
    *,
    report_year: int = 2020,
    cells: list[dict[str, object]] | None = None,
) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)

    documents = [
        {
            "doc_id": DOC_ID,
            "repo_id": "repo",
            "revision": "1",
            "relative_path": "ACB/2020/report.txt",
            "company_code": "ACB",
            "report_year": report_year,
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
            "table_id": TABLE_ID,
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
        }
    ]

    def cell(
        cell_id: str,
        row: int,
        col: int,
        *,
        row_label_raw: str | None,
        row_label_canonical: str | None = None,
        column_label_raw: str | None = None,
        value_raw: str = "",
        value_numeric: str | None = None,
        period: str | None = None,
        unit: str | None = None,
    ) -> dict[str, object]:
        return {
            "cell_id": cell_id,
            "table_id": TABLE_ID,
            "row_idx": row,
            "col_idx": col,
            "row_label_raw": row_label_raw,
            "row_label_canonical": row_label_canonical,
            "row_group_context_raw": None,
            "column_label_raw": column_label_raw,
            "column_label_canonical": None,
            "value_raw": value_raw,
            "value_numeric": Decimal(value_numeric) if value_numeric is not None else None,
            "period": period,
            "unit": unit,
            "source_line_start": row + 1,
            "source_line_end": row + 1,
            "extraction_confidence": 0.9,
        }

    if cells is None:
        cells = [
            # col_idx = 0: row label column, must be excluded (Day 18 plan §1.4).
            cell(
                "cell_label_row0",
                0,
                0,
                row_label_raw="Tien mat",
                value_raw="Tien mat",
            ),
            # explicit period, canonical YYYY.
            cell(
                "cell_explicit_2020",
                0,
                1,
                row_label_raw="Tien mat",
                row_label_canonical="cash",
                column_label_raw="Năm 2020",
                value_raw="100",
                value_numeric="100",
                period="2020",
                unit="VND",
            ),
            # explicit period, ISO date form (Day 18 plan §1.1).
            cell(
                "cell_explicit_iso",
                1,
                1,
                row_label_raw="Tai san khac",
                column_label_raw="Năm 2020",
                value_raw="50",
                value_numeric="50",
                period="2020-12-31",
                unit="VND",
            ),
            # no explicit period, "Số cuối năm" -> report_year.
            cell(
                "cell_closing",
                2,
                1,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                column_label_raw="Số cuối năm VND",
                value_raw="900",
                value_numeric="900",
                period=None,
                unit="VND",
            ),
            # no explicit period, "Số đầu năm" -> report_year - 1.
            cell(
                "cell_opening",
                2,
                2,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                column_label_raw="Số đầu năm VND",
                value_raw="800",
                value_numeric="800",
                period=None,
                unit="VND",
            ),
            # no explicit period, no recognizable column label -> period stays null.
            cell(
                "cell_unresolved",
                3,
                1,
                row_label_raw="Khoan muc khac",
                column_label_raw="Tổng cộng",
                value_raw="10",
                value_numeric="10",
                period=None,
                unit="VND",
            ),
            # numeric value but null-typed row: must be excluded (no value_numeric).
            cell(
                "cell_no_value",
                4,
                1,
                row_label_raw="Ghi chu",
                column_label_raw="Năm 2020",
                value_raw="-",
                value_numeric=None,
                period="2020",
                unit=None,
            ),
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
    return release_dir


def test_build_cell_frame_excludes_col_idx_zero_label_cells(tmp_path: Path) -> None:
    """col_idx == 0 holds the row label, never a value (Day 18 plan §1.4: only
    64/11,734 such cells across the whole corpus even carry value_numeric)."""
    release_dir = _write_release(tmp_path)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    assert (frame["col_idx"] == 0).sum() == 0


def test_build_cell_frame_excludes_null_numeric_value(tmp_path: Path) -> None:
    """A dash/placeholder cell must never enter the compiler's search space."""
    release_dir = _write_release(tmp_path)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    assert "cell_no_value" not in frame["cell_id"].tolist()


def test_build_cell_frame_normalizes_explicit_period_to_year(tmp_path: Path) -> None:
    """Day 18 plan §1.1: 37.7% of period values are ISO dates, not bare years;
    comparing to plan.periods (which are `^\\d{4}$`) requires normalization."""
    release_dir = _write_release(tmp_path)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_explicit_iso"]
    assert row["period"] == 2020
    assert bool(row["period_inferred"]) is False


def test_build_cell_frame_infers_period_from_closing_balance_column(tmp_path: Path) -> None:
    """ADR 0007 decision C2: "Số cuối năm" maps to the document's report_year."""
    release_dir = _write_release(tmp_path, report_year=2020)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_closing"]
    assert row["period"] == 2020
    assert bool(row["period_inferred"]) is True


def test_build_cell_frame_infers_period_from_opening_balance_column(tmp_path: Path) -> None:
    """ADR 0007 decision C2: "Số đầu năm" maps to report_year - 1."""
    release_dir = _write_release(tmp_path, report_year=2020)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_opening"]
    assert row["period"] == 2019
    assert bool(row["period_inferred"]) is True


def test_build_cell_frame_leaves_period_null_when_unresolvable(tmp_path: Path) -> None:
    """A column label with no period marker and no explicit `period` cannot be
    resolved; it must stay null rather than default to report_year, which
    would silently misattribute the value to the wrong year."""
    release_dir = _write_release(tmp_path)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_unresolved"]
    assert pd.isna(row["period"])


def test_build_cell_frame_preserves_explicit_period_over_column_label_marker(
    tmp_path: Path,
) -> None:
    """C2 priority order: an explicit `period` always wins even if the column
    label also happens to contain an opening/closing marker."""
    release_dir = _write_release(
        tmp_path,
        report_year=2099,
        cells=[
            {
                "cell_id": "cell_explicit_wins",
                "table_id": TABLE_ID,
                "row_idx": 0,
                "col_idx": 1,
                "row_label_raw": "X",
                "row_label_canonical": None,
                "row_group_context_raw": None,
                "column_label_raw": "Số cuối năm VND",
                "column_label_canonical": None,
                "value_raw": "5",
                "value_numeric": Decimal("5"),
                "period": "2021",
                "unit": "VND",
                "source_line_start": 1,
                "source_line_end": 1,
                "extraction_confidence": 0.9,
            }
        ],
    )
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_explicit_wins"]
    assert row["period"] == 2021
    assert bool(row["period_inferred"]) is False


def test_build_cell_frame_keeps_null_unit_as_null_not_nan_string(tmp_path: Path) -> None:
    """Day 20 plan Sec 1.3 / ADR 0009 decision C1: when a table mixes a cell
    with an explicit unit and a cell with `unit IS NULL`, DuckDB's pandas
    conversion turns the null cell's unit into a genuine float NaN rather
    than None -- `str(nan)` is the string `'nan'`, a fabricated unit that
    must never reach `CellMatch`. Reproduced with the minimal fixture below:
    a single-unit table does NOT trigger it, only a mixed table does."""
    release_dir = _write_release(
        tmp_path,
        cells=[
            {
                "cell_id": "cell_" + "a" * 64,
                "table_id": TABLE_ID,
                "row_idx": 0,
                "col_idx": 1,
                "row_label_raw": "X",
                "row_label_canonical": None,
                "row_group_context_raw": None,
                "column_label_raw": "Năm 2020",
                "column_label_canonical": None,
                "value_raw": "5",
                "value_numeric": Decimal("5"),
                "period": "2020",
                "unit": "VND",
                "source_line_start": 1,
                "source_line_end": 1,
                "extraction_confidence": 0.9,
            },
            {
                "cell_id": "cell_" + "b" * 64,
                "table_id": TABLE_ID,
                "row_idx": 1,
                "col_idx": 1,
                "row_label_raw": "Y",
                "row_label_canonical": None,
                "row_group_context_raw": None,
                "column_label_raw": "Năm 2020",
                "column_label_canonical": None,
                "value_raw": "7",
                "value_numeric": Decimal("7"),
                "period": "2020",
                "unit": None,
                "source_line_start": 2,
                "source_line_end": 2,
                "extraction_confidence": 0.9,
            },
        ],
    )
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_" + "b" * 64]
    assert pd.isna(row["unit"])
    assert str(row["unit"]) != "nan"


def test_build_cell_frame_rejects_empty_table_ids(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    with pytest.raises(ExecutionInputError):
        build_cell_frame(release_dir, [])


def test_hardened_connection_blocks_external_network_access() -> None:
    """ADR 0008 decision F1: Day 19 plan Sec 1.7 measured that
    `duckdb.connect(":memory:")` defaults to `enable_external_access=True`
    and autoloads/autoinstalls extensions -- disable all three so the engine
    that reads the release cannot reach the network, even though no plan
    field ever reaches raw SQL (candidate_table_ids are placeholder-bound,
    schema-constrained to `^tbl_[0-9a-f]{64}$`)."""
    import duckdb

    from financial_report_qa.execution.cell_frame import _hardened_connection

    connection = _hardened_connection()
    try:
        with pytest.raises(duckdb.Error):
            connection.execute("SELECT * FROM read_csv_auto('https://example.com/x.csv')")
    finally:
        connection.close()


def test_build_cell_frame_carries_row_label_and_value_columns(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    frame = build_cell_frame(release_dir, [TABLE_ID])
    row = frame.set_index("cell_id").loc["cell_explicit_2020"]
    assert row["row_label_raw"] == "Tien mat"
    assert row["row_label_canonical"] == "cash"
    assert row["value"] == Decimal("100")
    assert row["unit"] == "VND"
    assert row["table_id"] == TABLE_ID
    assert row["company_code"] == "ACB"
