"""Day 18 golden tests: the five scenarios plan.md names explicitly —
negative values, null cells, duplicate rows, mixed units, division by zero.

Each test is anchored to a specific measurement in
[docs/plans/day18-deterministic-compiler.md](../../../docs/plans/day18-deterministic-compiler.md)
so the scenario is not arbitrary: it reflects something actually observed in
the locked release, not a hypothetical edge case.
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64

_ALLOW_ALL = ExecutionSettings(
    timeout_seconds=5,
    max_rows=100000,
    allow_operations=(
        "lookup",
        "compare",
        "compare_companies",
        "difference",
        "growth_rate",
        "ratio",
        "average",
        "sum",
        "rank",
    ),
)


def _document(report_year: int = 2023) -> dict[str, object]:
    return {
        "doc_id": DOC_ID,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": "ACB/2023/report.txt",
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


def _table() -> dict[str, object]:
    return {
        "table_id": TABLE_ID,
        "doc_id": DOC_ID,
        "source_ordinal": 0,
        "title_raw": "Bang can doi ke toan",
        "statement_type": "balance_sheet",
        "unit_raw": "VND",
        "unit_normalized": "vnd",
        "line_start": 1,
        "line_end": 10,
        "row_count": 5,
        "column_count": 3,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _cell(
    cell_id: str,
    row: int,
    col: int,
    *,
    row_label_raw: str,
    row_label_canonical: str | None,
    value_raw: str,
    value_numeric: str | None,
    period: str | None,
    unit: str | None = "VND",
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": TABLE_ID,
        "row_idx": row,
        "col_idx": col,
        "row_label_raw": row_label_raw,
        "row_label_canonical": row_label_canonical,
        "row_group_context_raw": None,
        "column_label_raw": f"Năm {period}" if period else None,
        "column_label_canonical": None,
        "value_raw": value_raw,
        "value_numeric": Decimal(value_numeric) if value_numeric is not None else None,
        "period": period,
        "unit": unit,
        "source_line_start": row + 1,
        "source_line_end": row + 1,
        "extraction_confidence": 0.9,
    }


def _write_release(
    tmp_path: Path, cells: list[dict[str, object]], *, report_year: int = 2023
) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([_document(report_year)], schema=DOCUMENT_SCHEMA),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([_table()], schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def test_golden_negative_value_preserves_sign(tmp_path: Path) -> None:
    """Day 18 plan §1.5: 326,782/327,743 negative cells (99.7%) are stored via
    parenthesis notation in `value_raw` (e.g. `(1.234)`), but `value_numeric`
    already carries the correct sign — the compiler must read it as-is and
    never re-parse or flip it."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                1,
                1,
                row_label_raw="Loi nhuan sau thue",
                row_label_canonical="profit_after_tax",
                value_raw="(1.234)",
                value_numeric="-1234",
                period="2023",
            )
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="profit_after_tax"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("-1234")


def test_golden_null_cell_never_enters_the_search_space(tmp_path: Path) -> None:
    """Day 18 plan §1.5: the most common `value_raw` for a NULL `value_numeric`
    cell is a bare dash `-` (546,246 occurrences) or an empty string (546,637).
    Such a cell must be invisible to the locator — never coerced to 0, and
    never mistaken for the real value one row down."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                1,
                1,
                row_label_raw="Chi phi khac",
                row_label_canonical="other_income",
                value_raw="-",
                value_numeric=None,
                period="2023",
                unit=None,
            ),
            _cell(
                "cell_" + "b" * 64,
                2,
                1,
                row_label_raw="Chi phi khac",
                row_label_canonical="other_income",
                value_raw="500",
                value_numeric="500",
                period="2023",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="other_income"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("500")


def test_golden_duplicate_row_conflict_is_never_silently_resolved(tmp_path: Path) -> None:
    """Day 18 plan §1.4: 33,321/35,766 duplicate-row groups (93.2%) disagree on
    value. Picking the first physical row would produce a wrong answer with
    no indication anything was wrong; the compiler must refuse instead."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                1,
                1,
                row_label_raw="Tien va tuong duong tien",
                row_label_canonical="cash_and_cash_equivalents",
                value_raw="100",
                value_numeric="100",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                2,
                1,
                row_label_raw="Tien va tuong duong tien",
                row_label_canonical="cash_and_cash_equivalents",
                value_raw="150",
                value_numeric="150",
                period="2023",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "cell_ambiguous"
    assert result.answer is None


def test_golden_mixed_unit_rows_are_converted_before_arithmetic(tmp_path: Path) -> None:
    """Day 18 plan §1.4: 501/47,040 tables (1.06%) mix units across rows/columns
    within the same table. A difference between a VND_million cell and a raw
    VND cell must be computed on a shared scale, never as raw-digit
    subtraction, which would be off by six orders of magnitude."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                1,
                1,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                value_raw="12",
                value_numeric="12",
                period="2023",
                unit="VND_million",
            ),
            _cell(
                "cell_" + "b" * 64,
                2,
                1,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                value_raw="10000000",
                value_numeric="10000000",
                period="2022",
                unit="VND",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="difference",
        companies=("ACB",),
        periods=("2022", "2023"),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="total_assets"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("2")
    assert result.unit == "VND_million"


def test_golden_division_by_zero_is_a_typed_error_not_inf_or_nan(tmp_path: Path) -> None:
    """Day 18 plan §1.5: 4,649 stored cells are exactly `value_numeric = 0`
    across 1,257 tables. A growth_rate or ratio landing on one of them must
    fail with `division_by_zero`, never emit `Infinity`/`NaN` — plan.md
    explicitly bans both from the submission format."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                1,
                1,
                row_label_raw="Doanh thu",
                row_label_canonical="net_revenue",
                value_raw="500",
                value_numeric="500",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                2,
                1,
                row_label_raw="Doanh thu",
                row_label_canonical="net_revenue",
                value_raw="0",
                value_numeric="0",
                period="2022",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="growth_rate",
        companies=("ACB",),
        periods=("2022", "2023"),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="net_revenue"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "division_by_zero"
    assert result.answer is None
