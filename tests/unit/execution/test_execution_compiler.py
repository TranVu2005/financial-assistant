"""Tests for the Day 18 compile_plan orchestrator (ADR 0007)."""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_MBB = "tbl_" + "2" * 64
DOC_ID_ACB = "doc_" + "a" * 64
DOC_ID_MBB = "doc_" + "b" * 64


def _document(doc_id: str, company: str, year: int) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": f"{company}/{year}/report.txt",
        "company_code": company,
        "report_year": year,
        "statement_scope": "consolidated",
        "sha256": "0" * 64,
        "file_size_bytes": 10,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }


def _table(table_id: str, doc_id: str) -> dict[str, object]:
    return {
        "table_id": table_id,
        "doc_id": doc_id,
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


def _cell(
    cell_id: str,
    table_id: str,
    row: int,
    col: int,
    *,
    row_label_raw: str,
    row_label_canonical: str | None,
    value_numeric: str | None,
    period: str | None,
    unit: str | None = "VND",
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row,
        "col_idx": col,
        "row_label_raw": row_label_raw,
        "row_label_canonical": row_label_canonical,
        "row_group_context_raw": None,
        "column_label_raw": f"Năm {period}" if period else None,
        "column_label_canonical": None,
        "value_raw": value_numeric or "-",
        "value_numeric": Decimal(value_numeric) if value_numeric is not None else None,
        "period": period,
        "unit": unit,
        "source_line_start": row + 1,
        "source_line_end": row + 1,
        "extraction_confidence": 0.9,
    }


def _write_release(tmp_path: Path, cells: list[dict[str, object]]) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [_document(DOC_ID_ACB, "ACB", 2023), _document(DOC_ID_MBB, "MBB", 2023)]
    tables = [_table(TABLE_ID, DOC_ID_ACB), _table(TABLE_ID_MBB, DOC_ID_MBB)]
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


def test_compile_plan_lookup_answers_and_replays(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Tien mat",
                row_label_canonical="cash_and_cash_equivalents",
                value_numeric="100",
                period="2023",
            )
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
    assert result.status == "answered"
    assert result.answer == Decimal("100")
    assert result.unit == "VND"
    assert len(result.evidence) == 1


def test_compile_plan_lookup_negative_value(tmp_path: Path) -> None:
    """Day 18 plan §1.5: negative cells must pass through unmodified."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Loi nhuan",
                row_label_canonical="profit_after_tax",
                value_numeric="-500",
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
    assert result.answer == Decimal("-500")


def test_compile_plan_difference_across_units(tmp_path: Path) -> None:
    """Day 18 plan §1.4: mixed-unit rows must be converted before subtracting."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                value_numeric="2",
                period="2023",
                unit="VND_million",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID,
                2,
                1,
                row_label_raw="Tong tai san",
                row_label_canonical="total_assets",
                value_numeric="1000000",
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
    assert result.answer == Decimal("1")
    assert result.unit == "VND_million"


def test_compile_plan_growth_rate_zero_base_is_division_by_zero(tmp_path: Path) -> None:
    """Day 18 plan §1.5: 4,649 stored cells are exactly zero."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="120",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID,
                2,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
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
        metric=MetricSelector(canonical="profit_after_tax"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "division_by_zero"
    assert result.answer is None


def test_compile_plan_duplicate_conflicting_rows_is_cell_ambiguous(tmp_path: Path) -> None:
    """Day 18 plan §1.4: 93.2% of duplicate-row groups conflict; must never
    silently pick the first row."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID,
                2,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="200",
                period="2023",
            ),
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
    assert result.status == "error"
    assert result.error_code == "cell_ambiguous"


def test_compile_plan_missing_metric_is_metric_not_found(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
                period="2023",
            )
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="total_assets"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "metric_not_found"


def test_compile_plan_unresolvable_period_is_period_unresolved(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
                period="2021",
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
    assert result.status == "error"
    assert result.error_code == "period_unresolved"


def test_compile_plan_rejects_operation_outside_whitelist(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
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
    restricted = ExecutionSettings(
        timeout_seconds=5, max_rows=100000, allow_operations=("difference",)
    )
    result = compile_plan(plan, release_dir, execution_settings=restricted)
    assert result.status == "error"
    assert result.error_code == "operation_not_allowed"


def test_compile_plan_compare_companies(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="1000",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID_MBB,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="750",
                period="2023",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="compare_companies",
        companies=("ACB", "MBB"),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID, TABLE_ID_MBB),
        metric=MetricSelector(canonical="profit_after_tax"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("250")


def test_compile_plan_is_deterministic(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="X",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
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
    first = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    second = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert first == second
