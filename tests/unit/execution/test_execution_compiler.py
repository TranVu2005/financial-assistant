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


def _document(
    doc_id: str, company: str, year: int, *, statement_scope: str = "consolidated"
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": f"{company}/{year}/report.txt",
        "company_code": company,
        "report_year": year,
        "statement_scope": statement_scope,
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
    column_label_raw: str | None = None,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row,
        "col_idx": col,
        "row_label_raw": row_label_raw,
        "row_label_canonical": row_label_canonical,
        "row_group_context_raw": None,
        "column_label_raw": (
            column_label_raw
            if column_label_raw is not None
            else f"Năm {period}"
            if period
            else None
        ),
        "column_label_canonical": None,
        "value_raw": value_numeric or "-",
        "value_numeric": Decimal(value_numeric) if value_numeric is not None else None,
        "period": period,
        "unit": unit,
        "source_line_start": row + 1,
        "source_line_end": row + 1,
        "extraction_confidence": 0.9,
    }


def _write_release(
    tmp_path: Path,
    cells: list[dict[str, object]],
    *,
    extra_documents: list[dict[str, object]] | None = None,
    extra_tables: list[dict[str, object]] | None = None,
) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [
        _document(DOC_ID_ACB, "ACB", 2023),
        _document(DOC_ID_MBB, "MBB", 2023),
        *(extra_documents or ()),
    ]
    tables = [_table(TABLE_ID, DOC_ID_ACB), _table(TABLE_ID_MBB, DOC_ID_MBB), *(extra_tables or ())]
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


def test_compile_plan_answered_exposes_replay_rows_matching_pandas_query(tmp_path: Path) -> None:
    """A submission exporter must be able to materialize the exact DataFrame
    `pandas_query` (variable `df1`) operated on without re-deriving compiler
    internals -- `replay_rows` is that frame, in row form."""
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
    assert len(result.replay_rows) == 1
    row = result.replay_rows[0]
    assert row.company_code == "ACB"
    assert row.row_label_canonical == "cash_and_cash_equivalents"
    assert row.period == 2023
    assert row.value == Decimal("100")


def test_compile_plan_error_result_has_no_replay_rows(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, [])
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.replay_rows == ()


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


def test_compile_plan_column_refinement_is_preserved_in_replay_contract(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Thuế giá trị gia tăng",
                row_label_canonical=None,
                value_numeric="100",
                period="2025",
                column_label_raw="Số phải nộp đầu năm",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID,
                1,
                2,
                row_label_raw="Thuế giá trị gia tăng",
                row_label_canonical=None,
                value_numeric="200",
                period="2025",
                column_label_raw="Số phải nộp cuối năm",
            ),
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2025",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(
            raw_text="Thuế giá trị gia tăng",
            column_text="Số phải nộp cuối năm",
        ),
    )

    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)

    assert result.status == "answered"
    assert result.answer == Decimal("200")
    assert 'df1.column_label == "Số phải nộp cuối năm"' in result.pandas_query
    assert result.replay_rows[0].model_dump(mode="python") == {
        "company_code": "ACB",
        "row_label_canonical": None,
        "row_label_raw": "Thuế giá trị gia tăng",
        "column_label": "Số phải nộp cuối năm",
        "period": 2025,
        "value": Decimal("200"),
        # plan.md §14: unset on an unbound selector -- the rendered query for
        # this plan still filters by label, so `df1` carries no position.
        "table_id": None,
        "row_index": None,
    }


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


def test_compile_plan_average_varies_over_companies_not_only_first(tmp_path: Path) -> None:
    """Regression (Day 23 plan Step 2): the average/sum dispatch always
    iterated `plan.periods` at a fixed `companies[0]`, so a plan varying
    over companies (>1 company, 1 period -- exactly what
    `_validate_aggregate` already allows) silently averaged/summed just the
    first company, ignoring the rest, instead of raising or producing a
    length-1 no-op result that would at least be visibly wrong."""
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
        operation="average",
        companies=("ACB", "MBB"),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID, TABLE_ID_MBB),
        metric=MetricSelector(canonical="profit_after_tax"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("875")


def test_compile_plan_sum_varies_over_companies_not_only_first(tmp_path: Path) -> None:
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
        operation="sum",
        companies=("ACB", "MBB"),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID, TABLE_ID_MBB),
        metric=MetricSelector(canonical="profit_after_tax"),
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("1750")


def test_compile_plan_missing_unit_is_unit_missing_not_unit_incompatible(tmp_path: Path) -> None:
    """Day 20 plan Sec 1.3: a cell with no recorded unit (20.7% of numeric
    cells corpus-wide) must be reported as `unit_missing`, distinct from
    `unit_incompatible` (a mixed-unit conversion failure) -- ADR 0009 C1."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Thu nhap khac",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
                period="2023",
                unit=None,
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
    assert result.error_code == "unit_missing"


def test_compile_plan_rejects_rank_without_top_k(tmp_path: Path) -> None:
    """Day 19 plan Sec 1.10: `top_k` arity was only enforced when a caller
    happened to invoke `validate_plan_semantics`. `compile_plan` must not
    trust the caller (ADR 0008 decision E1)."""
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
        operation="rank",
        companies=("ACB", "MBB"),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="profit_after_tax"),
        top_k=None,
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "plan_rejected"


def test_compile_plan_rejects_frame_over_max_rows(tmp_path: Path) -> None:
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
                row_label_raw="Y",
                row_label_canonical="total_assets",
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
    tiny_limit = ExecutionSettings(timeout_seconds=5, max_rows=1, allow_operations=("lookup",))
    result = compile_plan(plan, release_dir, execution_settings=tiny_limit)
    assert result.status == "error"
    assert result.error_code == "row_limit_exceeded"


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


TABLE_ID_ACB_SEPARATE = "tbl_" + "3" * 64
DOC_ID_ACB_SEPARATE = "doc_" + "c" * 64


def _write_dual_scope_release(tmp_path: Path) -> Path:
    """Day 21 plan §1.2/§1.4: same company/metric/period, two real values
    split by statement_scope -- the exact shape measured on gold70 (e.g.
    CTG operating cash flow 2022: 84,420,878 separate vs 84,463,729
    consolidated)."""
    return _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="LNST",
                row_label_canonical="profit_after_tax",
                value_numeric="100",
                period="2023",
            ),
            _cell(
                "cell_" + "b" * 64,
                TABLE_ID_ACB_SEPARATE,
                1,
                1,
                row_label_raw="LNST",
                row_label_canonical="profit_after_tax",
                value_numeric="200",
                period="2023",
            ),
        ],
        extra_documents=[_document(DOC_ID_ACB_SEPARATE, "ACB", 2023, statement_scope="separate")],
        extra_tables=[_table(TABLE_ID_ACB_SEPARATE, DOC_ID_ACB_SEPARATE)],
    )


def _dual_scope_plan(*, statement_scope: str | None) -> FinancialQueryPlan:
    return FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID, TABLE_ID_ACB_SEPARATE),
        metric=MetricSelector(canonical="profit_after_tax"),
        statement_scope=statement_scope,  # type: ignore[arg-type]
    )


def test_compile_plan_without_scope_is_ambiguous_across_separate_and_consolidated(
    tmp_path: Path,
) -> None:
    """Baseline: without any scope information, two real candidate tables
    with genuinely different values must still abstain -- this is the
    Day 21 §1.1/§1.2 regression, and scope filtering must not silently
    resolve a real conflict."""
    release_dir = _write_dual_scope_release(tmp_path)
    plan = _dual_scope_plan(statement_scope=None)
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "cell_ambiguous"


def test_compile_plan_filters_candidates_by_plan_statement_scope(tmp_path: Path) -> None:
    """ADR 0010 decision A1: a plan that states its own scope resolves
    deterministically to that scope's value, with `scope_inferred=False`."""
    release_dir = _write_dual_scope_release(tmp_path)
    plan = _dual_scope_plan(statement_scope="separate")
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("200")
    assert result.scope_inferred is False


def test_compile_plan_applies_default_statement_scope_when_plan_unset(tmp_path: Path) -> None:
    """ADR 0010 decision B1: when the plan is silent, `ExecutionSettings.
    default_statement_scope` resolves the conflict but is flagged
    `scope_inferred=True` so verification can block it from being presented
    as a certain answer."""
    release_dir = _write_dual_scope_release(tmp_path)
    plan = _dual_scope_plan(statement_scope=None)
    settings = ExecutionSettings(
        timeout_seconds=5,
        max_rows=100000,
        allow_operations=("lookup",),
        default_statement_scope="consolidated",
    )
    result = compile_plan(plan, release_dir, execution_settings=settings)
    assert result.status == "answered"
    assert result.answer == Decimal("100")
    assert result.scope_inferred is True


def test_compile_plan_scope_filter_that_removes_every_candidate_is_an_error(
    tmp_path: Path,
) -> None:
    """When a plan states a scope no candidate table has, this must be a
    typed error, never a silent fall-through to the unfiltered set."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="LNST",
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
        statement_scope="separate",
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "candidate_table_ids_scope_empty"


def test_compile_plan_answered_result_has_scope_inferred_false_by_default(tmp_path: Path) -> None:
    """No scope stated and no default configured: unfiltered behavior is
    preserved exactly (pre-Day-21 tests all rely on this)."""
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
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.scope_inferred is False


def test_compile_plan_performs_unit_scale_conversion(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Doanh thu",
                row_label_canonical="net_revenue",
                value_numeric="5000",
                period="2023",
                unit="VND_million",
            )
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="net_revenue"),
        expected_unit="VND_billion",
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("5")
    assert result.unit == "VND_billion"


def test_compile_plan_rejects_incompatible_unit_conversion(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell(
                "cell_" + "a" * 64,
                TABLE_ID,
                1,
                1,
                row_label_raw="Biên lợi nhuận",
                row_label_canonical="profit_after_tax",
                value_numeric="0.45",
                period="2023",
                unit="percent",
            )
        ],
    )
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="profit_after_tax"),
        expected_unit="VND",
    )
    result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "error"
    assert result.error_code == "unit_incompatible"


def test_planners_propagate_expected_unit() -> None:
    from financial_report_qa.planning.entity_contracts import QueryEntities
    from financial_report_qa.planning.rule_planner import build_plan

    entities = QueryEntities(
        parser_version="entity-parser-v2",
        question="Doanh thu thuần của ACB năm 2023 là bao nhiêu tỷ đồng?",
        company_codes=("ACB",),
        periods=("2023",),
        metrics=("net_revenue",),
        requested_unit="billion_vnd",
    )
    result = build_plan(
        entities,
        candidate_table_ids=(TABLE_ID,),
        known_table_ids=frozenset([TABLE_ID]),
    )
    assert result.plan is not None
    assert result.plan.expected_unit == "VND_billion"



def test_compile_plan_position_bound_selector_resolves_a_row_label_collision(
    tmp_path: Path,
) -> None:
    """Spec 2026-08-21 §5.2/§7.1 end to end: two rows share one label and
    disagree, which an unbound plan can only report as `cell_ambiguous`.
    Grounding pins the row, the rendered query names the metric label and
    uses `row_idx` only to break this duplicate-label tie, and the replay of
    the rendered `df.loc[...]` string -- against a frame carrying the real
    corpus labels -- must agree with the compiled answer."""
    cells = [
        _cell(
            "cell_" + "a" * 64,
            TABLE_ID,
            3,
            1,
            row_label_raw="Tien mat",
            row_label_canonical="cash_and_cash_equivalents",
            value_numeric="100",
            period="2023",
        ),
        _cell(
            "cell_" + "b" * 64,
            TABLE_ID,
            14,
            1,
            row_label_raw="Tien mat",
            row_label_canonical="cash_and_cash_equivalents",
            value_numeric="900",
            period="2023",
        ),
    ]
    release_dir = _write_release(tmp_path, cells)
    unbound = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(TABLE_ID,),
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    assert compile_plan(unbound, release_dir, execution_settings=_ALLOW_ALL).error_code == (
        "cell_ambiguous"
    )

    bound = unbound.model_copy(
        update={
            "metric": MetricSelector(
                canonical="cash_and_cash_equivalents", table_id=TABLE_ID, row_index=14
            )
        }
    )
    result = compile_plan(bound, release_dir, execution_settings=_ALLOW_ALL)
    assert result.status == "answered"
    assert result.answer == Decimal("900")
    assert "df1.loc[" in result.pandas_query
    assert 'df1.row_label_canonical == "cash_and_cash_equivalents"' in result.pandas_query
    assert result.evidence[0].row_index == 14
    assert result.replay_rows[0].row_index == 14
    assert result.replay_rows[0].table_id == TABLE_ID
    # Replay rows carry corpus labels, never selector text: this selector has
    # no raw_text at all, yet the frame row it selected does.
    assert result.replay_rows[0].row_label_canonical == "cash_and_cash_equivalents"
    assert result.replay_rows[0].row_label_raw == "Tien mat"
