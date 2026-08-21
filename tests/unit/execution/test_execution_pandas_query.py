"""Tests for the Day 18 pandas_query renderer + whitelist replayer (ADR 0007 F1)."""

from decimal import Decimal

import pandas as pd
import pytest

from financial_report_qa.execution.pandas_query import render_pandas_query, replay_pandas_query
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

TABLE_IDS = ("tbl_" + "1" * 64,)


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["period"] = frame["period"].astype("Int64")
    return frame


def _row(
    *,
    company_code: str = "ACB",
    row_label_canonical: str | None = "cash_and_cash_equivalents",
    row_label_raw: str = "Tien mat",
    value: str,
    unit: str = "VND",
    period: int,
    column_label: str | None = None,
) -> dict[str, object]:
    return {
        "company_code": company_code,
        "row_label_canonical": row_label_canonical,
        "row_label_raw": row_label_raw,
        "column_label": column_label,
        "unit": unit,
        "value": Decimal(value),
        "period": period,
    }


def test_render_lookup_produces_readable_query() -> None:
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    assert "df1" in query
    assert "cash_and_cash_equivalents" in query
    assert "2023" in query
    assert "iloc[0]" in query
    assert "eval" not in query and "exec" not in query


@pytest.mark.parametrize(
    ("plan", "expected_column_predicates"),
    [
        (
            FinancialQueryPlan(
                operation="lookup",
                companies=("ACB",),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            1,
        ),
        (
            FinancialQueryPlan(
                operation="difference",
                companies=("ACB",),
                periods=("2024", "2025"),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            2,
        ),
        (
            FinancialQueryPlan(
                operation="growth_rate",
                companies=("ACB",),
                periods=("2024", "2025"),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            3,
        ),
        (
            FinancialQueryPlan(
                operation="compare",
                companies=("ACB",),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                metric_a=MetricSelector(raw_text="Metric A", column_text="Target column"),
                metric_b=MetricSelector(raw_text="Metric B", column_text="Target column"),
            ),
            2,
        ),
        (
            FinancialQueryPlan(
                operation="compare_companies",
                companies=("ACB", "MBB"),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            2,
        ),
        (
            FinancialQueryPlan(
                operation="ratio",
                companies=("ACB",),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                numerator_metric=MetricSelector(raw_text="Numerator", column_text="Target column"),
                denominator_metric=MetricSelector(
                    raw_text="Denominator", column_text="Target column"
                ),
            ),
            2,
        ),
        (
            FinancialQueryPlan(
                operation="average",
                companies=("ACB",),
                periods=("2024", "2025"),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            1,
        ),
        (
            FinancialQueryPlan(
                operation="sum",
                companies=("ACB", "MBB"),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
            ),
            1,
        ),
        (
            FinancialQueryPlan(
                operation="rank",
                companies=("ACB", "MBB"),
                periods=("2025",),
                candidate_table_ids=TABLE_IDS,
                metric=MetricSelector(raw_text="Metric", column_text="Target column"),
                top_k=1,
            ),
            1,
        ),
    ],
)
def test_render_all_operations_preserves_column_selector(
    plan: FinancialQueryPlan, expected_column_predicates: int
) -> None:
    query = render_pandas_query(plan)
    assert query.count("df1.column_label") == expected_column_predicates


def test_replay_lookup_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    frame = _frame([_row(value="100", period=2023)])
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("100")


def test_replay_lookup_with_column_is_independent_of_row_order() -> None:
    """A column-refined plan must encode the chosen column in the submitted
    query instead of relying on whichever matching row reaches iloc[0]."""
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("PC1",),
        periods=("2025",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            raw_text="Thuế giá trị gia tăng",
            column_text="Số phải nộp cuối năm",
        ),
    )
    query = render_pandas_query(plan)
    opening = _row(
        company_code="PC1",
        row_label_canonical=None,
        row_label_raw="Thuế giá trị gia tăng",
        column_label="Số phải nộp đầu năm",
        value="100",
        period=2025,
    )
    closing = _row(
        company_code="PC1",
        row_label_canonical=None,
        row_label_raw="Thuế giá trị gia tăng",
        column_label="Số phải nộp cuối năm",
        value="200",
        period=2025,
    )

    forward = replay_pandas_query(query, _frame([opening, closing]))
    reversed_result = replay_pandas_query(query, _frame([closing, opening]))

    assert forward == reversed_result == Decimal("200")


def test_replay_difference_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="difference",
        companies=("ACB",),
        periods=("2022", "2023"),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(value="800", period=2022),
            _row(value="900", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("100")


def test_replay_ratio_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="ratio",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        numerator_metric=MetricSelector(canonical="cash_and_cash_equivalents"),
        denominator_metric=MetricSelector(canonical="total_assets"),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(row_label_canonical="cash_and_cash_equivalents", value="50", period=2023),
            _row(row_label_canonical="total_assets", value="200", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("0.25")


def test_replay_average_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="average",
        companies=("ACB",),
        periods=("2021", "2022", "2023"),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(value="100", period=2021),
            _row(value="200", period=2022),
            _row(value="300", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("200")


def test_replay_average_over_companies_matches_compiled_answer() -> None:
    """Regression (Day 23 plan Step 2): `render_pandas_query`'s average/sum
    branch hardcoded `company=plan.companies[0]`, so a plan varying over
    companies (>1 company, 1 period) rendered a query that filtered to just
    the first company and averaged a length-1 selection -- silently
    ignoring the rest, never surfacing as a replay mismatch because the
    same wrong assumption was baked into both the compiler and the
    renderer."""
    plan = FinancialQueryPlan(
        operation="average",
        companies=("ACB", "MBB", "VCB"),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(company_code="ACB", value="100", period=2023),
            _row(company_code="MBB", value="200", period=2023),
            _row(company_code="VCB", value="300", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("200")


def test_replay_sum_over_companies_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="sum",
        companies=("ACB", "MBB", "VCB"),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(company_code="ACB", value="100", period=2023),
            _row(company_code="MBB", value="200", period=2023),
            _row(company_code="VCB", value="300", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("600")


def test_replay_rank_matches_compiled_answer() -> None:
    plan = FinancialQueryPlan(
        operation="rank",
        companies=("ACB", "MBB", "VCB"),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(canonical="cash_and_cash_equivalents"),
        top_k=2,
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(company_code="ACB", value="500", period=2023),
            _row(company_code="MBB", value="900", period=2023),
            _row(company_code="VCB", value="300", period=2023),
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("500")


def test_replay_rejects_deeply_nested_expression_without_recursion_error() -> None:
    """Day 19 plan Sec 1.8: a binop nested 1,000 levels deep raises Python's
    own RecursionError (an uncaught, undocumented failure mode) before this
    fix. The structural depth budget must reject it with a ValueError first."""
    frame = _frame([_row(value="1", period=2023)])
    nested = "1" + " + 1" * 1000
    with pytest.raises(ValueError):
        replay_pandas_query(nested, frame)


def test_replay_rejects_query_over_max_length() -> None:
    frame = _frame([_row(value="1", period=2023)])
    huge = "df1.period.isin([" + ", ".join(["2021"] * 2000) + "])"
    with pytest.raises(ValueError):
        replay_pandas_query(huge, frame)


def test_replay_accepts_query_within_budget() -> None:
    frame = _frame([_row(value="1", period=2023)])
    nested = "1" + " + 1" * 20
    assert replay_pandas_query(nested, frame) == Decimal("21")


def test_replay_rejects_unsupported_syntax() -> None:
    """The replayer must fail closed on anything outside its whitelist grammar,
    never fall back to `eval`/`exec` (ADR 0007 F1)."""
    frame = _frame([_row(value="1", period=2023)])
    with pytest.raises(ValueError):
        replay_pandas_query("__import__('os').system('echo hi')", frame)


def test_replay_rejects_assignment_style_injection() -> None:
    frame = _frame([_row(value="1", period=2023)])
    with pytest.raises((ValueError, SyntaxError)):
        replay_pandas_query("df1['value'].iloc[0]; import os", frame)


def test_render_and_replay_survive_real_corpus_label_with_quote() -> None:
    """Day 19 plan Sec 1.1: this exact label exists on a numeric cell in the
    locked release (PNJ, 2018). Naive f-string interpolation breaks Python
    string-literal syntax on the embedded `"` and raises an uncaught
    SyntaxError on valid data -- render must escape via json.dumps instead."""
    label = 'Khấu hao tài sản cố định ("TSCĐ")'
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("PNJ",),
        periods=("2018",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(raw_text=label),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(
                company_code="PNJ",
                row_label_canonical=None,
                row_label_raw=label,
                value="42",
                period=2018,
            )
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("42")


def test_render_escapes_backslash_in_raw_text() -> None:
    label = 'A\\B "C"'
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("PNJ",),
        periods=("2018",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(raw_text=label),
    )
    query = render_pandas_query(plan)
    frame = _frame(
        [
            _row(
                company_code="PNJ",
                row_label_canonical=None,
                row_label_raw=label,
                value="7",
                period=2018,
            )
        ]
    )
    replayed = replay_pandas_query(query, frame)
    assert replayed == Decimal("7")


def _bound_row(
    *,
    table_id: str = "tbl_" + "1" * 64,
    row_idx: int = 14,
    value: str,
    period: int,
    column_label: str | None = None,
) -> dict[str, object]:
    # Decoy labels: they deliberately do NOT match the selector's metric, so
    # any query that still resolves by label finds nothing and the test fails.
    row = _row(
        value=value,
        period=period,
        column_label=column_label,
        row_label_canonical="revenue",
        row_label_raw="Doanh thu",
    )
    row["table_id"] = table_id
    row["row_idx"] = row_idx
    return row


def test_render_position_bound_lookup_uses_df_loc_on_the_row_index() -> None:
    """plan.md §14: a grounded plan extracts by position -- `df.loc[row, col]`
    -- and never re-runs label matching inside Pandas."""
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            canonical="cash_and_cash_equivalents", table_id=TABLE_IDS[0], row_index=14
        ),
    )
    query = render_pandas_query(plan)
    assert "df1.loc[" in query
    assert "df1.row_idx == 14" in query
    assert "row_label_canonical" not in query
    assert "row_label_raw" not in query


def test_replay_position_bound_lookup_returns_the_grounded_cell() -> None:
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            canonical="cash_and_cash_equivalents", table_id=TABLE_IDS[0], row_index=14
        ),
    )
    frame = _frame(
        [
            _bound_row(row_idx=3, value="100", period=2023),
            _bound_row(row_idx=14, value="900", period=2023),
        ]
    )
    assert replay_pandas_query(render_pandas_query(plan), frame) == Decimal("900")


def test_replay_position_bound_difference_selects_both_periods_by_position() -> None:
    plan = FinancialQueryPlan(
        operation="difference",
        companies=("ACB",),
        periods=("2022", "2023"),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            canonical="cash_and_cash_equivalents", table_id=TABLE_IDS[0], row_index=14
        ),
    )
    frame = _frame(
        [
            _bound_row(row_idx=14, value="100", period=2022),
            _bound_row(row_idx=14, value="180", period=2023),
        ]
    )
    assert replay_pandas_query(render_pandas_query(plan), frame) == Decimal("80")


def test_replay_position_bound_average_aggregates_over_periods() -> None:
    plan = FinancialQueryPlan(
        operation="average",
        companies=("ACB",),
        periods=("2022", "2023"),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            canonical="cash_and_cash_equivalents", table_id=TABLE_IDS[0], row_index=14
        ),
    )
    frame = _frame(
        [
            _bound_row(row_idx=14, value="100", period=2022),
            _bound_row(row_idx=14, value="200", period=2023),
        ]
    )
    assert replay_pandas_query(render_pandas_query(plan), frame) == Decimal("150")


def test_replay_position_bound_query_keeps_the_column_predicate() -> None:
    """A row is not a cell: the PC1 tax note puts four amounts on one row, two
    of which resolve to the same period. The grounded column still narrows."""
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=TABLE_IDS,
        metric=MetricSelector(
            canonical="cash_and_cash_equivalents",
            column_text="Số phải nộp cuối năm",
            table_id=TABLE_IDS[0],
            row_index=14,
        ),
    )
    query = render_pandas_query(plan)
    assert "column_label" in query
    frame = _frame(
        [
            _bound_row(row_idx=14, value="10", period=2023, column_label="Số phải nộp đầu năm"),
            _bound_row(row_idx=14, value="70", period=2023, column_label="Số phải nộp cuối năm"),
        ]
    )
    assert replay_pandas_query(query, frame) == Decimal("70")
