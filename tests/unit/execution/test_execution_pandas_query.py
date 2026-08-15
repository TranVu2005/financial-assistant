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
) -> dict[str, object]:
    return {
        "company_code": company_code,
        "row_label_canonical": row_label_canonical,
        "row_label_raw": row_label_raw,
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
