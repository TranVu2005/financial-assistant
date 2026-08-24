"""Tests for the pandas_query whitelist replayer (ADR 0007 F1).

The plan-era renderer (`render_pandas_query`) was removed with the
operation-enum answering path; the shipped query strings now come from
`program_binding.render_program_pandas`, and what must keep working here is
the deny-by-default replay over exactly those strings.
"""

from decimal import Decimal

import pandas as pd
import pytest

from financial_report_qa.execution.pandas_query import _lit, replay_pandas_query

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
    table_id: str = "tbl_" + "1" * 64,
    row_idx: int = 0,
) -> dict[str, object]:
    return {
        "company_code": company_code,
        "row_label_canonical": row_label_canonical,
        "row_label_raw": row_label_raw,
        "column_label": column_label,
        "unit": unit,
        "value": Decimal(value),
        "period": period,
        "table_id": table_id,
        "row_idx": row_idx,
    }


def test_replay_semantic_lookup_matches_the_grounded_cell() -> None:
    """The masked-PAL lookup shape: label predicate first (it states which
    metric the answer reads), `table_id`/`row_idx` riding along to break ties
    between same-label duplicate rows (spec 2026-08-21 §5.2)."""
    query = (
        f"df1[(df1.row_label_raw == {_lit('Tien mat')})"
        f" & (df1.table_id == {_lit(TABLE_IDS[0])})"
        " & (df1.row_idx == 14)"
        ' & (df1.period == 2023)]["value"].iloc[0]'
    )
    frame = _frame([_row(value="900", period=2023, row_idx=14)])
    assert replay_pandas_query(query, frame) == Decimal("900")


def test_replay_position_bound_lookup_uses_df_loc_on_the_row_index() -> None:
    query = (
        f"df1.loc[(df1.row_label_raw == {_lit('Tien mat')})"
        f" & (df1.table_id == {_lit(TABLE_IDS[0])})"
        " & (df1.row_idx == 14)"
        ' & (df1.period == 2023), "value"].iloc[0]'
    )
    frame = _frame([_row(value="900", period=2023, row_idx=14)])
    assert replay_pandas_query(query, frame) == Decimal("900")


def test_replay_a_wrong_label_never_satisfies_the_predicate() -> None:
    """A row at the right position but with the wrong label must never
    satisfy a semantic-first query."""
    query = (
        f"df1[(df1.row_label_raw == {_lit('Tien mat')})"
        f" & (df1.table_id == {_lit(TABLE_IDS[0])})"
        " & (df1.row_idx == 14)"
        ' & (df1.period == 2023)]["value"].iloc[0]'
    )
    frame = _frame([_row(value="900", period=2023, row_idx=14, row_label_raw="Khac")])
    with pytest.raises(IndexError):
        replay_pandas_query(query, frame)


def test_replay_survives_real_corpus_label_with_quote() -> None:
    """Day 19 plan Sec 1.1: this exact label exists on a numeric cell in the
    locked release (PNJ, 2018). Naive f-string interpolation breaks Python
    string-literal syntax on the embedded `"` -- `_lit()` must escape via
    json.dumps instead."""
    label = 'Khấu hao tài sản cố định ("TSCĐ")'
    query = f'df1[(df1.row_label_raw == {_lit(label)}) & (df1.period == 2018)]["value"].iloc[0]'
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
    assert replay_pandas_query(query, frame) == Decimal("42")


def test_lit_escapes_backslash_in_raw_text() -> None:
    label = 'A\\B "C"'
    query = f'df1[(df1.row_label_raw == {_lit(label)}) & (df1.period == 2018)]["value"].iloc[0]'
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
    assert replay_pandas_query(query, frame) == Decimal("7")


def test_replay_difference_and_division_shapes() -> None:
    start = 'df1[(df1.row_label_raw == "Tien mat") & (df1.period == 2022)]["value"].iloc[0]'
    end = 'df1[(df1.row_label_raw == "Tien mat") & (df1.period == 2023)]["value"].iloc[0]'
    frame = _frame([_row(value="100", period=2022), _row(value="120", period=2023)])
    assert replay_pandas_query(f"{end} - {start}", frame) == Decimal("20")
    growth = replay_pandas_query(f"({end} - {start}) / abs({start})", frame)
    assert growth == Decimal("0.2")


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
