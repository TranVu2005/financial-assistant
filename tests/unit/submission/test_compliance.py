"""Kiểm tra 7 bất biến chống hardcode của bundle nộp bài."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_report_qa.submission.compliance import check_bundle, check_item
from financial_report_qa.submission.contracts import SubmissionEvidence, SubmissionItem


def _item(*, answer: float, query: str) -> SubmissionItem:
    return SubmissionItem.model_validate(
        {
            "id": 1,
            "question": "Doanh thu thuần năm 2023?",
            "answer": answer,
            "relevant_docs": ("VNM_2023",),
            "relevant_tables": ("VNM_2023|100",),
            "evidence": (SubmissionEvidence(variable="df1", csv_path="data/q000001_df1.csv"),),
            "pandas_query": query,
        }
    )


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["period"] = frame["period"].astype("Int64")
    return frame


_GOOD_ROWS = [
    {"company_code": "VNM", "row_label_raw": "Doanh thu thuần", "column_label": "2023",
     "period": 2023, "value": 1200.0},
    {"company_code": "VNM", "row_label_raw": "Lợi nhuận sau thuế", "column_label": "2023",
     "period": 2023, "value": 120.0},
]
_GOOD_QUERY = 'df1[(df1.row_label_raw == "Doanh thu thuần") & (df1.period == 2023)]["value"].iloc[0]'


def test_compliant_item_has_no_violations() -> None:
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert violations == ()


def test_c1_single_row_csv_is_a_violation() -> None:
    rows = [_GOOD_ROWS[0]]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C1" in {v.code for v in violations}


def test_c2_answer_equal_to_only_value_is_a_violation() -> None:
    rows = [_GOOD_ROWS[0]]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C2" in {v.code for v in violations}


def test_c3_answer_named_column_is_a_violation() -> None:
    rows = [dict(row, answer=row["value"]) for row in _GOOD_ROWS]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C3" in {v.code for v in violations}


def test_c4_answer_literal_in_query_is_a_violation() -> None:
    violations = check_item(
        _item(answer=1200.0, query="1200.0"), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C4" in {v.code for v in violations}


def test_c5_query_referencing_no_csv_column_is_a_violation() -> None:
    violations = check_item(
        _item(answer=1200.0, query="1200.0"), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C5" in {v.code for v in violations}


def test_c4_does_not_false_positive_on_row_idx_structural_literal() -> None:
    """Important 4: position-bound queries (plan.md §9/§14) routinely
    contain `df1.row_idx == N`. An answer that happens to numerically equal
    that row index must not trigger C4."""
    query = 'df1.loc[(df1.table_id == "t1") & (df1.row_idx == 19), "value"].iloc[0]'
    violations = check_item(
        _item(answer=19.0, query=query), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C4" not in {v.code for v in violations}


def test_c4_does_not_false_positive_on_period_structural_literal() -> None:
    query = 'df1[(df1.row_label_raw == "Doanh thu thuần") & (df1.period == 2023)]["value"].iloc[0]'
    violations = check_item(
        _item(answer=2023.0, query=query), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C4" not in {v.code for v in violations}


def test_c4_does_not_false_positive_on_digit_inside_quoted_column_label() -> None:
    """A `column_label` quoted string like `"2023"` can contain the answer's
    digits without the query hardcoding the answer as a bare literal."""
    query = (
        'df1[(df1.row_label_raw == "Doanh thu thuần") & '
        '(df1.column_label == "2023")]["value"].iloc[0]'
    )
    violations = check_item(
        _item(answer=2023.0, query=query), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C4" not in {v.code for v in violations}


def test_c4_still_catches_a_genuine_bare_literal_answer() -> None:
    """The fix must not blanket-disable C4 -- a real hardcoded literal that
    is neither quoted nor a structural comparison must still be caught."""
    violations = check_item(
        _item(answer=1200.0, query='df1.row_idx == 3 and 1200.0'),
        _frame(_GOOD_ROWS),
        timeout_seconds=5,
    )
    assert "C4" in {v.code for v in violations}


def test_c7_query_that_cannot_replay_is_a_violation() -> None:
    query = 'df1.loc[(df1.table_id == "t1") & (df1.row_idx == 3), "value"].iloc[0]'
    violations = check_item(
        _item(answer=1200.0, query=query), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C7" in {v.code for v in violations}


def test_c7_wrong_replay_value_is_a_violation() -> None:
    violations = check_item(
        _item(answer=999.0, query=_GOOD_QUERY), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C7" in {v.code for v in violations}


def _csv_row(
    *, row_label_raw: str, value: float, table_id: str = "tbl_" + "1" * 64, row_idx: int = 0
) -> dict[str, object]:
    return {
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": 1,
        "company_code": "VNM",
        "row_label_canonical": None,
        "row_label_raw": row_label_raw,
        "column_label": "2023",
        "period": 2023,
        "value": value,
    }


def test_check_bundle_validates_the_real_rendered_csv_not_the_in_memory_dicts() -> None:
    """Important 3 (2026-08-21 final review): `check_bundle` must render each
    item's rows through the exact same `_render_csv_bytes` the ZIP ships,
    then read them back with the same dtype-forcing `pd.read_csv` approach
    `validator.py` uses -- not build a `pd.DataFrame` straight from the
    in-memory dicts. A `row_label_raw` that looks like a pure number (e.g. a
    footnote reference such as "2") round-trips through a real CSV as
    pandas' *inferred* int64 unless the dtype is forced, which would
    silently break `df1.row_label_raw == "2"` (int compared to str never
    matches) -- exercised here via `check_bundle`'s own public entry point,
    not `check_item` directly, so the CSV-rendering step is actually
    exercised.
    """
    rows = [
        _csv_row(row_label_raw="2", value=100.0, row_idx=0),
        _csv_row(row_label_raw="3", value=200.0, row_idx=1),
    ]
    query = 'df1[df1.row_label_raw == "2"]["value"].iloc[0]'
    item = _item(answer=100.0, query=query)

    violations = check_bundle([item], {"data/q000001_df1.csv": rows}, timeout_seconds=5)

    assert violations == (), f"dtype round-trip phải giữ row_label_raw là chuỗi: {violations}"


def test_check_bundle_reports_missing_csv_as_c0() -> None:
    item = _item(answer=1200.0, query=_GOOD_QUERY)
    violations = check_bundle([item], {}, timeout_seconds=5)
    assert [v.code for v in violations] == ["C0"]


def test_c7_nan_replay_value_is_a_violation() -> None:
    """Test that replay producing NaN is flagged as C7 violation.

    Tạo DataFrame có giá trị NaN để test, truy vấn lấy ra giá trị NaN này.
    """
    # DataFrame có chứa NaN
    rows_with_nan = [
        {"company_code": "VNM", "row_label_raw": "Doanh thu thuần", "column_label": "2023",
         "period": 2023, "value": float('nan')},
        {"company_code": "VNM", "row_label_raw": "Lợi nhuận sau thuế", "column_label": "2023",
         "period": 2023, "value": 120.0},
    ]
    # Truy vấn lấy giá trị NaN
    query = 'df1[(df1.row_label_raw == "Doanh thu thuần") & (df1.period == 2023)]["value"].iloc[0]'
    violations = check_item(
        _item(answer=1200.0, query=query), _frame(rows_with_nan), timeout_seconds=5
    )
    assert "C7" in {v.code for v in violations}
    # Kiểm tra detail message chứa "NaN"
    c7_violations = [v for v in violations if v.code == "C7"]
    assert any("NaN" in v.detail for v in c7_violations)
