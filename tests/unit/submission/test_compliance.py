"""Kiểm tra 7 bất biến chống hardcode của bundle nộp bài."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_report_qa.submission.compliance import check_item
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
