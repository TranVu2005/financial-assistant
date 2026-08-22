"""Tie-break tất định thay cho abstain (design 2026-08-22 §6).

Mỗi hàm phải tất định: cùng input luôn cho cùng output, kể cả khi có hòa.
Đó là điều kiện để giải trình được lựa chọn, khác với `.iloc[0]` tuỳ ý.
"""

from __future__ import annotations

import pandas as pd

from financial_report_qa.execution.tiebreak import (
    dominant_value_rows,
    infer_unit_from_table,
    nearest_period_rows,
)


def _rows(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["period"] = frame["period"].astype("Int64")
    return frame


def _row(
    *,
    period: int | None = 2023,
    value: float = 100.0,
    unit: str | None = "VND",
    table_id: str = "t1",
    row_idx: int = 1,
    col_idx: int = 1,
    statutory_code: str | None = None,
) -> dict[str, object]:
    return {
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "period": period,
        "value": value,
        "unit": unit,
        "statutory_code": statutory_code,
    }


def test_nearest_period_picks_closest() -> None:
    rows = _rows([_row(period=2020, value=1.0), _row(period=2023, value=2.0)])
    picked = nearest_period_rows(rows, 2022)
    assert picked["period"].unique().tolist() == [2023]


def test_nearest_period_tie_prefers_the_later_period() -> None:
    """2021 và 2023 cách đều 2022 -- phải chọn 2023 (muộn hơn), tất định."""
    rows = _rows([_row(period=2021, value=1.0), _row(period=2023, value=2.0)])
    picked = nearest_period_rows(rows, 2022)
    assert picked["period"].unique().tolist() == [2023]


def test_nearest_period_returns_empty_when_no_period_at_all() -> None:
    rows = _rows([_row(period=None, value=1.0)])
    assert nearest_period_rows(rows, 2022).empty


def test_dominant_value_keeps_single_value_untouched() -> None:
    rows = _rows([_row(value=5.0, col_idx=1), _row(value=5.0, col_idx=2)])
    assert len(dominant_value_rows(rows)) == 2


def test_dominant_value_picks_the_most_frequent_pair() -> None:
    rows = _rows(
        [
            _row(value=5.0, col_idx=1),
            _row(value=5.0, col_idx=2),
            _row(value=9.0, col_idx=3),
        ]
    )
    picked = dominant_value_rows(rows)
    assert picked["value"].unique().tolist() == [5.0]


def test_dominant_value_tie_breaks_on_position_deterministically() -> None:
    """Hai giá trị cùng tần suất -- phải chọn cái xuất hiện trước theo
    (table_id, row_idx, col_idx), và phải cho kết quả giống nhau bất kể
    thứ tự dòng trong input."""
    forward = _rows([_row(value=5.0, col_idx=1), _row(value=9.0, col_idx=2)])
    reverse = _rows([_row(value=9.0, col_idx=2), _row(value=5.0, col_idx=1)])
    assert dominant_value_rows(forward)["value"].unique().tolist() == [5.0]
    assert dominant_value_rows(reverse)["value"].unique().tolist() == [5.0]


def test_infer_unit_returns_most_common_unit_of_that_table() -> None:
    frame = _rows(
        [
            _row(table_id="t1", col_idx=1, unit="VND"),
            _row(table_id="t1", col_idx=2, unit="VND"),
            _row(table_id="t1", col_idx=3, unit="trieu_VND"),
            _row(table_id="t2", col_idx=1, unit="USD"),
        ]
    )
    assert infer_unit_from_table(frame, "t1") == "VND"


def test_infer_unit_returns_none_when_table_has_no_unit() -> None:
    frame = _rows([_row(table_id="t1", unit=None)])
    assert infer_unit_from_table(frame, "t1") is None
