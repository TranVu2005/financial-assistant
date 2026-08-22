"""Chọn ứng viên tất định khi `locate()` gặp nhập nhằng.

Thiết kế 2026-08-22 §6. Answer Accuracy của cuộc thi tính theo
correct/**total**, nên một câu bỏ trống và một câu trả lời sai đều được 0
điểm -- abstain không mua được gì. Ba hàm ở đây biến ba điểm abstain của
`locator.py` thành lựa chọn có luật, giải trình được.

Mọi hàm phải **tất định**: cùng một tập dòng phải luôn cho cùng một kết quả
bất kể thứ tự dòng trong DataFrame đầu vào. Đó là khác biệt giữa "chọn theo
luật" và `.iloc[0]` tuỳ ý mà thiết kế này thay thế.
"""

from __future__ import annotations

import pandas as pd

_POSITION_COLUMNS = ["table_id", "row_idx", "col_idx"]


def nearest_period_rows(metric_rows: pd.DataFrame, period: int) -> pd.DataFrame:
    """Các dòng ở kỳ gần `period` nhất. Hòa thì ưu tiên kỳ **muộn hơn**.

    Kỳ muộn hơn được ưu tiên vì câu hỏi tài chính thường hỏi số liệu mới
    nhất; khi hệ thống đã không khớp đúng kỳ, đoán về phía gần hiện tại là
    lựa chọn ít sai hơn.
    """
    available = metric_rows["period"].dropna().unique()
    if len(available) == 0:
        return metric_rows.iloc[0:0]
    best = min(
        available, key=lambda value: (abs(int(value) - period), -int(value))
    )
    return metric_rows[metric_rows["period"] == best]


def dominant_value_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Thu hẹp một tập dòng xung đột về đúng một cặp `(value, unit)`.

    Chọn cặp xuất hiện nhiều nhất; hòa thì chọn cặp có vị trí
    `(table_id, row_idx, col_idx)` nhỏ nhất. Sắp xếp trước khi gom nhóm để
    kết quả không phụ thuộc thứ tự dòng đầu vào.
    """
    if rows["value"].nunique() <= 1:
        return rows
    ordered = rows.sort_values(_POSITION_COLUMNS, kind="stable")
    counts = ordered.groupby(
        ["value", "unit"], dropna=False, sort=False
    ).size()
    value, unit = counts.idxmax()
    matches = ordered["value"] == value
    matches &= (
        ordered["unit"].isna()
        if pd.isna(unit)
        else ordered["unit"] == unit
    )
    return ordered[matches]


def infer_unit_from_table(
    frame: pd.DataFrame, table_id: str
) -> str | None:
    """Đơn vị phổ biến nhất trong cùng bảng, hoặc `None` nếu bảng không ghi
    đơn vị ở bất kỳ ô nào.

    Một ô thiếu `unit` gần như luôn là lỗi trích xuất chứ không phải bảng
    thật sự không có đơn vị -- các ô còn lại cùng bảng là bằng chứng tốt
    nhất sẵn có. Hòa thì chọn theo thứ tự chữ cái để tất định.
    """
    same_table = frame[frame["table_id"] == table_id]
    units = same_table["unit"].dropna()
    if units.empty:
        return None
    return str(units.mode().sort_values().iloc[0])
