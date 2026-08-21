"""Chốt chặn chống hardcode cho bundle nộp bài.

Thể lệ Stage 2 quy định: "Kết quả của mỗi pandas query phải được tính toán
trực tiếp từ dữ liệu có trong các bảng CSV tại thời điểm thực thi. Không được
gán cứng, mã hóa hoặc lưu sẵn kết quả dưới bất kỳ hình thức nào. Các câu hỏi
vi phạm quy định này sẽ không được tính điểm." Mục VIII còn liệt kê "Hardcode
đáp án benchmark" là hành vi có thể bị loại đội thi.

Module này biến quy định đó thành phép kiểm mechanical. Nó chỉ đọc, không sửa
gì: `submission/cli.py` gọi nó và fail build khi có vi phạm.
"""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.contracts import SubmissionItem
from financial_report_qa.submission.exporter import _render_csv_bytes

# Forced dtype for every string-compared column (pandas_query.py
# `_ALLOWED_ATTRS`), mirroring `validator.py`'s own CSV round-trip: a
# `row_label_raw` that looks like a pure number (e.g. a footnote reference
# such as "2") would otherwise round-trip through the CSV as pandas'
# *inferred* int64, silently breaking `df1.row_label_raw == "2"` (int
# compared to str never matches). `table_id` is left to type inference --
# its values are always `tbl_<64 hex>`, never numeric-looking.
_STRING_DTYPE_COLUMNS = {
    "company_code": str,
    "row_label_canonical": str,
    "row_label_raw": str,
    "column_label": str,
}

_ANSWER_LIKE_COLUMNS = frozenset({"answer", "result", "ans", "expected"})
_NUMBER_LITERAL_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_VALUE_TOLERANCE = 1e-6

# C4 (2026-08-21 final review, Important 4): a quoted string literal in the
# query -- most commonly a `column_label` value rendered via `json.dumps`,
# e.g. `df1.column_label == "Năm 2023"` -- can contain digits that are not a
# numeric literal at all. Strip every quoted span before scanning for
# numbers.
_QUOTED_STRING_PATTERN = re.compile(r'"(?:[^"\\]|\\.)*"')

# `compile_grounded`/position binding (plan.md §9/§14) routinely emits
# `row_idx`/`col_idx`/`period` equality clauses (e.g. `df1.row_idx == 19`,
# `df1.period == 2023`) as structural row/column/period locators, not as a
# stand-in for the answer value. An answer that happens to numerically equal
# one of those locators must not trip C4. Strip these comparisons (after
# quoted strings are already gone) before the general literal scan.
_STRUCTURAL_COMPARISON_PATTERN = re.compile(r"\b(?:row_idx|col_idx|period)\s*==\s*-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class ComplianceViolation:
    """Một vi phạm cụ thể, gắn với đúng một câu hỏi."""

    question_id: int
    code: str
    detail: str


def _numbers_in(query: str) -> list[float]:
    """Literal numbers in `query` that are candidates for a C4 hardcode
    match -- i.e. NOT inside a quoted string, and NOT the right-hand side of
    a structural `row_idx`/`col_idx`/`period` equality (position-binding
    locators, not stand-ins for the answer)."""
    stripped = _QUOTED_STRING_PATTERN.sub("", query)
    stripped = _STRUCTURAL_COMPARISON_PATTERN.sub("", stripped)
    out: list[float] = []
    for token in _NUMBER_LITERAL_PATTERN.findall(stripped):
        try:
            out.append(float(token))
        except ValueError:  # pragma: no cover -- regex chỉ khớp số hợp lệ
            continue
    return out


def check_item(
    item: SubmissionItem, frame: pd.DataFrame, *, timeout_seconds: float
) -> tuple[ComplianceViolation, ...]:
    """Trả về mọi vi phạm của một câu. Rỗng nghĩa là hợp lệ."""
    violations: list[ComplianceViolation] = []
    query = item.pandas_query

    def add(code: str, detail: str) -> None:
        violations.append(ComplianceViolation(question_id=item.id, code=code, detail=detail))

    # C1: CSV phải là lát cắt bảng thật, không phải một ô dựng ngược từ đáp án.
    if len(frame) < 2:
        add("C1", f"CSV chỉ có {len(frame)} dòng dữ liệu (cần >= 2)")

    # C2: đáp án không được là giá trị duy nhất nằm sẵn trong CSV.
    if len(frame) == 1 and "value" in frame.columns:
        only = frame["value"].iloc[0]
        if isinstance(only, (int, float)) and math.isfinite(float(only)):
            if abs(float(only) - item.answer) <= _VALUE_TOLERANCE:
                add("C2", f"answer {item.answer} là giá trị duy nhất trong CSV")

    # C3: không được có cột mang sẵn đáp án.
    named = _ANSWER_LIKE_COLUMNS.intersection(str(c).lower() for c in frame.columns)
    if named:
        add("C3", f"CSV chứa cột mang sẵn đáp án: {sorted(named)}")

    # C4: đáp án không được xuất hiện dưới dạng hằng số trong query.
    for literal in _numbers_in(query):
        if abs(literal - item.answer) <= _VALUE_TOLERANCE:
            add("C4", f"pandas_query chứa literal {literal} trùng answer")
            break

    # C5: query phải thực sự đọc từ CSV.
    referenced = [str(c) for c in frame.columns if re.search(rf"\b{re.escape(str(c))}\b", query)]
    if not referenced:
        add("C5", "pandas_query không tham chiếu cột nào của CSV")

    # C6: nhãn dòng nêu trong query phải tồn tại trong CSV.
    for label_column in ("row_label_raw", "row_label_canonical"):
        if label_column not in frame.columns:
            continue
        quoted = re.findall(rf"{label_column}\s*==\s*\"([^\"]+)\"", query)
        present = {str(v) for v in frame[label_column].dropna().tolist()}
        for label in quoted:
            if label not in present:
                add("C6", f"{label_column}=={label!r} không có trong CSV")

    # C7: bằng chứng quyết định -- đáp án phải replay được từ chính CSV này.
    result = replay_in_sandbox(query, frame, timeout_seconds=timeout_seconds)
    if result.error_code is not None:
        add("C7", f"replay lỗi: {result.error_code}: {result.error_message}")
    elif result.value is None:
        add("C7", "replay không trả về giá trị")
    else:
        value_float = float(result.value)
        if math.isnan(value_float):
            add("C7", "replay trả về NaN")
        elif abs(value_float - item.answer) > _VALUE_TOLERANCE:
            add("C7", f"replay ra {value_float} nhưng answer là {item.answer}")

    return tuple(violations)


def check_bundle(
    items: Sequence[SubmissionItem],
    csv_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    timeout_seconds: float,
) -> tuple[ComplianceViolation, ...]:
    """Kiểm tra toàn bộ bundle. Trả về mọi vi phạm, sắp theo question_id.

    Quan trọng (Important 3, 2026-08-21 final review): mỗi CSV được kiểm
    tra bằng cách render qua `exporter._render_csv_bytes` -- CHÍNH bytes sẽ
    được ghi vào ZIP -- rồi đọc lại bằng `pd.read_csv`, thay vì dựng
    `pd.DataFrame` thẳng từ `csv_rows` trong bộ nhớ. Round-trip qua CSV thật
    có thể lặng lẽ đổi kiểu dữ liệu (vd. `row_label_raw` toàn chữ số như
    "2" suy luận thành int64), khiến một truy vấn khớp trên dict trong bộ
    nhớ nhưng KHÔNG khớp trên chính CSV sẽ ship -- nghĩa là chốt chặn này
    trước đây kiểm tra tiền thân của bundle, không phải bundle thật.
    """
    violations: list[ComplianceViolation] = []
    for item in items:
        csv_path = item.evidence[0].csv_path
        rows = csv_rows.get(csv_path)
        if rows is None:
            violations.append(
                ComplianceViolation(
                    question_id=item.id, code="C0", detail=f"thiếu CSV {csv_path}"
                )
            )
            continue
        csv_bytes = _render_csv_bytes(rows)
        frame = pd.read_csv(io.BytesIO(csv_bytes), dtype=_STRING_DTYPE_COLUMNS)
        if "period" in frame.columns:
            frame["period"] = frame["period"].astype("Int64")
        violations.extend(check_item(item, frame, timeout_seconds=timeout_seconds))
    return tuple(sorted(violations, key=lambda v: (v.question_id, v.code)))
