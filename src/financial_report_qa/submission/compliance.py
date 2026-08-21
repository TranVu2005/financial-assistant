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

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.contracts import SubmissionItem

_ANSWER_LIKE_COLUMNS = frozenset({"answer", "result", "ans", "expected"})
_NUMBER_LITERAL_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_VALUE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ComplianceViolation:
    """Một vi phạm cụ thể, gắn với đúng một câu hỏi."""

    question_id: int
    code: str
    detail: str


def _numbers_in(query: str) -> list[float]:
    out: list[float] = []
    for token in _NUMBER_LITERAL_PATTERN.findall(query):
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
    """Kiểm tra toàn bộ bundle. Trả về mọi vi phạm, sắp theo question_id."""
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
        frame = pd.DataFrame(list(rows))
        if "period" in frame.columns:
            frame["period"] = frame["period"].astype("Int64")
        violations.extend(check_item(item, frame, timeout_seconds=timeout_seconds))
    return tuple(sorted(violations, key=lambda v: (v.question_id, v.code)))
