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
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from financial_report_qa.core.errors import ProgramGuardError
from financial_report_qa.execution.masked_program import parse_program
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.contracts import SubmissionItem
from financial_report_qa.submission.exporter import _render_csv_bytes

_ANSWER_LIKE_COLUMNS = frozenset({"answer", "result", "ans", "expected"})
# C4 (2026-08-21 final review round 2, Important 1): a bare numeric literal
# must not be flagged when the digit is actually part of an identifier
# (`df1`, `df2` -- the variable-name convention used throughout this
# codebase's generated queries) rather than a standalone number. The
# lookbehind/lookahead reject any digit run that touches a word character
# (letter/digit/underscore) on either side, so `df1` never contributes a
# literal `1` and `1200.0` still matches in full.
_NUMBER_LITERAL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")
_VALUE_TOLERANCE = 1e-6

# C8 chỉ kiểm hình dạng chương trình, không kiểm số ứng viên -- việc đó thuộc
# về binding lúc chạy. Giới hạn rộng để `[NUM_i]` hợp lệ nào cũng qua được.
_MAX_NUM_PLACEHOLDERS = 1000

# C4 (2026-08-21 final review, Important 4): a quoted string literal in the
# query -- most commonly a `column_label` value rendered via `json.dumps`,
# e.g. `df1.column_label == "Năm 2023"` -- can contain digits that are not a
# numeric literal at all. Strip every quoted span before scanning for
# numbers.
_QUOTED_TOKEN = r'"(?:[^"\\]|\\.)*"'
_QUOTED_STRING_PATTERN = re.compile(_QUOTED_TOKEN)

# `compile_grounded`/position binding (plan.md §9/§14) routinely emits
# `row_idx`/`col_idx`/`period` equality clauses (e.g. `df1.row_idx == 19`,
# `df1.period == 2023`) as structural row/column/period locators, not as a
# stand-in for the answer value. An answer that happens to numerically equal
# one of those locators must not trip C4. Strip these comparisons (after
# quoted strings are already gone) before the general literal scan.
_STRUCTURAL_COMPARISON_PATTERN = re.compile(r"\b(?:row_idx|col_idx|period)\s*==\s*-?\d+(?:\.\d+)?")

# C4 (2026-08-21 final review round 2, Important 1): `.iloc[0]`/`.iat[0]`
# (and any other positional index) is a structural row-position accessor,
# not a stand-in for the answer -- it appears in virtually every generated
# query (see `_GOOD_QUERY` in the test suite). Strip the bracketed index of
# an `.iloc[...]`/`.iat[...]` accessor before scanning for literals, the same
# way `_STRUCTURAL_COMPARISON_PATTERN` strips `row_idx == N` clauses.
_POSITIONAL_INDEX_PATTERN = re.compile(r"\.(?:iloc|iat)\s*\[\s*-?\d+(?:\.\d+)?\s*\]")

# Hệ số đổi thang do `program_binding.render_program_pandas` nối vào cuối
# truy vấn (spec 2026-08-24 §4.5). Nó là hằng số của renderer, không phải của
# model, và không bao giờ là chỗ giấu đáp án -- nhưng một đáp án tình cờ bằng
# 100 sẽ trượt C4 nếu không strip. Danh sách đóng, khớp đúng `SCALE_SUFFIX`.
_SCALE_SUFFIX_PATTERN = re.compile(r"(?:\*\s*100|/\s*1000(?:000)?(?:000)?)\s*$")


@dataclass(frozen=True)
class ComplianceViolation:
    """Một vi phạm cụ thể, gắn với đúng một câu hỏi."""

    question_id: int
    code: str
    detail: str


def _numbers_in(query: str) -> list[float]:
    """Literal numbers in `query` that are candidates for a C4 hardcode
    match -- i.e. NOT inside a quoted string, NOT the right-hand side of a
    structural `row_idx`/`col_idx`/`period` equality (position-binding
    locators, not stand-ins for the answer), NOT a positional `.iloc[N]`/
    `.iat[N]` index, NOT a renderer-appended scale suffix (`* 100`,
    `/ 1000`, ...), and NOT a digit that is actually part of an identifier
    such as `df1`/`df2` (handled by `_NUMBER_LITERAL_PATTERN`'s word-boundary
    lookaround)."""
    stripped = _strip_structural_tokens(query)
    out: list[float] = []
    for token in _NUMBER_LITERAL_PATTERN.findall(stripped):
        try:
            out.append(float(token))
        except ValueError:  # pragma: no cover -- regex chỉ khớp số hợp lệ
            continue
    return out


def _strip_structural_tokens(query: str) -> str:
    """Bỏ mọi token cấu trúc trước khi quét literal cho C4."""
    stripped = _QUOTED_STRING_PATTERN.sub(" ", query)
    stripped = _SCALE_SUFFIX_PATTERN.sub(" ", stripped)
    stripped = _STRUCTURAL_COMPARISON_PATTERN.sub(" ", stripped)
    return _POSITIONAL_INDEX_PATTERN.sub(" ", stripped)


def check_program_literals(program: str) -> str | None:
    """C8: chương trình đã lưu phải qua được guard N4'. `None` là hợp lệ."""
    if not program:
        return None
    try:
        parse_program(program, value_count=_MAX_NUM_PLACEHOLDERS)
    except ProgramGuardError as error:
        return str(error)
    return None


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
    #
    # Nhãn được nhúng vào query qua `json.dumps(label, ensure_ascii=False)`
    # (backstop_answer.py, exporter.py), nên một nhãn chứa dấu ngoặc kép
    # thật (vd. `... ("BSC")`) tạo ra chuỗi query có `\"` đã escape. Regex
    # cũ `"([^"]+)"` dừng lại ở dấu `"` escape đó -- cắt cụt nhãn giữa
    # chừng -- rồi báo C6 sai (2026-08-22, phát hiện khi chạy full export
    # thật: 24/1012 câu bị flag sai vì lý do này). Bắt trọn token có escape
    # bằng `_QUOTED_TOKEN` rồi giải mã qua `json.loads` để lấy đúng chuỗi
    # gốc, thay vì so khớp trên văn bản query còn nguyên escape.
    for label_column in ("row_label_raw", "row_label_canonical"):
        if label_column not in frame.columns:
            continue
        quoted = re.findall(rf"{label_column}\s*==\s*({_QUOTED_TOKEN})", query)
        present = {str(v) for v in frame[label_column].dropna().tolist()}
        for raw_token in quoted:
            try:
                label = json.loads(raw_token)
            except ValueError:  # pragma: no cover -- _QUOTED_TOKEN chỉ khớp JSON hợp lệ
                label = raw_token[1:-1]
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

    # C8: N4' -- chương trình do LLM sinh ra không được chứa literal số.
    program_detail = check_program_literals(item.program)
    if program_detail is not None:
        add("C8", f"program vi phạm N4': {program_detail}")

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
    được ghi vào ZIP -- rồi đọc lại, thay vì dựng `pd.DataFrame` thẳng từ
    `csv_rows` trong bộ nhớ. Round-trip qua CSV thật có thể lặng lẽ đổi kiểu
    dữ liệu (vd. `row_label_raw` toàn chữ số như "2" suy luận thành int64),
    khiến một truy vấn khớp trên dict trong bộ nhớ nhưng KHÔNG khớp trên
    chính CSV sẽ ship -- nghĩa là chốt chặn này trước đây kiểm tra tiền thân
    của bundle, không phải bundle thật.

    Đọc lại bằng cách đọc TOÀN BỘ cột dưới dạng chuỗi thô
    (`keep_default_na=False`), rồi tự chuyển đổi `value`/`period`/
    `row_idx`/`col_idx` bằng `float()`/`int()` của Python thay vì để
    `pd.read_csv` tự suy luận kiểu. Hai lý do (phát hiện 2026-08-22 khi
    chạy full export thật, 34/1012 câu bị flag sai vì lý do này):

    1. `pd.read_csv`'s bộ phân tích số mặc định KHÔNG round-trip chính
       xác với số lớn nhiều chữ số có nghĩa (vd. "261095427438.99997" đọc
       lại thành 261095427439.0 -- sai khác 3e-5, đủ để C7 báo sai). `float()`
       của Python thì round-trip chính xác.
    2. `dtype=str` mà không tắt suy luận NA mặc định vẫn biến một ô rỗng
       thật (vd. `row_label_raw == ""` -- nhãn cột trống hợp lệ, không phải
       thiếu dữ liệu) thành NaN, khiến predicate `df1.row_label_raw == ""`
       không khớp gì trên chính CSV sẽ ship -> `.iloc[0]` ném IndexError.
       `_render_csv_bytes` đã gộp `None` thành `""` lúc ghi (không còn phân
       biệt được với chuỗi rỗng thật), nên với các cột nhãn, "" phải được
       giữ nguyên là "" khi đọc lại, không suy ra NaN.
    """
    violations: list[ComplianceViolation] = []
    for item in items:
        csv_path = item.evidence[0].csv_path
        rows = csv_rows.get(csv_path)
        if rows is None:
            violations.append(
                ComplianceViolation(question_id=item.id, code="C0", detail=f"thiếu CSV {csv_path}")
            )
            continue
        csv_bytes = _render_csv_bytes(rows)
        frame = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
        if "value" in frame.columns:
            frame["value"] = frame["value"].map(lambda s: float(s) if s != "" else float("nan"))
        for int_column in ("period", "row_idx", "col_idx"):
            if int_column not in frame.columns:
                continue
            frame[int_column] = pd.array(
                [int(v) if v != "" else pd.NA for v in frame[int_column]], dtype="Int64"
            )
        violations.extend(check_item(item, frame, timeout_seconds=timeout_seconds))
    return tuple(sorted(violations, key=lambda v: (v.question_id, v.code)))
