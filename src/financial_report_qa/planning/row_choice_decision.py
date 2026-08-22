"""Đọc quyết định chọn dòng của LLM và biến nó thành `MetricSelector`.

Thiết kế 2026-08-22 §5.3/§7.1. File quyết định chỉ mang
`{question_id, chosen_index}` -- không mang nhãn, không mang giá trị. Mọi
thông tin khác được tra lại từ danh sách ứng viên dựng ở local, nên một file
quyết định cũ hay hỏng không thể bơm dữ liệu mâu thuẫn với corpus vào bundle.

Selector trả về luôn là **position-bound** (`table_id` + `row_index`), nên
`locator._metric_mask` đi nhánh `_position_mask` -- không so khớp nhãn ở thời
điểm truy vấn. Theo bảng đo trong thiết kế 2026-08-21 §2.2, predicate theo vị
trí nhập nhằng 0.00% so với 31.58% của predicate ngữ nghĩa.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.planning.plan_contracts import MetricSelector
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DecisionSource = str


def load_decisions(path: Path) -> dict[int, int]:
    """Đọc JSONL quyết định thành `{question_id: chosen_index}`.

    Ném `FileNotFoundError` khi thiếu file: rơi âm thầm về đường khác sẽ
    khiến cả một lần chạy 3 tiếng dùng nhầm nguồn quyết định mà không ai
    biết.
    """
    text = path.read_text(encoding="utf-8")
    decisions: dict[int, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        decisions[int(record["question_id"])] = int(record["chosen_index"])
    return decisions


def _selector_from(candidate: RowFusedCandidate) -> MetricSelector | None:
    label = candidate.metadata.row_label_raw
    if label is None or not label.strip():
        return None
    # `MetricSelector.raw_text` is `RawMetricText`: max_length=512 and a
    # control-character-free pattern. `label` comes straight from raw corpus
    # text with no such bound -- a single OCR-merged or malformed label can
    # violate either constraint. Falling through to "no_candidates" for just
    # this one question beats letting `pydantic.ValidationError` propagate
    # uncaught and kill an entire in-progress multi-hour export run.
    try:
        return MetricSelector(
            raw_text=label,
            table_id=candidate.table_id,
            row_index=candidate.row_idx,
        )
    except ValidationError:
        return None


def selector_for(
    question_id: int,
    candidates: Sequence[RowFusedCandidate],
    decisions: Mapping[int, int],
) -> tuple[MetricSelector | None, DecisionSource]:
    """Selector cho một câu, kèm nguồn quyết định để ghi log.

    Nguồn có ba giá trị: `"llm"` (dùng chỉ số LLM trả về), `"fallback_rank1"`
    (thiếu quyết định hoặc chỉ số không hợp lệ -- chọn ứng viên hạng 1, tất
    định và tận dụng được retrieval), `"no_candidates"` (không dựng được
    selector; caller phải rơi xuống backstop).
    """
    if not candidates:
        return None, "no_candidates"

    index = decisions.get(question_id)
    source: DecisionSource = "llm"
    if index is None or not 0 <= index < len(candidates):
        index = 0
        source = "fallback_rank1"

    selector = _selector_from(candidates[index])
    if selector is None:
        return None, "no_candidates"
    return selector, source
