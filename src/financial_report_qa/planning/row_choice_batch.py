"""Dựng payload batch cho bước LLM chọn dòng (thiết kế 2026-08-22 §5.3).

LLM nhận câu hỏi + danh sách dòng ứng viên đã đánh số, và chỉ trả về **một
chỉ số nguyên**. Nó không bao giờ thấy giá trị số của ô, và không bao giờ
trả về giá trị -- đáp án luôn được tính lại từ CSV bằng pandas trong sandbox.
Đó là điều giữ cho compiler deterministic và compliance linter còn ý nghĩa.

Thứ tự phần tử trong `candidates` **là** hợp đồng: `row_choice_decision.py`
map `chosen_index` ngược về ứng viên bằng chính vị trí này.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


def _candidate_payload(index: int, candidate: RowFusedCandidate) -> dict[str, object]:
    metadata = candidate.metadata
    return {
        "index": index,
        "row_label": metadata.row_label_raw or "",
        "row_group_context": metadata.row_group_context_raw,
        "statement_type": metadata.statement_type,
        "table_title": metadata.title,
        "periods": list(metadata.periods),
        "units": list(metadata.units),
    }


def build_batch_payload(
    question_id: int,
    question: str,
    candidates: Sequence[RowFusedCandidate],
) -> dict[str, object]:
    """Một dòng JSONL: câu hỏi kèm các dòng ứng viên đã đánh số từ 0.

    `candidates` phải đã ở đúng thứ tự retrieval-rank (điểm cao nhất trước) --
    `RowFusionService.retrieve_rows` trả về đúng thứ tự đó. Không sắp lại.
    """
    return {
        "question_id": question_id,
        "question": question,
        "candidates": [
            _candidate_payload(index, candidate) for index, candidate in enumerate(candidates)
        ],
    }
