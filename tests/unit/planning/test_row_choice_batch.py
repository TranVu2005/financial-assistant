"""Payload batch gửi cho LLM chọn dòng (thiết kế 2026-08-22 §5.3)."""

from __future__ import annotations

from financial_report_qa.planning.row_choice_batch import build_batch_payload
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

# TableId bị ràng buộc ^tbl_[0-9a-f]{64}$ -- "t1" sẽ ném ValidationError.
_TABLE_ID = "tbl_" + "a" * 64


def _candidate(*, rank: int, row_idx: int, label: str) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=_TABLE_ID,
            row_idx=row_idx,
            row_label_raw=label,
            row_group_context_raw="IV. Tài sản ngắn hạn khác",
            statement_type="balance_sheet",
            title="BẢNG CÂN ĐỐI KẾ TOÁN",
            periods=("2023", "2022"),
            units=("VND",),
        ),
        snippet=label,
    )


def test_payload_has_question_and_indexed_candidates() -> None:
    payload = build_batch_payload(
        795,
        "Chi phí trả trước ngắn hạn khác cuối 2023?",
        [_candidate(rank=1, row_idx=3, label="Chi phí trả trước ngắn hạn khác")],
    )
    assert payload["question_id"] == 795
    assert payload["question"] == "Chi phí trả trước ngắn hạn khác cuối 2023?"
    assert payload["candidates"][0]["index"] == 0
    assert payload["candidates"][0]["row_label"] == "Chi phí trả trước ngắn hạn khác"


def test_candidate_indices_are_contiguous_and_follow_rank_order() -> None:
    """Index phải khớp vị trí trong danh sách -- Task 4 map ngược index về
    candidate bằng chính thứ tự này."""
    candidates = [
        _candidate(rank=1, row_idx=3, label="A"),
        _candidate(rank=2, row_idx=7, label="B"),
        _candidate(rank=3, row_idx=9, label="C"),
    ]
    payload = build_batch_payload(1, "câu hỏi", candidates)
    assert [c["index"] for c in payload["candidates"]] == [0, 1, 2]
    assert [c["row_label"] for c in payload["candidates"]] == ["A", "B", "C"]


def test_payload_never_leaks_a_cell_value() -> None:
    """Bất biến của thiết kế: LLM không được thấy giá trị số. Nó chỉ chọn
    dòng; đáp án luôn được tính lại từ CSV trong sandbox."""
    payload = build_batch_payload(
        1, "câu hỏi", [_candidate(rank=1, row_idx=3, label="A")]
    )
    serialized = repr(payload)
    assert "value" not in serialized
    assert "fused_score" not in serialized


def test_empty_candidates_produce_an_empty_list_not_an_error() -> None:
    payload = build_batch_payload(1, "câu hỏi", [])
    assert payload["candidates"] == []
