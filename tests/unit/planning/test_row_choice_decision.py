"""Đọc quyết định LLM và biến nó thành selector position-bound."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_report_qa.planning.row_choice_decision import load_decisions, selector_for
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
        metadata=RowMetadata(table_id=_TABLE_ID, row_idx=row_idx, row_label_raw=label),
        snippet=label,
    )


_CANDIDATES = [
    _candidate(rank=1, row_idx=3, label="A"),
    _candidate(rank=2, row_idx=7, label="B"),
]


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    target = tmp_path / "decisions.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return target


def test_load_decisions_maps_question_id_to_index(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"question_id": 795, "chosen_index": 1}])
    assert load_decisions(path) == {795: 1}


def test_load_decisions_ignores_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "decisions.jsonl"
    target.write_text('\n{"question_id": 1, "chosen_index": 0}\n\n', encoding="utf-8")
    assert load_decisions(target) == {1: 0}


def test_selector_uses_the_llm_choice(tmp_path: Path) -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: 1})
    assert source == "llm"
    assert selector is not None
    assert selector.is_position_bound
    assert selector.table_id == _TABLE_ID
    assert selector.row_index == 7
    assert selector.raw_text == "B"


def test_missing_decision_falls_back_to_rank_one() -> None:
    """§7.1: fallback tất định, giải trình được -- không gọi LLM lần hai."""
    selector, source = selector_for(795, _CANDIDATES, {})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_out_of_range_index_falls_back_to_rank_one() -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: 99})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_negative_index_falls_back_to_rank_one() -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: -1})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_no_candidates_yields_no_selector() -> None:
    selector, source = selector_for(795, [], {795: 0})
    assert selector is None
    assert source == "no_candidates"


def test_candidate_without_a_label_yields_no_selector() -> None:
    """MetricSelector đòi đúng một trong canonical/raw_text; một ứng viên
    không có nhãn không dựng được selector hợp lệ."""
    blank = RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_5",
        table_id=_TABLE_ID,
        row_idx=5,
        rank=1,
        fused_score=1.0,
        metadata=RowMetadata(table_id=_TABLE_ID, row_idx=5, row_label_raw=None),
        snippet="",
    )
    selector, source = selector_for(1, [blank], {1: 0})
    assert selector is None
    assert source == "no_candidates"


def test_overlength_label_yields_no_selector_instead_of_raising() -> None:
    """`MetricSelector.raw_text` is bounded (max_length=512, no control
    chars); raw corpus text carries no such bound. A malformed label must
    fall through to `no_candidates` for just this one question, not raise
    `pydantic.ValidationError` and kill the whole export run."""
    overlong = "x" * 600
    candidate = RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_9",
        table_id=_TABLE_ID,
        row_idx=9,
        rank=1,
        fused_score=1.0,
        metadata=RowMetadata(table_id=_TABLE_ID, row_idx=9, row_label_raw=overlong),
        snippet=overlong,
    )
    selector, source = selector_for(1, [candidate], {1: 0})
    assert selector is None
    assert source == "no_candidates"


def test_load_decisions_raises_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_decisions(tmp_path / "khong-ton-tai.jsonl")
