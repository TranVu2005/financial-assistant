"""Spec 2026-08-23 §6.2/§6.3: quyết định offline của LLM -> FinancialQueryPlan.

Module này là chỗ duy nhất biết cách map quyết định thành plan, và nó không
bao giờ được từ chối vì lý do ngữ nghĩa -- đó là cái cổng `rule_planner`
từng dựng lên và đã chặn 414/1012 câu.
"""

from __future__ import annotations

from pathlib import Path

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import (
    RowChoiceDecision,
    assemble_plan,
    load_decisions,
)
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64


def _candidate(
    rank: int,
    *,
    label: str = "Doanh thu thuần",
    table_id: str = TABLE_A,
    row_idx: int | None = None,
    company: str | None = "ACB",
) -> RowFusedCandidate:
    index = rank if row_idx is None else row_idx
    return RowFusedCandidate(
        row_id=f"{table_id}|row_{index}",
        table_id=table_id,
        row_idx=index,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=table_id,
            row_idx=index,
            company_code=company,
            row_label_raw=label,
            periods=("2023",),
        ),
        snippet=label,
    )


def _entities(question: str):
    return parse_query_entities(question)


def test_lookup_uses_the_chosen_row_position_bound() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1), _candidate(2, row_idx=7))
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="lookup", chosen=(1,)),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "lookup"
    assert plan.metric is not None
    assert plan.metric.is_position_bound
    assert plan.metric.row_index == 7


def test_missing_decision_falls_back_to_lookup_on_rank_1() -> None:
    """Thiếu quyết định phải cho ra plan, không phải abstain."""
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1), _candidate(2, row_idx=7))
    plan = assemble_plan(entities, None, candidates, (TABLE_A,))
    assert plan is not None
    assert plan.operation == "lookup"
    assert plan.metric is not None
    assert plan.metric.row_index == 1


def test_out_of_range_index_falls_back_to_rank_1_not_a_crash() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1),)
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="lookup", chosen=(99,)),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.metric is not None
    assert plan.metric.row_index == 1


def test_unknown_operation_degrades_to_lookup() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="teleport", chosen=(0,)),
        (_candidate(1),),
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "lookup"


def test_multi_company_rank_uses_a_label_selector_not_position() -> None:
    """`compile_rank` dùng MỘT selector cho MỌI công ty -- position-bound ghim
    đúng một công ty nên sẽ trả lời sai câu hỏi xếp hạng."""
    entities = _entities(
        "Xét VIC, VHM và VRE năm 2023, doanh thu thuần cao nhất là bao nhiêu tỷ đồng?"
    )
    assert len(entities.company_codes) >= 2
    candidates = tuple(
        _candidate(i + 1, table_id=TABLE_A, row_idx=i, company=code)
        for i, code in enumerate(entities.company_codes)
    )
    plan = assemble_plan(
        entities,
        RowChoiceDecision(
            question_id=2,
            operation="rank",
            chosen=tuple(range(len(entities.company_codes))),
            top_k=1,
        ),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "rank"
    assert plan.metric is not None
    assert not plan.metric.is_position_bound
    assert plan.metric.raw_text == "Doanh thu thuần"
    assert plan.top_k == 1


def test_rank_top_k_is_clamped_into_the_valid_range() -> None:
    entities = _entities(
        "Xét VIC, VHM và VRE năm 2023, doanh thu thuần cao nhất là bao nhiêu tỷ đồng?"
    )
    n = len(entities.company_codes)
    candidates = tuple(_candidate(i + 1, row_idx=i) for i in range(n))
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=2, operation="rank", chosen=tuple(range(n)), top_k=999),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.top_k is not None
    assert 1 <= plan.top_k < n


def test_ratio_consumes_two_chosen_rows_as_numerator_and_denominator() -> None:
    entities = _entities("Tỷ lệ lợi nhuận gộp trên doanh thu thuần của ACB năm 2023 là bao nhiêu?")
    candidates = (
        _candidate(1, label="Lợi nhuận gộp", row_idx=3),
        _candidate(2, label="Doanh thu thuần", row_idx=5),
    )
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=3, operation="ratio", chosen=(0, 1)),
        candidates,
        (TABLE_A,),
    )
    if plan is not None and plan.operation == "ratio":
        assert plan.numerator_metric is not None
        assert plan.denominator_metric is not None
        assert plan.numerator_metric.row_index == 3
        assert plan.denominator_metric.row_index == 5


def test_overlong_corpus_label_is_clamped_not_crashed_on() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=4, operation="lookup", chosen=(0,)),
        (_candidate(1, label="x" * 900),),
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.metric is not None
    assert plan.metric.raw_text is not None
    assert len(plan.metric.raw_text) == 512


def test_no_candidates_yields_no_plan() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    assert assemble_plan(entities, None, (), (TABLE_A,)) is None


def test_load_decisions_skips_a_corrupt_line_without_losing_the_file(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        '{"question_id": 1, "operation": "lookup", "chosen": [0]}\n'
        "not json at all\n"
        "\n"
        '{"question_id": 2, "operation": "rank", "chosen": [0, 1], "top_k": 1}\n',
        encoding="utf-8",
    )
    decisions = load_decisions(path)
    assert set(decisions) == {1, 2}
    assert decisions[2].operation == "rank"
    assert decisions[2].top_k == 1
