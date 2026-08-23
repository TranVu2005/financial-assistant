"""Nhánh 2 là một đường thẳng (spec 2026-08-23 §6, nguyên tắc N6).

Không thang tầng, không candidate switching, không context expansion. Một
câu hỏi đi qua đúng một chuỗi bước; hỏng ở đâu thì hỏng rõ ở đó.
"""

from __future__ import annotations

from pathlib import Path

from financial_report_qa.planning.cell_grounding import ground_question
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import RowChoiceDecision


def test_ground_question_reports_llm_decision_as_the_only_plan_source(
    release_dir: Path, execution_settings, fusion_rows, table_ids
) -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=RowChoiceDecision(question_id=1, operation="lookup", chosen=(0,)),
        fusion_rows=fusion_rows,
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.plan_source == "llm_decision"


def test_ground_question_fails_cleanly_when_no_plan_can_be_assembled(
    release_dir: Path, execution_settings, table_ids
) -> None:
    """Không ứng viên -> thất bại có mã, không exception, không tầng thứ hai."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=None,
        fusion_rows=(),
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.status == "failed"
    assert result.error_code == "no_row_candidates"


def test_cell_grounding_has_no_recovery_ladder_left() -> None:
    """Ghim N6: các tầng đã bỏ không được lặng lẽ quay lại."""
    from financial_report_qa.planning import cell_grounding

    for gone in (
        "ground_with_recovery",
        "_candidate_switching",
        "_context_expansion",
        "choose_row_label",
    ):
        assert not hasattr(cell_grounding, gone), f"{gone} thuộc thang tầng đã bỏ"
