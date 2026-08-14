"""Unit tests for Day 16 plan-evaluation metrics and reporting (§16.7)."""

from __future__ import annotations

import json
from pathlib import Path

from financial_report_qa.planning.plan_cases import PlanCase
from financial_report_qa.planning.plan_evaluation import (
    evaluate_plan_cases,
    evaluate_rule_planner_on_gold,
    write_held_out_plan_report,
    write_plan_case_report,
)
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion


def _case(case_id: str, question: str, **kwargs: object) -> PlanCase:
    return PlanCase(case_id=case_id, template_id="lookup_ticker", question=question, **kwargs)  # type: ignore[arg-type]


def test_correct_plan_case_scores_full_accuracy() -> None:
    cases = (
        _case(
            "a",
            "Tra cứu doanh thu thuần của NVL năm 2023.",
            expected_operation="lookup",
        ),
    )
    report = evaluate_plan_cases(cases, case_set_sha256="f" * 64)
    assert report.operation_accuracy == 1.0
    assert report.false_plan_rate == 0.0
    assert report.failures == ()


def test_operation_mismatch_is_recorded_as_a_failure() -> None:
    cases = (
        # expects growth_rate, but the wording ("So sánh"/2 years) actually
        # yields `difference` — a deliberate mismatch to exercise scoring.
        _case(
            "a",
            "So sánh doanh thu thuần của NVL giữa năm 2022 và năm 2023.",
            expected_operation="growth_rate",
        ),
    )
    report = evaluate_plan_cases(cases, case_set_sha256="f" * 64)
    assert report.operation_accuracy == 0.0
    assert len(report.failures) == 1
    assert report.failures[0].expected_operation == "growth_rate"
    assert report.failures[0].actual_operation == "difference"


def test_correctly_abstaining_case_counts_toward_abstain_recall() -> None:
    cases = (
        _case(
            "a",
            "Doanh thu thuần năm 2023 là bao nhiêu?",
            expected_abstain_code="entity_ambiguous",
        ),
    )
    report = evaluate_plan_cases(cases, case_set_sha256="f" * 64)
    assert report.abstain_recall == 1.0
    assert report.abstain_code_accuracy == 1.0
    assert report.false_plan_rate == 0.0


def test_wrongly_returning_a_plan_when_abstain_expected_raises_false_plan_rate() -> None:
    """The Day 16 DoD hard KPI: 0 trường hợp trả plan cho câu lẽ ra phải abstain.
    This test proves the metric actually detects a violation of that KPI."""
    cases = (
        # date_lookup shape resolves cleanly to lookup/period_grammar today, so
        # mislabel it as an ambiguous case to force a false-plan scenario.
        _case(
            "a",
            "Tra cứu doanh thu thuần của NVL năm 2023.",
            expected_abstain_code="entity_ambiguous",
        ),
    )
    report = evaluate_plan_cases(cases, case_set_sha256="f" * 64)
    assert report.false_plan_rate == 1.0
    assert len(report.failures) == 1


def test_write_plan_case_report_round_trips_through_json(tmp_path: Path) -> None:
    cases = (_case("a", "Tra cứu doanh thu thuần của NVL năm 2023.", expected_operation="lookup"),)
    report = evaluate_plan_cases(cases, case_set_sha256="f" * 64)
    json_path, markdown_path = write_plan_case_report(report, tmp_path)
    assert json_path.is_file()
    assert markdown_path.is_file()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["case_set_sha256"] == "f" * 64
    assert "operation_accuracy" in reloaded


def _gold_question(char: str, question: str, table_ids: tuple[str, ...]) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": f"retq_{char * 64}",
            "question": question,
            "intent": "lookup",
            "filters": {},
            "gold_table_ids": list(table_ids),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "X/report.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                }
                for table_id in table_ids
            ],
            "dataset_fingerprint": "c" * 64,
        }
    )


def test_held_out_report_is_descriptive_not_scored() -> None:
    """gold70's `intent` taxonomy does not map 1:1 onto Day 15 operations
    (Day 16 finding #2 / ADR 0005) — the held-out report can only describe
    plannability and operation distribution, never claim an accuracy number."""
    questions = (
        _gold_question("1", "Tra cứu doanh thu thuần của NVL năm 2023.", (f"tbl_{'1' * 64}",)),
    )
    report = evaluate_rule_planner_on_gold(questions)
    assert report.question_count == 1
    assert report.plannable_rate == 1.0
    assert report.operation_distribution.get("lookup") == 1


def test_write_held_out_plan_report_round_trips_through_json(tmp_path: Path) -> None:
    questions = (
        _gold_question("1", "Tra cứu doanh thu thuần của NVL năm 2023.", (f"tbl_{'1' * 64}",)),
    )
    report = evaluate_rule_planner_on_gold(questions)
    json_path, markdown_path = write_held_out_plan_report(report, tmp_path)
    assert json_path.is_file()
    assert markdown_path.is_file()
