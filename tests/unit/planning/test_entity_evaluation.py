"""Unit tests for entity-parser evaluation metrics and reporting."""

from __future__ import annotations

import json
from pathlib import Path

from financial_report_qa.planning.entity_cases import EntityCase
from financial_report_qa.planning.entity_contracts import AmbiguityCode
from financial_report_qa.planning.entity_evaluation import (
    evaluate_entity_cases,
    evaluate_entity_parser_on_gold,
    write_entity_case_report,
    write_held_out_report,
)
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion


def _case(
    case_id: str,
    question: str,
    *,
    company_codes: tuple[str, ...] = (),
    periods: tuple[str, ...] = (),
    metrics: tuple[str, ...] = (),
    statement_types: tuple[str, ...] = (),
    ambiguity: tuple[AmbiguityCode, ...] = (),
) -> EntityCase:
    return EntityCase(
        case_id=f"ecase_{case_id * 64}",
        template_id="lookup_ticker",
        category="lookup",
        question=question,
        expected_company_codes=company_codes,
        expected_periods=periods,
        expected_metrics=metrics,
        expected_statement_types=statement_types,
        expected_ambiguity=ambiguity,
    )


def test_perfect_parser_scores_are_all_one() -> None:
    cases = (
        _case(
            "a",
            "Tra cứu tổng tài sản của DBC năm 2023.",
            company_codes=("DBC",),
            periods=("2023",),
            metrics=("total_assets",),
        ),
    )
    report = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    assert report.exact_match_rate == 1.0
    assert report.by_field["company_codes"].precision == 1.0
    assert report.by_field["company_codes"].recall == 1.0
    assert report.failures == ()


def test_a_field_mismatch_is_recorded_as_a_failure_and_lowers_that_fields_metrics() -> None:
    # DBC's canonical metric for this question is total_assets, but we assert
    # against a wrong expected company code to force a false negative/positive.
    cases = (
        _case(
            "a",
            "Tra cứu tổng tài sản của DBC năm 2023.",
            company_codes=("KHG",),  # wrong on purpose
            periods=("2023",),
            metrics=("total_assets",),
        ),
    )
    report = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    assert report.exact_match_rate == 0.0
    company_metrics = report.by_field["company_codes"]
    assert company_metrics.true_positive == 0
    assert company_metrics.false_positive == 1
    assert company_metrics.false_negative == 1
    assert len(report.failures) == 1
    assert report.failures[0].expected_company_codes == ("KHG",)
    assert report.failures[0].actual_company_codes == ("DBC",)


def test_ambiguity_expected_correctly_is_scored_as_a_true_positive() -> None:
    cases = (
        _case(
            "a",
            "Doanh thu thuần năm 2023 là bao nhiêu?",
            periods=("2023",),
            metrics=("net_revenue",),
            ambiguity=("company_missing",),
        ),
    )
    report = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    assert report.exact_match_rate == 1.0
    assert report.by_field["ambiguity"].true_positive == 1


def test_evaluation_is_byte_stable_across_repeated_runs() -> None:
    cases = (
        _case(
            "a",
            "Tra cứu tổng tài sản của DBC năm 2023.",
            company_codes=("DBC",),
            periods=("2023",),
            metrics=("total_assets",),
        ),
        _case(
            "b",
            "Doanh thu thuần năm 2023 là bao nhiêu?",
            periods=("2023",),
            metrics=("net_revenue",),
            ambiguity=("company_missing",),
        ),
    )
    first = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    second = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    assert first.model_dump_json() == second.model_dump_json()


def test_write_entity_case_report_round_trips_through_json(tmp_path: Path) -> None:
    cases = (
        _case(
            "a",
            "Tra cứu tổng tài sản của DBC năm 2023.",
            company_codes=("DBC",),
            periods=("2023",),
            metrics=("total_assets",),
        ),
    )
    report = evaluate_entity_cases(cases, case_set_sha256="f" * 64)
    json_path, markdown_path = write_entity_case_report(report, tmp_path)
    assert json_path.is_file()
    assert markdown_path.is_file()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["case_set_sha256"] == "f" * 64
    assert "Exact-match rate" in markdown_path.read_text(encoding="utf-8")


def _gold_question(
    question_id_char: str, question: str, company: str, periods: tuple[str, ...]
) -> GoldRetrievalQuestion:
    table_id = f"tbl_{question_id_char * 64}"
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": f"retq_{question_id_char * 64}",
            "question": question,
            "intent": "lookup",
            "filters": {"company_codes": [company], "periods": list(periods)},
            "gold_table_ids": [table_id],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": f"{company}/report.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                }
            ],
            "dataset_fingerprint": "c" * 64,
        }
    )


def test_held_out_evaluation_scores_company_and_period_only() -> None:
    questions = (
        _gold_question("1", "Tra cứu tổng tài sản của DBC năm 2023.", "DBC", ("2023",)),
    )
    report = evaluate_entity_parser_on_gold(questions)
    assert report.exact_match_rate == 1.0
    assert report.company_metrics.true_positive == 1
    assert report.period_metrics.true_positive == 1
    assert report.failures == ()


def test_held_out_evaluation_records_a_failure_when_parser_misses_the_gold_company() -> None:
    questions = (
        # gold says NVL but the question text never mentions any company —
        # this is a case where the human-labeled filter isn't recoverable
        # from the question text alone, which the held-out report must show
        # as a failure rather than silently passing.
        _gold_question("1", "Tổng tài sản năm 2023 là bao nhiêu?", "NVL", ("2023",)),
    )
    report = evaluate_entity_parser_on_gold(questions)
    assert report.exact_match_rate == 0.0
    assert len(report.failures) == 1
    assert report.failures[0].expected_company_codes == ("NVL",)
    assert report.failures[0].actual_company_codes == ()


def test_write_held_out_report_round_trips_through_json(tmp_path: Path) -> None:
    questions = (
        _gold_question("1", "Tra cứu tổng tài sản của DBC năm 2023.", "DBC", ("2023",)),
    )
    report = evaluate_entity_parser_on_gold(questions)
    json_path, markdown_path = write_held_out_report(report, tmp_path)
    assert json_path.is_file()
    assert markdown_path.is_file()
