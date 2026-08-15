"""Tests for the Day 21 E2E pipeline contracts (ADR 0010 decision E1)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.pipeline.contracts import (
    PipelineQuestionResult,
    PipelineReport,
    ScopePolicyComparisonReport,
    ScopePolicyResult,
)

QUESTION_ID = "retq_" + "a" * 64


def _result(**overrides: object) -> PipelineQuestionResult:
    defaults: dict[str, object] = {
        "question_id": QUESTION_ID,
        "question": "Tra cứu tiền mặt của ACB năm 2023.",
        "gold_in_retrieved": True,
        "retrieved_table_count": 10,
        "stage": None,
        "code": None,
        "message": None,
        "answer": Decimal("100"),
    }
    defaults.update(overrides)
    return PipelineQuestionResult.model_validate(defaults)


def test_success_result_requires_answer() -> None:
    with pytest.raises(ValidationError):
        _result(stage=None, code=None, message=None, answer=None)


def test_success_result_constructs_with_answer() -> None:
    result = _result()
    assert result.stage is None
    assert result.answer == Decimal("100")


def test_failed_result_requires_code_and_message() -> None:
    with pytest.raises(ValidationError):
        _result(stage="planning", code=None, message=None, answer=None)


def test_failed_result_forbids_answer() -> None:
    with pytest.raises(ValidationError):
        _result(
            stage="planning",
            code="entity_ambiguous",
            message="ambiguous entities",
            answer=Decimal("100"),
        )


def test_failed_result_constructs_without_answer() -> None:
    result = _result(
        stage="planning",
        code="entity_ambiguous",
        message="ambiguous entities",
        answer=None,
    )
    assert result.stage == "planning"
    assert result.answer is None


def _report(**overrides: object) -> PipelineReport:
    defaults: dict[str, object] = {
        "dataset_fingerprint": "0" * 64,
        "rankings_source": "artifacts/evaluations/day14/v2/retrieval-v2-bm25-v4-422df141c935.json",
        "rankings_sha256": "1" * 64,
        "default_statement_scope": None,
        "question_count": 1,
        "verified_count": 1,
        "stage_counts": {},
        "scored_against_gold_count": 1,
        "correct_count": 1,
        "overconfident_wrong_count": 0,
        "results": (_result(),),
    }
    defaults.update(overrides)
    return PipelineReport.model_validate(defaults)


def test_report_rejects_question_count_mismatch() -> None:
    with pytest.raises(ValidationError):
        _report(question_count=2)


def test_report_rejects_duplicate_results() -> None:
    with pytest.raises(ValidationError):
        _report(question_count=2, results=(_result(), _result()))


def test_report_rejects_correct_exceeding_scored() -> None:
    with pytest.raises(ValidationError):
        _report(scored_against_gold_count=0, correct_count=1)


def test_report_accuracy_property() -> None:
    report = _report()
    assert report.accuracy_against_gold == 1.0


def test_report_accuracy_is_none_when_nothing_scored() -> None:
    report = _report(scored_against_gold_count=0, correct_count=0)
    assert report.accuracy_against_gold is None


def _policy_result(**overrides: object) -> ScopePolicyResult:
    defaults: dict[str, object] = {
        "policy": "none",
        "answered_count": 9,
        "scored_against_gold_count": 9,
        "correct_count": 6,
        "overconfident_wrong_count": 3,
    }
    defaults.update(overrides)
    return ScopePolicyResult.model_validate(defaults)


def test_scope_policy_result_accuracy_property() -> None:
    result = _policy_result()
    assert result.accuracy_against_gold == pytest.approx(6 / 9)


def test_scope_policy_result_rejects_correct_plus_wrong_over_scored() -> None:
    with pytest.raises(ValidationError):
        _policy_result(scored_against_gold_count=5, correct_count=3, overconfident_wrong_count=3)


def test_scope_policy_comparison_report_rejects_unsorted_policies() -> None:
    with pytest.raises(ValidationError):
        ScopePolicyComparisonReport(
            dataset_fingerprint="0" * 64,
            rankings_source="test.json",
            rankings_sha256="a" * 64,
            question_count=70,
            policies=(
                _policy_result(policy="none"),
                _policy_result(policy="abstain_when_unstated"),
                _policy_result(policy="default_consolidated"),
            ),
        )


def test_scope_policy_comparison_report_accepts_sorted_policies() -> None:
    report = ScopePolicyComparisonReport(
        dataset_fingerprint="0" * 64,
        rankings_source="test.json",
        rankings_sha256="a" * 64,
        question_count=70,
        policies=(
            _policy_result(policy="abstain_when_unstated"),
            _policy_result(policy="default_consolidated"),
            _policy_result(policy="none"),
        ),
    )
    assert len(report.policies) == 3
