"""Tests for the Day 22 submission-bundle contracts (plan.md §2.3/§2.4).

`SubmissionItem` is the one public contract written to `submission.json` --
`extra="forbid"` and exactly the seven fields plan.md specifies, nothing
internal (`status`, `run_id`, `cell_ids`, `page`, `bbox`) leaks through.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.submission.contracts import (
    QuestionOutcome,
    RawQuestion,
    SubmissionEvidence,
    SubmissionExportReport,
    SubmissionItem,
)


def _item(**overrides: object) -> dict[object, object]:
    defaults: dict[object, object] = {
        "id": 1,
        "question": "Doanh thu thuần của ACB năm 2023 là bao nhiêu?",
        "answer": 100.0,
        "relevant_docs": ["ACB_financial_statements_2023_consolidated_extracted"],
        "relevant_tables": ["ACB_financial_statements_2023_consolidated_extracted|12"],
        "evidence": [{"variable": "df1", "csv_path": "data/q000001_df1.csv"}],
        "pandas_query": 'df1[(df1.period == 2023)]["value"].iloc[0]',
    }
    defaults.update(overrides)
    return defaults


def test_submission_item_accepts_valid_record() -> None:
    item = SubmissionItem.model_validate(_item())
    assert item.id == 1
    assert item.evidence[0].variable == "df1"


def test_submission_item_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate({**_item(), "status": "answered"})


def test_submission_item_rejects_boolean_id() -> None:
    """plan.md §2.4 rule 1: `id` phải là integer thật, không nhận boolean."""
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(id=True))


def test_submission_item_rejects_nan_answer() -> None:
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(answer=float("nan")))


def test_submission_item_rejects_infinite_answer() -> None:
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(answer=float("inf")))


def test_submission_item_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(question=""))


def test_submission_item_rejects_malformed_relevant_table() -> None:
    """Must be `<report_id>|<line_start>` with a positive integer line_start."""
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(relevant_tables=["ACB_report_no_pipe"]))


def test_submission_item_rejects_zero_line_start() -> None:
    with pytest.raises(ValidationError):
        SubmissionItem.model_validate(_item(relevant_tables=["ACB_report|0"]))


def test_submission_evidence_rejects_invalid_variable_name() -> None:
    with pytest.raises(ValidationError):
        SubmissionEvidence.model_validate({"variable": "1df", "csv_path": "data/q1_df1.csv"})


def test_submission_evidence_rejects_path_traversal() -> None:
    with pytest.raises(ValidationError):
        SubmissionEvidence.model_validate({"variable": "df1", "csv_path": "data/../secret.csv"})


def test_submission_evidence_rejects_path_outside_data_dir() -> None:
    with pytest.raises(ValidationError):
        SubmissionEvidence.model_validate({"variable": "df1", "csv_path": "elsewhere/df1.csv"})


def test_submission_evidence_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        SubmissionEvidence.model_validate({"variable": "df1", "csv_path": "/data/df1.csv"})


def test_raw_question_rejects_boolean_id() -> None:
    with pytest.raises(ValidationError):
        RawQuestion.model_validate({"id": True, "question": "q?"})


def test_raw_question_accepts_valid_record() -> None:
    raw = RawQuestion.model_validate({"id": 7, "question": "q?"})
    assert raw.id == 7


def test_question_outcome_answered_requires_no_stage() -> None:
    outcome = QuestionOutcome.model_validate(
        {"id": 1, "question": "q?", "status": "answered", "stage": None, "code": None}
    )
    assert outcome.status == "answered"


def test_question_outcome_abstained_requires_stage_and_code() -> None:
    with pytest.raises(ValidationError):
        QuestionOutcome.model_validate(
            {"id": 1, "question": "q?", "status": "abstained", "stage": None, "code": None}
        )


def test_submission_export_report_counts_must_be_consistent() -> None:
    outcomes = (
        QuestionOutcome.model_validate(
            {"id": 1, "question": "q?", "status": "answered", "stage": None, "code": None}
        ),
        QuestionOutcome.model_validate(
            {
                "id": 2,
                "question": "q2?",
                "status": "abstained",
                "stage": "retrieval",
                "code": "no_candidate_tables",
            }
        ),
    )
    report = SubmissionExportReport.model_validate(
        {
            "dataset_fingerprint": "0" * 64,
            "question_count": 2,
            "answered_count": 1,
            "stage_counts": {"retrieval": 1},
            "outcomes": outcomes,
        }
    )
    assert report.answered_count == 1


def test_submission_export_report_rejects_inconsistent_answered_count() -> None:
    outcomes = (
        QuestionOutcome.model_validate(
            {"id": 1, "question": "q?", "status": "answered", "stage": None, "code": None}
        ),
    )
    with pytest.raises(ValidationError):
        SubmissionExportReport.model_validate(
            {
                "dataset_fingerprint": "0" * 64,
                "question_count": 1,
                "answered_count": 0,
                "stage_counts": {},
                "outcomes": outcomes,
            }
        )
