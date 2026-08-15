"""Tests for the Day 20 answer-package contracts (ADR 0009 decision E1).

`AnswerPackage` must be self-contained: it carries `retrieved_table_ids`
alongside `evidence` so a caller can verify evidence <= retrieved tables
without needing the original plan (Day 20 plan Sec 1.7 -- `CompiledQuery`
cannot do this on its own).
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.verification.contracts import (
    AnswerPackage,
    Citation,
    VerificationIssue,
    is_blocking_issue,
)

TABLE_ID = "tbl_" + "a" * 64
CELL_ID = "cell_" + "b" * 64
QUESTION_ID = "retq_" + "c" * 64


def _citation(**overrides: object) -> Citation:
    defaults: dict[str, object] = {
        "cell_id": CELL_ID,
        "table_id": TABLE_ID,
        "doc_relative_path": "ACB/2023/report.txt",
        "source_line_start": 10,
        "source_line_end": 10,
        "table_title": "Bang can doi ke toan",
        "value": Decimal("100"),
        "unit": "VND",
    }
    defaults.update(overrides)
    return Citation.model_validate(defaults)


def _package(**overrides: object) -> AnswerPackage:
    defaults: dict[str, object] = {
        "question_id": QUESTION_ID,
        "question": "Tra cuu tien mat cua ACB nam 2023.",
        "operation": "lookup",
        "answer": Decimal("100"),
        "unit": "VND",
        "display": "100 VND",
        "display_precision": 0,
        "answer_text": "Tiền mặt của ACB năm 2023 là 100 VND.",
        "evidence": (_citation(),),
        "retrieved_table_ids": (TABLE_ID,),
        "pandas_query": 'df1[(df1.period == 2023)]["value"].iloc[0]',
        "period_inferred": False,
        "verification_status": "verified",
        "verification_issues": (),
    }
    defaults.update(overrides)
    return AnswerPackage.model_validate(defaults)


def test_citation_requires_all_provenance_fields() -> None:
    citation = _citation()
    assert citation.table_id == TABLE_ID
    assert citation.source_line_start == 10


def test_citation_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Citation.model_validate({**_citation().model_dump(mode="json"), "extra": "x"})


def test_citation_accepts_null_table_title() -> None:
    """Day 22 plan: measured 7,643/146,011 (5.2%) real tables have a NULL
    `title_raw` in tables.parquet -- `build_citation_lookup` passes that
    through as-is, so Citation must not reject a real, otherwise-valid
    evidence cell just because its table has no title."""
    citation = _citation(table_title=None)
    assert citation.table_title is None


def test_answer_package_constructs_with_valid_fields() -> None:
    package = _package()
    assert package.answer == Decimal("100")
    assert package.evidence[0].table_id == TABLE_ID
    assert package.answer_text == "Tiền mặt của ACB năm 2023 là 100 VND."


def test_answer_package_rejects_empty_evidence() -> None:
    with pytest.raises(ValidationError):
        _package(evidence=())


def test_answer_package_rejects_empty_retrieved_table_ids() -> None:
    with pytest.raises(ValidationError):
        _package(retrieved_table_ids=())


def test_answer_package_rejects_negative_display_precision() -> None:
    with pytest.raises(ValidationError):
        _package(display_precision=-1)


def test_answer_package_rejected_status_requires_at_least_one_issue() -> None:
    with pytest.raises(ValidationError):
        _package(verification_status="rejected", verification_issues=())


def test_answer_package_rejected_status_accepts_issue() -> None:
    package = _package(
        verification_status="rejected",
        verification_issues=(
            VerificationIssue(code="recompute_mismatch", message="answer does not match replay"),
        ),
    )
    assert package.verification_status == "rejected"


def test_answer_package_verified_status_forbids_blocking_issue() -> None:
    """A `verified` package must not silently carry a blocking issue -- if
    any blocking check failed, `verification_status` must be `rejected`."""
    with pytest.raises(ValidationError):
        _package(
            verification_status="verified",
            verification_issues=(
                VerificationIssue(code="recompute_mismatch", message="answer mismatch"),
            ),
        )


def test_answer_package_verified_status_allows_period_inferred_warning() -> None:
    """`period_inferred_warning` is non-blocking (Day 20 plan Sec 1.5): 6/30
    gold70 answers rely on inferred periods and must still be verifiable."""
    package = _package(
        verification_status="verified",
        verification_issues=(
            VerificationIssue(
                code="period_inferred_warning", message="evidence relies on an inferred period"
            ),
        ),
    )
    assert package.verification_status == "verified"
    assert len(package.verification_issues) == 1


def test_scope_inferred_is_a_blocking_issue_code() -> None:
    """Day 21 plan §1.5/ADR 0010 decision B1: unlike `period_inferred_warning`
    (sign flip, small magnitude), an inferred statement_scope can flip the
    answer's VALUE (92.8% of two-scope groups disagree) -- must block."""
    assert is_blocking_issue("scope_inferred") is True


def test_answer_package_verified_status_forbids_scope_inferred_issue() -> None:
    with pytest.raises(ValidationError):
        _package(
            verification_status="verified",
            verification_issues=(
                VerificationIssue(
                    code="scope_inferred", message="statement_scope was inferred, not stated"
                ),
            ),
        )


def test_answer_package_rejected_status_accepts_scope_inferred_issue() -> None:
    package = _package(
        verification_status="rejected",
        verification_issues=(
            VerificationIssue(
                code="scope_inferred", message="statement_scope was inferred, not stated"
            ),
        ),
    )
    assert package.verification_status == "rejected"
