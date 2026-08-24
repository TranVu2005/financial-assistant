"""Tests for the `build_answer_package` orchestrator (ADR 0009).

One function, never a guessed or half-verified package: it runs the four
verification checks over one `ExecutedProgram` and maps the result onto
`AnswerPackage`, carrying the program and its confidence flags.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.verification.builder import build_answer_package
from financial_report_qa.verification.contracts import AnswerPackage, VerificationIssue

TABLE_ID = "tbl_" + "1" * 64
QUESTION_ID = "retq_" + "a" * 64


def _bound(value: str = "5310", num_index: int = 0) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=0,
        table_id=TABLE_ID,
        row_idx=3,
        col_idx=2,
        row_path="Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path="Năm_2023",
        period=2023,
        value=Decimal(value),
    )


def _executed(**overrides: object) -> ExecutedProgram:
    defaults: dict[str, object] = {
        "question_id": 7,
        "program": "[NUM_0]",
        "scale": "none",
        "bindings": (_bound(),),
        "answer": Decimal("5310"),
        "pandas_query": 'df1[(df1.row_idx == 3)]["value"].iloc[0]',
        "table_ids": (TABLE_ID,),
    }
    defaults.update(overrides)
    return ExecutedProgram(**defaults)


def test_verified_package_carries_the_program_and_confidence_flags() -> None:
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu doanh thu thuần năm 2023.",
        executed=_executed(regenerated=True, low_confidence=True),
        retrieved_table_ids=frozenset({TABLE_ID}),
    )

    assert isinstance(package, AnswerPackage)
    assert package.verification_status == "verified"
    assert package.program == "[NUM_0]"
    assert package.regenerated is True
    assert package.low_confidence is True
    assert package.answer == Decimal("5310")
    # The default display is the plain positional rendering of the answer.
    assert package.display == "5310"
    assert package.display_precision == 0


def test_negative_answer_keeps_its_sign_in_the_display() -> None:
    executed = _executed(
        program="[NUM_0] - [NUM_1]",
        bindings=(_bound("100", 0), _bound("350", 1)),
        answer=Decimal("-250"),
    )
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="So sánh.",
        executed=executed,
        retrieved_table_ids=frozenset({TABLE_ID}),
    )
    assert package.answer == Decimal("-250")
    assert package.display == "-250"


def test_evidence_outside_retrieval_is_rejected() -> None:
    other = "tbl_" + "b" * 64
    executed = _executed(table_ids=(other,))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu.",
        executed=executed,
        retrieved_table_ids=frozenset({TABLE_ID}),
    )
    assert package.verification_status == "rejected"
    codes = {issue.code for issue in package.verification_issues}
    assert codes == {"evidence_outside_retrieval"}


def test_drifted_answer_is_rejected_with_a_blocking_issue() -> None:
    executed = _executed(answer=Decimal("9999"))
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu.",
        executed=executed,
        retrieved_table_ids=frozenset({TABLE_ID}),
    )
    assert package.verification_status == "rejected"
    issue: VerificationIssue = package.verification_issues[0]
    assert issue.code == "recompute_mismatch"


def test_empty_retrieved_set_is_rejected() -> None:
    """A verified package must not silently carry evidence outside retrieval;
    with no retrieved tables at all, every table is outside."""
    package = build_answer_package(
        question_id=QUESTION_ID,
        question="Tra cứu.",
        executed=_executed(),
        retrieved_table_ids=frozenset(),
    )
    assert package.verification_status == "rejected"


def test_builder_requires_an_executed_program() -> None:
    with pytest.raises((TypeError, ValueError)):
        build_answer_package(  # type: ignore[call-arg]
            question_id=QUESTION_ID,
            question="Tra cứu.",
            retrieved_table_ids=frozenset({TABLE_ID}),
        )
