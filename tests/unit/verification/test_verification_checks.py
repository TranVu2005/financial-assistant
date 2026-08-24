"""Tests for the verification checks over an executed masked-PAL program
(ADR 0009 decisions B1/D1).

Each check is a pure function: (executed program and/or display fields) ->
`VerificationIssue | None`. `None` means the check passed.
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.verification.checks import (
    check_display_roundtrip_mismatch,
    check_evidence_outside_retrieval,
    check_recompute_mismatch,
    check_scale_not_presentable,
)

_TABLE_ID = "tbl_" + "a" * 64


def _bound(value: str = "5310", num_index: int = 0) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=0,
        table_id=_TABLE_ID,
        row_idx=3,
        col_idx=2,
        row_path="Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path="Năm_2023",
        period=2023,
        value=Decimal(value),
    )


def _executed(program: str = "[NUM_0]", values: tuple[str, ...] = ("5310",)) -> ExecutedProgram:
    return ExecutedProgram(
        question_id=7,
        program=program,
        scale="none",
        bindings=tuple(_bound(v, i) for i, v in enumerate(values)),
        answer=Decimal("5310"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )


def test_recompute_mismatch_passes_when_program_rederives_the_answer() -> None:
    assert check_recompute_mismatch(_executed("[NUM_0] + [NUM_0]", ("2655",))) is None
    assert check_recompute_mismatch(_executed()) is None


def test_recompute_mismatch_flags_a_drifted_answer() -> None:
    drifted = _executed().model_copy(update={"answer": Decimal("9999")})
    issue = check_recompute_mismatch(drifted)
    assert issue is not None
    assert issue.code == "recompute_mismatch"


def test_scale_not_presentable_accepts_known_scales() -> None:
    assert check_scale_not_presentable(_executed()) is None


def test_scale_not_presentable_flags_an_unknown_scale() -> None:
    """`ScaleName` is a Literal so the validating constructor already rejects
    unknown scales; `model_construct` simulates dynamically-built objects."""
    rogue = _executed().model_construct(
        **{**_executed().model_dump(), "scale": "giga"},
    )
    issue = check_scale_not_presentable(rogue)
    assert issue is not None
    assert issue.code == "unit_not_presentable"


def test_evidence_outside_retrieval_flags_tables_not_retrieved() -> None:
    executed = _executed()
    assert check_evidence_outside_retrieval(executed, frozenset({_TABLE_ID})) is None
    other = "tbl_" + "b" * 64
    issue = check_evidence_outside_retrieval(executed, frozenset({other}))
    assert issue is not None
    assert issue.code == "evidence_outside_retrieval"
    assert _TABLE_ID in issue.message


def test_display_roundtrip_parses_the_leading_number() -> None:
    assert (
        check_display_roundtrip_mismatch(Decimal("5310"), "5310 VND", display_precision=0) is None
    )


def test_display_roundtrip_flags_a_rounding_drift_beyond_precision() -> None:
    issue = check_display_roundtrip_mismatch(Decimal("5310"), "5.310 VND", display_precision=0)
    assert issue is not None
    assert issue.code == "display_roundtrip_mismatch"


def test_display_roundtrip_without_a_number_is_flagged() -> None:
    issue = check_display_roundtrip_mismatch(Decimal("5310"), "không có số", display_precision=0)
    assert issue is not None
    assert issue.code == "display_roundtrip_mismatch"
