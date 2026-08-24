from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.execution.program_contracts import (
    BoundValue,
    CellCandidate,
    ExecutedProgram,
    ProgramDecision,
    UseClaim,
)

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(**overrides: object) -> CellCandidate:
    defaults: dict[str, object] = {
        "index": 0,
        "table_id": _TABLE_ID,
        "row_idx": 3,
        "col_idx": 2,
        "row_path": "Doanh thu > Doanh thu thuần",
        "row_label_raw": "Doanh thu thuần",
        "col_path": "Năm_2023",
        "period": 2023,
    }
    return CellCandidate(**{**defaults, **overrides})  # type: ignore[arg-type]


def _bound(**overrides: object) -> BoundValue:
    defaults: dict[str, object] = {
        "num_index": 0,
        "candidate_index": 0,
        "table_id": _TABLE_ID,
        "row_idx": 3,
        "col_idx": 2,
        "row_path": "Doanh thu > Doanh thu thuần",
        "row_label_raw": "Doanh thu thuần",
        "col_path": "Năm_2023",
        "period": 2023,
        "value": Decimal("100"),
    }
    return BoundValue(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_cell_candidate_has_no_value_field() -> None:
    # N7: model chọn ô không bao giờ thấy một con số nào.
    assert "value" not in CellCandidate.model_fields
    with pytest.raises(ValidationError):
        _candidate(value=Decimal("1"))


def test_contracts_are_frozen() -> None:
    candidate = _candidate()
    with pytest.raises(ValidationError):
        candidate.index = 1  # type: ignore[misc]


def test_decision_requires_at_least_one_cell() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(), program="[NUM_0]")


def test_decision_rejects_a_negative_candidate_index() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(-1,), program="[NUM_0]")


def test_decision_defaults_to_no_scale() -> None:
    decision = ProgramDecision(question_id=1, cells=(4,), program="[NUM_0]")
    assert decision.scale == "none"
    assert decision.uses == ()


def test_decision_rejects_an_unknown_scale() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(4,), program="[NUM_0]", scale="dozen")


def test_use_claim_rejects_a_blank_row() -> None:
    with pytest.raises(ValidationError):
        UseClaim(num=0, row="  ", col="Năm 2023")


def test_executed_program_requires_a_binding() -> None:
    with pytest.raises(ValidationError):
        ExecutedProgram(
            question_id=1,
            program="[NUM_0]",
            scale="none",
            bindings=(),
            answer=Decimal("1"),
            pandas_query="df1[...]",
            table_ids=(_TABLE_ID,),
        )


def test_executed_program_defaults_are_confident() -> None:
    program = ExecutedProgram(
        question_id=1,
        program="[NUM_0]",
        scale="none",
        bindings=(_bound(),),
        answer=Decimal("100"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )
    assert program.regenerated is False
    assert program.low_confidence is False
    assert program.failure_code is None
