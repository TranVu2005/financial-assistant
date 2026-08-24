"""Frozen contracts for the masked-PAL answering branch (spec 2026-08-24).

`CellCandidate` deliberately has no value field: the model that picks cells
never sees a number, which is what makes N7 hold at the cell level. A value
appears for the first time on `BoundValue`, and only deterministic binding
from the release can produce one -- never the model.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    TableId,
    _FrozenModel,
)

ScaleName = Literal["none", "percent", "thousand", "million", "billion"]

ProgramFailureCode = Literal[
    "decision_unparseable",
    "candidate_index_out_of_range",
    "numeric_literal_in_program",
    "program_node_not_allowed",
    "division_by_zero",
    "non_finite_result",
    "use_binding_mismatch",
    "explanation_number_not_grounded",
    "no_cell_candidates",
]


class CellCandidate(_FrozenModel):
    """One numbered cell offered to the model. Carries no value (N7)."""

    index: int = Field(ge=0)
    table_id: TableId
    company_code: str | None = None
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    row_path: NonEmptyString
    row_label_raw: NonEmptyString
    row_label_canonical: str | None = None
    col_path: str
    period: int | None = None
    statement_type: str | None = None
    unit: str | None = None


class UseClaim(_FrozenModel):
    """What the model says `[NUM_<num>]` is, checked against the real binding."""

    num: int = Field(ge=0)
    row: NonEmptyString
    col: NonEmptyString


class ProgramDecision(_FrozenModel):
    """One offline decision: indices plus an expression, never a value."""

    question_id: int
    cells: tuple[int, ...] = Field(min_length=1)
    program: NonEmptyString
    uses: tuple[UseClaim, ...] = ()
    scale: ScaleName = "none"

    @field_validator("cells")
    @classmethod
    def validate_non_negative(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError("cells must be non-negative candidate indices")
        return values


class BoundValue(_FrozenModel):
    """One `[NUM_i]` after deterministic binding to a real cell."""

    num_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    table_id: TableId
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    row_path: NonEmptyString
    row_label_raw: NonEmptyString
    row_label_canonical: str | None = None
    col_path: str
    period: int | None = None
    value: Decimal
    unit: str | None = None


class ExecutedProgram(_FrozenModel):
    """One finished question: what ran, on which cells, and how confident."""

    question_id: int
    program: NonEmptyString
    scale: ScaleName
    bindings: tuple[BoundValue, ...] = Field(min_length=1)
    answer: Decimal
    pandas_query: NonEmptyString
    table_ids: tuple[TableId, ...] = Field(min_length=1)
    regenerated: bool = False
    low_confidence: bool = False
    failure_code: ProgramFailureCode | None = None
