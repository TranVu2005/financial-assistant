"""Immutable Day 18 contracts for the deterministic compiler.

`CellMatch` is the locator's output: one resolved metric-at-a-period, possibly
backed by several cells that agree on the same value (ADR 0007 decision D1).
`CompiledQuery` is the compiler's output: either an `answered` scalar with full
evidence, or a typed `error` — never a guessed value with a suppressed issue.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import ExpectedUnit, PlanOperation
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel

ExecutionIssueCode = Literal[
    "operation_not_allowed",
    "metric_not_found",
    "period_unresolved",
    "cell_ambiguous",
    "unit_incompatible",
    "division_by_zero",
    # Day 19 sandbox hardening codes (ADR 0008 decision G1).
    "plan_rejected",
    "query_rejected",
    "budget_exceeded",
    "row_limit_exceeded",
    # Day 20 answer-verification codes (ADR 0009 decision C1).
    "unit_missing",
]

CellId = Annotated[str, StringConstraints(pattern=r"^cell_[0-9a-f]{64}$")]


class CellMatch(_FrozenModel):
    """One resolved (metric, period) locate, with full cell-level provenance."""

    table_id: TableId
    cell_ids: tuple[CellId, ...]
    value: Decimal
    # ADR 0009 decision C1: constrained to the 6 real CanonicalUnit values so
    # a fabricated string (Day 20 plan Sec 1.3: `str(float('nan'))` == 'nan',
    # produced when DuckDB's pandas conversion turns a NULL unit into NaN)
    # can never pass as evidence.
    unit: CanonicalUnit
    period: int = Field(ge=1900, le=2100)
    period_inferred: bool

    @model_validator(mode="after")
    def validate_cell_ids_non_empty(self) -> Self:
        if not self.cell_ids:
            raise ValueError("cell_ids must not be empty")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must not contain duplicates")
        return self


class CompiledQuery(_FrozenModel):
    """One deterministic compilation result: a locked answer or a typed error."""

    operation: PlanOperation
    status: Literal["answered", "error"]
    answer: Decimal | None
    unit: ExpectedUnit | None
    evidence: tuple[CellMatch, ...]
    pandas_query: NonEmptyString
    error_code: ExecutionIssueCode | None
    error_message: NonEmptyString | None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        if self.status == "answered":
            if self.answer is None:
                raise ValueError("answered result requires answer")
            if not self.evidence:
                raise ValueError("answered result requires evidence")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("answered result must not carry error fields")
        else:
            if self.answer is not None:
                raise ValueError("error result must not carry answer")
            if self.error_code is None or self.error_message is None:
                raise ValueError("error result requires error_code and error_message")
        return self
