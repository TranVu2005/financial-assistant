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
    # Day 21 statement-scope codes (ADR 0010 decision A1).
    "candidate_table_ids_scope_empty",
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


class ReplayRow(_FrozenModel):
    """One row of the exact frame (pandas_query's `df1`) the compiler replayed
    against. Day 22 plan §1/§2 decision A: exposed so a submission exporter
    never re-derives this from `_dispatch`'s per-operation branches -- it is
    the compiler's own replay input, not a reconstruction."""

    company_code: NonEmptyString
    row_label_canonical: NonEmptyString | None
    row_label_raw: NonEmptyString | None
    period: int = Field(ge=1900, le=2100)
    value: Decimal


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
    # Day 21 plan §1.5/ADR 0010 decision B1: True only when the plan itself
    # left `statement_scope` unset and `ExecutionSettings.
    # default_statement_scope` resolved the candidate frame instead --
    # verification must not present such an answer as certain.
    scope_inferred: bool = False
    # Day 22 plan §2 decision A: only populated when status == "answered".
    replay_rows: tuple[ReplayRow, ...] = ()

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        if self.status == "answered":
            if self.answer is None:
                raise ValueError("answered result requires answer")
            if not self.evidence:
                raise ValueError("answered result requires evidence")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("answered result must not carry error fields")
            if not self.replay_rows:
                raise ValueError("answered result requires replay_rows")
        else:
            if self.answer is not None:
                raise ValueError("error result must not carry answer")
            if self.error_code is None or self.error_message is None:
                raise ValueError("error result requires error_code and error_message")
            if self.replay_rows:
                raise ValueError("error result must not carry replay_rows")
        return self
