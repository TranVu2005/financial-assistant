"""Day 20 answer-package contracts (ADR 0009 decision E1).

`AnswerPackage` is deliberately self-contained: it carries
`retrieved_table_ids` alongside `evidence`, unlike `CompiledQuery` (Day 18)
which does not carry `candidate_table_ids` at all. Day 20 plan Sec 1.7
measured that the "evidence <= retrieved tables" invariant held on gold70
by construction (`build_cell_frame` filters by table_ids) but had no
assertion guarding it and no way to check it from `CompiledQuery` alone --
a caller would have to keep the original plan around. `AnswerPackage` fixes
that: everything needed to independently re-verify one answer travels with
the package.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from financial_report_qa.execution.contracts import CellId
from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import PlanOperation
from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    QuestionId,
    TableId,
    _FrozenModel,
)

VerificationIssueCode = Literal[
    "recompute_mismatch",
    "unit_not_presentable",
    "evidence_outside_retrieval",
    "display_roundtrip_mismatch",
    "period_inferred_warning",
    "scope_inferred",
]

# Day 20 plan Sec 3 (task 20.5): four checks block verification, one is a
# non-blocking warning -- 6/30 gold70 answers rely on an inferred period
# (Day 20 plan Sec 1.5), so treating that as blocking would reject 20% of
# otherwise-correct answers.
#
# Day 21 plan §1.5/ADR 0010 decision B1: `scope_inferred` joins the blocking
# set, unlike `period_inferred_warning` -- an inferred statement_scope can
# flip the answer's VALUE (92.8% of two-scope groups disagree in value,
# Day 21 plan §1.4), not just shift a period by one year.
_BLOCKING_ISSUE_CODES = frozenset(
    {
        "recompute_mismatch",
        "unit_not_presentable",
        "evidence_outside_retrieval",
        "display_roundtrip_mismatch",
        "scope_inferred",
    }
)


def is_blocking_issue(code: VerificationIssueCode) -> bool:
    """Public membership test so `builder.py` doesn't need to duplicate or
    reach into `_BLOCKING_ISSUE_CODES` directly."""
    return code in _BLOCKING_ISSUE_CODES


class VerificationIssue(_FrozenModel):
    """One check result, typed for reporting -- never a silent downgrade."""

    code: VerificationIssueCode
    message: NonEmptyString


class Citation(_FrozenModel):
    """One evidence cell with everything needed to cite it independently."""

    cell_id: CellId
    table_id: TableId
    doc_relative_path: NonEmptyString
    source_line_start: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    # Day 22 plan: measured 7,643/146,011 (5.2%) real tables have NULL
    # title_raw in tables.parquet -- a real evidence cell must not be
    # rejected just because its table has no title.
    table_title: NonEmptyString | None
    value: Decimal
    unit: CanonicalUnit


class AnswerPackage(_FrozenModel):
    """A locked answer plus everything needed to independently re-verify it."""

    question_id: QuestionId
    question: NonEmptyString
    operation: PlanOperation
    answer: Decimal
    unit: CanonicalUnit
    display: NonEmptyString
    display_precision: int = Field(ge=0)
    answer_text: NonEmptyString
    evidence: tuple[Citation, ...]
    retrieved_table_ids: tuple[TableId, ...]
    pandas_query: NonEmptyString
    period_inferred: bool
    verification_status: Literal["verified", "rejected"]
    verification_issues: tuple[VerificationIssue, ...]

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        if not self.evidence:
            raise ValueError("evidence must not be empty")
        if not self.retrieved_table_ids:
            raise ValueError("retrieved_table_ids must not be empty")
        blocking = tuple(
            issue for issue in self.verification_issues if issue.code in _BLOCKING_ISSUE_CODES
        )
        if self.verification_status == "rejected" and not self.verification_issues:
            raise ValueError("rejected status requires at least one verification issue")
        if self.verification_status == "verified" and blocking:
            raise ValueError("verified status must not carry a blocking verification issue")
        return self
