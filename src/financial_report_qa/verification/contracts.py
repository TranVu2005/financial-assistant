"""Day 20 answer-package contracts (ADR 0009 decision E1).

`AnswerPackage` is deliberately self-contained: it carries
`retrieved_table_ids` alongside its evidence so everything needed to
independently re-verify one answer travels with the package.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    QuestionId,
    TableId,
    _FrozenModel,
)

# Cell-id provenance used to live in the compiler-era
# `execution/contracts.py` (spec 2026-08-24 §8.2 removed it); the pattern is
# kept here because `Citation` still cites by cell id.
CellId = Annotated[str, StringConstraints(pattern=r"^cell_[0-9a-f]{64}$")]

VerificationIssueCode = Literal[
    "recompute_mismatch",
    "unit_not_presentable",
    "evidence_outside_retrieval",
    "display_roundtrip_mismatch",
    "period_inferred_warning",
    "scope_inferred",
    # plan.md §15: per-fact re-location codes. Their producer
    # (`verification/checks.fact_checks.verify_fact`) died with the
    # operation-enum path; the literal retains both codes so reports and
    # answer packages serialized before that removal still validate.
    "fact_not_found",
    "fact_value_mismatch",
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
        # plan.md §15: a fact that cannot be independently re-confirmed is
        # exactly the failure mode the answering path must never present as
        # certain.
        "fact_not_found",
        "fact_value_mismatch",
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
    # The plan-era `PlanOperation` enum died with the operation-enum path;
    # a package built from an `ExecutedProgram` describes its computation by
    # the masked program string itself (`program`), so this is now a free
    # descriptive label.
    operation: NonEmptyString = ""
    answer: Decimal
    # The compiler-era canonical-unit declaration died with the operation
    # enum; an `ExecutedProgram`-built package leaves this unset unless a
    # caller knows the answer's canonical unit.
    unit: CanonicalUnit | None = None
    display: NonEmptyString
    display_precision: int = Field(ge=0)
    answer_text: NonEmptyString
    evidence: tuple[Citation, ...] = ()
    retrieved_table_ids: tuple[TableId, ...]
    pandas_query: NonEmptyString
    period_inferred: bool = False
    verification_status: Literal["verified", "rejected"]
    verification_issues: tuple[VerificationIssue, ...]
    # Day 24: records that this package was built under a policy that
    # deliberately accepts an inferred statement scope (see
    # `builder.build_answer_package`). Kept as an explicit, auditable field
    # rather than silently dropping the `scope_inferred` issue -- a reader
    # must always be able to tell that the scope was guessed.
    inferred_scope_accepted: bool = False
    # Spec 2026-08-24 §4.3: the masked-PAL program the package was built
    # from, plus the pipeline's own confidence flags on that execution.
    program: str = ""
    regenerated: bool = False
    low_confidence: bool = False

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        waived = {"scope_inferred"} if self.inferred_scope_accepted else set()
        blocking = tuple(
            issue
            for issue in self.verification_issues
            if issue.code in _BLOCKING_ISSUE_CODES and issue.code not in waived
        )
        if self.inferred_scope_accepted and not any(
            issue.code == "scope_inferred" for issue in self.verification_issues
        ):
            raise ValueError(
                "inferred_scope_accepted requires the scope_inferred issue to still be recorded"
            )
        if self.verification_status == "rejected" and not self.verification_issues:
            raise ValueError("rejected status requires at least one verification issue")
        if self.verification_status == "verified" and blocking:
            raise ValueError("verified status must not carry a blocking verification issue")
        return self
