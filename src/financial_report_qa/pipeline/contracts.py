"""Day 21 E2E pipeline contracts (ADR 0010 decision E1).

Historical note: these contracts fed `pipeline.evaluation.run_e2e_pipeline`
(Day 21, deleted 2026-08-23 with the planner ladder it measured). What
remains live is `PipelineStage` and the outcome models the submission
exporter uses to classify every non-answered question by stage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, model_validator

from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    QuestionId,
    _FrozenModel,
)

PipelineStage = Literal["retrieval", "planning", "normalization", "execution", "verification"]

# ADR 0010 decision E1: `ExecutionIssueCode`s that map to a fixed stage.
# `cell_ambiguous` is deliberately absent -- it is stage-ambiguous by nature
# (Day 21 plan §1.2: 22/24 `cell_ambiguous` failures under real retrieval had
# gold IN the candidate set, so a static "cell_ambiguous = retrieval" rule
# would misclassify the majority case) and was resolved per-question from
# `gold_in_retrieved` in the since-deleted Day 21 `evaluation.py`, not from
# this table.
NORMALIZATION_ISSUE_CODES: frozenset[str] = frozenset(
    {"metric_not_found", "period_unresolved", "unit_missing"}
)


class PipelineQuestionResult(_FrozenModel):
    """One question's outcome through retrieval -> plan -> execution ->
    verification. `stage is None` means a verified success; otherwise `stage`
    names where the chain stopped and `code`/`message` explain why.
    """

    question_id: QuestionId
    question: NonEmptyString
    gold_in_retrieved: bool
    retrieved_table_count: int = Field(ge=0)
    stage: PipelineStage | None
    code: NonEmptyString | None
    message: NonEmptyString | None
    answer: Decimal | None = None
    scope_inferred: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if (self.stage is None) != (self.code is None):
            raise ValueError("stage and code must both be set or both be unset")
        if (self.stage is None) != (self.message is None):
            raise ValueError("stage and message must both be set or both be unset")
        if self.stage is None and self.answer is None:
            raise ValueError("a successful (stage=None) result must carry an answer")
        if self.stage is not None and self.answer is not None:
            raise ValueError("a failed result must not carry an answer")
        return self


class PipelineReport(_FrozenModel):
    """One measurement pass over a gold question set through the real
    pipeline, with a saved retrieval ranking as provenance."""

    dataset_fingerprint: NonEmptyString
    rankings_source: NonEmptyString
    rankings_sha256: NonEmptyString
    default_statement_scope: Literal["separate", "consolidated"] | None
    question_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    stage_counts: dict[PipelineStage, int]
    scored_against_gold_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    overconfident_wrong_count: int = Field(ge=0)
    results: tuple[PipelineQuestionResult, ...]

    @model_validator(mode="after")
    def validate_complete(self) -> Self:
        if self.question_count != len(self.results):
            raise ValueError("question_count must equal len(results)")
        ids = tuple(result.question_id for result in self.results)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("results must be sorted and unique by question_id")
        if self.correct_count > self.scored_against_gold_count:
            raise ValueError("correct_count cannot exceed scored_against_gold_count")
        if self.verified_count > self.question_count:
            raise ValueError("verified_count cannot exceed question_count")
        return self

    @property
    def accuracy_against_gold(self) -> float | None:
        if self.scored_against_gold_count == 0:
            return None
        return self.correct_count / self.scored_against_gold_count


ScopePolicy = Literal["none", "default_consolidated", "abstain_when_unstated"]


class ScopePolicyResult(_FrozenModel):
    """One scope policy's coverage/accuracy/overconfident-wrong triple (Day
    21 plan §1.5/ADR 0010 decision G1: no single policy clears the gate, so
    the report must show the tradeoff, not one number).

    `answered_count` was measured at the compile-era raw `status ==
    "answered"` -- deliberately BEFORE the `scope_inferred` verification
    block (task 21.6). That block always rejects an inferred-scope answer by
    design, so measuring post-verification would make every non-`none`
    policy show 0 verified answers and hide the exact tradeoff (coverage vs.
    accuracy) this report exists to surface.
    """

    policy: ScopePolicy
    answered_count: int = Field(ge=0)
    scored_against_gold_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    overconfident_wrong_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.correct_count > self.scored_against_gold_count:
            raise ValueError("correct_count cannot exceed scored_against_gold_count")
        if self.correct_count + self.overconfident_wrong_count > self.scored_against_gold_count:
            raise ValueError(
                "correct_count + overconfident_wrong_count cannot exceed scored_against_gold_count"
            )
        return self

    @property
    def accuracy_against_gold(self) -> float | None:
        if self.scored_against_gold_count == 0:
            return None
        return self.correct_count / self.scored_against_gold_count


class ScopePolicyComparisonReport(_FrozenModel):
    """Runs the same question set through `run_e2e_pipeline` under all three
    scope policies and reports each one's tradeoff (ADR 0010 decision G1)."""

    dataset_fingerprint: NonEmptyString
    rankings_source: NonEmptyString
    rankings_sha256: NonEmptyString
    question_count: int = Field(ge=0)
    policies: tuple[ScopePolicyResult, ...]

    @model_validator(mode="after")
    def validate_policies(self) -> Self:
        names = tuple(result.policy for result in self.policies)
        if names != tuple(sorted(set(names))):
            raise ValueError("policies must be sorted and unique by policy name")
        return self
