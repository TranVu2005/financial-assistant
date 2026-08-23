"""Day 22 submission-bundle contracts (plan.md §2.3/§2.4).

`SubmissionItem` is the one public contract serialized into `submission.json`
-- `extra="forbid"` and exactly the seven fields plan.md specifies. It is
never built directly from an internal `AnswerPackage`; `submission/exporter.py`
is the only place that constructs one, and only from a verified `answered`
result.
"""

from __future__ import annotations

import math
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import Field, StrictInt, field_validator, model_validator

from financial_report_qa.pipeline.contracts import PipelineStage
from financial_report_qa.retrieval.contracts import Fingerprint, NonEmptyString, _FrozenModel


class RawQuestion(_FrozenModel):
    """One row of the official test question file (e.g.
    `data/raw/ViFinQA/questions/questions.jsonl`): `{"id": int, "question":
    str}`, nothing else -- no gold label exists for this release."""

    id: StrictInt = Field(gt=0)
    question: NonEmptyString


_VARIABLE_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


class SubmissionEvidence(_FrozenModel):
    """Maps one DataFrame variable used in `pandas_query` to its packaged CSV."""

    variable: NonEmptyString
    csv_path: NonEmptyString

    @field_validator("variable")
    @classmethod
    def validate_variable_name(cls, value: str) -> str:
        if not re.match(_VARIABLE_NAME_PATTERN, value):
            raise ValueError(f"variable must match {_VARIABLE_NAME_PATTERN!r}: {value!r}")
        return value

    @field_validator("csv_path")
    @classmethod
    def validate_safe_relative_csv_path(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("csv_path must use POSIX separators")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or ".." in path.parts
            or path.parts[:1] != ("data",)
        ):
            raise ValueError(f"csv_path must be a safe relative path under data/: {value!r}")
        return value


class SubmissionItem(_FrozenModel):
    """One row of `submission.json` -- plan.md §2.4's exact seven-field
    contract. No internal field (`status`, `run_id`, `cell_ids`, `page`,
    `bbox`, error detail) may appear here."""

    id: StrictInt = Field(gt=0)
    question: NonEmptyString
    answer: float = Field(allow_inf_nan=False)
    relevant_docs: tuple[NonEmptyString, ...]
    relevant_tables: tuple[NonEmptyString, ...]
    evidence: tuple[SubmissionEvidence, ...]
    pandas_query: NonEmptyString

    @field_validator("answer")
    @classmethod
    def validate_finite_answer(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("answer must be finite")
        return value

    @field_validator("relevant_tables")
    @classmethod
    def validate_relevant_tables(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            report_id, sep, line_start = value.rpartition("|")
            if not sep or not report_id:
                raise ValueError(
                    f"relevant_tables entry must be '<report_id>|<line_start>': {value!r}"
                )
            if not line_start.isdigit() or int(line_start) < 1:
                raise ValueError(
                    f"relevant_tables line_start must be a positive integer: {value!r}"
                )
        return values

    @field_validator("evidence")
    @classmethod
    def validate_unique_evidence_variables(
        cls, values: tuple[SubmissionEvidence, ...]
    ) -> tuple[SubmissionEvidence, ...]:
        variables = tuple(item.variable for item in values)
        if len(set(variables)) != len(variables):
            raise ValueError("evidence variable names must be unique within one item")
        return values


class QuestionOutcome(_FrozenModel):
    """One question's outcome through the live submission pipeline (retrieval
    -> plan -> execution -> verification). Mirrors
    `pipeline.contracts.PipelineQuestionResult`'s stage/code discipline, but
    against `RawQuestion` (no gold_table_ids exist for this release)."""

    id: StrictInt = Field(gt=0)
    question: NonEmptyString
    # "backstopped" (Day 23 plan, full-coverage strategy): the official
    # Dashboard rejects the *entire* ZIP if even one id is missing (plan.md
    # §2.4 rule 1), and its scoring macro-averages over the full 1.012-
    # question set -- a wrong answer already scores the same 0 credit as a
    # missing one. So every question still unanswered after every reasoning
    # tier gets a contract-valid but not-reasoned item from
    # `backstop_answer.build_backstop_item`; `stage`/`code` here record the
    # reasoning failure that triggered it (not "answered": that status is
    # reserved for a genuinely reasoned result).
    status: Literal["answered", "abstained", "error", "backstopped"]
    stage: PipelineStage | None
    code: NonEmptyString | None
    # Which planner actually produced the plan (Day 22 coverage-improvement
    # follow-up: route_plan tries the rule planner first, falling back to the
    # LLM planner only on abstain -- ADR 0006 decision A1). None whenever
    # planning was never reached (e.g. stage == "retrieval").
    # "rule_raw_grounded" (Day 23 plan Step 1): the rule planner still built
    # the plan, but only after `raw_metric_grounding.ground_raw_metric`
    # resolved a metric_unknown abstain -- kept distinct from "rule" so
    # coverage gained by this fallback is measurable on its own.
    # "llm_grounded" (Day 23 full-coverage strategy): the typed LLM planner
    # (vocabulary-free prompt) abstained, but a second attempt shown the
    # real candidate-table content (`llm_planner.build_plan_grounded`)
    # succeeded -- kept distinct so its yield is measurable on its own.
    # "llm_cell_grounded" (Day 25): the LLM chose only WHICH ROW from the
    # real candidate labels; the rule planner still built and validated
    # the plan. Distinct so this tier's yield is measurable on its own.
    # "llm_column_refined" (Day 26): row and period compiled to more than one
    # candidate cell (`cell_ambiguous`), and the LLM chose WHICH COLUMN from
    # that row's real headers. Distinct so this tier's yield is measurable on
    # its own, and so a column-narrowed answer is never mistaken for one the
    # deterministic locator resolved unaided.
    plan_source: (
        Literal[
            "rule",
            "rule_raw_grounded",
            "llm",
            "llm_grounded",
            # plan.md §7.1: Attempt 0's row decision, either a genuine LLM
            # pick ("llm_row_choice") or its deterministic rank-1 fallback
            # ("row_choice_fallback_rank1") when no decision applied. Kept
            # distinct from "llm_cell_grounded" (the live-LLM row choice
            # call) since this one can run with `llm_client=None` via an
            # offline decisions file.
            "llm_row_choice",
            "row_choice_fallback_rank1",
            "llm_cell_grounded",
            "llm_cell_grounded_recovered",
            "llm_cell_grounded_context_expanded",
            "llm_column_refined",
            # plan.md §12: the Evidence-Aware Planner chose `{operation,
            # operands}` over facts grounding had already resolved, rather
            # than emitting a whole typed plan of its own.
            "llm_evidence_planner",
            # spec 2026-08-23 §6: the single answering path -- `cell_grounding.
            # ground_question` assembled the plan from the offline LLM row
            # decision (or its rank-1 default). Replaces the ladder-era
            # sources above, which stay listed for already-written reports.
            "llm_decision",
        ]
        | None
    ) = None
    # plan.md §9: retrieval confidence backing the plan's row selector(s)
    # (min across every selector the plan uses), when the row went through
    # fusion at all. `None` for a deterministic canonical-dictionary match,
    # or whenever planning never resolved a row (abstained/erred earlier).
    grounding_score: float | None = None
    # plan.md §15: True only when `cell_grounding.ground_with_recovery`'s
    # threshold retry ran and never found a row ranked within
    # `max_grounding_rank` in fusion -- this answer is its best-effort
    # fallback, not a confidently-ranked match. Always False when the rule
    # or LLM planner answered directly without going through that ladder
    # (a direct rule-planner success is never threshold-checked today).
    low_confidence: bool = False

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.status == "answered":
            if self.stage is not None or self.code is not None:
                raise ValueError("an answered outcome must not carry stage/code")
        else:
            if self.stage is None or self.code is None:
                raise ValueError("a non-answered outcome requires stage and code")
        return self


class SubmissionExportReport(_FrozenModel):
    """Coverage report written *outside* the ZIP (plan.md §2.4 rule 9): which
    questions made it into `submission.json` and why the rest did not."""

    dataset_fingerprint: Fingerprint
    question_count: int = Field(ge=0)
    # Genuinely reasoned answers only (status == "answered") -- does NOT
    # count backstop fills, so this stays a meaningful coverage/accuracy
    # signal even though `submission.json` itself now always has one item
    # per question (Day 23 full-coverage strategy).
    answered_count: int = Field(ge=0)
    backstopped_count: int = Field(ge=0, default=0)
    stage_counts: dict[PipelineStage, int]
    outcomes: tuple[QuestionOutcome, ...]

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.question_count != len(self.outcomes):
            raise ValueError("question_count must equal len(outcomes)")
        ids = tuple(outcome.id for outcome in self.outcomes)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("outcomes must be sorted and unique by id")
        if self.answered_count + self.backstopped_count > self.question_count:
            raise ValueError("answered_count + backstopped_count cannot exceed question_count")
        actual_answered = sum(1 for outcome in self.outcomes if outcome.status == "answered")
        if actual_answered != self.answered_count:
            raise ValueError("answered_count must equal the number of answered outcomes")
        actual_backstopped = sum(1 for outcome in self.outcomes if outcome.status == "backstopped")
        if actual_backstopped != self.backstopped_count:
            raise ValueError("backstopped_count must equal the number of backstopped outcomes")
        return self


class ValidationIssue(_FrozenModel):
    """One offline-validator finding (plan.md §2.4 rule 9: validator runs
    before upload, never as part of the exporter's own self-report)."""

    code: NonEmptyString
    message: NonEmptyString


class ValidationReport(_FrozenModel):
    valid: bool
    item_count: int = Field(ge=0)
    issues: tuple[ValidationIssue, ...]

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.valid and self.issues:
            raise ValueError("a report with issues cannot be valid")
        if not self.valid and not self.issues:
            raise ValueError("an invalid report must carry at least one issue")
        return self
