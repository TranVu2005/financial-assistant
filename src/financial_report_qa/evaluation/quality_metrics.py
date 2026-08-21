"""plan.md §20: the four stage metrics `grep` found zero results for.

Before this module, `answer rate` was the only number anyone tracked --
exactly what §20 says not to do. Row Recall (`retrieval/row_fusion_evaluation.py`)
and answer accuracy (`scripts/gold_annotation/score_value.py`) already existed;
Cell Accuracy, Period Accuracy, Unit Accuracy and Valid Plan Rate did not.

`§20`'s own worked example is the reason these three are separated instead of
folded into one "grounding accuracy" number:

    Row Recall@10 = 95%, Cell Accuracy = 60%  -> the reranker/grounder is broken
    Row Recall@10 = 55%                       -> row retrieval itself is broken

Cell/Period/Unit Accuracy are scored per grounded fact, against
`dev-benchmark-v1.gold.jsonl` -- the row, column, period and unit an
annotator read straight from the source report text, independently of the
pipeline (ADR 0009 decision A2; see the gold set's own provenance doc for
why: scoring a pipeline's grounding against its own retrieved candidates
would only measure self-consistency, not correctness).

Valid Plan Rate needs no gold at all: it is the fraction of *attempted*
plans (a `FinancialQueryPlan` was actually built) that pass
`plan_validator.validate_plan_semantics` with zero issues -- the failure mode
plan.md §12 blames for 231 `llm_plan_invalid` questions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, model_validator

from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.retrieval.contracts import NonEmptyString, _FrozenModel


class GoldFactAnnotation(_FrozenModel):
    """One `dev-benchmark-v1.gold.jsonl` `annotated` record's primary fact.

    `row_labels`/`column_labels` are tuples, not single strings, because a
    question can carry `also_acceptable` alternate scope readings (the gold
    set's own provenance doc: 20 records where consolidated and separate
    reports genuinely disagree) -- either labeling can be considered
    correct. Scoped to the primary `gold_values[0]` fact only: every
    `annotated` record has exactly one (measured on the locked gold set), and
    a multi-step question's further facts live in `supporting_evidence`,
    which this pass does not score.
    """

    question_id: int = Field(gt=0)
    row_labels: tuple[NonEmptyString, ...]
    column_labels: tuple[NonEmptyString, ...] = ()
    period: int = Field(ge=1900, le=2100)
    unit: NonEmptyString

    @model_validator(mode="after")
    def validate_row_labels_non_empty(self) -> GoldFactAnnotation:
        if not self.row_labels:
            raise ValueError("row_labels must not be empty")
        return self


def load_gold_fact_annotations(path: Path) -> tuple[GoldFactAnnotation, ...]:
    """Read the `annotated` records of a gold JSONL file as gold facts.

    `needs_review` records are skipped: the gold set's own provenance doc
    records why each one was deliberately left unanswered rather than
    guessed, so scoring against one would grade the pipeline against a
    question that has no settled right answer.
    """
    annotations: list[GoldFactAnnotation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") != "annotated":
            continue
        values = record["gold_values"]
        annotations.append(
            GoldFactAnnotation.model_validate(
                {
                    "question_id": record["vifinqa_id"],
                    "row_labels": tuple(record["gold_rows"]),
                    "column_labels": tuple(record.get("gold_columns") or ()),
                    "period": record["period"],
                    "unit": values[0]["source_unit"],
                }
            )
        )
    return tuple(annotations)


class GroundingQualityReport(_FrozenModel):
    """plan.md §20's Cell/Period/Unit Accuracy, scored against gold facts."""

    question_count: int = Field(ge=0)
    # How many gold questions had a produced fact to grade at all -- a
    # question with none still counts against every accuracy below (see
    # `evaluate_grounding_quality`'s docstring), but this field keeps that
    # distinguishable from a genuinely wrong grounded fact.
    graded_count: int = Field(ge=0)
    cell_correct: int = Field(ge=0)
    period_correct: int = Field(ge=0)
    unit_correct: int = Field(ge=0)
    cell_accuracy: float = Field(ge=0, le=1)
    period_accuracy: float = Field(ge=0, le=1)
    unit_accuracy: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> GroundingQualityReport:
        if self.graded_count > self.question_count:
            raise ValueError("graded_count cannot exceed question_count")
        for correct in (self.cell_correct, self.period_correct, self.unit_correct):
            if correct > self.graded_count:
                raise ValueError("a correct count cannot exceed graded_count")
        return self


def _row_matches(fact: GroundedFact, gold: GoldFactAnnotation) -> bool:
    """Same normalization tolerance `locator.py`/`fact_grounding.py` already
    use for a selector's raw text against a corpus label: a gold label read
    from source text can differ from the pipeline's in casing, NFKC form or
    whitespace while naming the same row."""
    fact_key = normalized_key(fact.row_label)
    return any(fact_key == normalized_key(label) for label in gold.row_labels)


def evaluate_grounding_quality(
    golds: Sequence[GoldFactAnnotation],
    facts_by_question: Mapping[int, GroundedFact],
) -> GroundingQualityReport:
    """Score one grounded fact per gold question on row, period and unit.

    Denominator is every gold question, not just the ones a fact was
    produced for (mirrors `score_value.py`'s "accuracy on all
    gold-annotated"): a question the pipeline never grounded at all is a
    grounding failure, not something to quietly exclude. Column is
    deliberately not part of Cell Accuracy -- §20's table lists Cell/Period/
    Unit Accuracy as three separate rows, and column narrowing is already
    re-verified independently by `fact_checks.py` (plan.md §15).
    """
    graded = cell_correct = period_correct = unit_correct = 0
    for gold in golds:
        fact = facts_by_question.get(gold.question_id)
        if fact is None:
            continue
        graded += 1
        if _row_matches(fact, gold):
            cell_correct += 1
        if fact.period == gold.period:
            period_correct += 1
        if fact.unit == gold.unit:
            unit_correct += 1

    total = len(golds)

    def _rate(correct: int) -> float:
        return correct / total if total else 0.0

    return GroundingQualityReport(
        question_count=total,
        graded_count=graded,
        cell_correct=cell_correct,
        period_correct=period_correct,
        unit_correct=unit_correct,
        cell_accuracy=_rate(cell_correct),
        period_accuracy=_rate(period_correct),
        unit_accuracy=_rate(unit_correct),
    )


def valid_plan_rate(plans: Sequence[FinancialQueryPlan]) -> float:
    """plan.md §20 Planner-stage metric / §12's own KPI: the fraction of
    attempted plans (a `FinancialQueryPlan` was actually built) that pass
    `validate_plan_semantics` with zero issues.

    A plan that never got built at all (the planner abstained) is not
    counted here -- that is a coverage question, not a validity one. The
    failure this measures is narrower and matches §12's framing exactly: a
    plan the (typically LLM) planner *did* produce, but that names an arity
    or unit its own `operation` forbids.
    """
    if not plans:
        return 0.0
    valid = sum(
        1
        for plan in plans
        if not validate_plan_semantics(plan, known_table_ids=frozenset(plan.candidate_table_ids))
    )
    return valid / len(plans)
