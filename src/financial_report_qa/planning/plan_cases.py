"""Day 16 plan-case labels (§16.1): expected `PlanOperation`/abstain per template.

Reuses the frozen Day 10 `entity-cases-v1` corpus (1,400 cases, 14 templates)
instead of a parallel generator: `_LABEL_BY_TEMPLATE` is a pure, hand-written
mapping from `EntityCase.template_id` to the operation-or-abstain a "simple
question" of that shape must produce (Day 16 plan §1.9, ADR 0005). It is
checked against `rule_planner.build_plan` only by the evaluation harness
(`plan_evaluation.py`), never derived from it — a bug in the planner cannot
also corrupt its own answer key.

Scope note: this labels the *existing* 14 templates rather than adding new
ones for the operation vocabulary in Day 16 plan §1.8 (`gấp mấy lần`, `biên`,
...) — that vocabulary was measured to have zero occurrences in gold70 or the
entity-case corpus (Day 16 plan §1.8), so new templates for it would be
speculative. Extend `_LABEL_BY_TEMPLATE` when `entity_cases.TEMPLATE_INVENTORY`
grows new templates for observed vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import model_validator

from financial_report_qa.planning.entity_cases import EntityCase
from financial_report_qa.planning.plan_contracts import PlanOperation
from financial_report_qa.planning.rule_planner import PlanAbstainCode
from financial_report_qa.retrieval.contracts import NonEmptyString, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import (
    canonical_json_bytes,
    sha256_bytes,
    write_text_atomic,
)

# (expected_operation, expected_abstain_code) — exactly one side is non-None,
# enforced by PlanCase.validate_exactly_one below.
_LABEL_BY_TEMPLATE: dict[str, tuple[PlanOperation | None, PlanAbstainCode | None]] = {
    "lookup_ticker": ("lookup", None),
    "lookup_name": ("lookup", None),
    "statement_lookup": ("lookup", None),
    "compare_years": ("difference", None),
    "growth_years": ("growth_rate", None),
    "two_companies": ("compare_companies", None),
    "quarter_lookup": (None, "period_grammar_unsupported"),
    "date_lookup": (None, "period_grammar_unsupported"),
    "missing_company": (None, "entity_ambiguous"),
    "company_conflict": (None, "entity_ambiguous"),
    "relative_period": (None, "entity_ambiguous"),
    "incomplete_quarter": (None, "entity_ambiguous"),
    "unknown_metric": (None, "entity_ambiguous"),
    "missing_period": (None, "entity_ambiguous"),
}


class PlanCase(_FrozenModel):
    """One plan-case: a question plus the operation or abstain code it must yield."""

    case_id: str
    template_id: NonEmptyString
    question: NonEmptyString
    expected_operation: PlanOperation | None = None
    expected_abstain_code: PlanAbstainCode | None = None

    @model_validator(mode="after")
    def validate_exactly_one(self) -> Self:
        if (self.expected_operation is None) == (self.expected_abstain_code is None):
            raise ValueError("exactly one of expected_operation, expected_abstain_code must be set")
        return self


def _label_for_template(template_id: str) -> tuple[PlanOperation | None, PlanAbstainCode | None]:
    try:
        return _LABEL_BY_TEMPLATE[template_id]
    except KeyError as exc:
        raise ValueError(f"no Day 16 plan label registered for template '{template_id}'") from exc


def generate_plan_cases(entity_cases: tuple[EntityCase, ...]) -> tuple[PlanCase, ...]:
    """Derive one `PlanCase` per `EntityCase`, labeled purely from its template."""
    plan_cases = []
    for case in entity_cases:
        operation, abstain_code = _label_for_template(case.template_id)
        plan_cases.append(
            PlanCase(
                case_id=case.case_id,
                template_id=case.template_id,
                question=case.question,
                expected_operation=operation,
                expected_abstain_code=abstain_code,
            )
        )
    return tuple(plan_cases)


def plan_case_set_sha256(cases: tuple[PlanCase, ...]) -> str:
    """Return one fingerprint identifying the entire plan-case set."""
    payload = [case.model_dump(mode="json") for case in cases]
    return sha256_bytes(canonical_json_bytes(payload))


def write_plan_cases(cases: tuple[PlanCase, ...], path: Path) -> None:
    """Publish the plan-case set as JSONL, one case per line."""
    lines = (case.model_dump_json() for case in cases)
    write_text_atomic(path, "\n".join(lines) + "\n")


def load_plan_cases(path: Path) -> tuple[PlanCase, ...]:
    """Load a previously generated, immutable plan-case JSONL file."""
    cases = tuple(
        PlanCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Loaded plan cases contain duplicate case_id values")
    return cases
