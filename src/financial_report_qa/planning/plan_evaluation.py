"""Day 16 rule-planner evaluation (§16.7): two independent measurements.

- `evaluate_plan_cases` scores `rule_planner.build_plan` against
  `plan_cases.py`'s per-template answer key — the primary metric, iterated
  against while building the planner. Reports the Day 16 DoD hard KPI
  directly: `false_plan_rate` must be 0.0 (never return a plan for a question
  the answer key says must abstain).
- `evaluate_rule_planner_on_gold` is descriptive only, run once against
  gold70. gold70's `intent` taxonomy (lookup/compare/growth) does not map
  1:1 onto the 9 Day 15/16 `PlanOperation` values (Day 16 finding #2 / ADR
  0005 — e.g. `intent=compare` splits across `difference` and no operation at
  all), so there is no trustworthy per-question answer key to score against.
  It reports plannability and operation/abstain distribution, never accuracy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.plan_cases import PlanCase
from financial_report_qa.planning.rule_planner import PlanAbstainCode, build_plan
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic

_DUMMY_TABLE_ID = "tbl_" + "0" * 64
_DUMMY_KNOWN_TABLES = frozenset({_DUMMY_TABLE_ID})


class PlanCaseFailure(_FrozenModel):
    case_id: str
    question: str
    expected_operation: str | None
    actual_operation: str | None
    expected_abstain_code: str | None
    actual_abstain_codes: tuple[str, ...]


class PlanEvaluationReport(_FrozenModel):
    """Scored against `plan_cases.py`'s per-template answer key."""

    case_set_sha256: str
    case_count: int = Field(ge=0)
    operation_accuracy: float = Field(ge=0, le=1)
    abstain_recall: float = Field(ge=0, le=1)
    abstain_code_accuracy: float = Field(ge=0, le=1)
    false_plan_rate: float = Field(ge=0, le=1)
    exact_match_rate: float = Field(ge=0, le=1)
    failures: tuple[PlanCaseFailure, ...]


class HeldOutPlanReport(_FrozenModel):
    """Descriptive-only held-out report against gold70 (no operation answer key)."""

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    plannable_rate: float = Field(ge=0, le=1)
    operation_distribution: dict[str, int]
    abstain_code_distribution: dict[str, int]


def evaluate_plan_cases(cases: Sequence[PlanCase], *, case_set_sha256: str) -> PlanEvaluationReport:
    """Score the rule planner against the deterministic plan-case answer key."""
    operation_expected = operation_correct = 0
    abstain_expected = abstain_correct = abstain_code_correct = false_plan = 0
    exact_correct = 0
    failures: list[PlanCaseFailure] = []

    for case in sorted(cases, key=lambda item: item.case_id):
        entities = parse_query_entities(case.question)
        result = build_plan(
            entities, candidate_table_ids=(_DUMMY_TABLE_ID,), known_table_ids=_DUMMY_KNOWN_TABLES
        )
        actual_operation = result.plan.operation if result.plan is not None else None

        if case.expected_operation is not None:
            operation_expected += 1
            is_exact = actual_operation == case.expected_operation
            if is_exact:
                operation_correct += 1
                exact_correct += 1
        else:
            abstain_expected += 1
            if result.plan is None:
                abstain_correct += 1
                if case.expected_abstain_code in result.abstain_codes:
                    abstain_code_correct += 1
                    exact_correct += 1
            else:
                false_plan += 1
            is_exact = result.plan is None and case.expected_abstain_code in result.abstain_codes

        if not is_exact:
            failures.append(
                PlanCaseFailure(
                    case_id=case.case_id,
                    question=case.question,
                    expected_operation=case.expected_operation,
                    actual_operation=actual_operation,
                    expected_abstain_code=case.expected_abstain_code,
                    actual_abstain_codes=result.abstain_codes,
                )
            )

    total = len(cases)
    return PlanEvaluationReport(
        case_set_sha256=case_set_sha256,
        case_count=total,
        operation_accuracy=(operation_correct / operation_expected) if operation_expected else 1.0,
        abstain_recall=(abstain_correct / abstain_expected) if abstain_expected else 1.0,
        abstain_code_accuracy=(abstain_code_correct / abstain_expected)
        if abstain_expected
        else 1.0,
        false_plan_rate=(false_plan / abstain_expected) if abstain_expected else 0.0,
        exact_match_rate=(exact_correct / total) if total else 1.0,
        failures=tuple(sorted(failures, key=lambda item: item.case_id)),
    )


def evaluate_rule_planner_on_gold(
    questions: Sequence[GoldRetrievalQuestion],
) -> HeldOutPlanReport:
    """Describe (never score) the rule planner's behavior on gold70, once."""
    operation_counts: Counter[str] = Counter()
    abstain_counts: Counter[PlanAbstainCode] = Counter()

    ordered = sorted(questions, key=lambda item: item.question_id)
    for question in ordered:
        entities = parse_query_entities(question.question)
        known_table_ids = frozenset(question.gold_table_ids)
        result = build_plan(
            entities, candidate_table_ids=question.gold_table_ids, known_table_ids=known_table_ids
        )
        if result.plan is not None:
            operation_counts[result.plan.operation] += 1
        else:
            for code in result.abstain_codes:
                abstain_counts[code] += 1

    total = len(ordered)
    plannable = sum(operation_counts.values())
    return HeldOutPlanReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "",
        question_count=total,
        plannable_rate=(plannable / total) if total else 1.0,
        operation_distribution=dict(sorted(operation_counts.items())),
        abstain_code_distribution=dict(sorted(abstain_counts.items())),
    )


def _render_plan_case_markdown(report: PlanEvaluationReport) -> str:
    lines = [
        "# Day 16 Rule Planner Evaluation (Generated Plan Cases)",
        "",
        f"- Case set SHA-256: `{report.case_set_sha256}`",
        f"- Cases: {report.case_count}",
        f"- Operation accuracy (plan-expected cases): {report.operation_accuracy:.6f}",
        f"- Abstain recall (abstain-expected cases): {report.abstain_recall:.6f}",
        f"- Abstain code accuracy: {report.abstain_code_accuracy:.6f}",
        f"- **False-plan rate (DoD hard KPI, must be 0.0): {report.false_plan_rate:.6f}**",
        f"- Exact-match rate (overall): {report.exact_match_rate:.6f}",
        "",
        f"## Failures ({len(report.failures)})",
        "",
    ]
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.case_id}",
                "",
                f"- Question: {failure.question}",
                f"- Operation expected/actual: {failure.expected_operation} / "
                f"{failure.actual_operation}",
                f"- Abstain expected/actual: {failure.expected_abstain_code} / "
                f"{failure.actual_abstain_codes}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_plan_case_report(report: PlanEvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.case_set_sha256[:12]
    json_path = output_dir / f"plan-cases-{prefix}.json"
    markdown_path = output_dir / f"plan-cases-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_plan_case_markdown(report))
    return json_path, markdown_path


def _render_held_out_plan_markdown(report: HeldOutPlanReport) -> str:
    lines = [
        "# Day 16 Rule Planner Held-Out Report (retrieval-gold-v1)",
        "",
        "**Descriptive only — gold70's `intent` taxonomy does not map 1:1 onto",
        "`PlanOperation` (Day 16 finding #2 / ADR 0005), so there is no per-question",
        "answer key to score accuracy against.**",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Plannable rate (returned a plan, did not abstain): {report.plannable_rate:.6f}",
        "",
        "## Operation distribution (plannable questions)",
        "",
        "| Operation | Count |",
        "| --- | ---: |",
    ]
    for operation, count in report.operation_distribution.items():
        lines.append(f"| {operation} | {count} |")
    lines.extend(("", "## Abstain-code distribution (abstained questions)", ""))
    lines.extend(("| Code | Count |", "| --- | ---: |"))
    for code, count in report.abstain_code_distribution.items():
        lines.append(f"| {code} | {count} |")
    return "\n".join(lines) + "\n"


def write_held_out_plan_report(report: HeldOutPlanReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"plan-held-out-{prefix}.json"
    markdown_path = output_dir / f"plan-held-out-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_held_out_plan_markdown(report))
    return json_path, markdown_path
