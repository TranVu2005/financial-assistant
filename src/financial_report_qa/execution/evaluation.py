"""Day 18 compile-plans evaluation: route + compile every gold70 question.

Mirrors `planning/plan_evaluation.py::evaluate_rule_planner_on_gold` in shape,
but goes one step further than plannability: for every question the rule
planner can plan, it also compiles the plan against the locked release and
reports whether the compiler resolved it to a scalar answer or a typed error.
`resolved_rate` is Day 18 plan §1.3's headroom number (24/51 baseline,
30/51 with period inference) made reproducible by CLI.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.contracts import ExecutionIssueCode
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.rule_planner import build_plan
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic


class CompiledPlanFailure(_FrozenModel):
    question_id: str
    question: str
    operation: str
    error_code: ExecutionIssueCode
    error_message: str


class CompiledPlanReport(_FrozenModel):
    """One deterministic-compiler measurement pass over gold70."""

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    plannable_count: int = Field(ge=0)
    resolved_count: int = Field(ge=0)
    resolved_rate: float = Field(ge=0, le=1)
    error_code_distribution: dict[str, int]
    failures: tuple[CompiledPlanFailure, ...]


def evaluate_compiled_plans_on_gold(
    questions: Sequence[GoldRetrievalQuestion],
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
) -> CompiledPlanReport:
    """Route + compile every gold70 question once; never guesses (ADR 0007 D1)."""
    plannable = resolved = 0
    error_counts: Counter[str] = Counter()
    failures: list[CompiledPlanFailure] = []

    ordered = sorted(questions, key=lambda item: item.question_id)
    for question in ordered:
        entities = parse_query_entities(question.question)
        known_table_ids = frozenset(question.gold_table_ids)
        plan_result = build_plan(
            entities,
            candidate_table_ids=question.gold_table_ids,
            known_table_ids=known_table_ids,
        )
        if plan_result.plan is None:
            continue
        plannable += 1

        compiled = compile_plan(
            plan_result.plan, release_dir, execution_settings=execution_settings
        )
        if compiled.status == "answered":
            resolved += 1
        else:
            assert compiled.error_code is not None and compiled.error_message is not None
            error_counts[compiled.error_code] += 1
            failures.append(
                CompiledPlanFailure(
                    question_id=question.question_id,
                    question=question.question,
                    operation=compiled.operation,
                    error_code=compiled.error_code,
                    error_message=compiled.error_message,
                )
            )

    return CompiledPlanReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "",
        question_count=len(ordered),
        plannable_count=plannable,
        resolved_count=resolved,
        resolved_rate=(resolved / plannable) if plannable else 1.0,
        error_code_distribution=dict(sorted(error_counts.items())),
        failures=tuple(failures),
    )


def _render_compiled_plan_markdown(report: CompiledPlanReport) -> str:
    lines = [
        "# Day 18 Deterministic Compiler Evaluation (retrieval-gold-v1)",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Plannable (rule planner returned a plan): {report.plannable_count}",
        f"- Resolved to a scalar answer: {report.resolved_count}",
        f"- Resolved rate: {report.resolved_rate:.6f}",
        "",
        "## Error-code distribution (unresolved plans)",
        "",
        "| Code | Count |",
        "| --- | ---: |",
    ]
    for code, count in report.error_code_distribution.items():
        lines.append(f"| {code} | {count} |")
    lines.extend(("", f"## Failures ({len(report.failures)})", ""))
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.question_id}",
                "",
                f"- Question: {failure.question}",
                f"- Operation: {failure.operation}",
                f"- Error: `{failure.error_code}` — {failure.error_message}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_compiled_plan_report(report: CompiledPlanReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"compiled-plans-{prefix}.json"
    markdown_path = output_dir / f"compiled-plans-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_compiled_plan_markdown(report))
    return json_path, markdown_path
