"""Day 20 verify-answers evaluation report (plan Sec 3 task 20.10).

Mirrors `execution/evaluation.py::evaluate_compiled_plans_on_gold` in shape,
one step further: for every question the compiler resolves to a scalar
answer, this also builds and verifies an `AnswerPackage` and, if a hand-
labeled `answer-gold-v1.jsonl` value exists for that question (Day 20 plan
Sec 1.1/task 20.9), scores it for accuracy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import duckdb
from pydantic import Field

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.rule_planner import build_plan
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.verification.builder import build_answer_package
from financial_report_qa.verification.contracts import VerificationIssueCode


def load_answer_gold(path: Path) -> dict[str, Decimal]:
    """Load `answer-gold-v1.jsonl` (Day 20 plan Sec 1.1/task 20.9) into a
    `question_id -> Decimal` map. Extra fields per record (evidence,
    provenance, notes) are ignored here -- only `question_id`/`answer` are
    needed to score `compile_plan` output for accuracy.
    """
    gold: dict[str, Decimal] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            gold[record["question_id"]] = Decimal(record["answer"])
    return gold


def build_citation_lookup(
    release_dir: Path, cell_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    """Resolve citation provenance fields for a batch of cell ids.

    Mirrors `execution/cell_frame.py::_hardened_connection` (ADR 0008 F1):
    disable extension autoinstall/autoload so this read-only lookup cannot
    reach the network either.
    """
    if not cell_ids:
        return {}
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    try:
        rows = connection.execute(
            """
            SELECT c.cell_id, d.relative_path, c.source_line_start, c.source_line_end, t.title_raw
            FROM read_parquet(?) AS c
            JOIN read_parquet(?) AS t USING (table_id)
            JOIN read_parquet(?) AS d USING (doc_id)
            WHERE c.cell_id IN (SELECT UNNEST(?))
            """,
            [
                str(release_dir / "cells.parquet"),
                str(release_dir / "tables.parquet"),
                str(release_dir / "documents.parquet"),
                list(cell_ids),
            ],
        ).fetchall()
    finally:
        connection.close()
    return {
        cell_id: {
            "doc_relative_path": relative_path,
            "source_line_start": line_start,
            "source_line_end": line_end,
            "table_title": title,
        }
        for cell_id, relative_path, line_start, line_end, title in rows
    }


class AnswerVerificationFailure(_FrozenModel):
    question_id: str
    question: str
    reason: str


class AnswerVerificationReport(_FrozenModel):
    """One verification measurement pass over gold70."""

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    plannable_count: int = Field(ge=0)
    answered_count: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    issue_code_distribution: dict[str, int]
    accuracy_against_gold: float | None
    scored_against_gold_count: int = Field(ge=0)
    failures: tuple[AnswerVerificationFailure, ...]


def evaluate_answer_packages_on_gold(
    questions: Sequence[GoldRetrievalQuestion],
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
    answer_gold: dict[str, Decimal] | None,
) -> AnswerVerificationReport:
    """Route + compile + verify every gold70 question once."""
    answer_gold = answer_gold or {}
    plannable = answered = verified = rejected = 0
    scored = correct = 0
    issue_counts: Counter[VerificationIssueCode] = Counter()
    failures: list[AnswerVerificationFailure] = []

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
        if compiled.status != "answered":
            failures.append(
                AnswerVerificationFailure(
                    question_id=question.question_id,
                    question=question.question,
                    reason=f"not answered: {compiled.error_code}",
                )
            )
            continue
        answered += 1

        cell_ids = tuple(cid for cell in compiled.evidence for cid in cell.cell_ids)
        citation_lookup = build_citation_lookup(release_dir, cell_ids)
        package = build_answer_package(
            question_id=question.question_id,
            question=question.question,
            plan=plan_result.plan,
            compiled=compiled,
            retrieved_table_ids=known_table_ids,
            citation_lookup=citation_lookup,
        )
        for issue in package.verification_issues:
            issue_counts[issue.code] += 1
        if package.verification_status == "verified":
            verified += 1
        else:
            rejected += 1
            failures.append(
                AnswerVerificationFailure(
                    question_id=question.question_id,
                    question=question.question,
                    reason="; ".join(issue.code for issue in package.verification_issues),
                )
            )

        gold_answer = answer_gold.get(question.question_id)
        if gold_answer is not None:
            scored += 1
            if package.answer == gold_answer:
                correct += 1

    return AnswerVerificationReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "",
        question_count=len(ordered),
        plannable_count=plannable,
        answered_count=answered,
        verified_count=verified,
        rejected_count=rejected,
        issue_code_distribution=dict(sorted(issue_counts.items())),
        accuracy_against_gold=(correct / scored) if scored else None,
        scored_against_gold_count=scored,
        failures=tuple(failures),
    )


def _render_answer_verification_markdown(report: AnswerVerificationReport) -> str:
    lines = [
        "# Day 20 Answer Verification (retrieval-gold-v1)",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Plannable: {report.plannable_count}",
        f"- Answered by compiler: {report.answered_count}",
        f"- Verified: {report.verified_count}",
        f"- Rejected by verifier: {report.rejected_count}",
        f"- Scored against answer-gold-v1: {report.scored_against_gold_count}",
        f"- Accuracy against answer-gold-v1: {report.accuracy_against_gold}",
        "",
        "## Verification issue distribution",
        "",
        "| Code | Count |",
        "| --- | ---: |",
    ]
    for code, count in report.issue_code_distribution.items():
        lines.append(f"| {code} | {count} |")
    lines.extend(("", f"## Failures / rejections ({len(report.failures)})", ""))
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.question_id}",
                "",
                f"- Question: {failure.question}",
                f"- Reason: {failure.reason}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_answer_verification_report(
    report: AnswerVerificationReport, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"answer-verification-{prefix}.json"
    markdown_path = output_dir / f"answer-verification-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_answer_verification_markdown(report))
    return json_path, markdown_path
