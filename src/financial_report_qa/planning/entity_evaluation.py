"""Deterministic evaluation for the Day 10 entity parser.

Two independent measurements, never mixed into one number:

- `evaluate_entity_cases` scores the parser against the generated,
  structurally-labeled case set (`entity_cases.py`) — the primary metric,
  safe to iterate against while building the parser.
- `evaluate_entity_parser_on_gold` scores the parser against the 30
  human-reviewed `retrieval-gold-v1` questions. It is a **held-out** check:
  run once, reported as-is, never used to tune a rule.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from financial_report_qa.planning.entity_cases import EntityCase
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic

ScoredField = Literal["company_codes", "periods", "metrics", "statement_types", "ambiguity"]


class FieldMetrics(_FrozenModel):
    """Micro-averaged set precision/recall/F1 for one entity field."""

    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class EntityCaseFailure(_FrozenModel):
    case_id: str
    template_id: str
    question: str
    expected_company_codes: tuple[str, ...]
    actual_company_codes: tuple[str, ...]
    expected_periods: tuple[str, ...]
    actual_periods: tuple[str, ...]
    expected_metrics: tuple[str, ...]
    actual_metrics: tuple[str, ...]
    expected_statement_types: tuple[str, ...]
    actual_statement_types: tuple[str, ...]
    expected_ambiguity: tuple[str, ...]
    actual_ambiguity: tuple[str, ...]


class EntityEvaluationReport(_FrozenModel):
    case_set_sha256: str
    case_count: int = Field(ge=0)
    by_field: dict[ScoredField, FieldMetrics]
    exact_match_rate: float = Field(ge=0, le=1)
    exact_match_rate_by_category: dict[str, float]
    failures: tuple[EntityCaseFailure, ...]


class HeldOutFailure(_FrozenModel):
    question_id: str
    question: str
    expected_company_codes: tuple[str, ...]
    actual_company_codes: tuple[str, ...]
    expected_periods: tuple[str, ...]
    actual_periods: tuple[str, ...]


class HeldOutEntityReport(_FrozenModel):
    """Held-out score against the 30 human-reviewed gold questions.

    Only `company_codes` and `periods` are scored: they are the only two
    fields every gold question labels (5/30 label `statement_types`, none
    label a target `metrics` set), so scoring the other fields here would
    compare the parser against an incomplete answer key.
    """

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    company_metrics: FieldMetrics
    period_metrics: FieldMetrics
    exact_match_rate: float = Field(ge=0, le=1)
    failures: tuple[HeldOutFailure, ...]


def _field_metrics(pairs: Iterable[tuple[tuple[str, ...], tuple[str, ...]]]) -> FieldMetrics:
    true_positive = false_positive = false_negative = 0
    for expected, actual in pairs:
        expected_set, actual_set = set(expected), set(actual)
        true_positive += len(expected_set & actual_set)
        false_positive += len(actual_set - expected_set)
        false_negative += len(expected_set - actual_set)
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else (1.0 if false_negative == 0 else 0.0)
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return FieldMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_entity_cases(
    cases: Sequence[EntityCase], *, case_set_sha256: str
) -> EntityEvaluationReport:
    """Score the deterministic parser against the generated case set."""
    company_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    period_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    metric_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    statement_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    ambiguity_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    failures: list[EntityCaseFailure] = []
    exact_by_category: dict[str, list[bool]] = defaultdict(list)

    for case in sorted(cases, key=lambda item: item.case_id):
        entities = parse_query_entities(case.question)
        company_pairs.append((case.expected_company_codes, entities.company_codes))
        period_pairs.append((case.expected_periods, entities.periods))
        metric_pairs.append((case.expected_metrics, entities.metrics))
        statement_pairs.append((case.expected_statement_types, entities.statement_types))
        ambiguity_pairs.append((case.expected_ambiguity, entities.ambiguity))

        exact = (
            entities.company_codes == case.expected_company_codes
            and entities.periods == case.expected_periods
            and entities.metrics == case.expected_metrics
            and entities.statement_types == case.expected_statement_types
            and entities.ambiguity == case.expected_ambiguity
        )
        exact_by_category[case.category].append(exact)
        if not exact:
            failures.append(
                EntityCaseFailure(
                    case_id=case.case_id,
                    template_id=case.template_id,
                    question=case.question,
                    expected_company_codes=case.expected_company_codes,
                    actual_company_codes=entities.company_codes,
                    expected_periods=case.expected_periods,
                    actual_periods=entities.periods,
                    expected_metrics=case.expected_metrics,
                    actual_metrics=entities.metrics,
                    expected_statement_types=case.expected_statement_types,
                    actual_statement_types=entities.statement_types,
                    expected_ambiguity=case.expected_ambiguity,
                    actual_ambiguity=entities.ambiguity,
                )
            )

    total = len(cases)
    exact_count = total - len(failures)
    return EntityEvaluationReport(
        case_set_sha256=case_set_sha256,
        case_count=total,
        by_field={
            "company_codes": _field_metrics(company_pairs),
            "periods": _field_metrics(period_pairs),
            "metrics": _field_metrics(metric_pairs),
            "statement_types": _field_metrics(statement_pairs),
            "ambiguity": _field_metrics(ambiguity_pairs),
        },
        exact_match_rate=(exact_count / total) if total else 1.0,
        exact_match_rate_by_category={
            category: sum(values) / len(values)
            for category, values in sorted(exact_by_category.items())
        },
        failures=tuple(sorted(failures, key=lambda item: item.case_id)),
    )


def evaluate_entity_parser_on_gold(
    questions: Sequence[GoldRetrievalQuestion],
) -> HeldOutEntityReport:
    """Score the parser once against the 30 human-reviewed gold questions.

    Held-out by convention, not by mechanism: nothing here prevents a second
    call, but the Day 10 workflow runs this exactly once, at the end, and
    treats its output as reporting evidence rather than a tuning signal.
    """
    company_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    period_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    failures: list[HeldOutFailure] = []
    exact_flags: list[bool] = []

    ordered = sorted(questions, key=lambda item: item.question_id)
    for question in ordered:
        entities = parse_query_entities(question.question)
        expected_company = question.filters.company_codes
        expected_periods = question.filters.periods
        company_pairs.append((expected_company, entities.company_codes))
        period_pairs.append((expected_periods, entities.periods))
        exact = entities.company_codes == expected_company and entities.periods == expected_periods
        exact_flags.append(exact)
        if not exact:
            failures.append(
                HeldOutFailure(
                    question_id=question.question_id,
                    question=question.question,
                    expected_company_codes=expected_company,
                    actual_company_codes=entities.company_codes,
                    expected_periods=expected_periods,
                    actual_periods=entities.periods,
                )
            )

    return HeldOutEntityReport(
        dataset_fingerprint=ordered[0].dataset_fingerprint if ordered else "",
        question_count=len(ordered),
        company_metrics=_field_metrics(company_pairs),
        period_metrics=_field_metrics(period_pairs),
        exact_match_rate=(sum(exact_flags) / len(exact_flags)) if exact_flags else 1.0,
        failures=tuple(failures),
    )


def _render_case_markdown(report: EntityEvaluationReport) -> str:
    lines = [
        "# Day 10 Entity Parser Evaluation (Generated Cases)",
        "",
        f"- Case set SHA-256: `{report.case_set_sha256}`",
        f"- Cases: {report.case_count}",
        f"- Exact-match rate: {report.exact_match_rate:.6f}",
        "",
        "## By field",
        "",
        "| Field | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for field, metrics in report.by_field.items():
        lines.append(
            f"| {field} | {metrics.precision:.6f} | {metrics.recall:.6f} | {metrics.f1:.6f} | "
            f"{metrics.true_positive} | {metrics.false_positive} | {metrics.false_negative} |"
        )
    lines.extend(("", "## Exact-match rate by category", ""))
    lines.extend(("| Category | Rate |", "| --- | ---: |"))
    for category, rate in report.exact_match_rate_by_category.items():
        lines.append(f"| {category} | {rate:.6f} |")
    lines.extend(("", f"## Failures ({len(report.failures)})", ""))
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.case_id}",
                "",
                f"- Template: {failure.template_id}",
                f"- Question: {failure.question}",
                f"- Company expected/actual: {failure.expected_company_codes} / "
                f"{failure.actual_company_codes}",
                f"- Period expected/actual: {failure.expected_periods} / {failure.actual_periods}",
                f"- Metric expected/actual: {failure.expected_metrics} / {failure.actual_metrics}",
                f"- Statement expected/actual: {failure.expected_statement_types} / "
                f"{failure.actual_statement_types}",
                f"- Ambiguity expected/actual: {failure.expected_ambiguity} / "
                f"{failure.actual_ambiguity}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_entity_case_report(report: EntityEvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.case_set_sha256[:12]
    json_path = output_dir / f"entity-cases-{prefix}.json"
    markdown_path = output_dir / f"entity-cases-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_case_markdown(report))
    return json_path, markdown_path


def _render_held_out_markdown(report: HeldOutEntityReport) -> str:
    lines = [
        "# Day 10 Entity Parser Held-Out Evaluation (retrieval-gold-v1)",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Exact-match rate (company + period): {report.exact_match_rate:.6f}",
        f"- Company precision/recall/F1: {report.company_metrics.precision:.6f} / "
        f"{report.company_metrics.recall:.6f} / {report.company_metrics.f1:.6f}",
        f"- Period precision/recall/F1: {report.period_metrics.precision:.6f} / "
        f"{report.period_metrics.recall:.6f} / {report.period_metrics.f1:.6f}",
        "",
        f"## Failures ({len(report.failures)})",
        "",
    ]
    for failure in report.failures:
        lines.extend(
            (
                f"### {failure.question_id}",
                "",
                f"- Question: {failure.question}",
                f"- Company expected/actual: {failure.expected_company_codes} / "
                f"{failure.actual_company_codes}",
                f"- Period expected/actual: {failure.expected_periods} / {failure.actual_periods}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_held_out_report(report: HeldOutEntityReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"entity-held-out-{prefix}.json"
    markdown_path = output_dir / f"entity-held-out-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_held_out_markdown(report))
    return json_path, markdown_path
