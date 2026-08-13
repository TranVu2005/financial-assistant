"""Deterministic Day 13 export of manually classified retrieval failures."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from financial_report_qa.retrieval.contracts import (
    Fingerprint,
    QuestionId,
    RetrievalFilters,
    RetrievalIntent,
    RetrievalTrace,
    TableId,
)
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReportV2,
    RetrievalFailure,
)

FailureOnly = Literal[
    "no_eligible_documents",
    "no_index_tokens",
    "zero_gold_hits",
    "partial_gold_hits",
]
FailureRootCause = Literal[
    "missing_alias",
    "ocr_corruption",
    "filter_too_narrow",
    "filter_too_wide",
    "gold_label_error",
    "ranking_only",
    "unknown",
]
EvidenceNote = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_FAILURE_CATEGORIES: tuple[FailureOnly, ...] = (
    "no_eligible_documents",
    "no_index_tokens",
    "zero_gold_hits",
    "partial_gold_hits",
)
_ROOT_CAUSES: tuple[FailureRootCause, ...] = (
    "missing_alias",
    "ocr_corruption",
    "filter_too_narrow",
    "filter_too_wide",
    "gold_label_error",
    "ranking_only",
    "unknown",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailureRootCauseAnnotation(_FrozenModel):
    """Human classification kept separate from automatic retrieval failure logic."""

    question_id: QuestionId
    root_cause: FailureRootCause
    note: EvidenceNote


class RetrievalFailureCase(_FrozenModel):
    """One failed question with complete top-10 and diagnostic evidence."""

    question_id: QuestionId
    question: EvidenceNote
    intent: RetrievalIntent
    filters: RetrievalFilters
    predicted_table_ids: tuple[TableId, ...]
    gold_table_ids: tuple[TableId, ...]
    missing_gold_table_ids: tuple[TableId, ...]
    gold_rank_beyond_10: dict[str, int | None]
    failure: FailureOnly
    root_cause: FailureRootCause
    note: EvidenceNote
    trace: RetrievalTrace

    @model_validator(mode="after")
    def validate_evidence_consistency(self) -> RetrievalFailureCase:
        traced_ids = tuple(candidate.table_id for candidate in self.trace.results)
        if len(traced_ids) > 10:
            raise ValueError("failure trace must contain only top-10 predictions")
        if self.predicted_table_ids != traced_ids:
            raise ValueError("predicted_table_ids must match the failure trace")
        expected_missing = tuple(
            sorted(set(self.gold_table_ids).difference(self.predicted_table_ids))
        )
        if self.missing_gold_table_ids != expected_missing:
            raise ValueError("missing_gold_table_ids must match top-10 predictions")
        if set(self.gold_rank_beyond_10) != set(self.missing_gold_table_ids):
            raise ValueError("diagnostic ranks must cover every missing gold table exactly")
        if any(
            rank is not None and not 11 <= rank <= 100
            for rank in self.gold_rank_beyond_10.values()
        ):
            raise ValueError("diagnostic gold ranks must be between 11 and 100")
        return self


class RetrievalFailureReport(_FrozenModel):
    """Day 13 failure-only artifact used for source-backed Day 14 decisions."""

    dataset_fingerprint: Fingerprint
    diagnostic_k: Literal[100]
    evaluated_question_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    failure_counts: dict[FailureOnly, int]
    root_cause_counts: dict[FailureRootCause, int]
    failures: tuple[RetrievalFailureCase, ...]

    @model_validator(mode="after")
    def validate_summary(self) -> RetrievalFailureReport:
        question_ids = tuple(item.question_id for item in self.failures)
        if question_ids != tuple(sorted(set(question_ids))):
            raise ValueError("failures must be sorted and unique by question_id")
        if self.failure_count != len(self.failures):
            raise ValueError("failure_count must equal the number of failure cases")
        expected_failures = Counter(item.failure for item in self.failures)
        if self.failure_counts != {
            category: expected_failures[category] for category in _FAILURE_CATEGORIES
        }:
            raise ValueError("failure_counts do not match failure cases")
        expected_roots = Counter(item.root_cause for item in self.failures)
        if self.root_cause_counts != {
            category: expected_roots[category] for category in _ROOT_CAUSES
        }:
            raise ValueError("root_cause_counts do not match failure cases")
        return self


def load_failure_annotations(path: Path) -> tuple[FailureRootCauseAnnotation, ...]:
    """Load deterministic JSONL annotations maintained independently of rankings."""
    if not path.is_file():
        raise ValueError(f"Failure annotation file not found: {path}")
    annotations: list[FailureRootCauseAnnotation] = []
    previous_id: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"Failure annotations contain blank line {line_number}")
        try:
            annotation = FailureRootCauseAnnotation.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"Invalid failure annotation at line {line_number}") from exc
        if previous_id is not None and annotation.question_id <= previous_id:
            raise ValueError("Failure annotations must be sorted and unique by question_id")
        previous_id = annotation.question_id
        annotations.append(annotation)
    return tuple(annotations)


def build_failure_report(
    evaluation: RetrievalEvaluationReportV2,
    annotations: Iterable[FailureRootCauseAnnotation],
) -> RetrievalFailureReport:
    """Join automatic V2 failures with independently supplied manual classifications."""
    if evaluation.diagnostic_k != 100:
        raise ValueError("Day 13 failure export requires diagnostic_k=100")
    annotation_values = tuple(annotations)
    annotation_by_id = {item.question_id: item for item in annotation_values}
    if len(annotation_by_id) != len(annotation_values):
        raise ValueError("root-cause annotations must have unique question IDs")

    failed_results = tuple(
        result for result in evaluation.per_question if result.failure != "none"
    )
    failure_ids = {result.question_id for result in failed_results}
    if set(annotation_by_id) != failure_ids:
        raise ValueError("root-cause annotations must exactly match failure question IDs")

    failures: list[RetrievalFailureCase] = []
    for result in sorted(failed_results, key=lambda item: item.question_id):
        annotation = annotation_by_id[result.question_id]
        failure = _failure_only(result.failure)
        failures.append(
            RetrievalFailureCase(
                question_id=result.question_id,
                question=result.question,
                intent=cast(RetrievalIntent, result.intent),
                filters=result.filters,
                predicted_table_ids=result.predicted_table_ids,
                gold_table_ids=result.gold_table_ids,
                missing_gold_table_ids=result.missing_gold_table_ids,
                gold_rank_beyond_10=result.gold_rank_beyond_10,
                failure=failure,
                root_cause=annotation.root_cause,
                note=annotation.note,
                trace=result.trace,
            )
        )

    failure_counter = Counter(item.failure for item in failures)
    root_counter = Counter(item.root_cause for item in failures)
    return RetrievalFailureReport(
        dataset_fingerprint=evaluation.dataset_fingerprint,
        diagnostic_k=cast(Literal[100], evaluation.diagnostic_k),
        evaluated_question_count=evaluation.question_count,
        failure_count=len(failures),
        failure_counts={
            category: failure_counter[category] for category in _FAILURE_CATEGORIES
        },
        root_cause_counts={category: root_counter[category] for category in _ROOT_CAUSES},
        failures=tuple(failures),
    )


def _failure_only(failure: RetrievalFailure) -> FailureOnly:
    if failure == "none":
        raise ValueError("successful retrieval cannot be exported as a failure")
    return failure


def _json_inline(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _render_markdown(report: RetrievalFailureReport) -> str:
    lines = [
        "# Day 13 Retrieval Failure Cases",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Diagnostic cutoff: {report.diagnostic_k}",
        f"- Evaluated questions: {report.evaluated_question_count}",
        f"- Failures: {report.failure_count}",
        "",
        "## Failure counts",
        "",
        "| Failure | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {category} | {report.failure_counts[category]} |"
        for category in _FAILURE_CATEGORIES
    )
    lines.extend(("", "## Root-cause counts", "", "| Root cause | Count |", "| --- | ---: |"))
    lines.extend(
        f"| {category} | {report.root_cause_counts[category]} |"
        for category in _ROOT_CAUSES
    )
    lines.extend(("", "## Failure evidence", ""))
    for item in report.failures:
        lines.extend(
            (
                f"### {item.question_id}",
                "",
                f"- Question: {item.question}",
                f"- Intent: {item.intent}",
                f"- Filters: `{_json_inline(item.filters.model_dump(mode='json'))}`",
                f"- Failure: {item.failure}",
                f"- Root cause: {item.root_cause}",
                f"- Evidence note: {item.note}",
                f"- Gold table IDs: {', '.join(item.gold_table_ids)}",
                f"- Missing gold table IDs: {', '.join(item.missing_gold_table_ids)}",
                "- Gold ranks beyond 10: "
                f"`{_json_inline(item.gold_rank_beyond_10)}`",
                f"- Query tokens: {', '.join(item.trace.query_tokens) or '(none)'}",
                f"- Eligible documents: {item.trace.eligible_count}",
                f"- Empty reason: {item.trace.empty_reason or '(none)'}",
                "- Filter decisions: `"
                + _json_inline(
                    [
                        decision.model_dump(mode="json")
                        for decision in item.trace.filter_decisions
                    ]
                )
                + "`",
                "",
                "| Rank | Table ID | Score | Matched tokens | Metadata |",
                "| ---: | --- | ---: | --- | --- |",
            )
        )
        if item.trace.results:
            for candidate in item.trace.results:
                metadata = _json_inline(candidate.metadata.model_dump(mode="json")).replace(
                    "|", "\\|"
                )
                matched = ", ".join(candidate.matched_tokens) or "(none)"
                lines.append(
                    f"| {candidate.rank} | `{candidate.table_id}` | {candidate.score:.9g} | "
                    f"{matched} | `{metadata}` |"
                )
        else:
            lines.append("| - | (none) | - | (none) | `{}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_failure_report(
    report: RetrievalFailureReport, output_dir: Path
) -> tuple[Path, Path]:
    """Publish byte-stable JSON and Markdown failure artifacts atomically."""
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"failures-{prefix}.json"
    markdown_path = output_dir / f"failures-{prefix}.md"
    json_content = json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
    write_text_atomic(json_path, json_content + "\n")
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
