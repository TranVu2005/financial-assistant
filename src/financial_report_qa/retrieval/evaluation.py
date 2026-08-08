"""Fixed-denominator Retrieval@10 evaluation and deterministic report rendering."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalFilters,
    RetrievalTrace,
)

RetrievalFailure = Literal[
    "no_eligible_documents",
    "no_index_tokens",
    "zero_gold_hits",
    "partial_gold_hits",
    "none",
]


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    true_positive: int = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f2: float = Field(ge=0, le=1)


class RetrievalQuestionEvaluation(BaseModel):
    """All persisted evidence needed to inspect one evaluated retrieval query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    question: str
    intent: str
    filters: RetrievalFilters
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    missing_gold_table_ids: tuple[str, ...]
    metrics: RetrievalMetrics
    failure: RetrievalFailure
    trace: RetrievalTrace


class RetrievalEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    per_question: tuple[RetrievalQuestionEvaluation, ...]


class RetrievalClient(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int,
        question_id: str,
    ) -> RetrievalTrace: ...


def score_at_10(predicted: tuple[str, ...], gold: tuple[str, ...]) -> RetrievalMetrics:
    """Score fixed Retrieval@10 precision, recall, and F2."""
    if len(set(predicted)) != len(predicted):
        raise ValueError("predicted table IDs must be unique")
    if len(set(gold)) != len(gold):
        raise ValueError("gold table IDs must be unique")
    if not gold:
        raise ValueError("gold table IDs must not be empty")
    true_positive = len(set(predicted[:10]).intersection(gold))
    precision = true_positive / 10
    recall = true_positive / len(gold)
    denominator = 4 * precision + recall
    return RetrievalMetrics(
        true_positive=true_positive,
        precision=precision,
        recall=recall,
        f2=0.0 if denominator == 0 else 5 * precision * recall / denominator,
    )


def _average(metrics: Iterable[RetrievalMetrics]) -> RetrievalMetrics:
    values = tuple(metrics)
    if not values:
        return RetrievalMetrics(true_positive=0, precision=0, recall=0, f2=0)
    count = len(values)
    return RetrievalMetrics(
        true_positive=sum(item.true_positive for item in values),
        precision=sum(item.precision for item in values) / count,
        recall=sum(item.recall for item in values) / count,
        f2=sum(item.f2 for item in values) / count,
    )


def _failure_for(
    trace: RetrievalTrace, metrics: RetrievalMetrics, gold_count: int
) -> RetrievalFailure:
    if trace.empty_reason is not None:
        return trace.empty_reason
    if metrics.true_positive == 0:
        return "zero_gold_hits"
    if metrics.true_positive < gold_count:
        return "partial_gold_hits"
    return "none"


def evaluate_retrieval(
    retriever: RetrievalClient,
    questions: tuple[GoldRetrievalQuestion, ...],
    *,
    k: int = 10,
) -> RetrievalEvaluationReport:
    """Evaluate reviewed questions with a non-configurable Day 8 cutoff of ten."""
    if k != 10:
        raise ValueError("Day 8 evaluation is fixed at 10")
    results: list[RetrievalQuestionEvaluation] = []
    per_intent: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    for question in sorted(questions, key=lambda item: item.question_id):
        trace = retriever.retrieve(
            question.question, filters=question.filters, k=10, question_id=question.question_id
        )
        predicted = tuple(candidate.table_id for candidate in trace.results)
        metrics = score_at_10(predicted, question.gold_table_ids)
        missing_gold = tuple(sorted(set(question.gold_table_ids).difference(predicted[:10])))
        results.append(
            RetrievalQuestionEvaluation(
                question_id=question.question_id,
                question=question.question,
                intent=question.intent,
                filters=question.filters,
                predicted_table_ids=predicted,
                gold_table_ids=question.gold_table_ids,
                missing_gold_table_ids=missing_gold,
                metrics=metrics,
                failure=_failure_for(trace, metrics, len(question.gold_table_ids)),
                trace=trace,
            )
        )
        per_intent[question.intent].append(metrics)
    return RetrievalEvaluationReport(
        dataset_fingerprint=questions[0].dataset_fingerprint if questions else "",
        question_count=len(questions),
        macro=_average(result.metrics for result in results),
        by_intent={intent: _average(per_intent[intent]) for intent in sorted(per_intent)},
        per_question=tuple(results),
    )


def _render_markdown(report: RetrievalEvaluationReport) -> str:
    macro = report.macro
    lines = [
        "# Day 8 BM25 Retrieval Evaluation",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Precision@10: {macro.precision:.6f}",
        f"- Recall@10: {macro.recall:.6f}",
        f"- F2@10: {macro.f2:.6f}",
        "",
        "## Metrics by intent",
        "",
        "| Intent | Precision@10 | Recall@10 | F2@10 | True positives |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for intent, metrics in report.by_intent.items():
        lines.append(
            f"| {intent} | {metrics.precision:.6f} | {metrics.recall:.6f} | "
            f"{metrics.f2:.6f} | {metrics.true_positive} |"
        )
    lines.extend(("", "## Query evidence", ""))
    for result in report.per_question:
        lines.extend(
            (
                f"### {result.question_id}",
                "",
                f"- Question: {result.question}",
                f"- Intent: {result.intent}",
                f"- Failure: {result.failure}",
                f"- Predicted table IDs: {', '.join(result.predicted_table_ids) or '(none)'}",
                f"- Gold table IDs: {', '.join(result.gold_table_ids)}",
                f"- Missing gold table IDs: {', '.join(result.missing_gold_table_ids) or '(none)'}",
                f"- Eligible documents: {result.trace.eligible_count}",
                f"- Empty reason: {result.trace.empty_reason or '(none)'}",
                "- Filter counts: "
                + (
                    "; ".join(
                        f"{item.field}={item.matched_count_before_intersection}/"
                        f"{item.eligible_count_after_intersection}"
                        for item in result.trace.filter_decisions
                    )
                    or "(no filters)"
                ),
                "- Scores and matched tokens: "
                + (
                    "; ".join(
                        f"{candidate.table_id}={candidate.score:.6f} "
                        f"[{', '.join(candidate.matched_tokens)}]"
                        for candidate in result.trace.results
                    )
                    or "(none)"
                ),
                "",
            )
        )
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, content: str) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def write_report(report: RetrievalEvaluationReport, output_dir: Path) -> tuple[Path, Path]:
    """Publish byte-stable JSON and Markdown evaluation artifacts atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-day8-{prefix}.json"
    markdown_path = output_dir / f"retrieval-day8-{prefix}.md"
    json_content = json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
    _write_atomic(json_path, json_content + "\n")
    _write_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
