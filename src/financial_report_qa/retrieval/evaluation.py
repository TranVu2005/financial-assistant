"""Fixed-denominator Retrieval@10 evaluation and deterministic report rendering."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class RetrievalMetricsExtended(BaseModel):
    """Versioned metrics with explicit cutoffs alongside the Day 8 measures."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_positive: int = Field(ge=0)
    precision_at_10: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    f2_at_10: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    precision_at_r: float = Field(ge=0, le=1)
    f2_at_r: float = Field(ge=0, le=1)


class RetrievalQuestionEvaluationV2(BaseModel):
    """Top-10 evidence plus diagnostic ranks for one reviewed question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str
    question: str
    intent: str
    filters: RetrievalFilters
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    missing_gold_table_ids: tuple[str, ...]
    gold_rank_beyond_10: dict[str, int | None]
    metrics: RetrievalMetricsExtended
    failure: RetrievalFailure
    trace: RetrievalTrace


class RetrievalEvaluationReportV2(BaseModel):
    """Day 13 report kept separate from strict persisted Day 8--12 models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    diagnostic_k: int = Field(ge=10)
    macro: RetrievalMetricsExtended
    by_intent: dict[str, RetrievalMetricsExtended]
    by_gold_cardinality: dict[str, RetrievalMetricsExtended]
    by_period_cardinality: dict[str, RetrievalMetricsExtended]
    by_statement_filter: dict[str, RetrievalMetricsExtended]
    by_report_era: dict[str, RetrievalMetricsExtended]
    per_question: tuple[RetrievalQuestionEvaluationV2, ...]

    @model_validator(mode="after")
    def validate_complete_question_evidence(self) -> RetrievalEvaluationReportV2:
        if self.question_count != len(self.per_question):
            raise ValueError("question_count must equal per_question length")
        return self


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


def score_extended_at_10(
    predicted: tuple[str, ...], gold: tuple[str, ...]
) -> RetrievalMetricsExtended:
    """Score Day 13 metrics; candidates after rank ten are diagnostics only."""
    legacy = score_at_10(predicted, gold)
    top_10 = predicted[:10]
    gold_set = set(gold)

    def recall_at(cutoff: int) -> float:
        return len(set(top_10[:cutoff]).intersection(gold_set)) / len(gold)

    first_gold_rank = next(
        (rank for rank, table_id in enumerate(top_10, start=1) if table_id in gold_set), None
    )
    r = len(gold)
    precision_at_r = len(set(top_10[:r]).intersection(gold_set)) / r
    f2_r_denominator = 4 * precision_at_r + legacy.recall
    return RetrievalMetricsExtended(
        true_positive=legacy.true_positive,
        precision_at_10=legacy.precision,
        recall_at_3=recall_at(3),
        recall_at_5=recall_at(5),
        recall_at_10=legacy.recall,
        f2_at_10=legacy.f2,
        mrr=0.0 if first_gold_rank is None else 1 / first_gold_rank,
        precision_at_r=precision_at_r,
        f2_at_r=(
            0.0
            if f2_r_denominator == 0
            else 5 * precision_at_r * legacy.recall / f2_r_denominator
        ),
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


def _average_extended(metrics: Iterable[RetrievalMetricsExtended]) -> RetrievalMetricsExtended:
    values = tuple(metrics)
    if not values:
        return RetrievalMetricsExtended(
            true_positive=0,
            precision_at_10=0,
            recall_at_3=0,
            recall_at_5=0,
            recall_at_10=0,
            f2_at_10=0,
            mrr=0,
            precision_at_r=0,
            f2_at_r=0,
        )
    count = len(values)
    return RetrievalMetricsExtended(
        true_positive=sum(item.true_positive for item in values),
        precision_at_10=sum(item.precision_at_10 for item in values) / count,
        recall_at_3=sum(item.recall_at_3 for item in values) / count,
        recall_at_5=sum(item.recall_at_5 for item in values) / count,
        recall_at_10=sum(item.recall_at_10 for item in values) / count,
        f2_at_10=sum(item.f2_at_10 for item in values) / count,
        mrr=sum(item.mrr for item in values) / count,
        precision_at_r=sum(item.precision_at_r for item in values) / count,
        f2_at_r=sum(item.f2_at_r for item in values) / count,
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


def _gold_cardinality_v2(question: GoldRetrievalQuestion) -> str:
    count = len(question.gold_table_ids)
    if count == 1:
        return "one_table"
    if count == 2:
        return "two_tables"
    return "three_or_more"


def _period_cardinality_v2(question: GoldRetrievalQuestion) -> str:
    return "one_period" if len(question.filters.periods) == 1 else "multiple_periods"


def _statement_filter_v2(question: GoldRetrievalQuestion) -> str:
    return "filtered" if question.filters.statement_types else "unfiltered"


def _report_era_v2(question: GoldRetrievalQuestion) -> str | None:
    if not question.filters.periods:
        return None
    latest_period = max(int(period) for period in question.filters.periods)
    if not 2015 <= latest_period <= 2025:
        raise ValueError("report-era breakdown requires periods from 2015 through 2025")
    if latest_period <= 2019:
        return "2015_2019"
    if latest_period <= 2023:
        return "2020_2023"
    return "2024_2025"


def _diagnostic_gold_ranks(
    predicted: tuple[str, ...], missing_gold: tuple[str, ...]
) -> dict[str, int | None]:
    ranks = {table_id: rank for rank, table_id in enumerate(predicted, start=1)}
    return {
        table_id: ranks.get(table_id)
        for table_id in missing_gold
    }


def evaluate_retrieval_v2(
    retriever: RetrievalClient,
    questions: tuple[GoldRetrievalQuestion, ...],
    *,
    k: int = 10,
    diagnostic_k: int = 100,
) -> RetrievalEvaluationReportV2:
    """Evaluate fixed top-10 metrics and retain deeper ranks only for diagnosis."""
    if k != 10:
        raise ValueError("Day 13 evaluation metrics are fixed at 10")
    if diagnostic_k < 10:
        raise ValueError("diagnostic_k must be at least 10")

    results: list[RetrievalQuestionEvaluationV2] = []
    per_intent: dict[str, list[RetrievalMetricsExtended]] = defaultdict(list)
    per_gold: dict[str, list[RetrievalMetricsExtended]] = defaultdict(list)
    per_period: dict[str, list[RetrievalMetricsExtended]] = defaultdict(list)
    per_statement: dict[str, list[RetrievalMetricsExtended]] = defaultdict(list)
    per_era: dict[str, list[RetrievalMetricsExtended]] = defaultdict(list)

    for question in sorted(questions, key=lambda item: item.question_id):
        retrieved_at_10 = retriever.retrieve(
            question.question,
            filters=question.filters,
            k=10,
            question_id=question.question_id,
        )
        predicted = tuple(candidate.table_id for candidate in retrieved_at_10.results[:10])
        metrics = score_extended_at_10(predicted, question.gold_table_ids)
        missing_gold = tuple(sorted(set(question.gold_table_ids).difference(predicted)))
        metric_trace = retrieved_at_10.model_copy(
            update={"results": retrieved_at_10.results[:10]}
        )
        diagnostic_trace = metric_trace
        if missing_gold and diagnostic_k > 10:
            diagnostic_trace = retriever.retrieve(
                question.question,
                filters=question.filters,
                k=diagnostic_k,
                question_id=question.question_id,
            )
        diagnostic_predicted = tuple(
            candidate.table_id for candidate in diagnostic_trace.results
        )
        if diagnostic_predicted[:10] != predicted:
            raise ValueError(
                "diagnostic ranking must preserve the metric top-10 prefix"
            )
        failure = _failure_for(
            metric_trace,
            RetrievalMetrics(
                true_positive=metrics.true_positive,
                precision=metrics.precision_at_10,
                recall=metrics.recall_at_10,
                f2=metrics.f2_at_10,
            ),
            len(question.gold_table_ids),
        )
        results.append(
            RetrievalQuestionEvaluationV2(
                question_id=question.question_id,
                question=question.question,
                intent=question.intent,
                filters=question.filters,
                predicted_table_ids=predicted,
                gold_table_ids=question.gold_table_ids,
                missing_gold_table_ids=missing_gold,
                gold_rank_beyond_10=_diagnostic_gold_ranks(
                    diagnostic_predicted, missing_gold
                ),
                metrics=metrics,
                failure=failure,
                trace=metric_trace,
            )
        )
        per_intent[question.intent].append(metrics)
        per_gold[_gold_cardinality_v2(question)].append(metrics)
        per_period[_period_cardinality_v2(question)].append(metrics)
        per_statement[_statement_filter_v2(question)].append(metrics)
        report_era = _report_era_v2(question)
        if report_era is not None:
            per_era[report_era].append(metrics)

    def averaged(
        groups: dict[str, list[RetrievalMetricsExtended]],
    ) -> dict[str, RetrievalMetricsExtended]:
        return {key: _average_extended(groups[key]) for key in sorted(groups)}

    return RetrievalEvaluationReportV2(
        dataset_fingerprint=questions[0].dataset_fingerprint if questions else "",
        question_count=len(questions),
        diagnostic_k=diagnostic_k,
        macro=_average_extended(result.metrics for result in results),
        by_intent=averaged(per_intent),
        by_gold_cardinality=averaged(per_gold),
        by_period_cardinality=averaged(per_period),
        by_statement_filter=averaged(per_statement),
        by_report_era=averaged(per_era),
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


def _render_markdown_v2(report: RetrievalEvaluationReportV2) -> str:
    macro = report.macro
    lines = [
        "# Day 13 Retrieval Evaluation V2",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Diagnostic cutoff: {report.diagnostic_k}",
        f"- Recall@10: {macro.recall_at_10:.6f}",
        f"- F2@R: {macro.f2_at_r:.6f}",
    ]

    def append_breakdown(
        title: str, values: dict[str, RetrievalMetricsExtended]
    ) -> None:
        lines.extend(
            (
                "",
                f"## {title}",
                "",
                "| Group | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R | TP |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for label, metrics in values.items():
            lines.append(
                f"| {label} | {metrics.precision_at_10:.6f} | "
                f"{metrics.recall_at_3:.6f} | {metrics.recall_at_5:.6f} | "
                f"{metrics.recall_at_10:.6f} | {metrics.f2_at_10:.6f} | "
                f"{metrics.mrr:.6f} | {metrics.precision_at_r:.6f} | "
                f"{metrics.f2_at_r:.6f} | {metrics.true_positive} |"
            )

    append_breakdown("By intent", report.by_intent)
    append_breakdown("By gold cardinality", report.by_gold_cardinality)
    append_breakdown("By period cardinality", report.by_period_cardinality)
    append_breakdown("By statement filter", report.by_statement_filter)
    append_breakdown("By report era", report.by_report_era)
    lines.extend(("", "## Per-question evidence", ""))
    for item in report.per_question:
        ranks = json.dumps(item.gold_rank_beyond_10, ensure_ascii=False, sort_keys=True)
        lines.append(
            f"- `{item.question_id}`: failure={item.failure}; "
            f"R@10={item.metrics.recall_at_10:.6f}; diagnostic_gold_ranks=`{ranks}`"
        )
    return "\n".join(lines) + "\n"


def write_report_v2(
    report: RetrievalEvaluationReportV2, output_dir: Path
) -> tuple[Path, Path]:
    """Publish a complete byte-stable V2 JSON/Markdown report pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-v2-{prefix}.json"
    markdown_path = output_dir / f"retrieval-v2-{prefix}.md"
    json_content = json.dumps(
        report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2
    )
    _write_atomic(json_path, json_content + "\n")
    _write_atomic(markdown_path, _render_markdown_v2(report))
    return json_path, markdown_path
