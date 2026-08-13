"""Deterministic Day 12 graph-expansion grid evaluation."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalFilters,
    _FrozenModel,
)
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.dense_contracts import LatencySummary
from financial_report_qa.retrieval.dense_evaluation import (
    MetricDelta,
    _delta,
    _latency_summary,
    _validate_bm25_reference,
)
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
    score_at_10,
)
from financial_report_qa.retrieval.expansion import GraphExpansionService
from financial_report_qa.retrieval.expansion_contracts import (
    PRE_REGISTERED_EXPANSION_GRID,
    ExpansionParams,
    ExpansionTrace,
)
from financial_report_qa.retrieval.graph_service import TableGraphService
from financial_report_qa.retrieval.service import RetrievalService

ExpansionFailure = Literal[
    "no_eligible_documents", "zero_gold_hits", "partial_gold_hits", "full_gold_hits"
]


class ExpansionQuestionEvaluation(_FrozenModel):
    question_id: str
    question: str
    intent: str
    filters: RetrievalFilters
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    missing_gold_table_ids: tuple[str, ...]
    metrics: RetrievalMetrics
    failure: ExpansionFailure
    trace: ExpansionTrace


class ExpansionGridPoint(_FrozenModel):
    params: ExpansionParams
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    by_gold_cardinality: dict[str, RetrievalMetrics]
    by_period_cardinality: dict[str, RetrievalMetrics]
    failure_counts: dict[ExpansionFailure, int]
    delta_vs_bm25: MetricDelta
    latency: LatencySummary
    per_question: tuple[ExpansionQuestionEvaluation, ...]


class ExpansionGridReport(_FrozenModel):
    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    bm25_reference: RetrievalMetrics
    grid: tuple[ExpansionGridPoint, ...]
    best_params: ExpansionParams
    decision_reason: str
    evidence_caveat: str


def _average(metrics: Iterable[RetrievalMetrics]) -> RetrievalMetrics:
    values = tuple(metrics)
    if not values:
        return RetrievalMetrics(true_positive=0, precision=0, recall=0, f2=0)
    return RetrievalMetrics(
        true_positive=sum(item.true_positive for item in values),
        precision=sum(item.precision for item in values) / len(values),
        recall=sum(item.recall for item in values) / len(values),
        f2=sum(item.f2 for item in values) / len(values),
    )


def _failure(trace: ExpansionTrace, metrics: RetrievalMetrics, gold_count: int) -> ExpansionFailure:
    if trace.empty_reason == "no_eligible_documents":
        return "no_eligible_documents"
    if metrics.true_positive == 0:
        return "zero_gold_hits"
    if metrics.true_positive < gold_count:
        return "partial_gold_hits"
    return "full_gold_hits"


def _gold_cardinality(question: GoldRetrievalQuestion) -> str:
    return "one_table" if len(question.gold_table_ids) == 1 else "multiple_tables"


def _period_cardinality(question: GoldRetrievalQuestion) -> str:
    return "one_period" if len(question.filters.periods) == 1 else "multiple_periods"


def _evaluate_at_params(
    service: GraphExpansionService, questions: Sequence[GoldRetrievalQuestion]
) -> tuple[
    RetrievalMetrics,
    dict[str, RetrievalMetrics],
    dict[str, RetrievalMetrics],
    dict[str, RetrievalMetrics],
    dict[ExpansionFailure, int],
    LatencySummary,
    tuple[ExpansionQuestionEvaluation, ...],
]:
    per_intent: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    per_gold_cardinality: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    per_period_cardinality: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    failures: dict[ExpansionFailure, int] = {
        "full_gold_hits": 0,
        "partial_gold_hits": 0,
        "zero_gold_hits": 0,
        "no_eligible_documents": 0,
    }
    latency: list[float] = []
    results: list[ExpansionQuestionEvaluation] = []
    for question in sorted(questions, key=lambda item: item.question_id):
        start = time.perf_counter()
        trace = service.retrieve(
            question.question, filters=question.filters, k=10, question_id=question.question_id
        )
        latency.append(time.perf_counter() - start)
        predicted = tuple(candidate.table_id for candidate in trace.results)
        metrics = score_at_10(predicted, question.gold_table_ids)
        failure = _failure(trace, metrics, len(question.gold_table_ids))
        failures[failure] += 1
        results.append(
            ExpansionQuestionEvaluation(
                question_id=question.question_id,
                question=question.question,
                intent=question.intent,
                filters=question.filters,
                predicted_table_ids=predicted,
                gold_table_ids=question.gold_table_ids,
                missing_gold_table_ids=tuple(
                    sorted(set(question.gold_table_ids).difference(predicted[:10]))
                ),
                metrics=metrics,
                failure=failure,
                trace=trace,
            )
        )
        per_intent[question.intent].append(metrics)
        per_gold_cardinality[_gold_cardinality(question)].append(metrics)
        per_period_cardinality[_period_cardinality(question)].append(metrics)
    return (
        _average(item.metrics for item in results),
        {key: _average(per_intent[key]) for key in sorted(per_intent)},
        {key: _average(per_gold_cardinality[key]) for key in sorted(per_gold_cardinality)},
        {key: _average(per_period_cardinality[key]) for key in sorted(per_period_cardinality)},
        failures,
        _latency_summary(latency),
        tuple(results),
    )


def evaluate_expansion_grid(
    base: RetrievalService,
    graph_service: TableGraphService,
    questions: Sequence[GoldRetrievalQuestion],
    bm25_report: RetrievalEvaluationReport,
    *,
    grid: tuple[ExpansionParams, ...] = PRE_REGISTERED_EXPANSION_GRID,
) -> ExpansionGridReport:
    """Evaluate each pre-registered Day 12 point; this function selects no default system."""
    _validate_bm25_reference(bm25_report)
    points: list[ExpansionGridPoint] = []
    for params in grid:
        service = GraphExpansionService(base, graph_service, params)
        macro, by_intent, by_gold, by_period, failures, latency, per_question = _evaluate_at_params(
            service, questions
        )
        points.append(
            ExpansionGridPoint(
                params=params,
                macro=macro,
                by_intent=by_intent,
                by_gold_cardinality=by_gold,
                by_period_cardinality=by_period,
                failure_counts=failures,
                delta_vs_bm25=_delta(macro, bm25_report.macro),
                latency=latency,
                per_question=per_question,
            )
        )
    if not points:
        raise ValueError("expansion grid must not be empty")
    best = max(
        points,
        key=lambda point: (
            point.macro.f2,
            point.macro.recall,
            -point.params.alpha,
            point.params.relations,
            not point.params.expand_non_seeds,
        ),
    )
    return ExpansionGridReport(
        dataset_fingerprint=bm25_report.dataset_fingerprint,
        question_count=len(questions),
        bm25_reference=bm25_report.macro,
        grid=tuple(points),
        best_params=best.params,
        decision_reason=(
            "Day 12 records the best pre-registered point only; Day 14 decides whether "
            "graph expansion is retained."
        ),
        evidence_caveat=(
            "Only 4/30 questions have headroom; they involve only two distinct missing tables, "
            "both already present in BM25 top-50. This evaluation does not establish "
            "a default system."
        ),
    )


def deterministic_projection(report: ExpansionGridReport) -> dict[str, object]:
    """Scrub timing measurements so independently replayed evidence compares byte-for-byte."""
    projection = report.model_dump(mode="json")
    for point in projection["grid"]:
        point.pop("latency")
    return projection


def _render_markdown(report: ExpansionGridReport) -> str:
    lines = [
        "# Day 12 Graph Expansion Evaluation",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- Decision: {report.decision_reason}",
        f"- Caveat: {report.evidence_caveat}",
        "",
        "## Full pre-registered grid",
        "",
        "| Relations | Alpha | Expand non-seeds | Precision@10 | Recall@10 | F2@10 | "
        "Latency p95 (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for point in report.grid:
        lines.append(
            "| "
            + " | ".join(
                (
                    ", ".join(point.params.relations),
                    f"{point.params.alpha:g}",
                    str(point.params.expand_non_seeds),
                    f"{point.macro.precision:.6f}",
                    f"{point.macro.recall:.6f}",
                    f"{point.macro.f2:.6f}",
                    f"{point.latency.p95_seconds:.6f}",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_day12_expansion(report: ExpansionGridReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-day12-expansion-{prefix}.json"
    markdown_path = output_dir / f"retrieval-day12-expansion-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
