"""Deterministic Day 10 fusion-grid evaluation against the locked gold set.

Every point in `financial_report_qa.retrieval.fusion_contracts.PRE_REGISTERED_WEIGHT_GRID`
is scored and reported — none are cherry-picked. `default_system` becomes
`"fusion"` only when the best grid point is at or above BM25 v3 on **both**
F2@10 and Recall@10; otherwise BM25 v3 stays the default and the shortfall is
recorded as-is. This mirrors the guard-then-report shape of
`financial_report_qa.retrieval.dense_evaluation`.
"""

from __future__ import annotations

import json
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
from financial_report_qa.retrieval.dense_evaluation import (
    MetricDelta,
    _delta,
    _validate_bm25_reference,
)
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
    score_at_10,
)
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.fusion_contracts import (
    PRE_REGISTERED_WEIGHT_GRID,
    FusionTrace,
    FusionWeights,
)
from financial_report_qa.retrieval.service import RetrievalService

FusionFailure = Literal[
    "no_eligible_documents", "zero_gold_hits", "partial_gold_hits", "full_gold_hits"
]
DefaultSystem = Literal["bm25-v3", "fusion"]


class FusionQuestionEvaluation(_FrozenModel):
    question_id: str
    question: str
    intent: str
    filters: RetrievalFilters
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    missing_gold_table_ids: tuple[str, ...]
    metrics: RetrievalMetrics
    failure: FusionFailure
    trace: FusionTrace


class FusionGridPoint(_FrozenModel):
    weights: FusionWeights
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    failure_counts: dict[FusionFailure, int]
    delta_vs_bm25: MetricDelta
    per_question: tuple[FusionQuestionEvaluation, ...]


class FusionGridReport(_FrozenModel):
    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    bm25_reference: RetrievalMetrics
    grid: tuple[FusionGridPoint, ...]
    best_weights: FusionWeights
    default_system: DefaultSystem
    decision_reason: str


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


def _fusion_failure(
    trace: FusionTrace, metrics: RetrievalMetrics, gold_count: int
) -> FusionFailure:
    if trace.empty_reason == "no_eligible_documents":
        return "no_eligible_documents"
    if metrics.true_positive == 0:
        return "zero_gold_hits"
    if metrics.true_positive < gold_count:
        return "partial_gold_hits"
    return "full_gold_hits"


def _evaluate_at_weights(
    service: FusionService, questions: Sequence[GoldRetrievalQuestion]
) -> tuple[
    RetrievalMetrics,
    dict[str, RetrievalMetrics],
    dict[FusionFailure, int],
    tuple[FusionQuestionEvaluation, ...],
]:
    results: list[FusionQuestionEvaluation] = []
    per_intent: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    failure_counts: dict[FusionFailure, int] = {
        "full_gold_hits": 0,
        "partial_gold_hits": 0,
        "zero_gold_hits": 0,
        "no_eligible_documents": 0,
    }
    for question in sorted(questions, key=lambda item: item.question_id):
        trace = service.retrieve(
            question.question, filters=question.filters, k=10, question_id=question.question_id
        )
        predicted = tuple(candidate.table_id for candidate in trace.results)
        metrics = score_at_10(predicted, question.gold_table_ids)
        failure = _fusion_failure(trace, metrics, len(question.gold_table_ids))
        failure_counts[failure] += 1
        missing = tuple(sorted(set(question.gold_table_ids).difference(predicted[:10])))
        results.append(
            FusionQuestionEvaluation(
                question_id=question.question_id,
                question=question.question,
                intent=question.intent,
                filters=question.filters,
                predicted_table_ids=predicted,
                gold_table_ids=question.gold_table_ids,
                missing_gold_table_ids=missing,
                metrics=metrics,
                failure=failure,
                trace=trace,
            )
        )
        per_intent[question.intent].append(metrics)
    macro = _average(result.metrics for result in results)
    by_intent = {intent: _average(per_intent[intent]) for intent in sorted(per_intent)}
    return macro, by_intent, failure_counts, tuple(results)


def evaluate_fusion_grid(
    bm25: RetrievalService,
    dense: DenseRetrievalService,
    questions: Sequence[GoldRetrievalQuestion],
    bm25_report: RetrievalEvaluationReport,
    *,
    weight_grid: tuple[FusionWeights, ...] = PRE_REGISTERED_WEIGHT_GRID,
) -> FusionGridReport:
    """Evaluate every pre-registered weight point and apply the pinned decision rule."""
    _validate_bm25_reference(bm25_report)
    bm25_metrics = bm25_report.macro

    points: list[FusionGridPoint] = []
    for weights in weight_grid:
        service = FusionService(bm25, dense, weights)
        macro, by_intent, failure_counts, per_question = _evaluate_at_weights(service, questions)
        points.append(
            FusionGridPoint(
                weights=weights,
                macro=macro,
                by_intent=by_intent,
                failure_counts=failure_counts,
                delta_vs_bm25=_delta(macro, bm25_metrics),
                per_question=per_question,
            )
        )

    # Tie-break prefers the simplest weighting: lower dense weight first (less
    # machinery to keep BM25-equivalent behavior), then higher bm25 weight.
    # In practice PRE_REGISTERED_WEIGHT_GRID has no duplicate (f2, recall)
    # pairs, so this only matters as a documented, deterministic fallback.
    def _tie_break_key(point: FusionGridPoint) -> tuple[float, float, float, float]:
        return (point.macro.f2, point.macro.recall, -point.weights.dense, point.weights.bm25)

    best = max(points, key=_tie_break_key)
    reaches_reference = (
        best.macro.f2 >= bm25_metrics.f2 and best.macro.recall >= bm25_metrics.recall
    )
    # dense=0 makes the grid point BM25 verbatim (see FusionService: a
    # zero-weighted branch contributes no candidates), so it always ties the
    # reference trivially. Reaching the reference is necessary but not
    # sufficient -- "fusion" must also carry a real, weighted dense
    # contribution, or the label just means "BM25 plus unused dense
    # infrastructure".
    meets_gate = reaches_reference and best.weights.dense > 0
    default_system: DefaultSystem = "fusion" if meets_gate else "bm25-v3"
    if meets_gate:
        reason = (
            f"weights bm25={best.weights.bm25}/dense={best.weights.dense} reach "
            f"F2={best.macro.f2:.6f} (>= {bm25_metrics.f2:.6f}) and "
            f"Recall={best.macro.recall:.6f} (>= {bm25_metrics.recall:.6f}) with a real dense "
            f"contribution (dense={best.weights.dense} > 0); fusion becomes default"
        )
    elif reaches_reference:
        reason = (
            f"best grid point bm25={best.weights.bm25}/dense={best.weights.dense} reaches BM25 "
            f"v3 on both F2 and Recall, but uses no dense weight (dense=0.0), so it carries no "
            f"real dense contribution -- it is BM25 v3 itself; BM25 v3 stays default"
        )
    else:
        reason = (
            f"no grid point reached BM25 v3 on both F2 and Recall; best was "
            f"bm25={best.weights.bm25}/dense={best.weights.dense} F2={best.macro.f2:.6f} "
            f"Recall={best.macro.recall:.6f} vs reference F2={bm25_metrics.f2:.6f} "
            f"Recall={bm25_metrics.recall:.6f}; BM25 v3 stays default"
        )

    return FusionGridReport(
        dataset_fingerprint=bm25_report.dataset_fingerprint,
        question_count=len(questions),
        bm25_reference=bm25_metrics,
        grid=tuple(points),
        best_weights=best.weights,
        default_system=default_system,
        decision_reason=reason,
    )


def deterministic_projection(report: FusionGridReport) -> dict[str, object]:
    """Return the replay-relevant projection (currently the full report).

    Nothing in `FusionGridReport` is non-deterministic today — dense scores
    come from the query-embedding cache, which guarantees identical vectors
    regardless of cache hit/miss, and `FusionTrace` never surfaces the raw
    `cache_hit` flag. Kept as an explicit function, matching
    `financial_report_qa.retrieval.dense_evaluation.deterministic_projection`,
    so a future field that *is* non-deterministic has an obvious place to be
    scrubbed instead of silently leaking into a replay comparison.
    """
    projection = report.model_dump(mode="json")
    if not isinstance(projection, dict):
        raise ValueError("day 10 deterministic projection must be an object")
    return projection


def _render_markdown(report: FusionGridReport) -> str:
    lines = [
        "# Day 10 Fusion Grid Evaluation",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        f"- Questions: {report.question_count}",
        f"- BM25 v3 reference: Precision@10={report.bm25_reference.precision:.6f} "
        f"Recall@10={report.bm25_reference.recall:.6f} F2@10={report.bm25_reference.f2:.6f}",
        f"- Default system: **{report.default_system}**",
        f"- Best weights: bm25={report.best_weights.bm25} dense={report.best_weights.dense} "
        f"(rrf_k={report.best_weights.rrf_k}, depth={report.best_weights.depth})",
        f"- Decision reason: {report.decision_reason}",
        "",
        "## Full pre-registered grid",
        "",
        "| bm25 | dense | Precision@10 | Recall@10 | F2@10 | ΔF2 vs BM25 | ΔRecall vs BM25 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in report.grid:
        lines.append(
            f"| {point.weights.bm25} | {point.weights.dense} | {point.macro.precision:.6f} | "
            f"{point.macro.recall:.6f} | {point.macro.f2:.6f} | {point.delta_vs_bm25.f2:+.6f} | "
            f"{point.delta_vs_bm25.recall:+.6f} |"
        )
    lines.extend(("", "## Failure counts by weight point", ""))
    lines.append("| bm25 | dense | full | partial | zero | no_eligible |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for point in report.grid:
        counts = point.failure_counts
        lines.append(
            f"| {point.weights.bm25} | {point.weights.dense} | {counts['full_gold_hits']} | "
            f"{counts['partial_gold_hits']} | {counts['zero_gold_hits']} | "
            f"{counts['no_eligible_documents']} |"
        )
    return "\n".join(lines) + "\n"


def write_day10_fusion(report: FusionGridReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-day10-fusion-{prefix}.json"
    markdown_path = output_dir / f"retrieval-day10-fusion-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
