"""Deterministic Day 9 dense-retrieval evaluation and BM25 comparison."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    RetrievalFilters,
)
from financial_report_qa.retrieval.dense_artifacts import write_text_atomic
from financial_report_qa.retrieval.dense_contracts import (
    DenseEncoderSpec,
    DenseRetrievalTrace,
    EncoderName,
    LatencySummary,
)
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
    score_at_10,
)

DenseFailure = Literal[
    "no_eligible_documents",
    "zero_gold_hits",
    "partial_gold_hits",
    "full_gold_hits",
]
DENSE_ENCODER_NAMES: tuple[EncoderName, EncoderName] = ("bge-m3", "multilingual-e5-small")
LOCKED_DATASET_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DenseQuestionEvaluation(_FrozenModel):
    question_id: str
    question: str
    intent: str
    filters: RetrievalFilters
    predicted_table_ids: tuple[str, ...]
    gold_table_ids: tuple[str, ...]
    missing_gold_table_ids: tuple[str, ...]
    metrics: RetrievalMetrics
    failure: DenseFailure
    trace: DenseRetrievalTrace


class DenseEvaluationReport(_FrozenModel):
    dataset_fingerprint: str
    question_count: int = Field(ge=0)
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    failure_counts: dict[DenseFailure, int]
    per_question: tuple[DenseQuestionEvaluation, ...]


class DenseEvaluationRun(_FrozenModel):
    encoder_name: EncoderName
    encoder_spec_sha256: str
    cold_report: DenseEvaluationReport
    warm_report: DenseEvaluationReport
    build_seconds: float = Field(ge=0)
    index_byte_size: int = Field(ge=0)
    cold_latency: LatencySummary
    warm_latency: LatencySummary


class MetricDelta(_FrozenModel):
    true_positive: int
    precision: float
    recall: float
    f2: float


class Day9SystemSummary(_FrozenModel):
    macro: RetrievalMetrics
    by_intent: dict[str, RetrievalMetrics]
    delta_vs_bm25: MetricDelta | None = None
    delta_by_intent: dict[str, MetricDelta] | None = None
    failure_counts: dict[DenseFailure, int] | None = None


class Day9ComparisonReport(_FrozenModel):
    dataset_fingerprint: str
    systems: dict[str, Day9SystemSummary]
    dense_runs: dict[EncoderName, DenseEvaluationRun]


class DenseRetrievalClient(Protocol):
    @property
    def encoder_spec(self) -> DenseEncoderSpec: ...

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int,
        question_id: str,
    ) -> DenseRetrievalTrace: ...


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


def _failure(
    trace: DenseRetrievalTrace, metrics: RetrievalMetrics, gold_count: int
) -> DenseFailure:
    if trace.empty_reason == "no_eligible_documents":
        return "no_eligible_documents"
    if metrics.true_positive == 0:
        return "zero_gold_hits"
    if metrics.true_positive < gold_count:
        return "partial_gold_hits"
    return "full_gold_hits"


def _evaluate(
    retriever: DenseRetrievalClient,
    questions: Sequence[GoldRetrievalQuestion],
    *,
    capture_latency: bool,
) -> tuple[DenseEvaluationReport, tuple[float, ...]]:
    results: list[DenseQuestionEvaluation] = []
    per_intent: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    failure_counts: dict[DenseFailure, int] = {
        "full_gold_hits": 0,
        "partial_gold_hits": 0,
        "zero_gold_hits": 0,
        "no_eligible_documents": 0,
    }
    latency: list[float] = []
    for question in sorted(questions, key=lambda item: item.question_id):
        start = time.perf_counter() if capture_latency else 0.0
        trace = retriever.retrieve(
            question.question,
            filters=question.filters,
            k=10,
            question_id=question.question_id,
        )
        if capture_latency:
            latency.append(time.perf_counter() - start)
        predicted = tuple(candidate.table_id for candidate in trace.results)
        metrics = score_at_10(predicted, question.gold_table_ids)
        failure = _failure(trace, metrics, len(question.gold_table_ids))
        failure_counts[failure] += 1
        missing = tuple(sorted(set(question.gold_table_ids).difference(predicted[:10])))
        result = DenseQuestionEvaluation(
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
        results.append(result)
        per_intent[question.intent].append(metrics)
    return (
        DenseEvaluationReport(
            dataset_fingerprint=questions[0].dataset_fingerprint if questions else "",
            question_count=len(questions),
            macro=_average(result.metrics for result in results),
            by_intent={intent: _average(per_intent[intent]) for intent in sorted(per_intent)},
            failure_counts=failure_counts,
            per_question=tuple(results),
        ),
        tuple(latency),
    )


def evaluate_dense_retrieval(
    retriever: DenseRetrievalClient,
    questions: Sequence[GoldRetrievalQuestion],
    *,
    k: int = 10,
) -> DenseEvaluationReport:
    """Evaluate a dense retriever at the fixed Day 9 cutoff of ten."""
    if k != 10:
        raise ValueError("Day 9 evaluation is fixed at 10")
    report, _ = _evaluate(retriever, questions, capture_latency=False)
    return report


def _latency_summary(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(sample_count=0, p50_seconds=0, p95_seconds=0)
    sorted_values = sorted(values)
    return LatencySummary(
        sample_count=len(sorted_values),
        p50_seconds=statistics.median(sorted_values),
        p95_seconds=sorted_values[math.ceil(0.95 * len(sorted_values)) - 1],
    )


def _assert_cache_state(report: DenseEvaluationReport, *, expected: bool) -> None:
    for result in report.per_question:
        if result.trace.eligible_count and result.trace.cache_hit is not expected:
            label = "warm" if expected else "cold"
            raise ValueError(f"{label} dense evaluation has an unexpected cache state")


def _equivalent_without_cache_state(
    cold: DenseEvaluationReport,
    warm: DenseEvaluationReport,
) -> bool:
    cold_projection = cold.model_dump(mode="json")
    warm_projection = warm.model_dump(mode="json")
    for report in (cold_projection, warm_projection):
        for question in report["per_question"]:
            question["trace"].pop("cache_hit")
    return cold_projection == warm_projection


def evaluate_cold_and_warm(
    service: DenseRetrievalClient,
    questions: Sequence[GoldRetrievalQuestion],
    *,
    build_seconds: float = 0,
    index_byte_size: int = 0,
) -> DenseEvaluationRun:
    """Measure two equivalent passes whose only permitted difference is cache state."""
    cold_report, cold_latency = _evaluate(service, questions, capture_latency=True)
    warm_report, warm_latency = _evaluate(service, questions, capture_latency=True)
    _assert_cache_state(cold_report, expected=False)
    _assert_cache_state(warm_report, expected=True)
    if not _equivalent_without_cache_state(cold_report, warm_report):
        raise ValueError("cold and warm dense evaluations are not equivalent")
    return DenseEvaluationRun(
        encoder_name=service.encoder_spec.name,
        encoder_spec_sha256=cold_report.per_question[0].trace.encoder_spec_sha256
        if cold_report.per_question
        else "0" * 64,
        cold_report=cold_report,
        warm_report=warm_report,
        build_seconds=build_seconds,
        index_byte_size=index_byte_size,
        cold_latency=_latency_summary(cold_latency),
        warm_latency=_latency_summary(warm_latency),
    )


def _delta(value: RetrievalMetrics, reference: RetrievalMetrics) -> MetricDelta:
    return MetricDelta(
        true_positive=value.true_positive - reference.true_positive,
        precision=value.precision - reference.precision,
        recall=value.recall - reference.recall,
        f2=value.f2 - reference.f2,
    )


def _validate_bm25_reference(report: RetrievalEvaluationReport) -> None:
    expected = (0.1499999999999999, 0.880952380952381, 0.4224545295973871)
    observed = (report.macro.precision, report.macro.recall, report.macro.f2)
    if report.dataset_fingerprint != LOCKED_DATASET_FINGERPRINT:
        raise ValueError("BM25 reference does not match the locked Day 8 fingerprint")
    if report.question_count != 70 or any(
        not math.isclose(value, target, abs_tol=5e-8, rel_tol=0.0)
        for value, target in zip(observed, expected, strict=True)
    ):
        raise ValueError("BM25 reference does not match the locked Day 8 evaluation")


def build_day9_comparison(
    bm25_report: RetrievalEvaluationReport,
    bge_run: DenseEvaluationRun,
    e5_run: DenseEvaluationRun,
) -> Day9ComparisonReport:
    """Combine validated BM25 observations with both dense baselines."""
    _validate_bm25_reference(bm25_report)
    runs = {bge_run.encoder_name: bge_run, e5_run.encoder_name: e5_run}
    if set(runs) != {"bge-m3", "multilingual-e5-small"}:
        raise ValueError("Day 9 comparison requires one run for each approved dense encoder")
    if any(
        run.cold_report.dataset_fingerprint != bm25_report.dataset_fingerprint
        for run in runs.values()
    ):
        raise ValueError("Dense and BM25 reports must have the same dataset fingerprint")
    systems: dict[str, Day9SystemSummary] = {
        "bm25-v3": Day9SystemSummary(macro=bm25_report.macro, by_intent=bm25_report.by_intent)
    }
    for name in DENSE_ENCODER_NAMES:
        run = runs[name]
        systems[name] = Day9SystemSummary(
            macro=run.cold_report.macro,
            by_intent=run.cold_report.by_intent,
            delta_vs_bm25=_delta(run.cold_report.macro, bm25_report.macro),
            delta_by_intent={
                intent: _delta(metrics, bm25_report.by_intent[intent])
                for intent, metrics in run.cold_report.by_intent.items()
                if intent in bm25_report.by_intent
            },
            failure_counts=run.cold_report.failure_counts,
        )
    return Day9ComparisonReport(
        dataset_fingerprint=bm25_report.dataset_fingerprint,
        systems=systems,
        dense_runs=runs,
    )


def deterministic_projection(report: Day9ComparisonReport) -> dict[str, object]:
    """Return replay-relevant evidence while excluding non-deterministic operations data."""
    projection = report.model_dump(mode="json")

    def scrub(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: scrub(item)
                for key, item in value.items()
                if key not in {"build_seconds", "cold_latency", "warm_latency", "cache_hit"}
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    scrubbed = scrub(projection)
    if not isinstance(scrubbed, dict):
        raise ValueError("day 9 deterministic projection must be an object")
    return scrubbed


def _render_markdown(report: Day9ComparisonReport) -> str:
    lines = [
        "# Day 9 Dense Retrieval Comparison",
        "",
        f"- Dataset fingerprint: `{report.dataset_fingerprint}`",
        "",
        "| System | Precision@10 | Recall@10 | F2@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("bm25-v3", *DENSE_ENCODER_NAMES):
        metrics = report.systems[name].macro
        lines.append(
            f"| {name} | {metrics.precision:.6f} | {metrics.recall:.6f} | {metrics.f2:.6f} |"
        )
    lines.extend(("", "## Dense operational evidence", ""))
    for name in DENSE_ENCODER_NAMES:
        run = report.dense_runs[name]
        lines.extend(
            (
                f"### {name}",
                "",
                f"- Encoder spec SHA-256: `{run.encoder_spec_sha256}`",
                f"- Build seconds: {run.build_seconds:.6f}",
                f"- Index byte size: {run.index_byte_size}",
                f"- Cold p50/p95 seconds: {run.cold_latency.p50_seconds:.6f}/"
                f"{run.cold_latency.p95_seconds:.6f}",
                f"- Warm p50/p95 seconds: {run.warm_latency.p50_seconds:.6f}/"
                f"{run.warm_latency.p95_seconds:.6f}",
                "",
            )
        )
    return "\n".join(lines) + "\n"


def write_day9_comparison(report: Day9ComparisonReport, output_dir: Path) -> tuple[Path, Path]:
    """Write stable Day 9 JSON and Markdown comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = report.dataset_fingerprint[:12]
    json_path = output_dir / f"retrieval-day9-dense-{prefix}.json"
    markdown_path = output_dir / f"retrieval-day9-dense-{prefix}.md"
    write_text_atomic(
        json_path,
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    write_text_atomic(markdown_path, _render_markdown(report))
    return json_path, markdown_path
