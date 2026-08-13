from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    GoldRetrievalQuestion,
    RetrievalFilters,
    TableMetadata,
)
from financial_report_qa.retrieval.dense_contracts import (
    DenseEncoderSpec,
    DenseRetrievalCandidate,
    DenseRetrievalTrace,
    EncoderName,
    LatencySummary,
)
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec, encoder_spec_sha256
from financial_report_qa.retrieval.dense_evaluation import (
    DenseEvaluationRun,
    build_day9_comparison,
    deterministic_projection,
    evaluate_cold_and_warm,
    evaluate_dense_retrieval,
    write_day9_comparison,
)
from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
)

_LOCKED_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"


def _table_id(value: str) -> str:
    return "tbl_" + value * 64


def _question() -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": "retq_" + "1" * 64,
            "question": "Doanh thu cua VCB la bao nhieu?",
            "intent": "lookup",
            "filters": {"company_codes": ["VCB"]},
            "gold_table_ids": [_table_id("a"), _table_id("b")],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": _table_id("a"),
                    "relative_path": "VCB/report.txt",
                    "line_start": 1,
                    "line_end": 1,
                    "verified": True,
                },
                {
                    "table_id": _table_id("b"),
                    "relative_path": "VCB/report.txt",
                    "line_start": 2,
                    "line_end": 2,
                    "verified": True,
                },
            ],
            "dataset_fingerprint": _LOCKED_FINGERPRINT,
        }
    )


@dataclass
class _DenseRetrieverFixture:
    encoder_spec: DenseEncoderSpec
    cache_hit: bool = False

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int,
        question_id: str,
    ) -> DenseRetrievalTrace:
        assert query == "Doanh thu cua VCB la bao nhieu?"
        assert filters.company_codes == ("VCB",)
        assert k == 10
        table_id = _table_id("a")
        return DenseRetrievalTrace(
            question_id=question_id,
            query=query,
            normalized_query_sha256="d" * 64,
            encoder_spec_sha256=encoder_spec_sha256(self.encoder_spec),
            cache_hit=self.cache_hit,
            eligible_count=2,
            filter_decisions=(
                FilterDecision(
                    field="company_codes",
                    requested_values=("VCB",),
                    matched_count_before_intersection=2,
                    eligible_count_after_intersection=2,
                ),
            ),
            results=(
                DenseRetrievalCandidate(
                    row_id=0,
                    table_id=table_id,
                    score=0.75,
                    rank=1,
                    metadata=TableMetadata(
                        table_id=table_id,
                        doc_id="doc-a",
                        company_code="VCB",
                        source_path="VCB/report.txt",
                        line_start=1,
                        line_end=1,
                    ),
                    snippet="Doanh thu",
                ),
            ),
        )


@dataclass
class _CachingDenseRetriever(_DenseRetrieverFixture):
    calls: int = 0

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int,
        question_id: str,
    ) -> DenseRetrievalTrace:
        self.cache_hit = self.calls > 0
        self.calls += 1
        return super().retrieve(query, filters=filters, k=k, question_id=question_id)


def _run(name: EncoderName, *, cold_p95: float, cache_hit: bool = False) -> DenseEvaluationRun:
    spec = approved_encoder_spec(name).model_copy(update={"dimension": 2})
    report = evaluate_dense_retrieval(_DenseRetrieverFixture(spec, cache_hit), (_question(),))
    latency = LatencySummary(sample_count=1, p50_seconds=cold_p95, p95_seconds=cold_p95)
    return DenseEvaluationRun(
        encoder_name=name,
        encoder_spec_sha256=encoder_spec_sha256(spec),
        cold_report=report,
        warm_report=report,
        build_seconds=cold_p95,
        index_byte_size=42,
        cold_latency=latency,
        warm_latency=latency,
    )


def _bm25_reference() -> RetrievalEvaluationReport:
    metrics = RetrievalMetrics(
        true_positive=44,
        precision=0.14666666666666667,
        recall=0.8833333333333333,
        f2=0.4312169312169312,
    )
    return RetrievalEvaluationReport(
        dataset_fingerprint=_LOCKED_FINGERPRINT,
        question_count=30,
        macro=metrics,
        by_intent={"lookup": metrics},
        per_question=(),
    )


def test_dense_evaluation_reuses_fixed_day8_metric_math() -> None:
    """Changing denominator or dense failure taxonomy must alter the reported evidence."""
    report = evaluate_dense_retrieval(
        _DenseRetrieverFixture(approved_encoder_spec("multilingual-e5-small")), (_question(),)
    )

    assert report.macro.precision == pytest.approx(0.1)
    assert report.macro.recall == pytest.approx(0.5)
    assert report.macro.f2 == pytest.approx(5 * 0.1 * 0.5 / 0.9)
    assert report.failure_counts == {
        "full_gold_hits": 0,
        "partial_gold_hits": 1,
        "zero_gold_hits": 0,
        "no_eligible_documents": 0,
    }


def test_cold_and_warm_predictions_match_but_cache_states_change() -> None:
    """A warm cache must not change ranking or falsely report a cold cache hit."""
    run = evaluate_cold_and_warm(
        _CachingDenseRetriever(approved_encoder_spec("multilingual-e5-small")), (_question(),)
    )

    assert run.cold_report.per_question[0].trace.cache_hit is False
    assert run.warm_report.per_question[0].trace.cache_hit is True
    assert run.cold_report.per_question[0].predicted_table_ids == (
        run.warm_report.per_question[0].predicted_table_ids
    )


def test_day9_comparison_reports_dense_minus_bm25_delta() -> None:
    """Comparison must retain the locked BM25 metric and expose dense's signed delta."""
    bge = _run("bge-m3", cold_p95=0.1)
    e5 = _run("multilingual-e5-small", cold_p95=0.2)

    comparison = build_day9_comparison(_bm25_reference(), bge, e5)

    assert comparison.systems["bm25-v3"].macro.recall == pytest.approx(0.8833333333333333)
    delta = comparison.systems["bge-m3"].delta_vs_bm25
    assert delta is not None
    assert delta.recall == pytest.approx(bge.cold_report.macro.recall - 0.8833333333333333)
    intent_deltas = comparison.systems["bge-m3"].delta_by_intent
    assert intent_deltas is not None
    assert intent_deltas["lookup"].recall == pytest.approx(
        bge.cold_report.by_intent["lookup"].recall - 0.8833333333333333
    )


def test_day9_comparison_rejects_an_unlocked_bm25_fingerprint() -> None:
    """A compatible but unapproved BM25 file must not become the Day 9 reference silently."""
    bge = _run("bge-m3", cold_p95=0.1)
    e5 = _run("multilingual-e5-small", cold_p95=0.2)
    unapproved_fingerprint = "a" * 64
    bm25 = _bm25_reference().model_copy(update={"dataset_fingerprint": unapproved_fingerprint})
    bge = bge.model_copy(
        update={
            "cold_report": bge.cold_report.model_copy(
                update={"dataset_fingerprint": unapproved_fingerprint}
            ),
            "warm_report": bge.warm_report.model_copy(
                update={"dataset_fingerprint": unapproved_fingerprint}
            ),
        }
    )
    e5 = e5.model_copy(
        update={
            "cold_report": e5.cold_report.model_copy(
                update={"dataset_fingerprint": unapproved_fingerprint}
            ),
            "warm_report": e5.warm_report.model_copy(
                update={"dataset_fingerprint": unapproved_fingerprint}
            ),
        }
    )

    with pytest.raises(ValueError, match="locked"):
        build_day9_comparison(bm25, bge, e5)


def test_deterministic_projection_excludes_wall_clock_and_cache_state(tmp_path: Path) -> None:
    """Replay identity must ignore operational timings while artifacts keep them for inspection."""
    first = build_day9_comparison(
        _bm25_reference(), _run("bge-m3", cold_p95=0.1), _run("multilingual-e5-small", cold_p95=0.2)
    )
    second = build_day9_comparison(
        _bm25_reference(),
        _run("bge-m3", cold_p95=9.9, cache_hit=True),
        _run("multilingual-e5-small", cold_p95=8.8, cache_hit=True),
    )

    json_path, markdown_path = write_day9_comparison(first, tmp_path)

    assert deterministic_projection(first) == deterministic_projection(second)
    assert json_path.name == "retrieval-day9-dense-422df141c935.json"
    assert markdown_path.name == "retrieval-day9-dense-422df141c935.md"
    assert json_path.read_bytes().endswith(b"\n")
    assert markdown_path.read_bytes().endswith(b"\n")
