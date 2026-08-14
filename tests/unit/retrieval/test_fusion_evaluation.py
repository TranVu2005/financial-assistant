"""Unit tests for the Day 10 fusion-grid evaluation and decision rule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from financial_report_qa.retrieval.contracts import (
    GoldRetrievalQuestion,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.dense_index import build_dense_index
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReport, RetrievalMetrics
from financial_report_qa.retrieval.fusion_contracts import PRE_REGISTERED_WEIGHT_GRID, FusionWeights
from financial_report_qa.retrieval.fusion_evaluation import (
    deterministic_projection,
    evaluate_fusion_grid,
    write_day10_fusion,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.reference import (
    CURRENT_BM25_REFERENCE,
    load_bm25_reference_report,
)
from financial_report_qa.retrieval.service import RetrievalService


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def _document(character: str, *, text: str, company: str) -> TableDocument:
    table_id = _table_id(character)
    return TableDocument(
        table_id=table_id,
        doc_id=f"doc_{character}",
        text=text,
        metadata=TableMetadata(
            table_id=table_id,
            doc_id=f"doc_{character}",
            company_code=company,
            source_path=f"{company}/{character}.txt",
            line_start=1,
            line_end=2,
        ),
    )


@dataclass
class _FixedVectorEncoder:
    spec: DenseEncoderSpec
    document_vectors: dict[str, tuple[float, float]]
    query_vector: tuple[float, float] = (1.0, 0.0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.document_vectors[text] for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.query_vector, dtype=np.float32)


_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
_LOCKED_METRICS = CURRENT_BM25_REFERENCE.macro


def _bm25_reference_report() -> RetrievalEvaluationReport:
    path = (
        Path(__file__).parents[3]
        / "artifacts/evaluations/day13/bm25/retrieval-day8-422df141c935.json"
    )
    return load_bm25_reference_report(path).report


def _gold_question(gold_table_ids: tuple[str, ...]) -> GoldRetrievalQuestion:
    return GoldRetrievalQuestion.model_validate(
        {
            "question_id": "retq_" + "1" * 64,
            "question": "doanh thu",
            "intent": "lookup",
            "filters": {},
            "gold_table_ids": list(gold_table_ids),
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": table_id,
                    "relative_path": "VCB/a.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                }
                for table_id in gold_table_ids
            ],
            "dataset_fingerprint": _FINGERPRINT,
        }
    )


def _services(tmp_path: Path) -> tuple[RetrievalService, DenseRetrievalService]:
    """One document: `score_at_10`'s fixed denominator of 10 caps precision at
    0.1 here, so no grid point can reach the locked BM25 v3 reference."""
    documents = (_document("a", text="doanh thu", company="VCB"),)
    bm25 = RetrievalService(build_bm25_index(documents, dataset_fingerprint=_FINGERPRINT))
    corpus = build_dense_corpus(
        documents, dataset_fingerprint=_FINGERPRINT, release_lock_sha256="e" * 64
    )
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _FixedVectorEncoder(spec, {"doanh thu": (1.0, 0.0)})
    dense_index = build_dense_index(corpus, encoder)
    dense = DenseRetrievalService(dense_index, encoder, QueryEmbeddingCache(tmp_path, spec))
    return bm25, dense


def _high_recall_services(tmp_path: Path) -> tuple[RetrievalService, DenseRetrievalService]:
    """Ten identical, all-gold documents: every top-10 prediction is a perfect
    precision=recall=1.0 hit regardless of internal ranking order, so every
    grid point trivially beats the locked BM25 v3 reference."""
    documents = tuple(_document(str(index), text="doanh thu", company="VCB") for index in range(10))
    bm25 = RetrievalService(build_bm25_index(documents, dataset_fingerprint=_FINGERPRINT))
    corpus = build_dense_corpus(
        documents, dataset_fingerprint=_FINGERPRINT, release_lock_sha256="e" * 64
    )
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _FixedVectorEncoder(spec, {"doanh thu": (1.0, 0.0)})
    dense_index = build_dense_index(corpus, encoder)
    dense = DenseRetrievalService(dense_index, encoder, QueryEmbeddingCache(tmp_path, spec))
    return bm25, dense


def test_grid_covers_every_pre_registered_weight_point(tmp_path: Path) -> None:
    bm25, dense = _services(tmp_path)
    question = _gold_question((_table_id("a"),))
    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    assert len(report.grid) == len(PRE_REGISTERED_WEIGHT_GRID)
    assert {point.weights for point in report.grid} == set(PRE_REGISTERED_WEIGHT_GRID)


def test_tied_trivial_grid_keeps_bm25_default_even_though_every_point_reaches_the_reference(
    tmp_path: Path,
) -> None:
    """Every grid point ties the locked reference exactly here (all 10 documents
    are gold, so any top-10 ranking is a perfect hit). The tie-break prefers the
    lowest dense weight, which uniquely picks the dense=0 point -- that point is
    BM25 verbatim (a zero-weighted branch contributes no candidates), so it
    carries no real dense contribution and must not be labeled "fusion"."""
    bm25, dense = _high_recall_services(tmp_path)
    question = _gold_question(tuple(_table_id(str(index)) for index in range(10)))
    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    assert report.default_system == "bm25-v3"
    assert report.best_weights == FusionWeights(bm25=1, dense=0, rrf_k=60, depth=50)
    assert "no real dense contribution" in report.decision_reason
    for point in report.grid:
        assert point.macro.f2 == pytest.approx(1.0)


def test_grid_selects_fusion_only_when_the_winning_point_has_a_real_dense_contribution(
    tmp_path: Path,
) -> None:
    """BM25 finds nothing (no query-term overlap with any document), so every
    weight point's BM25 branch is empty regardless of its bm25 weight; only the
    dense branch can find the 10 gold documents. The winning point must have
    dense > 0, and only then may default_system become "fusion"."""
    documents = tuple(
        _document(str(index), text="chi phí quản lý điều hành", company="VCB")
        for index in range(10)
    )
    bm25 = RetrievalService(build_bm25_index(documents, dataset_fingerprint=_FINGERPRINT))
    corpus = build_dense_corpus(
        documents, dataset_fingerprint=_FINGERPRINT, release_lock_sha256="e" * 64
    )
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _FixedVectorEncoder(spec, {"chi phí quản lý điều hành": (1.0, 0.0)})
    dense_index = build_dense_index(corpus, encoder)
    dense = DenseRetrievalService(dense_index, encoder, QueryEmbeddingCache(tmp_path, spec))

    question = _gold_question(tuple(_table_id(str(index)) for index in range(10)))
    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())

    assert report.default_system == "fusion"
    assert report.best_weights.dense > 0
    zero_dense_point = next(p for p in report.grid if p.weights.dense == 0)
    assert zero_dense_point.macro.f2 == pytest.approx(0.0)  # no BM25 match, no dense contribution


def test_decision_rule_keeps_bm25_default_when_no_grid_point_reaches_the_reference(
    tmp_path: Path,
) -> None:
    bm25, dense = _services(tmp_path)
    question = _gold_question((_table_id("a"),))
    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    assert report.default_system == "bm25-v3"
    assert "no grid point reached" in report.decision_reason
    assert all(point.macro.f2 < _LOCKED_METRICS.f2 for point in report.grid)


def test_evaluation_rejects_a_bm25_reference_that_does_not_match_the_locked_values(
    tmp_path: Path,
) -> None:
    bm25, dense = _services(tmp_path)
    question = _gold_question((_table_id("a"),))
    wrong = _bm25_reference_report().model_copy(
        update={"macro": RetrievalMetrics(true_positive=1, precision=0.1, recall=0.1, f2=0.1)}
    )
    with pytest.raises(ValueError, match="BM25 reference"):
        evaluate_fusion_grid(bm25, dense, (question,), wrong)


def test_grid_evaluation_is_deterministic_across_repeated_runs(tmp_path: Path) -> None:
    bm25, dense = _services(tmp_path)
    question = _gold_question((_table_id("a"),))
    first = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    second = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    assert deterministic_projection(first) == deterministic_projection(second)


def test_write_day10_fusion_round_trips_through_json(tmp_path: Path) -> None:
    bm25, dense = _services(tmp_path)
    question = _gold_question((_table_id("a"),))
    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())
    json_path, markdown_path = write_day10_fusion(report, tmp_path / "out")
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Full pre-registered grid" in markdown_path.read_text(encoding="utf-8")


def test_fusion_weights_reject_two_zero_weights() -> None:
    with pytest.raises(ValueError):
        FusionWeights(bm25=0, dense=0)


def test_contradiction_tier_activates_through_the_full_evaluation_harness_with_soft_filters(
    tmp_path: Path,
) -> None:
    """On the real Day 10 gold set every question supplies a hard company+period
    filter, which excludes any metadata-contradicting document via
    `eligible_positions` before `_contradictions` ever runs -- so
    `contradiction_count` is 0 for every candidate, on every question, at every
    grid point (see artifacts/evaluations/day10/retrieval-day10-fusion-*.json).
    That does not mean the mechanism is dead: with a softer filter -- no
    company_codes filter here -- a document whose metadata disagrees with an
    unambiguous parsed entity is still demoted, exercised end-to-end through
    `evaluate_fusion_grid` (not just `FusionService.retrieve` directly)."""
    documents = (
        # Higher raw relevance (more term hits + closer dense vector) but wrong company.
        _document("1", text="tổng tài sản tổng tài sản của công ty", company="VCB"),
        # Lower raw relevance but matches the explicitly named company.
        _document("2", text="tổng tài sản", company="ACB"),
    )
    bm25 = RetrievalService(build_bm25_index(documents, dataset_fingerprint=_FINGERPRINT))
    corpus = build_dense_corpus(
        documents, dataset_fingerprint=_FINGERPRINT, release_lock_sha256="e" * 64
    )
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _FixedVectorEncoder(
        spec,
        {
            "tổng tài sản tổng tài sản của công ty": (1.0, 0.0),
            "tổng tài sản": (0.9, (1 - 0.9**2) ** 0.5),
        },
    )
    dense_index = build_dense_index(corpus, encoder)
    dense = DenseRetrievalService(dense_index, encoder, QueryEmbeddingCache(tmp_path, spec))

    question = GoldRetrievalQuestion.model_validate(
        {
            "question_id": "retq_" + "2" * 64,
            "question": "Tra cứu tổng tài sản của ACB.",
            "intent": "lookup",
            "filters": {},  # soft: no company_codes/periods filter, unlike real gold data
            "gold_table_ids": [_table_id("2")],
            "reviewed_by": "reviewer",
            "reviewed_at": "2026-08-08T00:00:00+00:00",
            "gold_evidence": [
                {
                    "table_id": _table_id("2"),
                    "relative_path": "ACB/2.txt",
                    "line_start": 1,
                    "line_end": 2,
                    "verified": True,
                }
            ],
            "dataset_fingerprint": _FINGERPRINT,
        }
    )

    report = evaluate_fusion_grid(bm25, dense, (question,), _bm25_reference_report())

    assert report.grid  # sanity: the full pre-registered grid still ran
    for point in report.grid:
        by_table = {
            candidate.table_id: candidate for candidate in point.per_question[0].trace.results
        }
        assert by_table[_table_id("1")].contradiction_count == 1
        assert by_table[_table_id("1")].contradicted_fields == ("company",)
        assert by_table[_table_id("2")].contradiction_count == 0
