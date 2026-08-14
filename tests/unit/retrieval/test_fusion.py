"""Unit tests for weighted-RRF BM25/dense fusion with a contradiction penalty."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from financial_report_qa.core.errors import FusionInputError
from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_corpus import build_dense_corpus
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec
from financial_report_qa.retrieval.dense_index import build_dense_index
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.fusion_contracts import (
    PRE_REGISTERED_WEIGHT_GRID,
    FusionWeights,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.service import RetrievalService


def _table_id(character: str) -> str:
    return f"tbl_{character * 64}"


def _document(
    character: str, *, text: str, company: str, periods: tuple[str, ...] = ()
) -> TableDocument:
    table_id = _table_id(character)
    return TableDocument(
        table_id=table_id,
        doc_id=f"doc_{character}",
        text=text,
        metadata=TableMetadata(
            table_id=table_id,
            doc_id=f"doc_{character}",
            company_code=company,
            periods=periods,
            source_path=f"{company}/{character}.txt",
            line_start=1,
            line_end=2,
        ),
    )


@dataclass
class _FixedVectorEncoder:
    """Assigns a hand-chosen unit vector per document TEXT and a fixed query vector."""

    spec: DenseEncoderSpec
    document_vectors: dict[str, tuple[float, float]]
    query_vector: tuple[float, float] = (1.0, 0.0)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self.document_vectors[text] for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.query_vector, dtype=np.float32)


def _services(
    tmp_path: Path,
    documents: tuple[TableDocument, ...],
    document_vectors: dict[str, tuple[float, float]],
) -> tuple[RetrievalService, DenseRetrievalService]:
    bm25_index = build_bm25_index(documents, dataset_fingerprint="f" * 64)
    bm25 = RetrievalService(bm25_index)

    corpus = build_dense_corpus(
        documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64
    )
    spec = approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    encoder = _FixedVectorEncoder(spec, document_vectors)
    dense_index = build_dense_index(corpus, encoder)
    dense = DenseRetrievalService(dense_index, encoder, QueryEmbeddingCache(tmp_path, spec))
    return bm25, dense


def test_rrf_score_matches_hand_computed_arithmetic(tmp_path: Path) -> None:
    # a: bm25 rank 1 (2 term hits), dense rank 2 (cos 0.9 with query (1,0)).
    # b: bm25 rank 2 (1 term hit), dense rank 3 (cos 0.5).
    # c: bm25 score 0 (no query terms at all), dense rank 1 (cos 1.0, exact match).
    documents = (
        _document("a", text="doanh thu doanh thu tăng trưởng", company="VCB"),
        _document("b", text="doanh thu ổn định", company="VCB"),
        _document("c", text="chi phí quản lý", company="VCB"),
    )
    vectors = {
        "doanh thu doanh thu tăng trưởng": (0.9, (1 - 0.9**2) ** 0.5),
        "doanh thu ổn định": (0.5, (1 - 0.5**2) ** 0.5),
        "chi phí quản lý": (1.0, 0.0),
    }
    bm25, dense = _services(tmp_path, documents, vectors)

    # Confirm the calibration assumptions the hand computation below depends on.
    bm25_trace = bm25.retrieve("doanh thu", filters=RetrievalFilters(), k=10)
    assert [c.table_id for c in bm25_trace.results if c.score > 0] == [
        _table_id("a"),
        _table_id("b"),
    ]
    dense_trace = dense.retrieve("doanh thu", filters=RetrievalFilters(), k=10)
    assert [c.table_id for c in dense_trace.results] == [
        _table_id("c"),
        _table_id("a"),
        _table_id("b"),
    ]

    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=1, rrf_k=60, depth=50))
    trace = service.retrieve("doanh thu", filters=RetrievalFilters(), k=10)

    by_table = {candidate.table_id: candidate for candidate in trace.results}
    assert by_table[_table_id("a")].fused_score == pytest.approx(1 / 61 + 1 / 62)
    assert by_table[_table_id("b")].fused_score == pytest.approx(1 / 62 + 1 / 63)
    assert by_table[_table_id("c")].fused_score == pytest.approx(1 / 61)
    assert by_table[_table_id("a")].bm25_rank == 1
    assert by_table[_table_id("a")].dense_rank == 2
    assert by_table[_table_id("b")].bm25_rank == 2
    assert by_table[_table_id("b")].dense_rank == 3
    assert by_table[_table_id("c")].bm25_rank is None
    assert by_table[_table_id("c")].dense_rank == 1
    assert [c.table_id for c in trace.results] == [_table_id("a"), _table_id("b"), _table_id("c")]
    assert [c.rank for c in trace.results] == [1, 2, 3]


def test_zero_score_bm25_candidates_never_contribute_rrf_mass(tmp_path: Path) -> None:
    """A document with no matching query term must not be treated as a BM25 hit,
    even though bm25s would otherwise rank it (score 0) among eligible documents."""
    documents = (
        _document("a", text="doanh thu", company="VCB"),
        _document("d", text="hoàn toàn không liên quan", company="VCB"),
    )
    vectors = {"doanh thu": (1.0, 0.0), "hoàn toàn không liên quan": (0.0, 1.0)}
    bm25, dense = _services(tmp_path, documents, vectors)
    # dense weight 0 isolates the BM25 branch: "d"'s only possible non-zero
    # contribution would come from a wrongly-counted BM25 hit.
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=0, rrf_k=60, depth=50))
    trace = service.retrieve("doanh thu", filters=RetrievalFilters(), k=10)
    assert trace.bm25_candidate_count == 1  # only "a" survives the zero-score filter
    assert trace.dense_candidate_count == 0  # dense weight 0 -> no dense candidates at all
    by_table = {candidate.table_id: candidate for candidate in trace.results}
    # "d" has no evidence from either branch (dense is weighted 0) and must be
    # absent, not merely ranked below "a" with a fused_score of 0.0.
    assert _table_id("d") not in by_table
    assert by_table[_table_id("a")].fused_score == pytest.approx(1 / 61)


def test_zero_weight_branch_contributes_no_candidates(tmp_path: Path) -> None:
    """A document only reachable through a zero-weighted branch must never appear."""
    documents = (
        _document("a", text="doanh thu", company="VCB"),
        _document("e", text="chi phí", company="VCB"),
    )
    vectors = {"doanh thu": (1.0, 0.0), "chi phí": (0.0, 1.0)}
    bm25, dense = _services(tmp_path, documents, vectors)

    # "e" only matches the dense query vector (1,0) weakly and has no BM25 term
    # overlap with "doanh thu" -> BM25-only branch must never surface it.
    bm25_only = FusionService(bm25, dense, FusionWeights(bm25=1, dense=0, rrf_k=60, depth=50))
    trace = bm25_only.retrieve("doanh thu", filters=RetrievalFilters(), k=10)
    assert {c.table_id for c in trace.results} == {_table_id("a")}

    # Symmetric case: dense-only branch must never surface a BM25-only hit.
    dense_only = FusionService(bm25, dense, FusionWeights(bm25=0, dense=1, rrf_k=60, depth=50))
    trace = dense_only.retrieve("doanh thu", filters=RetrievalFilters(), k=10)
    assert {c.table_id for c in trace.results} == {_table_id("a"), _table_id("e")}


def test_evidence_free_candidate_never_outranks_a_contradicted_real_hit(tmp_path: Path) -> None:
    """A candidate with zero evidence from either branch must not enter the fused
    pool at all, even when the only real hit is demoted by a contradiction."""
    documents = (
        _document("1", text="tổng tài sản", company="VCB"),
        _document("2", text="hoàn toàn không liên quan", company="ACB"),
    )
    vectors = {"tổng tài sản": (0.1, 0.99498743710662), "hoàn toàn không liên quan": (1.0, 0.0)}
    bm25, dense = _services(tmp_path, documents, vectors)
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=0, rrf_k=60, depth=50))

    trace = service.retrieve("Tổng tài sản của ACB là bao nhiêu?", filters=RetrievalFilters(), k=10)

    by_table = {candidate.table_id: candidate for candidate in trace.results}
    assert _table_id("2") not in by_table  # zero-score BM25, dense weighted 0
    assert by_table[_table_id("1")].contradiction_count == 1
    assert [c.table_id for c in trace.results] == [_table_id("1")]
    assert [c.rank for c in trace.results] == [1]


def test_every_fused_candidate_has_positive_score(tmp_path: Path) -> None:
    documents = (
        _document("a", text="doanh thu doanh thu tăng trưởng", company="VCB"),
        _document("b", text="doanh thu ổn định", company="VCB"),
        _document("c", text="chi phí quản lý", company="VCB"),
    )
    vectors = {
        "doanh thu doanh thu tăng trưởng": (0.9, (1 - 0.9**2) ** 0.5),
        "doanh thu ổn định": (0.5, (1 - 0.5**2) ** 0.5),
        "chi phí quản lý": (1.0, 0.0),
    }
    bm25, dense = _services(tmp_path, documents, vectors)
    for weights in PRE_REGISTERED_WEIGHT_GRID:
        trace = FusionService(bm25, dense, weights).retrieve(
            "doanh thu", filters=RetrievalFilters(), k=10
        )
        assert all(candidate.fused_score > 0 for candidate in trace.results)


def test_contradiction_demotes_a_candidate_below_a_lower_scoring_one(tmp_path: Path) -> None:
    """An unambiguous parsed company that disagrees with a candidate's metadata
    must push that candidate to a lower tier regardless of its raw fused score."""
    documents = (
        # Higher raw relevance (more term hits + closer dense vector) but wrong company.
        _document("1", text="tổng tài sản tổng tài sản của công ty", company="VCB"),
        # Lower raw relevance but matches the explicitly named company.
        _document("2", text="tổng tài sản", company="ACB"),
    )
    vectors = {
        "tổng tài sản tổng tài sản của công ty": (1.0, 0.0),
        "tổng tài sản": (0.9, (1 - 0.9**2) ** 0.5),
    }
    bm25, dense = _services(tmp_path, documents, vectors)
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=1, rrf_k=60, depth=50))

    trace = service.retrieve("Tra cứu tổng tài sản của ACB.", filters=RetrievalFilters(), k=10)

    by_table = {candidate.table_id: candidate for candidate in trace.results}
    assert by_table[_table_id("1")].contradiction_count == 1
    assert by_table[_table_id("1")].contradicted_fields == ("company",)
    assert by_table[_table_id("2")].contradiction_count == 0
    # y ranks first even though x has the higher fused_score.
    assert by_table[_table_id("1")].fused_score > by_table[_table_id("2")].fused_score
    assert [c.table_id for c in trace.results] == [_table_id("2"), _table_id("1")]
    assert [c.rank for c in trace.results] == [1, 2]


def test_ambiguous_entities_never_penalize_any_candidate(tmp_path: Path) -> None:
    """A genuinely ambiguous parse (explicit ticker contradicting a matched
    name, i.e. company_conflict) must never demote any candidate: an entity
    the parser itself distrusts cannot be used as evidence against a table."""
    documents = (
        _document("1", text="tổng tài sản", company="VCB"),
        _document("2", text="tổng tài sản", company="ACB"),
    )
    vectors = {"tổng tài sản": (1.0, 0.0)}
    bm25, dense = _services(tmp_path, documents, vectors)
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=1, rrf_k=60, depth=50))

    # "mã CK: KHG" contradicts the matched name "Ngân hàng TMCP Á Châu" (ACB) ->
    # company_conflict ambiguity -> company_codes empty -> no contradiction check.
    trace = service.retrieve(
        "Tổng tài sản của Ngân hàng TMCP Á Châu (mã CK: KHG) là bao nhiêu?",
        filters=RetrievalFilters(),
        k=10,
    )
    assert trace.entities.ambiguity == ("company_conflict", "period_missing")
    assert all(candidate.contradiction_count == 0 for candidate in trace.results)


def test_no_eligible_documents_short_circuits_without_calling_either_branch(tmp_path: Path) -> None:
    documents = (_document("a", text="doanh thu", company="VCB"),)
    vectors = {"doanh thu": (1.0, 0.0)}
    bm25, dense = _services(tmp_path, documents, vectors)
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=1))

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("ZZZ",)), k=10)
    assert trace.empty_reason == "no_eligible_documents"
    assert trace.results == ()
    assert trace.eligible_count == 0


def test_k_must_be_positive(tmp_path: Path) -> None:
    documents = (_document("a", text="doanh thu", company="VCB"),)
    vectors = {"doanh thu": (1.0, 0.0)}
    bm25, dense = _services(tmp_path, documents, vectors)
    service = FusionService(bm25, dense, FusionWeights(bm25=1, dense=1))
    with pytest.raises(FusionInputError):
        service.retrieve("doanh thu", filters=RetrievalFilters(), k=0)
