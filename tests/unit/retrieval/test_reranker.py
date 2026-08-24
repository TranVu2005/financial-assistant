import numpy as np
import pytest

from financial_report_qa.core.errors import RerankInputError, RerankModelError
from financial_report_qa.retrieval.contracts import TableMetadata
from financial_report_qa.retrieval.fusion_contracts import FusedCandidate
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec
from financial_report_qa.retrieval.reranker import approved_reranker_spec, rerank_candidates


class _FakeReranker:
    """Cho điểm theo bảng tra cứu table_id -> score, đếm số lần được gọi."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.spec = RerankerSpec(
            name="qwen3-reranker-4b",
            model_id="Qwen/Qwen3-Reranker-4B",
            revision="a" * 40,
            batch_size=4,
        )
        self._scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, documents):  # type: ignore[no-untyped-def]
        self.calls.append((query, tuple(documents)))
        return np.asarray([self._scores[doc] for doc in documents], dtype=np.float32)


def _table_id(suffix: str) -> str:
    return "tbl_" + suffix * 64


def _candidate(suffix: str, *, fused_rank: int, fused_score: float) -> FusedCandidate:
    table_id = _table_id(suffix)
    return FusedCandidate(
        table_id=table_id,
        rank=fused_rank,
        fused_score=fused_score,
        contradiction_count=0,
        metadata=TableMetadata(
            table_id=table_id,
            doc_id="doc_" + "0" * 64,
            source_path="X/2023/x/x_extracted.txt",
            line_start=1,
            line_end=2,
        ),
        snippet=f"snippet-{suffix}",
    )


def test_reranking_reorders_by_cross_encoder_score_not_fused_rank() -> None:
    candidates = (
        _candidate("a", fused_rank=1, fused_score=0.9),
        _candidate("b", fused_rank=2, fused_score=0.5),
    )
    reranker = _FakeReranker({"snippet-a": 0.1, "snippet-b": 0.8})

    trace = rerank_candidates("q", candidates, reranker, k=2)

    assert [item.table_id for item in trace.results] == [_table_id("b"), _table_id("a")]
    assert [item.rank for item in trace.results] == [1, 2]
    # Gốc RRF được giữ lại nguyên vẹn để giải trình quyết định đảo thứ tự.
    assert [item.fused_rank for item in trace.results] == [2, 1]


def test_reranking_only_sees_the_top_depth_candidates() -> None:
    candidates = tuple(
        _candidate(chr(ord("a") + i), fused_rank=i + 1, fused_score=1.0 - i / 10)
        for i in range(5)
    )
    reranker = _FakeReranker({f"snippet-{chr(ord('a') + i)}": float(i) for i in range(5)})

    trace = rerank_candidates("q", candidates, reranker, k=2, depth=3)

    assert trace.input_count == 3
    assert len(reranker.calls) == 1
    assert reranker.calls[0][1] == ("snippet-a", "snippet-b", "snippet-c")


def test_ties_break_on_table_id_so_the_order_is_deterministic() -> None:
    candidates = (
        _candidate("b", fused_rank=1, fused_score=0.9),
        _candidate("a", fused_rank=2, fused_score=0.5),
    )
    reranker = _FakeReranker({"snippet-b": 0.5, "snippet-a": 0.5})

    trace = rerank_candidates("q", candidates, reranker, k=2)

    assert [item.table_id for item in trace.results] == [_table_id("a"), _table_id("b")]


def test_empty_candidate_list_returns_an_explicit_empty_reason() -> None:
    reranker = _FakeReranker({})

    trace = rerank_candidates("q", (), reranker, k=10)

    assert trace.results == ()
    assert trace.empty_reason == "no_fused_candidates"
    assert reranker.calls == []


def test_k_must_be_positive() -> None:
    with pytest.raises(RerankInputError):
        rerank_candidates("q", (), _FakeReranker({}), k=0)


def test_a_non_finite_model_score_is_rejected_loudly() -> None:
    class _NaNReranker(_FakeReranker):
        def score(self, query: str, documents):  # type: ignore[no-untyped-def]
            return np.asarray([float("nan")] * len(documents), dtype=np.float32)

    with pytest.raises(RerankModelError):
        rerank_candidates("q", (_candidate("a", fused_rank=1, fused_score=0.9),),
                          _NaNReranker({}), k=1)


def test_score_count_mismatch_is_rejected() -> None:
    class _ShortReranker(_FakeReranker):
        def score(self, query: str, documents):  # type: ignore[no-untyped-def]
            return np.asarray([0.5], dtype=np.float32)

    candidates = (
        _candidate("a", fused_rank=1, fused_score=0.9),
        _candidate("b", fused_rank=2, fused_score=0.5),
    )
    with pytest.raises(RerankModelError):
        rerank_candidates("q", candidates, _ShortReranker({}), k=2)


def test_approved_spec_is_pinned() -> None:
    spec = approved_reranker_spec("qwen3-reranker-4b")
    assert spec.model_id == "Qwen/Qwen3-Reranker-4B"
    assert spec.dtype == "float32"
