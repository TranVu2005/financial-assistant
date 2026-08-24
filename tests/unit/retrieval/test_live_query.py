import numpy as np
import pytest

from financial_report_qa.core.errors import RerankInputError
from financial_report_qa.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalTrace,
    TableMetadata,
)
from financial_report_qa.retrieval.fusion_contracts import (
    FusedCandidate,
    FusionTrace,
    FusionWeights,
)
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec
from financial_report_qa.planning.entity_parser import parse_query_entities

_QUESTION = "Doanh thu thuần của VCB năm 2023 là bao nhiêu?"


def _table_id(suffix: str) -> str:
    return "tbl_" + suffix * 64


def _metadata(suffix: str) -> TableMetadata:
    return TableMetadata(
        table_id=_table_id(suffix),
        doc_id="doc_" + "0" * 64,
        company_code="VCB",
        periods=("2023",),
        source_path="VCB/2023/x/x_extracted.txt",
        line_start=1,
        line_end=2,
    )


class _FakeBm25Service:
    def __init__(self, suffixes: tuple[str, ...]) -> None:
        self._suffixes = suffixes
        self.last_k: int | None = None
        self.last_filters = None

    def retrieve(self, query, *, filters, k=10, question_id=None):  # type: ignore[no-untyped-def]
        self.last_k = k
        self.last_filters = filters
        return RetrievalTrace(
            query=query,
            query_tokens=(),
            eligible_count=len(self._suffixes),
            filter_decisions=(),
            results=tuple(
                RetrievalCandidate(
                    table_id=_table_id(suffix),
                    score=1.0 - index / 10,
                    rank=index + 1,
                    metadata=_metadata(suffix),
                    snippet=f"snippet-{suffix}",
                    matched_tokens=("doanh",),
                )
                for index, suffix in enumerate(self._suffixes[:k])
            ),
        )


class _FakeFusionService:
    def __init__(self, suffixes: tuple[str, ...]) -> None:
        self._suffixes = suffixes
        self.last_k: int | None = None

    def retrieve(self, query, *, filters, k=10, question_id=None):  # type: ignore[no-untyped-def]
        self.last_k = k
        return FusionTrace(
            query=query,
            weights=FusionWeights(bm25=1, dense=1),
            entities=parse_query_entities(query),
            eligible_count=len(self._suffixes),
            bm25_candidate_count=len(self._suffixes),
            dense_candidate_count=len(self._suffixes),
            results=tuple(
                FusedCandidate(
                    table_id=_table_id(suffix),
                    rank=index + 1,
                    fused_score=1.0 - index / 10,
                    contradiction_count=0,
                    metadata=_metadata(suffix),
                    snippet=f"snippet-{suffix}",
                )
                for index, suffix in enumerate(self._suffixes[:k])
            ),
        )


class _ReversingReranker:
    """Cho điểm ngược lại thứ tự đầu vào, để thấy rõ reranker có tác dụng."""

    def __init__(self) -> None:
        self.spec = RerankerSpec(
            name="qwen3-reranker-4b",
            model_id="Qwen/Qwen3-Reranker-4B",
            revision="a" * 40,
            batch_size=4,
        )

    def score(self, query, documents):  # type: ignore[no-untyped-def]
        return np.asarray(range(len(documents)), dtype=np.float32)


def test_bm25_only_path_is_unchanged_when_no_reranker_is_supplied() -> None:
    service = _FakeBm25Service(("a", "b", "c"))

    table_ids = retrieve_candidate_table_ids(_QUESTION, service, k=2)

    assert table_ids == (_table_id("a"), _table_id("b"))
    assert service.last_k == 2


def test_filters_are_derived_from_the_question_text() -> None:
    service = _FakeBm25Service(("a",))

    retrieve_candidate_table_ids(_QUESTION, service, k=1)

    assert service.last_filters is not None
    assert service.last_filters.company_codes == ("VCB",)


def test_reranker_reorders_the_fused_top_and_truncates_to_k() -> None:
    service = _FakeFusionService(("a", "b", "c"))

    table_ids = retrieve_candidate_table_ids(
        _QUESTION, service, k=2, reranker=_ReversingReranker(), rerank_depth=3
    )

    assert table_ids == (_table_id("c"), _table_id("b"))


def test_reranking_asks_the_retriever_for_the_full_rerank_depth_not_just_k() -> None:
    service = _FakeFusionService(tuple("0123456789"))

    retrieve_candidate_table_ids(
        _QUESTION, service, k=2, reranker=_ReversingReranker(), rerank_depth=8
    )

    assert service.last_k == 8


def test_rerank_depth_below_k_is_rejected() -> None:
    with pytest.raises(RerankInputError):
        retrieve_candidate_table_ids(
            _QUESTION, _FakeFusionService(("a",)), k=10,
            reranker=_ReversingReranker(), rerank_depth=5,
        )
