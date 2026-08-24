"""Retrieval for a raw, never-before-seen question.

Full target pipeline (§5.1 of the 2026-08-23 target architecture):

    question -> entities -> metadata filters -> candidate tables
             -> BM25 + dense -> weighted RRF -> top-N
             -> cross-encoder rerank -> top-k

The metadata-filter step is not implemented here: every retriever already
filters first (`filtering.py::eligible_positions`, shared by BM25, dense and
fusion), so the only thing this module adds is deriving those filters from
raw question text -- `to_retrieval_filters` drops any field the entity parser
itself flagged as ambiguous, so an uncertain parse widens the candidate set
rather than silently emptying it.

`service` is deliberately typed as a Protocol, not as `RetrievalService`:
`FusionService` (BM25 + dense under one RRF) satisfies the same shape, so
switching the live path from BM25-only to fused retrieval is a wiring change
at the call site, not a change here.
"""

from __future__ import annotations

from typing import Protocol

from financial_report_qa.core.errors import RerankInputError
from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import RetrievalFilters, TableId
from financial_report_qa.retrieval.rerank_contracts import DEFAULT_RERANK_DEPTH
from financial_report_qa.retrieval.reranker import Reranker, rerank_candidates


class _RankedResult(Protocol):
    table_id: str


class _RetrievalTraceLike(Protocol):
    results: tuple[_RankedResult, ...]


class TableRetriever(Protocol):
    """Anything that ranks tables under metadata filters: BM25 or fusion."""

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> _RetrievalTraceLike: ...


def retrieve_candidate_table_ids(
    question: str,
    service: TableRetriever,
    *,
    k: int = 10,
    reranker: Reranker | None = None,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
) -> tuple[TableId, ...]:
    """Rank candidate tables for one raw question, in retrieval-rank order.

    With a `reranker`, the retriever is asked for `rerank_depth` candidates
    (not `k`) so the cross-encoder has a real pool to reorder; the reranked
    list is then cut to `k`.
    """
    if reranker is not None and rerank_depth < k:
        raise RerankInputError("rerank_depth must be at least k")

    entities = parse_query_entities(question)
    filters = to_retrieval_filters(entities)
    depth = rerank_depth if reranker is not None else k
    trace = service.retrieve(question, filters=filters, k=depth)

    if reranker is None:
        return tuple(candidate.table_id for candidate in trace.results)

    rerank_trace = rerank_candidates(
        question,
        trace.results,  # type: ignore[arg-type]
        reranker,
        k=k,
        depth=rerank_depth,
    )
    return tuple(candidate.table_id for candidate in rerank_trace.results)
