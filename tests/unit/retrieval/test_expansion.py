"""Behavioral tests for Day 12 deterministic graph reranking."""

from __future__ import annotations

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import (
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.expansion import (
    GraphExpansionService,
    seeds_from_fusion_trace,
    seeds_from_retrieval_trace,
)
from financial_report_qa.retrieval.expansion_contracts import ExpansionParams
from financial_report_qa.retrieval.fusion_contracts import (
    FusedCandidate,
    FusionTrace,
    FusionWeights,
)
from financial_report_qa.retrieval.graph import build_graph
from financial_report_qa.retrieval.graph_service import TableGraphService


def _id(token: str) -> str:
    return "tbl_" + token * 64


def _doc(
    token: str, *, doc_id: str = "report", company: str | None = "ACB", line: int = 1
) -> TableDocument:
    table_id = _id(token)
    return TableDocument(
        table_id=table_id,
        doc_id=doc_id,
        text=f"table {token}",
        metadata=TableMetadata(
            table_id=table_id,
            doc_id=doc_id,
            company_code=company,
            source_path="a.txt",
            line_start=line,
            line_end=line,
        ),
    )


class _Base:
    def __init__(self, documents: tuple[TableDocument, ...], order: tuple[str, ...]) -> None:
        self.documents = {document.table_id: document for document in documents}
        self.order = order

    def retrieve(
        self, query: str, *, filters: RetrievalFilters, k: int, question_id: str | None = None
    ) -> RetrievalTrace:
        results = tuple(
            RetrievalCandidate(
                table_id=table_id,
                rank=rank,
                score=10 - rank,
                metadata=self.documents[table_id].metadata,
                snippet=self.documents[table_id].text,
            )
            for rank, table_id in enumerate(self.order[:k], 1)
        )
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=("x",),
            eligible_count=len(results),
            filter_decisions=(),
            results=results,
        )


def _service(
    documents: tuple[TableDocument, ...],
    order: tuple[str, ...],
    *,
    alpha: float,
    expand_non_seeds: bool = False,
) -> GraphExpansionService:
    graph = build_graph(documents, dataset_fingerprint="f" * 64, release_lock_sha256="e" * 64)
    return GraphExpansionService(
        _Base(documents, order),
        TableGraphService(graph),
        ExpansionParams(
            relations=("same_document",),
            fan_out=2,
            seed_depth=50,
            alpha=alpha,
            expand_non_seeds=expand_non_seeds,
        ),
    )


def test_alpha_zero_reproduces_the_base_retriever_exactly() -> None:
    documents = (_doc("a", line=1), _doc("b", line=2), _doc("c", line=3))
    trace = _service(documents, (_id("c"), _id("a"), _id("b")), alpha=0).retrieve(
        "x", filters=RetrievalFilters(), k=3
    )
    assert tuple(candidate.table_id for candidate in trace.results) == (
        _id("c"),
        _id("a"),
        _id("b"),
    )


def test_expansion_filters_and_demotes_contradictions() -> None:
    documents = (
        _doc("a", company="ACB", line=1),
        _doc("b", company="ACB", line=2),
        _doc("c", company="VCB", line=3),
    )
    trace = _service(documents, (_id("a"), _id("c")), alpha=1, expand_non_seeds=True).retrieve(
        "ACB", filters=RetrievalFilters(company_codes=("ACB",)), k=3
    )
    assert _id("c") not in {candidate.table_id for candidate in trace.results}
    assert trace.dropped_out_of_filter >= 1


def test_expand_non_seeds_false_does_not_introduce_a_neighbor_and_support_sums() -> None:
    documents = (_doc("a", line=1), _doc("b", line=2), _doc("c", line=3))
    only_seeds = _service(
        documents, (_id("a"), _id("b")), alpha=1, expand_non_seeds=False
    ).retrieve("x", filters=RetrievalFilters(), k=3)
    assert {candidate.table_id for candidate in only_seeds.results} == {_id("a"), _id("b")}
    expanded = _service(documents, (_id("a"), _id("b")), alpha=1, expand_non_seeds=True).retrieve(
        "x", filters=RetrievalFilters(), k=3
    )
    candidate = next(item for item in expanded.results if item.table_id == _id("c"))
    assert candidate.features.supporting_seed_count == 2
    assert candidate.features.neighbor_rrf == 1 / 61 + 1 / 62


def test_results_tie_break_by_table_id() -> None:
    documents = (_doc("a"), _doc("b"))
    trace = _service(documents, (_id("b"), _id("a")), alpha=0).retrieve(
        "x", filters=RetrievalFilters(), k=2
    )
    # alpha=0 retains the seed ranks, so this is intentionally not a score tie.
    assert tuple(candidate.table_id for candidate in trace.results) == (_id("b"), _id("a"))


def test_seed_adapters_normalize_bm25_and_fusion_traces_identically() -> None:
    document = _doc("a")
    retrieval = RetrievalTrace(
        query="x",
        query_tokens=("x",),
        eligible_count=1,
        filter_decisions=(),
        results=(
            RetrievalCandidate(
                table_id=document.table_id,
                rank=1,
                score=1,
                metadata=document.metadata,
                snippet=document.text,
            ),
        ),
    )
    fusion = FusionTrace(
        query="x",
        weights=FusionWeights(bm25=1, dense=0),
        entities=parse_query_entities("x"),
        eligible_count=1,
        bm25_candidate_count=1,
        dense_candidate_count=0,
        results=(
            FusedCandidate(
                table_id=document.table_id,
                rank=1,
                fused_score=1,
                bm25_rank=1,
                bm25_score=1,
                contradiction_count=0,
                metadata=document.metadata,
                snippet=document.text,
            ),
        ),
    )
    assert seeds_from_retrieval_trace(retrieval) == seeds_from_fusion_trace(fusion)
