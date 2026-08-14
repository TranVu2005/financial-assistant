"""Day 12 one-hop graph expansion with deterministic rank-only reranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import RetrievalFilters, RetrievalTrace
from financial_report_qa.retrieval.expansion_contracts import (
    ExpansionParams,
    ExpansionTrace,
    RerankedCandidate,
    RerankFeatures,
    SeedCandidate,
)
from financial_report_qa.retrieval.filtering import eligible_positions
from financial_report_qa.retrieval.fusion import _contradictions
from financial_report_qa.retrieval.fusion_contracts import FusionTrace
from financial_report_qa.retrieval.graph_contracts import GraphRelation
from financial_report_qa.retrieval.graph_service import TableGraphService


class SeedRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int,
        question_id: str | None = None,
    ) -> object: ...


def seeds_from_retrieval_trace(trace: RetrievalTrace) -> tuple[SeedCandidate, ...]:
    return tuple(
        SeedCandidate(
            table_id=item.table_id,
            rank=item.rank,
            metadata=item.metadata,
            snippet=item.snippet,
        )
        for item in trace.results
    )


def seeds_from_fusion_trace(trace: FusionTrace) -> tuple[SeedCandidate, ...]:
    return tuple(
        SeedCandidate(
            table_id=item.table_id,
            rank=item.rank,
            metadata=item.metadata,
            snippet=item.snippet,
        )
        for item in trace.results
    )


def _seeds(trace: object) -> tuple[SeedCandidate, ...]:
    if isinstance(trace, RetrievalTrace):
        return seeds_from_retrieval_trace(trace)
    if isinstance(trace, FusionTrace):
        return seeds_from_fusion_trace(trace)
    raise TypeError("seed retriever must return RetrievalTrace or FusionTrace")


class GraphExpansionService:
    """Use graph edges as auditable features, optionally admitting non-seed nodes."""

    def __init__(
        self,
        base: SeedRetriever,
        graph_service: TableGraphService,
        params: ExpansionParams,
    ) -> None:
        self._base = base
        self._graph = graph_service
        self._params = params
        self._documents = {document.table_id: document for document in graph_service.documents}

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> ExpansionTrace:
        if k < 1:
            raise ValueError("k must be positive")
        base_trace = self._base.retrieve(
            query, filters=filters, k=self._params.seed_depth, question_id=question_id
        )
        seeds = _seeds(base_trace)
        entities = parse_query_entities(query)
        if not seeds:
            return ExpansionTrace(
                question_id=question_id,
                query=query,
                params=self._params,
                entities=entities,
                eligible_count=0,
                seed_count=0,
                expanded_count=0,
                dropped_out_of_filter=0,
                dropped_contradicted=0,
                empty_reason="no_eligible_documents",
            )

        seed_ids = {seed.table_id for seed in seeds}
        candidates = set(seed_ids)
        supports: dict[str, list[tuple[int, GraphRelation]]] = defaultdict(list)
        for seed in seeds:
            for relation in self._params.relations:
                for edge in self._graph.neighbors(
                    seed.table_id, relation=relation, limit=self._params.fan_out
                ):
                    candidates.add(edge.dst_table_id)
                    supports[edge.dst_table_id].append((seed.rank, relation))

        all_ids = tuple(sorted(candidates))
        candidate_documents = tuple(self._documents[table_id] for table_id in all_ids)
        eligible_positions_, _ = eligible_positions(candidate_documents, filters)
        eligible_ids = {candidate_documents[position].table_id for position in eligible_positions_}
        dropped_out_of_filter = len(candidates - eligible_ids)
        # Contradictions intentionally form a ranking tier, as in Day 10; they are not discarded.
        eligible = [table_id for table_id in all_ids if table_id in eligible_ids]
        by_id = {seed.table_id: seed for seed in seeds}
        reranked: list[RerankedCandidate] = []
        for table_id in eligible:
            seed_candidate = by_id.get(table_id)
            seed_rrf = (
                0.0 if seed_candidate is None else 1 / (self._params.rrf_k + seed_candidate.rank)
            )
            support = supports.get(table_id, [])
            neighbor_rrf = sum(1 / (self._params.rrf_k + rank) for rank, _ in support)
            relations = tuple(sorted({relation for _, relation in support}))
            metadata = self._documents[table_id].metadata
            contradicted_fields, contradiction_count = _contradictions(entities, metadata)
            reranked.append(
                RerankedCandidate(
                    table_id=table_id,
                    rank=1,
                    score=seed_rrf + self._params.alpha * neighbor_rrf,
                    source="seed" if seed_candidate is not None else "graph",
                    features=RerankFeatures(
                        seed_rrf=seed_rrf,
                        neighbor_rrf=neighbor_rrf,
                        supporting_seed_count=len({rank for rank, _ in support}),
                        relations=relations,
                    ),
                    contradiction_count=contradiction_count,
                    contradicted_fields=contradicted_fields,
                    metadata=metadata,
                    snippet=seed_candidate.snippet
                    if seed_candidate is not None
                    else self._documents[table_id].text[:500],
                )
            )
        if not self._params.expand_non_seeds:
            reranked = [item for item in reranked if item.table_id in seed_ids]
        ordered = sorted(
            reranked, key=lambda item: (item.contradiction_count, -item.score, item.table_id)
        )[:k]
        results = tuple(
            item.model_copy(update={"rank": rank}) for rank, item in enumerate(ordered, 1)
        )
        return ExpansionTrace(
            question_id=question_id,
            query=query,
            params=self._params,
            entities=entities,
            eligible_count=len(eligible_ids),
            seed_count=len(seeds),
            expanded_count=len(candidates),
            dropped_out_of_filter=dropped_out_of_filter,
            dropped_contradicted=sum(item.contradiction_count > 0 for item in reranked),
            results=results,
            empty_reason="no_eligible_documents" if not eligible_ids else None,
        )
