"""Weighted-RRF fusion of BM25 and dense table retrieval, with a contradiction penalty.

Both branches search under the *same* metadata filters (filter-first, like
every other Day 8/9 retriever) and are combined by weighted Reciprocal Rank
Fusion. A candidate whose table metadata contradicts an unambiguous parsed
entity (a different company, a disjoint period) is demoted to a lower tier
rather than scored down by an arbitrary constant — see `_contradictions`.

Invariant: every result has `fused_score > 0`. A branch weighted at 0
contributes no candidates -- not even its top hit -- so a fused candidate
always carries positive rank contribution from a weighted branch that
actually matched it.
"""

from __future__ import annotations

from dataclasses import dataclass

from financial_report_qa.core.errors import FusionInputError
from financial_report_qa.planning.entity_contracts import (
    _COMPANY_AMBIGUITY,
    _PERIOD_AMBIGUITY,
    QueryEntities,
)
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import RetrievalFilters, TableMetadata
from financial_report_qa.retrieval.dense_service import DenseRetrievalService
from financial_report_qa.retrieval.fusion_contracts import (
    ContradictedField,
    FusedCandidate,
    FusionTrace,
    FusionWeights,
)
from financial_report_qa.retrieval.service import RetrievalService


@dataclass(frozen=True)
class _BranchHit:
    rank: int
    score: float
    metadata: TableMetadata
    snippet: str


def _contradictions(
    entities: QueryEntities, metadata: TableMetadata
) -> tuple[tuple[ContradictedField, ...], int]:
    """Count unambiguous parsed entities that contradict a candidate's metadata.

    Only fields free of an ambiguity flag are checked — an entity the parser
    itself was unsure about can never be used to demote a candidate.
    """
    flags = set(entities.ambiguity)
    fields: list[ContradictedField] = []
    if not (flags & _COMPANY_AMBIGUITY) and entities.company_codes:
        company_code = metadata.company_code
        if company_code is not None and company_code not in entities.company_codes:
            fields.append("company")
    if not (flags & _PERIOD_AMBIGUITY) and entities.periods:
        if metadata.periods and not set(entities.periods).intersection(metadata.periods):
            fields.append("period")
    return tuple(fields), len(fields)


class FusionService:
    """Combine a BM25 and a dense retriever under one set of pre-registered weights."""

    def __init__(
        self,
        bm25: RetrievalService,
        dense: DenseRetrievalService,
        weights: FusionWeights,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._weights = weights

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> FusionTrace:
        if k < 1:
            raise FusionInputError("k must be positive")
        depth = max(self._weights.depth, k)
        bm25_trace = self._bm25.retrieve(query, filters=filters, k=depth, question_id=question_id)
        dense_trace = self._dense.retrieve(query, filters=filters, k=depth, question_id=question_id)
        entities = parse_query_entities(query)
        eligible_count = max(bm25_trace.eligible_count, dense_trace.eligible_count)

        if eligible_count == 0:
            return FusionTrace(
                question_id=question_id,
                query=query,
                weights=self._weights,
                entities=entities,
                eligible_count=0,
                bm25_candidate_count=0,
                dense_candidate_count=0,
                results=(),
                empty_reason="no_eligible_documents",
            )

        # A zero BM25 score means no query term matched the document at all;
        # such a "candidate" only exists because BM25 padded the result list
        # by table_id, and RRF must never reward that as if it were evidence.
        # A branch weighted at 0 must contribute no candidates at all -- not
        # even its top hit -- otherwise a document with zero real evidence
        # from either branch still enters the fused pool at fused_score=0.0
        # and can outrank a genuine, if contradicted, hit from the other
        # branch (the contradiction tier sorts above raw score).
        bm25_candidates = (
            tuple(
                candidate
                for candidate in bm25_trace.results
                if candidate.score > 0 and candidate.matched_tokens
            )
            if self._weights.bm25 > 0
            else ()
        )
        dense_candidates = dense_trace.results if self._weights.dense > 0 else ()

        bm25_by_table = {
            candidate.table_id: _BranchHit(
                rank, candidate.score, candidate.metadata, candidate.snippet
            )
            for rank, candidate in enumerate(bm25_candidates, start=1)
        }
        dense_by_table = {
            candidate.table_id: _BranchHit(
                rank, candidate.score, candidate.metadata, candidate.snippet
            )
            for rank, candidate in enumerate(dense_candidates, start=1)
        }

        fused: list[FusedCandidate] = []
        for table_id in sorted(set(bm25_by_table) | set(dense_by_table)):
            bm25_hit = bm25_by_table.get(table_id)
            dense_hit = dense_by_table.get(table_id)
            source = bm25_hit or dense_hit
            assert source is not None  # table_id came from the union of both maps
            fused_score = 0.0
            if bm25_hit is not None:
                fused_score += self._weights.bm25 / (self._weights.rrf_k + bm25_hit.rank)
            if dense_hit is not None:
                fused_score += self._weights.dense / (self._weights.rrf_k + dense_hit.rank)
            contradicted_fields, contradiction_count = _contradictions(entities, source.metadata)
            fused.append(
                FusedCandidate(
                    table_id=table_id,
                    rank=1,  # placeholder; overwritten after the final sort below
                    fused_score=fused_score,
                    bm25_rank=bm25_hit.rank if bm25_hit else None,
                    bm25_score=bm25_hit.score if bm25_hit else None,
                    dense_rank=dense_hit.rank if dense_hit else None,
                    dense_score=dense_hit.score if dense_hit else None,
                    contradiction_count=contradiction_count,
                    contradicted_fields=contradicted_fields,
                    metadata=source.metadata,
                    snippet=source.snippet,
                )
            )

        ordered = sorted(
            fused,
            key=lambda candidate: (
                candidate.contradiction_count,
                -candidate.fused_score,
                candidate.table_id,
            ),
        )[:k]
        results = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered, start=1)
        )
        return FusionTrace(
            question_id=question_id,
            query=query,
            weights=self._weights,
            entities=entities,
            eligible_count=eligible_count,
            bm25_candidate_count=len(bm25_candidates),
            dense_candidate_count=len(dense_candidates),
            results=results,
        )
