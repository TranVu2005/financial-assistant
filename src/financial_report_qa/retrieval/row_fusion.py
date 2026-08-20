"""Weighted-RRF fusion of BM25, dense, fuzzy, and alias-dictionary row
retrieval (plan.md §7).

All four branches search under the *same* candidate table set (filter-first,
inheriting the scoping from the table retrieval stage) and are combined by
weighted Reciprocal Rank Fusion. ``fuzzy``/``alias`` default to weight 0 in
`RowFusionWeights`, so a caller that never passes those services or weights
gets exactly the old BM25+dense behavior.

Unlike table-level fusion, there is no contradiction penalty here — every
row already belongs to a validated candidate table, so entity contradictions
are handled at the table retrieval layer.

Invariant: every result has ``fused_score > 0``.  A branch weighted at 0
contributes no candidates — not even its top hit — so a fused candidate
always carries positive rank contribution from a weighted branch that
actually matched it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from financial_report_qa.core.errors import FusionInputError
from financial_report_qa.retrieval.row_dense_service import RowDenseRetrievalService
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import (
    RowFusedCandidate,
    RowFusionTrace,
    RowFusionWeights,
)
from financial_report_qa.retrieval.row_lexical import (
    RowAliasRetrievalService,
    RowFuzzyRetrievalService,
)
from financial_report_qa.retrieval.row_service import RowRetrievalService


@dataclass(frozen=True)
class _BranchHit:
    rank: int
    score: float
    row_idx: int
    metadata: RowMetadata
    snippet: str


def _branch_hits(
    candidates: Sequence[object],
) -> dict[str, _BranchHit]:
    return {
        candidate.row_id: _BranchHit(  # type: ignore[attr-defined]
            rank,
            candidate.score,  # type: ignore[attr-defined]
            candidate.row_idx,  # type: ignore[attr-defined]
            candidate.metadata,  # type: ignore[attr-defined]
            candidate.snippet,  # type: ignore[attr-defined]
        )
        for rank, candidate in enumerate(candidates, start=1)
    }


class RowFusionService:
    """Combine BM25, dense, fuzzy, and alias row retrievers under one set of
    weights. ``fuzzy``/``alias`` are optional -- omit them (or leave their
    weight at 0) to get plain BM25+dense fusion."""

    def __init__(
        self,
        bm25: RowRetrievalService,
        dense: RowDenseRetrievalService | None,
        weights: RowFusionWeights,
        *,
        fuzzy: RowFuzzyRetrievalService | None = None,
        alias: RowAliasRetrievalService | None = None,
    ) -> None:
        self._bm25 = bm25
        self._dense = dense
        self._fuzzy = fuzzy
        self._alias = alias
        self._weights = weights

    def retrieve_rows(
        self,
        query: str,
        *,
        candidate_table_ids: Sequence[str],
        k: int = 10,
    ) -> RowFusionTrace:
        if k < 1:
            raise FusionInputError("k must be positive")

        canonical_table_ids = tuple(sorted(set(candidate_table_ids)))

        if not canonical_table_ids:
            return RowFusionTrace(
                query=query,
                weights=self._weights,
                candidate_table_ids=(),
                bm25_candidate_count=0,
                dense_candidate_count=0,
                fuzzy_candidate_count=0,
                alias_candidate_count=0,
                results=(),
                empty_reason="no_candidates",
            )

        depth = max(self._weights.depth, k)

        bm25_results = (
            self._bm25.retrieve_rows(query, candidate_table_ids=canonical_table_ids, k=depth)
            if self._weights.bm25 > 0 and self._bm25 is not None
            else ()
        )
        dense_results = (
            self._dense.retrieve_rows(query, candidate_table_ids=canonical_table_ids, k=depth)
            if self._weights.dense > 0 and self._dense is not None
            else ()
        )
        fuzzy_results = (
            self._fuzzy.retrieve_rows(query, candidate_table_ids=canonical_table_ids, k=depth)
            if self._weights.fuzzy > 0 and self._fuzzy is not None
            else ()
        )
        alias_results = (
            self._alias.retrieve_rows(query, candidate_table_ids=canonical_table_ids, k=depth)
            if self._weights.alias > 0 and self._alias is not None
            else ()
        )

        # A zero BM25 score means no query term matched the document at all;
        # such a "candidate" only exists because BM25 padded the result list,
        # and RRF must never reward that as if it were evidence.
        # A branch weighted at 0 must contribute no candidates at all. Dense,
        # fuzzy, and alias candidates already come pre-filtered to positive
        # scores by their own services.
        bm25_candidates = (
            tuple(
                candidate
                for candidate in bm25_results
                if candidate.score > 0 and candidate.matched_tokens
            )
            if self._weights.bm25 > 0 and self._bm25 is not None
            else ()
        )
        dense_candidates = (
            dense_results if self._weights.dense > 0 and self._dense is not None else ()
        )
        fuzzy_candidates = fuzzy_results
        alias_candidates = alias_results

        bm25_by_row = _branch_hits(bm25_candidates)
        dense_by_row = _branch_hits(dense_candidates)
        fuzzy_by_row = _branch_hits(fuzzy_candidates)
        alias_by_row = _branch_hits(alias_candidates)

        fused: list[RowFusedCandidate] = []
        all_row_ids = set(bm25_by_row) | set(dense_by_row) | set(fuzzy_by_row) | set(alias_by_row)
        for row_id in sorted(all_row_ids):
            bm25_hit = bm25_by_row.get(row_id)
            dense_hit = dense_by_row.get(row_id)
            fuzzy_hit = fuzzy_by_row.get(row_id)
            alias_hit = alias_by_row.get(row_id)
            source = bm25_hit or dense_hit or fuzzy_hit or alias_hit
            assert source is not None  # row_id came from the union of all four maps

            fused_score = 0.0
            if bm25_hit is not None:
                fused_score += self._weights.bm25 / (self._weights.rrf_k + bm25_hit.rank)
            if dense_hit is not None:
                fused_score += self._weights.dense / (self._weights.rrf_k + dense_hit.rank)
            if fuzzy_hit is not None:
                fused_score += self._weights.fuzzy / (self._weights.rrf_k + fuzzy_hit.rank)
            if alias_hit is not None:
                fused_score += self._weights.alias / (self._weights.rrf_k + alias_hit.rank)

            # Derive table_id from the row_id (format: "{table_id}|row_{row_idx}")
            table_id = row_id.rsplit("|", 1)[0]

            fused.append(
                RowFusedCandidate(
                    row_id=row_id,
                    table_id=table_id,
                    row_idx=source.row_idx,
                    rank=1,  # placeholder; overwritten after the final sort below
                    fused_score=fused_score,
                    bm25_rank=bm25_hit.rank if bm25_hit else None,
                    bm25_score=bm25_hit.score if bm25_hit else None,
                    dense_rank=dense_hit.rank if dense_hit else None,
                    dense_score=dense_hit.score if dense_hit else None,
                    fuzzy_rank=fuzzy_hit.rank if fuzzy_hit else None,
                    fuzzy_score=fuzzy_hit.score if fuzzy_hit else None,
                    alias_rank=alias_hit.rank if alias_hit else None,
                    alias_score=alias_hit.score if alias_hit else None,
                    metadata=source.metadata,
                    snippet=source.snippet,
                )
            )

        ordered = sorted(
            fused,
            key=lambda candidate: (-candidate.fused_score, candidate.row_id),
        )[:k]
        results = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered, start=1)
        )
        return RowFusionTrace(
            query=query,
            weights=self._weights,
            candidate_table_ids=canonical_table_ids,
            bm25_candidate_count=len(bm25_candidates),
            dense_candidate_count=len(dense_candidates),
            fuzzy_candidate_count=len(fuzzy_candidates),
            alias_candidate_count=len(alias_candidates),
            results=results,
        )
