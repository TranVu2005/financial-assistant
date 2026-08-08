"""Filter-first retrieval that never ranks ineligible tables."""

from __future__ import annotations

import math
from typing import Literal, cast

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
)
from financial_report_qa.retrieval.index import BM25Index, tokenize_query


class RetrievalService:
    def __init__(self, index: BM25Index) -> None:
        self._index = index

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> RetrievalTrace:
        if k < 1:
            raise ValueError("k must be positive")
        eligible, decisions = self._eligible_positions(filters)
        query_tokens = tuple(tokenize_query(query))
        index_tokens = tuple(
            token for token in query_tokens if token in self._index.retriever.vocab_dict
        )
        if not eligible:
            return RetrievalTrace(
                question_id=question_id,
                query=query,
                query_tokens=index_tokens,
                eligible_count=0,
                filter_decisions=decisions,
                results=(),
                empty_reason="no_eligible_documents",
            )
        if not index_tokens:
            return RetrievalTrace(
                question_id=question_id,
                query=query,
                query_tokens=index_tokens,
                eligible_count=len(eligible),
                filter_decisions=decisions,
                results=(),
                empty_reason="no_index_tokens",
            )
        scores = self._index.retriever.get_scores(list(index_tokens))
        ranked_positions = sorted(
            eligible,
            key=lambda position: (
                -float(scores[position]),
                self._index.documents[position].table_id,
            ),
        )[:k]
        candidates: list[RetrievalCandidate] = []
        for rank, position in enumerate(ranked_positions, start=1):
            score = float(scores[position])
            if not math.isfinite(score):
                raise ValueError("BM25 produced a non-finite score")
            document = self._index.documents[position]
            matched_tokens = tuple(
                sorted(set(index_tokens).intersection(tokenize_query(document.text)))
            )
            candidates.append(
                RetrievalCandidate(
                    table_id=document.table_id,
                    score=score,
                    rank=rank,
                    metadata=document.metadata,
                    snippet=document.text[:500],
                    matched_tokens=matched_tokens,
                )
            )
        return RetrievalTrace(
            question_id=question_id,
            query=query,
            query_tokens=index_tokens,
            eligible_count=len(eligible),
            filter_decisions=decisions,
            results=tuple(candidates),
        )

    def _eligible_positions(
        self, filters: RetrievalFilters
    ) -> tuple[tuple[int, ...], tuple[FilterDecision, ...]]:
        eligible = set(range(len(self._index.documents)))
        decisions: list[FilterDecision] = []
        fields = (
            ("company_codes", filters.company_codes, "company_code"),
            ("periods", filters.periods, "period"),
            ("statement_types", filters.statement_types, "statement_type"),
        )
        for field, requested_values, metadata_field in fields:
            if not requested_values:
                continue
            matched = {
                position
                for position, document in enumerate(self._index.documents)
                if getattr(document.metadata, metadata_field) in requested_values
            }
            eligible.intersection_update(matched)
            decisions.append(
                FilterDecision(
                    field=cast(Literal["company_codes", "periods", "statement_types"], field),
                    requested_values=requested_values,
                    matched_count_before_intersection=len(matched),
                    eligible_count_after_intersection=len(eligible),
                )
            )
        return tuple(sorted(eligible)), tuple(decisions)
