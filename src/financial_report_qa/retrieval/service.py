"""Filter-first retrieval that never ranks ineligible tables."""

from __future__ import annotations

import math
from typing import Literal, cast

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    MetricExpansion,
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalTrace,
)
from financial_report_qa.retrieval.index import BM25Index, tokenize_text
from financial_report_qa.retrieval.metric_aliases import (
    build_metric_alias_lexicon,
    expand_metric_query,
)


class RetrievalService:
    def __init__(self, index: BM25Index) -> None:
        self._index = index
        self._metric_alias_lexicon = build_metric_alias_lexicon(index.documents)

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
        base_tokens = tokenize_text(query)
        expanded_tokens, metric_expansions = expand_metric_query(
            base_tokens, self._metric_alias_lexicon
        )
        index_tokens = tuple(
            token for token in expanded_tokens if token in self._index.retriever.vocab_dict
        )
        effective_expansions = tuple(
            MetricExpansion(
                alias_tokens=expansion.alias_tokens,
                canonical_metric=expansion.canonical_metric,
                added_tokens=tuple(
                    token
                    for token in expansion.added_tokens
                    if token in self._index.retriever.vocab_dict
                ),
            )
            for expansion in metric_expansions
            if any(
                token in self._index.retriever.vocab_dict
                for token in expansion.added_tokens
            )
        )
        if not eligible:
            return RetrievalTrace(
                question_id=question_id,
                query=query,
                query_tokens=index_tokens,
                eligible_count=0,
                filter_decisions=decisions,
                results=(),
                metric_expansions=effective_expansions,
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
                metric_expansions=effective_expansions,
                empty_reason="no_index_tokens",
            )
        scores = self._index.retriever.get_scores(list(index_tokens))
        if any(not math.isfinite(float(scores[position])) for position in eligible):
            raise ValueError("BM25 produced a non-finite score")
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
            document = self._index.documents[position]
            matched_tokens = tuple(
                sorted(set(index_tokens).intersection(tokenize_text(document.text)))
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
            metric_expansions=effective_expansions,
        )

    def _eligible_positions(
        self, filters: RetrievalFilters
    ) -> tuple[tuple[int, ...], tuple[FilterDecision, ...]]:
        eligible = set(range(len(self._index.documents)))
        decisions: list[FilterDecision] = []
        fields = (
            ("company_codes", filters.company_codes, "company_code"),
            ("periods", filters.periods, "periods"),
            ("statement_types", filters.statement_types, "statement_type"),
        )
        for field, requested_values, metadata_field in fields:
            if not requested_values:
                continue
            if field == "periods":
                matched = {
                    position
                    for position, document in enumerate(self._index.documents)
                    if set(document.metadata.periods).intersection(requested_values)
                }
            else:
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
