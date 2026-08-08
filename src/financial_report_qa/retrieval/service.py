"""Filter-first retrieval that never ranks ineligible tables."""

from __future__ import annotations

from typing import Literal, cast

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    FilterFieldDecision,
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
        eligible, field_decisions = self._eligible_documents(filters)
        decision = FilterDecision(
            filters=filters,
            indexed_count=len(self._index.documents),
            eligible_count=len(eligible),
            excluded_count=len(self._index.documents) - len(eligible),
            field_decisions=field_decisions,
        )
        tokens = tokenize_query(query)
        index_tokens = [token for token in tokens if token in self._index.retriever.vocab_dict]
        if not index_tokens or not eligible:
            return RetrievalTrace(
                question_id=question_id, query=query, filter_decision=decision, results=()
            )
        scores = self._index.retriever.get_scores(index_tokens)
        candidates = [
            RetrievalCandidate(
                table_id=document.table_id,
                score=float(scores[position]),
                rank=1,
                metadata=document.metadata,
                snippet=document.text[:500],
            )
            for position, document in enumerate(self._index.documents)
            if self._matches(document, filters)
        ]
        ordered = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.table_id))[
            :k
        ]
        ranked = tuple(
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(ordered, start=1)
        )
        return RetrievalTrace(
            question_id=question_id, query=query, filter_decision=decision, results=ranked
        )

    @staticmethod
    def _matches(document: object, filters: RetrievalFilters) -> bool:
        metadata = getattr(document, "metadata")
        return (
            (not filters.company_codes or metadata.company_code in filters.company_codes)
            and (not filters.periods or metadata.period in filters.periods)
            and (not filters.statement_types or metadata.statement_type in filters.statement_types)
        )

    def _eligible_documents(
        self, filters: RetrievalFilters
    ) -> tuple[tuple[object, ...], tuple[FilterFieldDecision, ...]]:
        eligible = set(range(len(self._index.documents)))
        decisions: list[FilterFieldDecision] = []
        fields = (
            ("company_codes", filters.company_codes, "company_code"),
            ("periods", filters.periods, "period"),
            ("statement_types", filters.statement_types, "statement_type"),
        )
        for field_name, requested_values, metadata_field in fields:
            if not requested_values:
                continue
            matched = {
                index
                for index, document in enumerate(self._index.documents)
                if getattr(document.metadata, metadata_field) in requested_values
            }
            eligible.intersection_update(matched)
            decisions.append(
                FilterFieldDecision(
                    field_name=cast(
                        Literal["company_codes", "periods", "statement_types"], field_name
                    ),
                    requested_values=requested_values,
                    matched_count_before_intersection=len(matched),
                    eligible_count_after_intersection=len(eligible),
                )
            )
        return (
            tuple(self._index.documents[index] for index in sorted(eligible)),
            tuple(decisions),
        )
