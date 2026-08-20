"""Filter-first row retrieval scoped to candidate tables."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import TableId, _FrozenModel
from financial_report_qa.retrieval.index import tokenize_text
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_index import RowBM25Index


class RowRetrievalCandidate(_FrozenModel):
    row_id: str
    table_id: TableId
    row_idx: int
    score: float
    rank: int = Field(ge=1)
    metadata: RowMetadata
    snippet: str
    matched_tokens: tuple[str, ...] = ()

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RowRetrievalService:
    def __init__(self, index: RowBM25Index) -> None:
        self._index = index

    def retrieve_rows(
        self,
        query: str,
        *,
        candidate_table_ids: Sequence[str],
        k: int = 10,
    ) -> tuple[RowRetrievalCandidate, ...]:
        if k < 1:
            raise ValueError("k must be positive")

        if not candidate_table_ids:
            return ()

        # Tokenize query
        base_tokens = tokenize_text(query)
        index_tokens = tuple(
            token for token in base_tokens if token in self._index.retriever.vocab_dict
        )
        if not index_tokens:
            return ()

        # Find eligible row positions belonging to candidate tables
        candidate_table_set = frozenset(candidate_table_ids)
        eligible_positions = [
            i
            for i, doc in enumerate(self._index.documents)
            if doc.table_id in candidate_table_set
        ]
        if not eligible_positions:
            return ()

        # Retrieve BM25 scores
        scores = self._index.retriever.get_scores(list(index_tokens))
        if any(not math.isfinite(float(scores[pos])) for pos in eligible_positions):
            raise ValueError("BM25 produced a non-finite score")

        # Rank eligible positions
        ranked_positions = sorted(
            [pos for pos in eligible_positions if float(scores[pos]) > 0.0],
            key=lambda pos: (
                -float(scores[pos]),
                self._index.documents[pos].row_id,
            ),
        )[:k]

        candidates: list[RowRetrievalCandidate] = []
        for rank, position in enumerate(ranked_positions, start=1):
            score = float(scores[position])
            document = self._index.documents[position]
            matched_tokens = tuple(
                sorted(set(index_tokens).intersection(tokenize_text(document.text)))
            )
            candidates.append(
                RowRetrievalCandidate(
                    row_id=document.row_id,
                    table_id=document.table_id,
                    row_idx=document.row_idx,
                    score=score,
                    rank=rank,
                    metadata=document.metadata,
                    snippet=document.text[:500],
                    matched_tokens=matched_tokens,
                )
            )

        return tuple(candidates)
