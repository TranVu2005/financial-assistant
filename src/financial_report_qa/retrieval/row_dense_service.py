"""Filter-first exact dense row retrieval scoped to candidate tables."""

from __future__ import annotations

import math
from collections.abc import Sequence

import faiss
import numpy as np
from pydantic import Field, field_validator

from financial_report_qa.core.errors import DenseArtifactError, DenseInputError
from financial_report_qa.retrieval.contracts import TableId, _FrozenModel
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_encoder import DenseEncoder, encoder_spec_sha256
from financial_report_qa.retrieval.row_dense_index import RowDenseIndex
from financial_report_qa.retrieval.row_documents import RowMetadata


class RowDenseRetrievalCandidate(_FrozenModel):
    """One exact cosine-scored row candidate."""

    row_id: str
    table_id: TableId
    row_idx: int
    score: float
    rank: int = Field(ge=1)
    metadata: RowMetadata
    snippet: str

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RowDenseRetrievalService:
    """Search exactly and exclusively within rows eligible for candidate tables."""

    def __init__(
        self,
        index: RowDenseIndex,
        encoder: DenseEncoder,
        cache: QueryEmbeddingCache,
    ) -> None:
        spec_hash = encoder_spec_sha256(encoder.spec)
        if spec_hash != index.manifest.encoder_spec_sha256:
            raise DenseArtifactError("Row dense index and encoder specifications do not match")
        if spec_hash != cache.encoder_spec_sha256:
            raise DenseArtifactError("Dense cache and encoder specifications do not match")
        self._index = index
        self._encoder = encoder
        self._cache = cache

    def retrieve_rows(
        self,
        query: str,
        *,
        candidate_table_ids: Sequence[str],
        k: int = 10,
    ) -> tuple[RowDenseRetrievalCandidate, ...]:
        if k < 1:
            raise DenseInputError("k must be positive")
        if not candidate_table_ids:
            return ()

        # Map candidate tables to 0-indexed row positions in the corpus
        candidate_table_set = frozenset(candidate_table_ids)
        eligible = tuple(
            i
            for i, doc in enumerate(self._index.corpus.documents)
            if doc.table_id in candidate_table_set
        )
        if not eligible:
            return ()

        cached = self._cache.get_or_encode(query, self._encoder)
        positions = self._search_eligible(cached.vector, eligible)
        ranked = sorted(
            positions,
            key=lambda item: (-item[1], self._index.corpus.documents[item[0]].row_id),
        )[:k]
        results = tuple(
            RowDenseRetrievalCandidate(
                row_id=document.row_id,
                table_id=document.table_id,
                row_idx=document.row_idx,
                score=score,
                rank=rank,
                metadata=document.metadata,
                snippet=document.text[:500],
            )
            for rank, (position, score) in enumerate(ranked, start=1)
            for document in (self._index.corpus.documents[position],)
        )
        return results

    def _search_eligible(
        self,
        vector: np.ndarray,
        eligible: tuple[int, ...],
    ) -> tuple[tuple[int, float], ...]:
        selector = faiss.IDSelectorBatch(np.asarray(eligible, dtype=np.int64))
        parameters = faiss.SearchParameters()
        parameters.sel = selector
        scores, row_ids = self._index.faiss_index.search(
            np.ascontiguousarray(vector.reshape(1, -1)),
            len(eligible),
            params=parameters,
        )
        pairs = tuple(zip(row_ids[0].tolist(), scores[0].tolist(), strict=True))
        returned = tuple(row_id for row_id, _ in pairs)
        if (
            len(pairs) != len(eligible)
            or set(returned) != set(eligible)
            or len(set(returned)) != len(returned)
            or any(row_id < 0 or not math.isfinite(score) for row_id, score in pairs)
        ):
            raise DenseArtifactError(
                "Row FAISS search did not return exactly the eligible documents"
            )
        return tuple((row_id, float(score)) for row_id, score in pairs)
