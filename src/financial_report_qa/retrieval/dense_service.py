"""Filter-first exact dense table retrieval."""

from __future__ import annotations

import math

import faiss
import numpy as np

from financial_report_qa.core.errors import DenseArtifactError, DenseInputError
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import (
    DenseRetrievalCandidate,
    DenseRetrievalTrace,
)
from financial_report_qa.retrieval.dense_encoder import DenseEncoder, encoder_spec_sha256
from financial_report_qa.retrieval.dense_index import DenseIndex
from financial_report_qa.retrieval.filtering import eligible_positions


class DenseRetrievalService:
    """Search exactly and exclusively within documents eligible for supplied metadata filters."""

    def __init__(
        self,
        index: DenseIndex,
        encoder: DenseEncoder,
        cache: QueryEmbeddingCache,
    ) -> None:
        spec_hash = encoder_spec_sha256(encoder.spec)
        if spec_hash != index.manifest.encoder_spec_sha256:
            raise DenseArtifactError("Dense index and encoder specifications do not match")
        if spec_hash != cache.encoder_spec_sha256:
            raise DenseArtifactError("Dense cache and encoder specifications do not match")
        self._index = index
        self._encoder = encoder
        self._cache = cache

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters,
        k: int = 10,
        question_id: str | None = None,
    ) -> DenseRetrievalTrace:
        if k < 1:
            raise DenseInputError("k must be positive")
        eligible, decisions = eligible_positions(self._index.corpus.documents, filters)
        if not eligible:
            return DenseRetrievalTrace(
                question_id=question_id,
                query=query,
                encoder_spec_sha256=self._index.manifest.encoder_spec_sha256,
                eligible_count=0,
                filter_decisions=decisions,
                empty_reason="no_eligible_documents",
            )

        cached = self._cache.get_or_encode(query, self._encoder)
        positions = self._search_eligible(cached.vector, eligible)
        ranked = sorted(
            positions,
            key=lambda item: (-item[1], self._index.corpus.documents[item[0]].table_id),
        )[:k]
        results = tuple(
            DenseRetrievalCandidate(
                row_id=position,
                table_id=document.table_id,
                score=score,
                rank=rank,
                metadata=document.metadata,
                snippet=document.text[:500],
            )
            for rank, (position, score) in enumerate(ranked, start=1)
            for document in (self._index.corpus.documents[position],)
        )
        return DenseRetrievalTrace(
            question_id=question_id,
            query=query,
            normalized_query_sha256=cached.query_sha256,
            encoder_spec_sha256=self._index.manifest.encoder_spec_sha256,
            cache_hit=cached.cache_hit,
            eligible_count=len(eligible),
            filter_decisions=decisions,
            results=results,
        )

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
            raise DenseArtifactError("FAISS search did not return exactly the eligible documents")
        return tuple((row_id, float(score)) for row_id, score in pairs)
