"""Persisted cross-encoder scores, so a reranked run costs GPU exactly once.

`ProgramDecision.cells` are positions in a candidate list built from
`retrieved`, so `submission row-batches` and `submission export` must rank
tables under identical settings -- which means the reranker runs twice over
the same 1012 x 50 pairs for a single submission. The scores are a pure
function of (pinned model, query, candidate snippets), so the second pass has
nothing to compute.

`CachedReranker` satisfies the `Reranker` protocol, so it drops in anywhere a
reranker goes. It builds the real model **lazily**: a fully-cached pass never
constructs it, which is what lets the export step run on a machine with no
GPU at all after the batch step warmed the cache.

A miss with no model available raises rather than silently returning an
unranked order -- retrieval rank is a scoring invariant, and a run that
quietly skipped reranking would look identical to one that did it.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from financial_report_qa.core.errors import RerankModelError
from financial_report_qa.retrieval.dense_artifacts import (
    canonical_json_bytes,
    write_numpy_atomic,
)
from financial_report_qa.retrieval.rerank_contracts import (
    RerankerSpec,
    reranker_spec_sha256,
)
from financial_report_qa.retrieval.reranker import Reranker


def normalize_rerank_query(query: str) -> str:
    """Same normalization the dense query cache applies, for the same reason:
    whitespace and Unicode form must not fork the cache key."""
    return " ".join(unicodedata.normalize("NFKC", query).split())


@dataclass(frozen=True)
class RerankCacheStats:
    """How much of a run the cache actually served."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses


class CachedReranker:
    """A `Reranker` that reads scores from disk and only then computes them."""

    def __init__(
        self,
        cache_dir: Path,
        spec: RerankerSpec,
        *,
        factory: Callable[[], Reranker] | None = None,
    ) -> None:
        self.spec = spec
        self._spec_hash = reranker_spec_sha256(spec)
        self._root = cache_dir / self._spec_hash[:12]
        self._factory = factory
        self._inner: Reranker | None = None
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> RerankCacheStats:
        return RerankCacheStats(hits=self._hits, misses=self._misses)

    def _digest(self, query: str, documents: Sequence[str]) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "documents": list(documents),
                    "normalization_version": "v1",
                    "query": normalize_rerank_query(query),
                    "reranker_spec_sha256": self._spec_hash,
                }
            )
        ).hexdigest()

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        path = self._root / f"{self._digest(query, documents)}.npy"
        if path.is_file():
            cached: np.ndarray = np.load(path, allow_pickle=False)
            if cached.shape != (len(documents),):
                raise RerankModelError(
                    f"cached rerank scores have shape {cached.shape}, "
                    f"expected ({len(documents)},): {path}"
                )
            self._hits += 1
            served: np.ndarray = cached.astype(np.float32)
            return served

        if self._inner is None:
            if self._factory is None:
                raise RerankModelError(
                    "rerank cache miss and no model was supplied -- refusing to "
                    "return an unreranked order; re-run with the reranker model "
                    f"available to fill {path}"
                )
            self._inner = self._factory()

        computed: np.ndarray = np.asarray(
            self._inner.score(query, documents), dtype=np.float32
        )
        write_numpy_atomic(path, computed)
        self._misses += 1
        return computed
