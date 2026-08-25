"""Unit tests for the persisted cross-encoder score cache.

The cache exists so a submission pays GPU for reranking once instead of twice
(`row-batches` then `export` must rank under identical settings). These tests
pin the three properties that makes it safe to rely on: a second identical
call never touches the model, a different input never reuses another's scores,
and a miss with no model fails loudly rather than silently skipping rerank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from financial_report_qa.core.errors import RerankModelError
from financial_report_qa.retrieval.rerank_cache import CachedReranker
from financial_report_qa.retrieval.rerank_contracts import RerankerSpec

_SPEC = RerankerSpec(
    name="qwen3-reranker-4b",
    model_id="Qwen/Qwen3-Reranker-4B",
    revision="a" * 40,
    batch_size=4,
)


class _CountingReranker:
    """Scores by document length; counts how often it is actually asked."""

    def __init__(self) -> None:
        self.spec = _SPEC
        self.calls = 0

    def score(self, query: str, documents):  # type: ignore[no-untyped-def]
        self.calls += 1
        return np.asarray([float(len(d)) for d in documents], dtype=np.float32)


def test_a_repeated_call_is_served_from_disk(tmp_path: Path) -> None:
    inner = _CountingReranker()
    cache = CachedReranker(tmp_path, _SPEC, factory=lambda: inner)
    documents = ("snippet-a", "snippet-bb")

    first = cache.score("doanh thu 2023", documents)
    second = cache.score("doanh thu 2023", documents)

    assert inner.calls == 1
    assert np.array_equal(first, second)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_a_warm_cache_never_builds_the_model(tmp_path: Path) -> None:
    documents = ("snippet-a", "snippet-bb")
    CachedReranker(tmp_path, _SPEC, factory=_CountingReranker).score("q", documents)

    def _explode() -> _CountingReranker:  # pragma: no cover -- must not run
        raise AssertionError("factory must not be called on a full cache hit")

    warm = CachedReranker(tmp_path, _SPEC, factory=_explode)  # type: ignore[arg-type]
    scores = warm.score("q", documents)

    assert scores.tolist() == [9.0, 10.0]
    assert warm.stats == warm.stats.__class__(hits=1, misses=0)


def test_a_different_query_does_not_reuse_scores(tmp_path: Path) -> None:
    inner = _CountingReranker()
    cache = CachedReranker(tmp_path, _SPEC, factory=lambda: inner)
    documents = ("snippet-a",)

    cache.score("câu hỏi một", documents)
    cache.score("câu hỏi hai", documents)

    assert inner.calls == 2


def test_a_different_candidate_list_does_not_reuse_scores(tmp_path: Path) -> None:
    inner = _CountingReranker()
    cache = CachedReranker(tmp_path, _SPEC, factory=lambda: inner)

    cache.score("q", ("snippet-a", "snippet-b"))
    cache.score("q", ("snippet-b", "snippet-a"))

    # Order is part of the key: rerank output is a ranking over *this* list.
    assert inner.calls == 2


def test_whitespace_and_unicode_form_do_not_fork_the_key(tmp_path: Path) -> None:
    inner = _CountingReranker()
    cache = CachedReranker(tmp_path, _SPEC, factory=lambda: inner)

    cache.score("doanh  thu\n2023", ("s",))
    cache.score("doanh thu 2023", ("s",))

    assert inner.calls == 1


def test_a_different_spec_does_not_reuse_scores(tmp_path: Path) -> None:
    inner_a = _CountingReranker()
    inner_b = _CountingReranker()
    other = _SPEC.model_copy(update={"batch_size": 8})

    CachedReranker(tmp_path, _SPEC, factory=lambda: inner_a).score("q", ("s",))
    CachedReranker(tmp_path, other, factory=lambda: inner_b).score("q", ("s",))

    assert inner_a.calls == 1
    assert inner_b.calls == 1


def test_a_miss_without_a_model_fails_loudly(tmp_path: Path) -> None:
    # Silently returning the fused order would be indistinguishable from a
    # real rerank in the output, and retrieval rank is a scoring invariant.
    cache = CachedReranker(tmp_path, _SPEC)

    with pytest.raises(RerankModelError, match="cache miss"):
        cache.score("q", ("s",))


def test_a_corrupt_cache_entry_is_rejected(tmp_path: Path) -> None:
    inner = _CountingReranker()
    cache = CachedReranker(tmp_path, _SPEC, factory=lambda: inner)
    cache.score("q", ("s1", "s2"))

    entry = next((tmp_path).rglob("*.npy"))
    np.save(entry, np.asarray([1.0], dtype=np.float32))

    with pytest.raises(RerankModelError, match="expected"):
        CachedReranker(tmp_path, _SPEC, factory=lambda: inner).score("q", ("s1", "s2"))
