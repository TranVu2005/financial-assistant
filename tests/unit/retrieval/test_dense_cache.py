from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from financial_report_qa.retrieval.dense_cache import QueryEmbeddingCache
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import approved_encoder_spec


@dataclass
class CountingEncoder:
    spec: DenseEncoderSpec
    query_calls: int = 0

    def encode_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_query_cache_reuses_normalized_query_for_same_encoder(tmp_path: Path) -> None:
    """Skipping NFKC/whitespace normalization would encode equivalent queries twice."""
    encoder = CountingEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(update={"dimension": 2})
    )
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)

    first = cache.get_or_encode("  Lợi  nhuận  ", encoder)
    second = cache.get_or_encode("Lợi nhuận", encoder)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.query_sha256 == second.query_sha256
    assert encoder.query_calls == 1
    np.testing.assert_array_equal(first.vector, second.vector)
