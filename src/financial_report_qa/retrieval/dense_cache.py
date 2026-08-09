"""Integrity-checked query embedding cache."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.retrieval.dense_artifacts import canonical_json_bytes, write_numpy_atomic
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import DenseEncoder, encoder_spec_sha256


@dataclass(frozen=True)
class CachedQueryEmbedding:
    vector: np.ndarray
    cache_hit: bool
    query_sha256: str
    normalized_query: str
    path: Path


def normalize_dense_query(query: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", query).split())


class QueryEmbeddingCache:
    def __init__(self, root: Path, spec: DenseEncoderSpec) -> None:
        self._root = root / encoder_spec_sha256(spec)[:12]
        self._spec = spec
        self._spec_hash = encoder_spec_sha256(spec)

    @property
    def encoder_spec_sha256(self) -> str:
        """Return the specification hash that scopes every cache entry."""
        return self._spec_hash

    def get_or_encode(self, query: str, encoder: DenseEncoder) -> CachedQueryEmbedding:
        if encoder_spec_sha256(encoder.spec) != self._spec_hash:
            raise DenseArtifactError("Query cache encoder specification does not match")
        normalized = normalize_dense_query(query)
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "encoder_spec_sha256": self._spec_hash,
                    "normalization_version": "v1",
                    "query": normalized,
                }
            )
        ).hexdigest()
        path = self._root / f"{digest}.npy"
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            self._validate(vector)
            return CachedQueryEmbedding(vector, True, digest, normalized, path)
        vector = np.asarray(encoder.encode_query(normalized), dtype=np.float32)
        self._validate(vector)
        write_numpy_atomic(path, vector)
        return CachedQueryEmbedding(vector, False, digest, normalized, path)

    def _validate(self, vector: np.ndarray) -> None:
        if vector.dtype != np.float32:
            raise DenseArtifactError("Query cache vector dtype must be float32")
        if vector.shape != (self._spec.dimension,) or not np.isfinite(vector).all():
            raise DenseArtifactError("Query cache vector shape or values are invalid")
        if not np.isclose(float(np.linalg.norm(vector)), 1.0, rtol=0.0, atol=1e-5):
            raise DenseArtifactError("Query cache vector must have unit norm")
