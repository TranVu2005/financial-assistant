"""Frozen public contracts for reproducible dense table retrieval."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from financial_report_qa.retrieval.contracts import (
    FilterDecision,
    Fingerprint,
    NonEmptyString,
    QuestionId,
    TableId,
    TableMetadata,
    _FrozenModel,
)

EncoderName = Literal["bge-m3", "multilingual-e5-small", "qwen3-embedding-4b"]
DenseEmptyReason = Literal["no_eligible_documents"]
ModelRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class DenseEncoderSpec(_FrozenModel):
    """Pinned encoder behavior that identifies a dense index."""

    name: EncoderName
    model_id: NonEmptyString
    revision: ModelRevision
    dimension: int = Field(gt=0)
    max_sequence_length: Literal[512, 1024, 2048, 8192] = 512
    query_prefix: str
    document_prefix: str
    pooling: Literal["sentence_transformers"] = "sentence_transformers"
    normalize_embeddings: Literal[True] = True
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(gt=0)


class DenseCorpusManifest(_FrozenModel):
    """Identity and integrity data for a persisted shared corpus."""

    schema_version: Literal["dense-corpus-v1"] = "dense-corpus-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_count: int = Field(ge=0)
    document_sha256: Fingerprint
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)


class DenseIndexManifest(_FrozenModel):
    """Identity and integrity data for one exact FAISS dense index."""

    schema_version: Literal["dense-index-v1"] = "dense-index-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_sha256: Fingerprint
    encoder: DenseEncoderSpec
    encoder_spec_sha256: Fingerprint
    document_count: int = Field(ge=0)
    dimension: int = Field(gt=0)
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    metric: Literal["inner_product"] = "inner_product"
    dtype: Literal["float32"] = "float32"
    normalized: Literal[True] = True
    index_byte_size: int = Field(ge=0)
    library_versions: dict[str, NonEmptyString]
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)


class DenseRetrievalCandidate(_FrozenModel):
    """One exact cosine-scored table candidate."""

    row_id: int = Field(ge=0)
    table_id: TableId
    score: float
    rank: int = Field(ge=1)
    metadata: TableMetadata
    snippet: str

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class DenseRetrievalTrace(_FrozenModel):
    """Auditable filter-first dense-retrieval outcome for one query."""

    question_id: QuestionId | None = None
    query: str
    normalized_query_sha256: Fingerprint | None = None
    encoder_spec_sha256: Fingerprint
    cache_hit: bool = False
    eligible_count: int = Field(ge=0)
    filter_decisions: tuple[FilterDecision, ...] = ()
    results: tuple[DenseRetrievalCandidate, ...] = ()
    empty_reason: DenseEmptyReason | None = None


class LatencySummary(_FrozenModel):
    """Measured query latency distribution for one cache state."""

    sample_count: int = Field(ge=0)
    p50_seconds: float = Field(ge=0)
    p95_seconds: float = Field(ge=0)


class DenseRunObservation(_FrozenModel):
    """Operational metadata excluded from deterministic evaluation projections."""

    encoder_name: EncoderName
    encoder_spec_sha256: Fingerprint
    dataset_fingerprint: Fingerprint
    build_seconds: float = Field(ge=0)
    index_byte_size: int = Field(ge=0)
    cold_latency: LatencySummary
    warm_latency: LatencySummary
