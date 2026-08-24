"""Frozen contracts for cross-encoder reranking of fused table candidates.

The reranker never sees the whole corpus: it rescoring only the top-N output
of weighted RRF (`fusion.py`), which is what keeps it runnable on a CPU-only
local machine. Both the fused rank and the rerank score are kept on every
candidate so a reordering decision stays auditable after the fact.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import (
    Fingerprint,
    NonEmptyString,
    QuestionId,
    TableId,
    TableMetadata,
    _FrozenModel,
)
from financial_report_qa.retrieval.dense_artifacts import canonical_json_bytes
from financial_report_qa.retrieval.dense_contracts import ModelRevision

RerankerName = Literal["qwen3-reranker-4b"]
RerankEmptyReason = Literal["no_fused_candidates"]

#: Số ứng viên RRF đưa vào reranker. Spec §5.3: rerank top-50, không phải
#: toàn corpus -- đây là điều làm bước này chạy được trên CPU local.
DEFAULT_RERANK_DEPTH = 50


class RerankerSpec(_FrozenModel):
    """Pinned cross-encoder behavior that identifies a rerank run."""

    name: RerankerName
    model_id: NonEmptyString
    revision: ModelRevision
    max_sequence_length: Literal[512, 1024, 2048, 8192] = 2048
    # N5: điểm số liên tục dùng để xếp hạng -- không bao giờ quantize.
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu", "cuda"] = "cpu"
    batch_size: int = Field(gt=0)


def reranker_spec_sha256(spec: RerankerSpec) -> str:
    """Return the canonical identity digest for one pinned reranker spec."""
    return hashlib.sha256(canonical_json_bytes(spec.model_dump(mode="json"))).hexdigest()


class RerankedCandidate(_FrozenModel):
    """One candidate after cross-encoder rescoring, with its fused origin kept."""

    table_id: TableId
    rank: int = Field(ge=1)
    rerank_score: float
    fused_rank: int = Field(ge=1)
    fused_score: float
    metadata: TableMetadata
    snippet: str

    @field_validator("rerank_score", "fused_score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class RerankTrace(_FrozenModel):
    """Auditable rerank outcome for one query."""

    question_id: QuestionId | None = None
    query: str
    reranker_spec_sha256: Fingerprint
    input_count: int = Field(ge=0)
    results: tuple[RerankedCandidate, ...] = ()
    empty_reason: RerankEmptyReason | None = None
