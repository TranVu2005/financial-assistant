"""Immutable contracts for weighted-RRF row-level BM25/dense fusion."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from financial_report_qa.retrieval.contracts import TableId, _FrozenModel
from financial_report_qa.retrieval.row_documents import RowMetadata


class RowFusionWeights(_FrozenModel):
    """One point in the row-level weighted-RRF grid.

    ``rrf_k`` and ``depth`` are pinned at their conventional defaults
    (Cormack et al.) so only the branch weights vary across the grid.

    ``fuzzy`` and ``alias`` (plan.md §7) are secondary signals -- character
    similarity and metric-dictionary matches -- that default to 0 (off) so
    existing bm25/dense-only callers see no behavior change. At least one of
    ``bm25``/``dense`` must still be positive: fuzzy/alias contribute to
    score, never gate whether a query gets retrieval at all.
    """

    bm25: float = Field(ge=0)
    dense: float = Field(ge=0)
    fuzzy: float = Field(default=0.0, ge=0)
    alias: float = Field(default=0.0, ge=0)
    rrf_k: int = Field(default=60, gt=0)
    depth: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def validate_nonzero_weight(self) -> RowFusionWeights:
        if self.bm25 == 0 and self.dense == 0:
            raise ValueError("at least one of bm25/dense weight must be positive")
        return self


RowFusionEmptyReason = Literal["no_candidates"]


class RowFusedCandidate(_FrozenModel):
    """One fused row candidate with both branch contributions kept for audit."""

    row_id: str
    table_id: TableId
    row_idx: int
    rank: int = Field(ge=1)
    fused_score: float
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    fuzzy_rank: int | None = Field(default=None, ge=1)
    fuzzy_score: float | None = None
    alias_rank: int | None = Field(default=None, ge=1)
    alias_score: float | None = None
    metadata: RowMetadata
    snippet: str

    @field_validator("fused_score")
    @classmethod
    def validate_finite_fused_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fused_score must be finite")
        return value

    @field_validator("bm25_score", "dense_score", "fuzzy_score", "alias_score")
    @classmethod
    def validate_finite_branch_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("branch score must be finite")
        return value


class RowFusionTrace(_FrozenModel):
    """Auditable filter-first row fusion outcome for one query at one weight point."""

    query: str
    weights: RowFusionWeights
    candidate_table_ids: tuple[str, ...]
    bm25_candidate_count: int = Field(ge=0)
    dense_candidate_count: int = Field(ge=0)
    fuzzy_candidate_count: int = Field(default=0, ge=0)
    alias_candidate_count: int = Field(default=0, ge=0)
    results: tuple[RowFusedCandidate, ...] = ()
    empty_reason: RowFusionEmptyReason | None = None
