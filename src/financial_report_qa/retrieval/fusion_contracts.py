"""Immutable contracts for Day 10 weighted-RRF BM25/dense fusion."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.retrieval.contracts import (
    QuestionId,
    TableId,
    TableMetadata,
    _FrozenModel,
)

FusionEmptyReason = Literal["no_eligible_documents"]
ContradictedField = Literal["company", "period"]


class FusionWeights(_FrozenModel):
    """One point in the pre-registered weighted-RRF grid.

    `rrf_k` and `depth` are pinned at their conventional defaults (Cormack
    et al.) so only `bm25`/`dense` vary across the grid; nothing here is
    fit to the 30-question gold set after the fact.
    """

    bm25: float = Field(ge=0)
    dense: float = Field(ge=0)
    rrf_k: int = Field(default=60, gt=0)
    depth: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def validate_nonzero_weight(self) -> FusionWeights:
        if self.bm25 == 0 and self.dense == 0:
            raise ValueError("at least one of bm25/dense weight must be positive")
        return self


PRE_REGISTERED_WEIGHT_GRID: tuple[FusionWeights, ...] = tuple(
    FusionWeights(bm25=bm25, dense=dense)
    for bm25, dense in ((1, 0), (1, 1), (2, 1), (3, 1), (4, 1), (0, 1))
)


class FusedCandidate(_FrozenModel):
    """One fused candidate with both branch contributions kept for audit."""

    table_id: TableId
    rank: int = Field(ge=1)
    fused_score: float
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = None
    contradiction_count: int = Field(ge=0)
    contradicted_fields: tuple[ContradictedField, ...] = ()
    metadata: TableMetadata
    snippet: str

    @field_validator("fused_score")
    @classmethod
    def validate_finite_fused_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("fused_score must be finite")
        return value

    @field_validator("bm25_score", "dense_score")
    @classmethod
    def validate_finite_branch_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("branch score must be finite")
        return value


class FusionTrace(_FrozenModel):
    """Auditable filter-first fusion outcome for one query at one weight point."""

    question_id: QuestionId | None = None
    query: str
    weights: FusionWeights
    entities: QueryEntities
    eligible_count: int = Field(ge=0)
    bm25_candidate_count: int = Field(ge=0)
    dense_candidate_count: int = Field(ge=0)
    results: tuple[FusedCandidate, ...] = ()
    empty_reason: FusionEmptyReason | None = None
