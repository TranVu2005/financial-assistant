"""Immutable contracts for Day 12 one-hop graph expansion and reranking."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.retrieval.contracts import (
    QuestionId,
    TableId,
    TableMetadata,
    _canonical_tuple,
    _FrozenModel,
)
from financial_report_qa.retrieval.fusion_contracts import ContradictedField
from financial_report_qa.retrieval.graph_contracts import GRAPH_RELATIONS, GraphRelation

ExpansionEmptyReason = Literal["no_eligible_documents"]
CandidateSource = Literal["seed", "graph"]


class SeedCandidate(_FrozenModel):
    """Score-independent representation shared by every seed retriever."""

    table_id: TableId
    rank: int = Field(ge=1)
    metadata: TableMetadata
    snippet: str


class ExpansionParams(_FrozenModel):
    """One fixed graph-expansion and reranking configuration."""

    relations: tuple[GraphRelation, ...] = GRAPH_RELATIONS
    alpha: float = Field(default=0.5, ge=0)
    fan_out: int = Field(default=25, gt=0)
    seed_depth: int = Field(default=50, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    expand_non_seeds: bool = False

    @field_validator("relations")
    @classmethod
    def validate_relations(cls, values: tuple[GraphRelation, ...]) -> tuple[GraphRelation, ...]:
        _canonical_tuple(values, label="relations", allow_empty=False)
        return values


_ALL_RELATIONS: tuple[GraphRelation, ...] = GRAPH_RELATIONS
_SAME_DOCUMENT_ONLY: tuple[GraphRelation, ...] = ("same_document",)

PRE_REGISTERED_EXPANSION_GRID: tuple[ExpansionParams, ...] = (
    ExpansionParams(alpha=0, relations=_ALL_RELATIONS),
    *tuple(
        ExpansionParams(
            relations=relations,
            alpha=alpha,
            expand_non_seeds=expand_non_seeds,
        )
        for relations in (_SAME_DOCUMENT_ONLY, _ALL_RELATIONS)
        for alpha in (0.25, 0.5, 1.0)
        for expand_non_seeds in (False, True)
    ),
)
"""The fixed 13-point grid; no parameter is fitted after seeing gold results."""


class RerankFeatures(_FrozenModel):
    """Auditable rank-based evidence used to score a graph candidate."""

    seed_rrf: float = Field(ge=0)
    neighbor_rrf: float = Field(ge=0)
    supporting_seed_count: int = Field(ge=0)
    relations: tuple[GraphRelation, ...] = ()

    @field_validator("seed_rrf", "neighbor_rrf")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RRF features must be finite")
        return value

    @field_validator("relations")
    @classmethod
    def validate_relations(cls, values: tuple[GraphRelation, ...]) -> tuple[GraphRelation, ...]:
        _canonical_tuple(values, label="relations")
        return values


class RerankedCandidate(_FrozenModel):
    """One final candidate with all reranking evidence preserved."""

    table_id: TableId
    rank: int = Field(ge=1)
    score: float
    source: CandidateSource
    features: RerankFeatures
    contradiction_count: int = Field(ge=0)
    contradicted_fields: tuple[ContradictedField, ...] = ()
    metadata: TableMetadata
    snippet: str

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class ExpansionTrace(_FrozenModel):
    """Auditable result of one seed, expand, filter, and rerank operation."""

    question_id: QuestionId | None = None
    query: str
    params: ExpansionParams
    entities: QueryEntities
    eligible_count: int = Field(ge=0)
    seed_count: int = Field(ge=0)
    expanded_count: int = Field(ge=0)
    dropped_out_of_filter: int = Field(ge=0)
    dropped_contradicted: int = Field(ge=0)
    results: tuple[RerankedCandidate, ...] = ()
    empty_reason: ExpansionEmptyReason | None = None
