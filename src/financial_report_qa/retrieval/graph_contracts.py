"""Immutable contracts for the Day 11 GTR-lite table-relation graph.

Every edge is derived deterministically from `TableMetadata`/`metric_labels`
already present on a locked `TableDocument` -- nothing here re-reads raw
cells. `same_company` and `same_period` were measured and deliberately
excluded from the relation set: on the locked 146,011-table release they
produce 117,156,769 and 26,243,938 undirected pairs respectively, and every
reviewed gold question already hard-filters `company_codes` and `periods`
before ranking (see `retrieval/filtering.py`), so those two relations would
only ever connect tables already inside the eligible pool -- no new
information. `ExcludedRelation` records that decision, with the measured
pair count, in the persisted manifest instead of dropping it silently.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from financial_report_qa.retrieval.contracts import (
    Fingerprint,
    NonEmptyString,
    TableId,
    _FrozenModel,
)

GraphRelation = Literal[
    "adjacent_period",
    "explained_by_note",
    "same_document",
    "same_statement_type",
    "shared_metric",
]

SYMMETRIC_RELATIONS: tuple[GraphRelation, ...] = (
    "adjacent_period",
    "same_document",
    "same_statement_type",
    "shared_metric",
)
ASYMMETRIC_RELATIONS: tuple[GraphRelation, ...] = ("explained_by_note",)
GRAPH_RELATIONS: tuple[GraphRelation, ...] = tuple(
    sorted(SYMMETRIC_RELATIONS + ASYMMETRIC_RELATIONS)
)


class GraphEvidence(_FrozenModel):
    """Why one edge exists -- always populated with the fields its relation needs."""

    doc_id: NonEmptyString | None = None
    company_code: NonEmptyString | None = None
    shared_metrics: tuple[NonEmptyString, ...] = ()
    period_pairs: tuple[tuple[NonEmptyString, NonEmptyString], ...] = ()
    statement_type: NonEmptyString | None = None
    src_statement_type: NonEmptyString | None = None
    line_gap: int | None = Field(default=None, ge=0)

    @field_validator("shared_metrics")
    @classmethod
    def validate_shared_metrics(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("shared_metrics must be sorted and unique")
        return values

    @field_validator("period_pairs")
    @classmethod
    def validate_period_pairs(
        cls, values: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        if values != tuple(sorted(set(values))):
            raise ValueError("period_pairs must be sorted and unique")
        return values


_REQUIRED_EVIDENCE_FIELDS: dict[GraphRelation, tuple[str, ...]] = {
    "adjacent_period": ("company_code", "period_pairs"),
    "explained_by_note": ("doc_id", "src_statement_type", "statement_type"),
    "same_document": ("doc_id", "line_gap"),
    "same_statement_type": ("company_code", "statement_type"),
    "shared_metric": ("company_code", "shared_metrics"),
}


class GraphEdge(_FrozenModel):
    """One directed GTR-lite edge; symmetric relations are emitted both ways."""

    src_table_id: TableId
    dst_table_id: TableId
    relation: GraphRelation
    weight: float
    evidence: GraphEvidence

    @field_validator("weight")
    @classmethod
    def validate_finite_positive_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("weight must be finite and positive")
        return value

    @model_validator(mode="after")
    def validate_edge_shape(self) -> GraphEdge:
        if self.src_table_id == self.dst_table_id:
            raise ValueError("a graph edge must not be a self-loop")
        missing = [
            field
            for field in _REQUIRED_EVIDENCE_FIELDS[self.relation]
            if getattr(self.evidence, field) is None or getattr(self.evidence, field) == ()
        ]
        if missing:
            raise ValueError(f"relation {self.relation!r} requires evidence fields {missing}")
        return self


class ExcludedRelation(_FrozenModel):
    """A relation considered and rejected, with the measurement that justified it."""

    name: NonEmptyString
    reason: NonEmptyString
    measured_pair_count: int = Field(ge=0)


class GraphManifest(_FrozenModel):
    """Identity and integrity data for a persisted table-relation graph."""

    schema_version: Literal["graph-v1"] = "graph-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_count: int = Field(ge=0)
    document_sha256: Fingerprint
    relations: tuple[GraphRelation, ...] = GRAPH_RELATIONS
    excluded_relations: tuple[ExcludedRelation, ...] = ()
    bucket_counts: dict[GraphRelation, int] = Field(default_factory=dict)
    membership_counts: dict[GraphRelation, int] = Field(default_factory=dict)
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)
