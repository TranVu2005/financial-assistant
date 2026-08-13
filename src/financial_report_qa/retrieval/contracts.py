"""Immutable public contracts for the Day 8 retrieval baseline."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

RetrievalIntent = Literal["lookup", "compare", "growth"]
EmptyReason = Literal["no_eligible_documents", "no_index_tokens"]
TableId = Annotated[str, StringConstraints(pattern=r"^tbl_[0-9a-f]{64}$")]
QuestionId = Annotated[str, StringConstraints(pattern=r"^retq_[0-9a-f]{64}$")]
Fingerprint = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_tuple(
    values: tuple[str, ...], *, label: str, allow_empty: bool = True
) -> tuple[str, ...]:
    if not allow_empty and not values:
        raise ValueError(f"{label} must not be empty")
    if any(not value or value != value.strip() for value in values):
        raise ValueError(f"{label} must not contain blank values")
    canonical = tuple(sorted(set(values)))
    if values != canonical:
        raise ValueError(f"{label} must be sorted and unique")
    return values


class RetrievalFilters(_FrozenModel):
    """Explicit Day 8 metadata filters; OR within a field, AND across fields."""

    company_codes: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    statement_types: tuple[str, ...] = ()

    @field_validator("company_codes", "periods", "statement_types")
    @classmethod
    def validate_canonical_values(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _canonical_tuple(values, label=str(getattr(info, "field_name", "filter")))


class GoldTableEvidence(_FrozenModel):
    table_id: TableId
    relative_path: NonEmptyString
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    verified: Literal[True]

    @field_validator("relative_path")
    @classmethod
    def validate_safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or PureWindowsPath(value).drive
            or ".." in path.parts
            or "\\" in value
        ):
            raise ValueError("relative_path must be a safe POSIX-relative path")
        return value

    @field_validator("line_end")
    @classmethod
    def validate_span(cls, value: int, info: object) -> int:
        if value < getattr(info, "data", {}).get("line_start", 1):
            raise ValueError("line_end must be at least line_start")
        return value


class GoldRetrievalQuestion(_FrozenModel):
    question_id: QuestionId
    question: NonEmptyString
    intent: RetrievalIntent
    filters: RetrievalFilters
    gold_table_ids: tuple[TableId, ...]
    reviewed_by: NonEmptyString
    reviewed_at: datetime
    gold_evidence: tuple[GoldTableEvidence, ...]
    dataset_fingerprint: Fingerprint

    @field_validator("gold_table_ids")
    @classmethod
    def validate_gold_table_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_tuple(values, label="gold_table_ids", allow_empty=False)

    @field_validator("gold_evidence")
    @classmethod
    def validate_gold_evidence(
        cls, values: tuple[GoldTableEvidence, ...]
    ) -> tuple[GoldTableEvidence, ...]:
        if not values:
            raise ValueError("gold_evidence must not be empty")
        identifiers = tuple(item.table_id for item in values)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("gold_evidence must be sorted and contain one item per table")
        return values

    @field_validator("reviewed_at")
    @classmethod
    def validate_review_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value


class TableMetadata(_FrozenModel):
    table_id: TableId
    doc_id: str
    company_code: str | None = None
    periods: tuple[str, ...] = ()
    statement_type: str | None = None
    title: str | None = None
    source_path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    group_contexts: tuple[str, ...] = ()

    @field_validator("periods")
    @classmethod
    def validate_periods(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_tuple(values, label="periods")

    @field_validator("group_contexts")
    @classmethod
    def validate_group_contexts(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_tuple(values, label="group_contexts")


class MetricLabelObservation(_FrozenModel):
    """One canonical metric identity and its optional observed corpus label."""

    canonical: NonEmptyString
    raw: NonEmptyString | None = None


class MetricExpansion(_FrozenModel):
    """Canonical query tokens added after matching a corpus-observed alias."""

    alias_tokens: tuple[str, ...]
    canonical_metric: NonEmptyString
    added_tokens: tuple[str, ...]


class TableDocument(_FrozenModel):
    table_id: TableId
    doc_id: str
    text: NonEmptyString
    metadata: TableMetadata
    metric_labels: tuple[MetricLabelObservation, ...] = ()

    @field_validator("metric_labels")
    @classmethod
    def validate_metric_labels(
        cls, values: tuple[MetricLabelObservation, ...]
    ) -> tuple[MetricLabelObservation, ...]:
        ordering = tuple((item.canonical, item.raw or "") for item in values)
        if ordering != tuple(sorted(set(ordering))):
            raise ValueError("metric_labels must be sorted and unique")
        return values


class RetrievalCandidate(_FrozenModel):
    table_id: TableId
    score: float
    rank: int = Field(ge=1)
    metadata: TableMetadata
    snippet: str
    matched_tokens: tuple[str, ...] = ()

    @field_validator("score")
    @classmethod
    def validate_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class FilterDecision(_FrozenModel):
    field: Literal["company_codes", "periods", "statement_types"]
    requested_values: tuple[str, ...]
    matched_count_before_intersection: int = Field(ge=0)
    eligible_count_after_intersection: int = Field(ge=0)


class RetrievalTrace(_FrozenModel):
    question_id: QuestionId | None = None
    query: str
    query_tokens: tuple[str, ...]
    eligible_count: int = Field(ge=0)
    filter_decisions: tuple[FilterDecision, ...]
    results: tuple[RetrievalCandidate, ...]
    metric_expansions: tuple[MetricExpansion, ...] = ()
    empty_reason: EmptyReason | None = None


class BM25IndexManifest(_FrozenModel):
    schema_version: Literal["bm25-index-v3"] = "bm25-index-v3"
    builder_version: Literal["v3"] = "v3"
    tokenizer_version: Literal["v1"] = "v1"
    query_expansion_version: Literal["v1"] = "v1"
    dtype: Literal["float32"] = "float32"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint | None = None
    document_count: int = Field(ge=0)
    document_sha256: Fingerprint
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)
    bm25s_version: NonEmptyString
    k1: float
    b: float
    delta: float
    method: NonEmptyString
