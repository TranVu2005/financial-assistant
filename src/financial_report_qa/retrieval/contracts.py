"""Immutable contracts shared by the Day 8 retrieval pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
_QUESTION_ID_PATTERN = r"^retq_[a-z0-9_]+$"
_FINGERPRINT_PATTERN = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _normalize_values(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(sorted(set(values)))
    if any(not value.strip() for value in normalized):
        raise ValueError("filter values must not be blank")
    return normalized


class RetrievalFilters(_FrozenModel):
    """Exact metadata restrictions, OR within each field and AND across fields."""

    company_codes: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    statement_types: tuple[str, ...] = ()

    @field_validator("company_codes", "periods", "statement_types")
    @classmethod
    def normalize_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return _normalize_values(values)


class GoldTableEvidence(_FrozenModel):
    table_id: str = Field(pattern=_TABLE_ID_PATTERN)
    relative_path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    verified: Literal[True]

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must stay within repository")
        return path.as_posix()

    @field_validator("end_line")
    @classmethod
    def valid_line_span(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        if value < data.get("start_line", 1):
            raise ValueError("end_line must be at least start_line")
        return value


class GoldRetrievalQuestion(_FrozenModel):
    question_id: str = Field(pattern=_QUESTION_ID_PATTERN)
    question: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    filters: RetrievalFilters
    gold_table_ids: tuple[str, ...] = Field(min_length=1)
    gold_evidence: tuple[GoldTableEvidence, ...] = Field(min_length=1)
    reviewed_by: str = Field(min_length=1)
    reviewed_at: datetime
    dataset_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)

    @field_validator("gold_table_ids")
    @classmethod
    def normalize_gold_table_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(set(values)))
        if not normalized or any(not value.strip() for value in normalized):
            raise ValueError("gold_table_ids must contain non-blank values")
        return normalized

    @field_validator("reviewed_at")
    @classmethod
    def reviewed_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return value


class TableMetadata(_FrozenModel):
    table_id: str = Field(pattern=_TABLE_ID_PATTERN)
    company_code: str | None = None
    period: str | None = None
    statement_type: str | None = None
    title: str | None = None
    source_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class TableDocument(_FrozenModel):
    table_id: str = Field(pattern=_TABLE_ID_PATTERN)
    text: str = Field(min_length=1)
    metadata: TableMetadata


class RetrievalCandidate(_FrozenModel):
    table_id: str
    score: float
    rank: int = Field(ge=1)
    metadata: TableMetadata
    snippet: str


class FilterFieldDecision(_FrozenModel):
    field_name: Literal["company_codes", "periods", "statement_types"]
    requested_values: tuple[str, ...]
    matched_count_before_intersection: int = Field(ge=0)
    eligible_count_after_intersection: int = Field(ge=0)


class FilterDecision(_FrozenModel):
    filters: RetrievalFilters
    indexed_count: int = Field(ge=0)
    eligible_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    field_decisions: tuple[FilterFieldDecision, ...] = ()


class RetrievalTrace(_FrozenModel):
    question_id: str | None
    query: str
    filter_decision: FilterDecision
    results: tuple[RetrievalCandidate, ...]


class BM25IndexManifest(_FrozenModel):
    dataset_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    document_count: int = Field(ge=0)
    document_sha256: str = Field(pattern=_FINGERPRINT_PATTERN)
    bm25s_version: str = Field(min_length=1)
    k1: float
    b: float
    delta: float
    method: str
