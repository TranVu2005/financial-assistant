"""Immutable contracts for Day 10 deterministic entity parsing.

Mirrors the frozen-model discipline of `financial_report_qa.retrieval.contracts`
so entity output can be embedded in fusion traces and replayed byte-for-byte.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    RetrievalFilters,
    _canonical_tuple,
    _FrozenModel,
)

EntityField = Literal["company", "period", "metric", "statement_type", "statement_scope"]

# Day 21 plan §1.6/ADR 0010 decision A1: "riêng"/"công ty mẹ" -> separate,
# "hợp nhất"/"toàn tập đoàn" -> consolidated. Distinct from `statement_type`
# (balance sheet vs. income statement vs. ...), a different corpus dimension.
StatementScope = Literal["separate", "consolidated"]

# Reuses `financial_report_qa.schemas.normalization.NormalizationIssueCode` names
# where the same evidence conflict applies to free-text questions
# (`company_conflict`, `period_ambiguous`, `period_incomplete`, `metric_unknown`,
# `statement_conflict`); adds question-specific codes for evidence a table cell
# never needs (a whole field missing, or a period phrased relative to "today").
AmbiguityCode = Literal[
    "company_conflict",
    "company_missing",
    "period_ambiguous",
    "period_incomplete",
    "period_missing",
    "period_relative_unresolved",
    "metric_conflict",
    "metric_unknown",
    "statement_conflict",
    "statement_type_missing",
]

_COMPANY_AMBIGUITY: frozenset[AmbiguityCode] = frozenset({"company_conflict", "company_missing"})
_PERIOD_AMBIGUITY: frozenset[AmbiguityCode] = frozenset(
    {
        "period_ambiguous",
        "period_incomplete",
        "period_missing",
        "period_relative_unresolved",
    }
)
_STATEMENT_AMBIGUITY: frozenset[AmbiguityCode] = frozenset(
    {"statement_conflict", "statement_type_missing"}
)


class ParsedSpan(_FrozenModel):
    """One surface match in the NFKC-normalized question text.

    Offsets let every parse decision be traced back to the literal characters
    that produced it, independent of any dictionary lookup result.
    """

    field: EntityField
    surface: NonEmptyString
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("end")
    @classmethod
    def validate_span(cls, value: int, info: object) -> int:
        if value <= getattr(info, "data", {}).get("start", -1):
            raise ValueError("end must be greater than start")
        return value


class QueryEntities(_FrozenModel):
    """Deterministic entity-parser output for one natural-language question."""

    parser_version: Literal["entity-parser-v1", "entity-parser-v2"] = "entity-parser-v2"
    question: NonEmptyString
    company_codes: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    metric_phrases: tuple[str, ...] = ()
    operation: str | None = None
    requested_unit: str | None = None
    statement_types: tuple[str, ...] = ()
    statement_scope: StatementScope | None = None
    ambiguity: tuple[AmbiguityCode, ...] = ()
    spans: tuple[ParsedSpan, ...] = ()

    @property
    def metric_phrase(self) -> str | None:
        return self.metric_phrases[0] if self.metric_phrases else None

    @field_validator("company_codes", "periods", "metrics", "statement_types")
    @classmethod
    def validate_canonical_values(cls, values: tuple[str, ...], info: object) -> tuple[str, ...]:
        return _canonical_tuple(values, label=str(getattr(info, "field_name", "entities")))

    @field_validator("ambiguity")
    @classmethod
    def validate_ambiguity(cls, values: tuple[AmbiguityCode, ...]) -> tuple[AmbiguityCode, ...]:
        if tuple(sorted(set(values))) != values:
            raise ValueError("ambiguity codes must be sorted and unique")
        return values


def to_retrieval_filters(entities: QueryEntities) -> RetrievalFilters:
    """Project only fields free of an ambiguity flag into a hard retrieval filter.

    A field with an ambiguity code never contributes a filter value: guessing a
    company or period the parser itself flagged as unresolved would silently
    narrow retrieval on evidence the parser does not trust.
    """
    flags = set(entities.ambiguity)
    return RetrievalFilters(
        company_codes=() if flags & _COMPANY_AMBIGUITY else entities.company_codes,
        periods=() if flags & _PERIOD_AMBIGUITY else entities.periods,
        statement_types=() if flags & _STATEMENT_AMBIGUITY else entities.statement_types,
    )
