"""Immutable Day 15 FinancialQueryPlan schema.

Structural well-formedness only (field types, enums, `extra="forbid"`). Operation
arity (how many companies/periods/metrics an operation needs) and release-bound
existence checks (`candidate_table_ids` in the corpus) belong to the semantic
validator in `plan_validator.py`, not here — see ADR 0004 for why the split.
"""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import model_validator

from financial_report_qa.normalization.metrics import METRIC_ALIASES
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel

CANONICAL_METRICS: frozenset[str] = frozenset(METRIC_ALIASES.values())
_PERIOD_PATTERN = re.compile(r"^\d{4}$")

PlanOperation = Literal[
    "lookup",
    "compare",
    "compare_companies",
    "difference",
    "growth_rate",
    "ratio",
    "average",
    "sum",
    "rank",
]

ExpectedUnit = Literal["VND", "VND_thousand", "VND_million", "VND_billion", "percent", "ratio"]


class MetricSelector(_FrozenModel):
    """Locate one row across candidate tables: a canonical metric, or a verbatim
    source label copied from a table the planner actually saw (ADR 0004 Option C).
    """

    canonical: NonEmptyString | None = None
    raw_text: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_exactly_one_branch(self) -> Self:
        if (self.canonical is None) == (self.raw_text is None):
            raise ValueError("metric selector requires exactly one of canonical, raw_text")
        if self.canonical is not None and self.canonical not in CANONICAL_METRICS:
            raise ValueError(f"'{self.canonical}' is not a canonical metric")
        return self


class FinancialQueryPlan(_FrozenModel):
    """One deterministically-compilable Pandas computation over candidate tables."""

    operation: PlanOperation
    companies: tuple[NonEmptyString, ...]
    periods: tuple[NonEmptyString, ...]
    candidate_table_ids: tuple[TableId, ...]
    metric: MetricSelector | None = None
    metric_a: MetricSelector | None = None
    metric_b: MetricSelector | None = None
    numerator_metric: MetricSelector | None = None
    denominator_metric: MetricSelector | None = None
    top_k: int | None = None
    expected_unit: ExpectedUnit | None = None

    @model_validator(mode="after")
    def validate_shared_structure(self) -> Self:
        if not self.companies:
            raise ValueError("companies must not be empty")
        if len(set(self.companies)) != len(self.companies):
            raise ValueError("companies must not contain duplicates")
        if not self.periods:
            raise ValueError("periods must not be empty")
        if len(set(self.periods)) != len(self.periods):
            raise ValueError("periods must not contain duplicates")
        if any(not _PERIOD_PATTERN.match(period) for period in self.periods):
            raise ValueError("periods must be canonical 'YYYY', not a full date")
        if not 1 <= len(self.candidate_table_ids) <= 12:
            raise ValueError("candidate_table_ids must have between 1 and 12 entries")
        if len(set(self.candidate_table_ids)) != len(self.candidate_table_ids):
            raise ValueError("candidate_table_ids must not contain duplicates")
        return self
