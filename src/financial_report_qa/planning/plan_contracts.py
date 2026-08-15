"""Immutable Day 15 FinancialQueryPlan schema.

Structural well-formedness only (field types, enums, `extra="forbid"`). Operation
arity (how many companies/periods/metrics an operation needs) and release-bound
existence checks (`candidate_table_ids` in the corpus) belong to the semantic
validator in `plan_validator.py`, not here — see ADR 0004 for why the split.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import StringConstraints, model_validator

from financial_report_qa.normalization.metrics import METRIC_ALIASES
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel

CANONICAL_METRICS: frozenset[str] = frozenset(METRIC_ALIASES.values())
_PERIOD_PATTERN = re.compile(r"^\d{4}$")

# Day 19 plan Sec 1.2/1.10: these two fields are the only free-form strings that
# survive plan validation to reach the execution sandbox. Bounds are measured,
# not guessed: control chars ban costs 1/5,353,511 cells; 512-char cap costs 433
# numeric cells (0.008%); the company allowlist costs 0/1,971 documents.
RawMetricText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=512, pattern=r"^[^\x00-\x1f]+$"
    ),
]
CompanyCode = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]{1,16}$")]

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

# Day 21 plan §1.3/§1.4: `documents.statement_scope` distinguishes separate
# ("riêng"/"công ty mẹ") from consolidated ("hợp nhất") reports; 64.9% of
# (company, period, metric) groups exist in both with 92.8% disagreeing in
# value. ADR 0010 decision A1: this must be a plan field, not an implicit
# filter applied downstream where no validator or audit trail can see it.
StatementScope = Literal["separate", "consolidated"]


class MetricSelector(_FrozenModel):
    """Locate one row across candidate tables: a canonical metric, or a verbatim
    source label copied from a table the planner actually saw (ADR 0004 Option C).
    """

    canonical: NonEmptyString | None = None
    raw_text: RawMetricText | None = None

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
    companies: tuple[CompanyCode, ...]
    periods: tuple[NonEmptyString, ...]
    candidate_table_ids: tuple[TableId, ...]
    metric: MetricSelector | None = None
    metric_a: MetricSelector | None = None
    metric_b: MetricSelector | None = None
    numerator_metric: MetricSelector | None = None
    denominator_metric: MetricSelector | None = None
    top_k: int | None = None
    expected_unit: ExpectedUnit | None = None
    statement_scope: StatementScope | None = None

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
