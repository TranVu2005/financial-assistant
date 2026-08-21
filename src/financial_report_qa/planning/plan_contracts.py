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


def map_requested_unit(req_unit: str | None) -> ExpectedUnit | None:
    if not req_unit:
        return None
    mapping: dict[str, ExpectedUnit] = {
        "vnd": "VND",
        "VND": "VND",
        "VND_thousand": "VND_thousand",
        "million_vnd": "VND_million",
        "VND_million": "VND_million",
        "billion_vnd": "VND_billion",
        "VND_billion": "VND_billion",
        "percent": "percent",
        "ratio": "ratio",
    }
    return mapping.get(req_unit)


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
    # A row alone does not identify a cell. Vietnamese statement tables put
    # several semantically distinct amounts on one row -- the PC1 2025 tax note
    # carries "Số phải nộp đầu năm", "Số phải nộp trong năm", "Số đã thực nộp"
    # and "Số phải nộp cuối năm" against the same "Thuế giá trị gia tăng" row.
    # Selecting by row and period alone collapses them (two of those columns
    # both resolve to the closing year) and abstains as `cell_ambiguous`.
    # Measured on the locked release: `column_label_raw` is present on 96.9% of
    # numeric cells and only 4.4% of tables lack it entirely, so the column is a
    # dimension the release can actually support. Optional because most
    # questions name only a row; when absent, period remains the sole narrowing.
    column_text: RawMetricText | None = None
    # plan.md §9/§14: once cell grounding has decided which physical row the
    # question means, the selector stops being a search key and becomes a
    # pointer. `table_id` + `row_index` together pin exactly one row, and the
    # locator switches to positional extraction -- no label matching at query
    # time at all. Both unset on an unbound selector, which keeps every
    # pre-grounding plan (rule planner, LLM planner, stored fixtures) valid
    # and behaving exactly as before.
    table_id: TableId | None = None
    row_index: int | None = None

    @model_validator(mode="after")
    def validate_exactly_one_branch(self) -> Self:
        if (self.canonical is None) == (self.raw_text is None):
            raise ValueError("metric selector requires exactly one of canonical, raw_text")
        if self.canonical is not None and self.canonical not in CANONICAL_METRICS:
            raise ValueError(f"'{self.canonical}' is not a canonical metric")
        # A row index without its table is not a position: `row_idx` is only
        # unique within one table, so half a binding would silently select
        # row 14 of whatever table happened to be scanned.
        if (self.table_id is None) != (self.row_index is None):
            raise ValueError("table_id and row_index must be set together or both omitted")
        if self.row_index is not None and self.row_index < 0:
            raise ValueError("row_index must not be negative")
        return self

    @property
    def is_position_bound(self) -> bool:
        """True when grounding has pinned this selector to a physical row."""
        return self.table_id is not None and self.row_index is not None


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
