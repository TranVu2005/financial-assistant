"""plan.md §9 Cell Grounding provenance: `GroundedFact`.

A `GroundedFact` is the object grounding hands to everything downstream. Its
identity is positional -- `(table_id, row_index)` plus the column -- not a
label string. The labels it carries are provenance for audit and for §15's
per-fact verification, never the thing a later stage re-matches on.

That distinction is the whole point of §9/§14: while a plan identified its
row by a fuzzy label, the locator had to re-run semantic matching at query
time against `row_label_raw`/`row_label_canonical`, and the 144-question dev
benchmark measured what that costs -- 71 of 88 wrong answers picked a
plausible-looking number off the wrong row entirely, with no unit error at
all. Once the row is pinned to an integer index, Pandas has nothing left to
guess about.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Annotated

from pydantic import Field, StringConstraints, field_validator

from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import CompanyCode
from financial_report_qa.retrieval.contracts import NonEmptyString, TableId, _FrozenModel

FactId = Annotated[str, StringConstraints(pattern=r"^F[0-9]+$")]


class GroundedFact(_FrozenModel):
    """One cell resolved to an exact position, with full provenance."""

    fact_id: FactId
    table_id: TableId
    # plan.md §9/§14: `row_index` is `cells.row_idx` -- the row's physical
    # position in the extracted table, which is what makes the later
    # `df.loc[row_index, column]` extraction deterministic.
    row_index: int = Field(ge=0)
    row_label: NonEmptyString
    column: NonEmptyString | None = None
    # plan.md §12: which issuer this cell belongs to. A fact enumerated for
    # the planner has to say whose figure it is -- and turning the chosen
    # facts back into an executable plan needs `companies` filled from
    # somewhere other than a re-parse of the question.
    company_code: CompanyCode | None = None
    period: int = Field(ge=1900, le=2100)
    raw_value: Decimal
    # Constrained to the 6 real CanonicalUnit values for the same reason
    # `CellMatch.unit` is (ADR 0009 decision C1): a NULL unit surfaces as the
    # fabricated string 'nan' through DuckDB's pandas conversion, and that
    # must never pass as evidence.
    unit: CanonicalUnit
    # `None` when the row was never scored by row fusion -- e.g. a
    # deterministic canonical-alias match, which needs no confidence number.
    grounding_score: float | None = None

    @field_validator("grounding_score")
    @classmethod
    def validate_finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("grounding_score must be finite")
        return value
