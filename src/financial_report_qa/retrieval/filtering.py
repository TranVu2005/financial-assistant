"""Shared explicit metadata eligibility for all table retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from financial_report_qa.retrieval.contracts import FilterDecision, RetrievalFilters, TableDocument


def eligible_positions(
    documents: Sequence[TableDocument], filters: RetrievalFilters
) -> tuple[tuple[int, ...], tuple[FilterDecision, ...]]:
    eligible = set(range(len(documents)))
    decisions: list[FilterDecision] = []
    for field, requested, attribute in (
        ("company_codes", filters.company_codes, "company_code"),
        ("periods", filters.periods, "periods"),
        ("statement_types", filters.statement_types, "statement_type"),
    ):
        if not requested:
            continue
        matched = (
            {
                position
                for position, document in enumerate(documents)
                if set(document.metadata.periods).intersection(requested)
            }
            if field == "periods"
            else {
                position
                for position, document in enumerate(documents)
                if getattr(document.metadata, attribute) in requested
            }
        )
        eligible.intersection_update(matched)
        decisions.append(
            FilterDecision(
                field=cast(Literal["company_codes", "periods", "statement_types"], field),
                requested_values=requested,
                matched_count_before_intersection=len(matched),
                eligible_count_after_intersection=len(eligible),
            )
        )
    return tuple(sorted(eligible)), tuple(decisions)
