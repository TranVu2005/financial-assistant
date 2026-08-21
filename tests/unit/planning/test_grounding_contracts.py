"""Tests for the plan.md §9 `GroundedFact` provenance contract."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.planning.grounding_contracts import GroundedFact

TABLE_ID = "tbl_" + "1" * 64


def _fact(**overrides: object) -> GroundedFact:
    payload: dict[str, object] = {
        "fact_id": "F1",
        "table_id": TABLE_ID,
        "row_index": 14,
        "row_label": "Doanh thu thuần về bán hàng và CCDV",
        "column": "Năm 2023",
        "period": 2023,
        "raw_value": Decimal("63075000"),
        "unit": "VND_million",
        "grounding_score": 0.94,
    }
    payload.update(overrides)
    return GroundedFact.model_validate(payload)


def test_grounded_fact_identifies_its_row_by_integer_index() -> None:
    """plan.md §9: a fact is pinned by `row_index`, not by a label string --
    the label is provenance carried alongside it, never the identifier."""
    fact = _fact()
    assert fact.row_index == 14
    assert fact.row_label == "Doanh thu thuần về bán hàng và CCDV"
    assert fact.column == "Năm 2023"
    assert fact.raw_value == Decimal("63075000")
    assert fact.unit == "VND_million"
    assert fact.grounding_score == 0.94


def test_grounded_fact_allows_missing_column_and_score() -> None:
    """Not every grounded row names a column, and a deterministic canonical
    match never went through fusion, so it carries no retrieval score."""
    fact = _fact(column=None, grounding_score=None)
    assert fact.column is None
    assert fact.grounding_score is None


def test_grounded_fact_rejects_negative_row_index() -> None:
    with pytest.raises(ValidationError):
        _fact(row_index=-1)


def test_grounded_fact_rejects_malformed_fact_id() -> None:
    with pytest.raises(ValidationError):
        _fact(fact_id="fact-one")


def test_grounded_fact_rejects_unknown_unit() -> None:
    """§9's `source_unit` is a real CanonicalUnit, never a fabricated string."""
    with pytest.raises(ValidationError):
        _fact(unit="nan")


def test_grounded_fact_is_frozen() -> None:
    fact = _fact()
    with pytest.raises(ValidationError):
        fact.row_index = 7  # type: ignore[misc]
