from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.units import (
    CanonicalUnit,
    convert_scale,
    economic_value,
    has_unit_evidence,
    normalize_unit,
    resolve_unit,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Đơn vị tính: VND", "VND"),
        ("ĐVT: nghìn đồng", "VND_thousand"),
        ("Triệu VND", "VND_million"),
        ("tỷ đồng", "VND_billion"),
        ("%", "percent"),
        ("lần", "ratio"),
    ],
)
def test_normalize_unit_aliases(raw: str, expected: CanonicalUnit) -> None:
    assert normalize_unit(raw) == Decision(value=expected)


def test_resolve_unit_prefers_more_specific_agreeing_evidence() -> None:
    assert resolve_unit(cell_hint="percent", column_raw="Tỷ lệ (%)", table_raw=None) == Decision(
        value="percent"
    )


def test_resolve_unit_rejects_conflicting_evidence() -> None:
    assert resolve_unit(
        cell_hint=None, column_raw="ĐVT: triệu đồng", table_raw="ĐVT: tỷ đồng"
    ) == Decision(value=None, issue_code="unit_conflict")


VND_UNITS = st.sampled_from(["VND", "VND_thousand", "VND_million", "VND_billion"])


@given(
    coefficient=st.integers(min_value=-(10**18), max_value=10**18),
    source=VND_UNITS,
    target=VND_UNITS,
)
def test_scale_conversion_preserves_economic_value(
    coefficient: int, source: CanonicalUnit, target: CanonicalUnit
) -> None:
    value = Decimal(coefficient)
    converted = convert_scale(value, source=source, target=target)
    assert economic_value(converted, target) == economic_value(value, source)


def test_economic_value_and_invalid_conversions() -> None:
    assert economic_value(Decimal("1500"), "VND_million") == Decimal("1500000000")
    with pytest.raises(ValueError, match="incompatible scale conversion"):
        convert_scale(Decimal("100"), source="VND", target="percent")


def test_resolve_unit_ignores_year_column_labels_without_unit_markers() -> None:
    """Regression: 'Năm 2024' or '2024' in column headers should not trigger unit_unknown."""
    decision = resolve_unit(cell_hint=None, column_raw="2024", table_raw=None)
    assert decision.value is None
    assert decision.issue_code is None

    decision_year = resolve_unit(cell_hint=None, column_raw="Năm 2024", table_raw=None)
    assert decision_year.value is None
    assert decision_year.issue_code is None


def test_unit_evidence_rejects_year_header() -> None:
    assert has_unit_evidence("2024") is False
    assert has_unit_evidence("Đơn vị: triệu đồng") is True
