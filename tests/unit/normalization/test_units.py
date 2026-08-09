from decimal import Decimal

import pytest

from financial_report_qa.normalization.units import (
    convert_scale,
    economic_value,
    has_unit_evidence,
    normalize_unit,
    resolve_unit,
    unit_multiplier,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("VND", "VND"),
        ("VNĐ", "VND"),
        ("Đơn vị tính: nghìn đồng", "VND_thousand"),
        ("ĐVT: triệu VNĐ", "VND_million"),
        ("Đơn vị: tỷ đồng", "VND_billion"),
        ("tỷ lệ (%)", "percent"),
        ("lần", "ratio"),
    ],
)
def test_normalize_unit_supports_financial_unit_aliases(raw: str, canonical: str) -> None:
    decision = normalize_unit(raw)

    assert decision.value == canonical
    assert decision.issue_code is None


@pytest.mark.parametrize("raw", [None, "", "   ", "2024", "Năm 2024", "Quý IV"])
def test_normalize_unit_ignores_values_without_unit_meaning(raw: str | None) -> None:
    decision = normalize_unit(raw)

    assert decision.value is None
    assert decision.issue_code is None


def test_normalize_unit_reports_unknown_when_unit_evidence_cannot_be_resolved() -> None:
    decision = normalize_unit("Đơn vị tính: nghìn USD")

    assert decision.value is None
    assert decision.issue_code == "unit_unknown"


def test_has_unit_evidence_does_not_treat_period_headers_as_units() -> None:
    assert has_unit_evidence("Số tiền (triệu đồng)") is True
    assert has_unit_evidence("Năm 2024") is False
    assert has_unit_evidence(None) is False


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("2020VND", "VND"),
        ("2025Triệu VND", "VND_million"),
        ("1/1/2017\nGiá gốc VND", "VND"),
        ("Ngàn VNDNăm trước", "VND_thousand"),
        ("31/12/2024\nVND", "VND"),
    ],
)
def test_normalize_unit_extracts_unit_from_composite_headers(raw: str, canonical: str) -> None:
    decision = normalize_unit(raw)

    assert decision.value == canonical
    assert decision.issue_code is None


@pytest.mark.parametrize("raw", ["Công ty Vinpearl", "Công ty liên kết", "Mối quan hệ"])
def test_has_unit_evidence_ignores_company_and_relationship_text(raw: str) -> None:
    assert has_unit_evidence(raw) is False


def test_resolve_unit_extracts_composite_column_scale() -> None:
    decision = resolve_unit(
        cell_hint=None,
        column_raw="2025Triệu VND",
        table_raw=None,
    )

    assert decision.value == "VND_million"
    assert decision.issue_code is None


def test_resolve_unit_prefers_cell_percentage_over_table_monetary_scale() -> None:
    decision = resolve_unit(
        cell_hint="percent",
        column_raw="Năm 2024",
        table_raw="Đơn vị tính: triệu đồng",
    )

    assert decision.value == "percent"
    assert decision.issue_code is None


def test_resolve_unit_prefers_column_scale_over_table_default() -> None:
    decision = resolve_unit(
        cell_hint=None,
        column_raw="ĐVT: nghìn đồng",
        table_raw="Đơn vị tính: triệu đồng",
    )

    assert decision.value == "VND_thousand"
    assert decision.issue_code is None


def test_resolve_unit_returns_no_issue_without_any_unit_evidence() -> None:
    decision = resolve_unit(cell_hint=None, column_raw="Năm 2024", table_raw=None)

    assert decision.value is None
    assert decision.issue_code is None


def test_unit_scaling_uses_decimal_arithmetic() -> None:
    assert unit_multiplier("VND_million") == Decimal("1000000")
    assert economic_value(Decimal("1.25"), "VND_million") == Decimal("1250000.00")
    assert convert_scale(Decimal("1500"), "VND_thousand", "VND_million") == Decimal("1.5")


@pytest.mark.parametrize(
    ("source", "target"),
    [("percent", "ratio"), ("VND", "percent")],
)
def test_convert_scale_rejects_incompatible_units(source: str, target: str) -> None:
    with pytest.raises(ValueError, match="incompatible scale conversion"):
        convert_scale(Decimal("1"), source, target)  # type: ignore[arg-type]
