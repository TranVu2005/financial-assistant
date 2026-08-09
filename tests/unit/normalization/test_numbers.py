from decimal import Decimal

import pytest

from financial_report_qa.normalization.numbers import (
    is_missing_number,
    is_numeric_candidate,
    parse_number,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", Decimal("0")),
        ("1.234.567", Decimal("1234567")),
        ("1,234,567", Decimal("1234567")),
        ("1 234 567", Decimal("1234567")),
        ("1.234,56", Decimal("1234.56")),
        ("1,234.56", Decimal("1234.56")),
        ("12,5", Decimal("12.5")),
        ("(1.250)", Decimal("-1250")),
        ("-42", Decimal("-42")),
        ("+42", Decimal("42")),
        ("123O", Decimal("1230")),
    ],
)
def test_parse_number_supports_vietnamese_and_english_formats(raw: str, expected: Decimal) -> None:
    decision = parse_number(raw)

    assert decision.value == expected
    assert decision.issue_code is None


def test_parse_number_keeps_percentage_as_explicit_unit_hint() -> None:
    decision = parse_number("12,5%")

    assert decision.value == Decimal("12.5")
    assert decision.unit_hint == "percent"
    assert decision.issue_code is None


@pytest.mark.parametrize("raw", ["-", "—", "N/A", "null", "", "."])
def test_parse_number_preserves_missing_values(raw: str) -> None:
    decision = parse_number(raw)

    assert is_missing_number(raw) is True
    assert decision.value is None
    assert decision.issue_code == "number_missing"


@pytest.mark.parametrize("raw", ["(123", "123)", "12 34", "12A", "1.2.34"])
def test_parse_number_rejects_malformed_values(raw: str) -> None:
    decision = parse_number(raw)

    assert decision.value is None
    assert decision.issue_code == "number_invalid"


@pytest.mark.parametrize("raw", ["0,123", "0.123", "1,234", "1.234"])
def test_parse_number_does_not_guess_a_single_three_digit_separator(raw: str) -> None:
    decision = parse_number(raw)

    assert decision.value is None
    assert decision.issue_code == "number_ambiguous"


def test_numeric_candidate_filter_excludes_labels_and_missing_markers() -> None:
    assert is_numeric_candidate("1.234,56") is True
    assert is_numeric_candidate("(123)") is True
    assert is_numeric_candidate("123O") is True
    assert is_numeric_candidate("Doanh thu 2024") is False
    assert is_numeric_candidate("—") is False


@pytest.mark.parametrize("raw", ["4 - 5", "10 - 39", "31.12.2021"])
def test_numeric_candidate_excludes_ranges_and_dates(raw: str) -> None:
    assert is_numeric_candidate(raw) is False


def test_numeric_candidate_keeps_malformed_merged_percent_for_audit() -> None:
    assert is_numeric_candidate("50%30%") is True
    assert parse_number("50%30%").issue_code == "number_invalid"


def test_monetary_context_resolves_single_three_digit_group() -> None:
    assert parse_number("1.764", context="monetary").value == Decimal("1764")


def test_percent_context_resolves_decimal_comma() -> None:
    decision = parse_number("99,999%", context="percent")

    assert decision.value == Decimal("99.999")
    assert decision.unit_hint == "percent"


def test_unknown_context_preserves_separator_ambiguity() -> None:
    assert parse_number("25.967", context="unknown").issue_code == "number_ambiguous"
