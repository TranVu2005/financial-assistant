"""Tests for the numeric guard (ADR 0009 decision F1): generated text must
never introduce a number outside the whitelist of already-known numbers.

Day 20 plan Sec 1.8 measured that every numeric token in gold70 questions is
a year, and that naive tokenization captures trailing punctuation
(`'2023.'`) -- both are exercised below.
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.verification.numeric_guard import (
    extract_number_tokens,
    guard_generated_text,
)


def test_extract_number_tokens_handles_thousands_and_decimals() -> None:
    assert extract_number_tokens("doanh thu 84,420,878.5 VND năm 2023.") == (
        "84,420,878.5",
        "2023",
    )


def test_guard_rejects_a_number_outside_the_whitelist() -> None:
    result = guard_generated_text(
        "Doanh thu là 100 VND, không phải 200 VND.",
        whitelist=frozenset({Decimal("100")}),
    )
    assert result.allowed is False
    assert result.disallowed_numbers == ("200",)


def test_guard_accepts_text_within_the_whitelist() -> None:
    result = guard_generated_text(
        "Doanh thu năm 2023 là 84,420,878 VND.",
        whitelist=frozenset({Decimal("84420878"), Decimal("2023")}),
    )
    assert result.allowed is True
    assert result.disallowed_numbers == ()


def test_guard_treats_an_unparseable_token_as_disallowed() -> None:
    result = guard_generated_text("giá trị 1..2 lạ", whitelist=frozenset({Decimal("1")}))
    assert result.allowed is False
