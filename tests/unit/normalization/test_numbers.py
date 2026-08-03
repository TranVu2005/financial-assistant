from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from financial_report_qa.normalization.numbers import NumberDecision, parse_number


@pytest.mark.parametrize(
    ("raw", "value", "unit_hint", "issue"),
    [
        ("1.500", Decimal("1500"), None, None),
        ("1,500", Decimal("1500"), None, None),
        ("1 500 000", Decimal("1500000"), None, None),
        ("1.500,25", Decimal("1500.25"), None, None),
        ("1,500.25", Decimal("1500.25"), None, None),
        ("(1.500)", Decimal("-1500"), None, None),
        ("+12,5", Decimal("12.5"), None, None),
        ("12,5%", Decimal("12.5"), "percent", None),
        ("-", None, None, "number_missing"),
        ("N/A", None, None, "number_missing"),
        ("1.50.0", None, None, "number_invalid"),
        ("(100", None, None, "number_invalid"),
        ("1,23,456", None, None, "number_ambiguous"),
    ],
)
def test_parse_number_examples(
    raw: str,
    value: Decimal | None,
    unit_hint: str | None,
    issue: str | None,
) -> None:
    assert parse_number(raw) == NumberDecision(
        value=value, unit_hint=unit_hint, issue_code=issue  # type: ignore[arg-type]
    )


@given(
    value=st.decimals(
        min_value=Decimal("-1000000000000000000"),
        max_value=Decimal("1000000000000000000"),
        allow_nan=False,
        allow_infinity=False,
        places=2,
    ),
)
def test_controlled_decimal_rendering_round_trips(value: Decimal) -> None:
    raw = format(value, "f")
    assert parse_number(raw).value == value


@given(raw=st.text(max_size=40))
def test_parse_number_never_mutates_input(raw: str) -> None:
    before = raw
    parse_number(raw)
    assert raw == before
