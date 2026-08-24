from decimal import Decimal

import pytest

from financial_report_qa.core.errors import ProgramEvalError, ProgramGuardError
from financial_report_qa.execution.masked_program import (
    apply_scale,
    parse_program,
    run_program,
    substitute_placeholders,
)


def test_placeholders_become_parseable_identifiers() -> None:
    assert substitute_placeholders("([NUM_1] - [NUM_0]) / [NUM_0]") == "(NUM_1 - NUM_0) / NUM_0"


def test_a_bare_lookup_is_a_valid_program() -> None:
    # Câu tra cứu thuần là biểu thức ngắn nhất, không phải một operation riêng.
    assert run_program("[NUM_0]", [Decimal("4500")]) == Decimal("4500")


def test_growth_rate_evaluates_exactly() -> None:
    result = run_program("([NUM_1] - [NUM_0]) / [NUM_0]", [Decimal("4500"), Decimal("5310")])
    assert result == Decimal("0.18")


def test_abs_is_allowed() -> None:
    assert run_program("abs([NUM_0] - [NUM_1])", [Decimal("3"), Decimal("10")]) == Decimal("7")


def test_unary_minus_is_allowed() -> None:
    assert run_program("-[NUM_0]", [Decimal("7")]) == Decimal("-7")


@pytest.mark.parametrize(
    "program",
    [
        "[NUM_0] * 100",
        "[NUM_0] / 1000",
        "[NUM_0] + 0",
        "abs([NUM_0] - 1)",
        "[NUM_0] * 1e3",
        "-1 * [NUM_0]",
    ],
)
def test_every_numeric_literal_is_rejected(program: str) -> None:
    # N4': không ngoại lệ nào, kể cả hệ số đổi thang hay số 0 vô hại.
    with pytest.raises(ProgramGuardError):
        parse_program(program, value_count=1)


@pytest.mark.parametrize(
    "program",
    [
        "[NUM_0] ** [NUM_1]",
        "[NUM_0] // [NUM_1]",
        "[NUM_0] % [NUM_1]",
        "round([NUM_0])",
        "sum([[NUM_0]])",
        "[NUM_0].real",
        "[NUM_0][0]",
        "lambda: [NUM_0]",
        "[NUM_0] if [NUM_1] else [NUM_0]",
        "[NUM_0] > [NUM_1]",
        "__import__('os')",
        "df1",
    ],
)
def test_nodes_outside_the_whitelist_are_rejected(program: str) -> None:
    with pytest.raises(ProgramGuardError):
        parse_program(program, value_count=2)


def test_a_placeholder_beyond_the_bound_values_is_rejected() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("[NUM_2]", value_count=2)


def test_a_syntax_error_is_reported_as_a_guard_error() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("([NUM_0] - ", value_count=1)


def test_a_statement_is_not_an_expression() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("ans = [NUM_0]", value_count=1)


def test_division_by_zero_is_reported_with_its_own_code() -> None:
    with pytest.raises(ProgramEvalError, match="division_by_zero"):
        run_program("[NUM_0] / [NUM_1]", [Decimal("1"), Decimal("0")])


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        ("none", Decimal("0.18")),
        ("percent", Decimal("18.00")),
        ("thousand", Decimal("0.00018")),
        ("million", Decimal("0.00000018")),
        ("billion", Decimal("0.00000000018")),
    ],
)
def test_scale_factors(scale: str, expected: Decimal) -> None:
    assert apply_scale(Decimal("0.18"), scale) == expected  # type: ignore[arg-type]
