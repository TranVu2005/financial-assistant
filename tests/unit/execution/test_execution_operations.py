"""Tests for the Day 18 per-operation compiler functions (ADR 0007 decision E1)."""

from decimal import Decimal

import pytest

from financial_report_qa.execution import operations
from financial_report_qa.execution.contracts import CellMatch

TABLE_ID = "tbl_" + "1" * 64


def _cell(cell_id_char: str, *, value: str, unit: str = "VND", period: int = 2020) -> CellMatch:
    return CellMatch(
        table_id=TABLE_ID,
        cell_ids=("cell_" + cell_id_char * 64,),
        value=Decimal(value),
        unit=unit,
        period=period,
        period_inferred=False,
    )


def test_compile_lookup_returns_cell_value_and_unit() -> None:
    answer, unit = operations.compile_lookup(_cell("a", value="100"))
    assert answer == Decimal("100")
    assert unit == "VND"


def test_compile_difference_subtracts_same_unit() -> None:
    end = _cell("a", value="900", period=2023)
    start = _cell("b", value="800", period=2022)
    answer, unit = operations.compile_difference(end, start)
    assert answer == Decimal("100")
    assert unit == "VND"


def test_compile_difference_preserves_negative_result() -> None:
    """Day 18 plan §1.5: 327,743 stored cells are negative; the compiler must
    not lose or flip that sign during arithmetic."""
    end = _cell("a", value="50", period=2023)
    start = _cell("b", value="800", period=2022)
    answer, _ = operations.compile_difference(end, start)
    assert answer == Decimal("-750")


def test_compile_difference_converts_mismatched_scale() -> None:
    """1,000,000 VND (start) subtracted from 2 VND_million (end) is 1 VND_million,
    even though the stored cells use different scales."""
    end = _cell("a", value="2", unit="VND_million", period=2023)
    start = _cell("b", value="1000000", unit="VND", period=2022)
    answer, unit = operations.compile_difference(end, start)
    assert answer == Decimal("1")
    assert unit == "VND_million"


def test_compile_difference_rejects_incompatible_units() -> None:
    """A currency cell can never be differenced against a percent cell; this
    must fail loudly (unit_incompatible), not silently subtract raw numbers."""
    end = _cell("a", value="10", unit="percent", period=2023)
    start = _cell("b", value="800", unit="VND", period=2022)
    with pytest.raises(ValueError):
        operations.compile_difference(end, start)


def test_compile_growth_rate_computes_fraction() -> None:
    end = _cell("a", value="120", period=2023)
    start = _cell("b", value="100", period=2022)
    answer, unit = operations.compile_growth_rate(end, start)
    assert answer == Decimal("0.2")
    assert unit == "ratio"


def test_compile_growth_rate_raises_on_zero_base() -> None:
    """Day 18 plan §1.5: 4,649 stored cells are exactly zero; dividing by one
    of them as a growth-rate base must fail with a typed error, not inf/NaN."""
    end = _cell("a", value="120", period=2023)
    start = _cell("b", value="0", period=2022)
    with pytest.raises(ZeroDivisionError):
        operations.compile_growth_rate(end, start)


def test_compile_compare_subtracts_two_metrics_same_period() -> None:
    metric_a = _cell("a", value="500")
    metric_b = _cell("b", value="300")
    answer, unit = operations.compile_compare(metric_a, metric_b)
    assert answer == Decimal("200")
    assert unit == "VND"


def test_compile_compare_companies_subtracts_across_companies() -> None:
    company_a = _cell("a", value="1000")
    company_b = _cell("b", value="750")
    answer, unit = operations.compile_compare_companies(company_a, company_b)
    assert answer == Decimal("250")
    assert unit == "VND"


def test_compile_ratio_divides_numerator_by_denominator() -> None:
    numerator = _cell("a", value="50")
    denominator = _cell("b", value="200")
    answer, unit = operations.compile_ratio(numerator, denominator)
    assert answer == Decimal("0.25")
    assert unit == "ratio"


def test_compile_ratio_raises_on_zero_denominator() -> None:
    numerator = _cell("a", value="50")
    denominator = _cell("b", value="0")
    with pytest.raises(ZeroDivisionError):
        operations.compile_ratio(numerator, denominator)


def test_compile_average_computes_mean_across_periods() -> None:
    cells = (
        _cell("a", value="100", period=2021),
        _cell("b", value="200", period=2022),
        _cell("c", value="300", period=2023),
    )
    answer, unit = operations.compile_average(cells)
    assert answer == Decimal("200")
    assert unit == "VND"


def test_compile_sum_adds_across_periods() -> None:
    cells = (
        _cell("a", value="100", period=2021),
        _cell("b", value="200", period=2022),
    )
    answer, unit = operations.compile_sum(cells)
    assert answer == Decimal("300")
    assert unit == "VND"


def test_compile_rank_returns_value_at_top_k_position() -> None:
    cells = (
        _cell("a", value="500"),
        _cell("b", value="900"),
        _cell("c", value="300"),
    )
    answer, unit = operations.compile_rank(cells, top_k=2)
    assert answer == Decimal("500")
    assert unit == "VND"


def test_compile_rank_top_1_returns_highest_value() -> None:
    cells = (
        _cell("a", value="500"),
        _cell("b", value="900"),
        _cell("c", value="300"),
    )
    answer, _ = operations.compile_rank(cells, top_k=1)
    assert answer == Decimal("900")
