"""Day 18 per-operation compiler functions (ADR 0007 decision E1).

Each `PlanOperation` gets its own public function so that one operation's
arithmetic can never accidentally reuse another's. All unit conversion is
delegated to `normalization/units.py::convert_scale`, unchanged from Day 5 —
it already raises `ValueError` on an incompatible conversion (e.g. currency
vs. percent), which the caller maps to the `unit_incompatible` error code.
Division uses plain `/`, so a zero denominator raises the built-in
`ZeroDivisionError`, mapped by the caller to `division_by_zero`.
"""

from __future__ import annotations

from decimal import Decimal
from typing import get_args

from financial_report_qa.execution.contracts import CellMatch
from financial_report_qa.normalization.units import CanonicalUnit, convert_scale

_CANONICAL_UNITS = frozenset(get_args(CanonicalUnit))


def _as_canonical_unit(unit: str) -> CanonicalUnit:
    if unit not in _CANONICAL_UNITS:
        raise ValueError(f"'{unit}' is not a canonical unit")
    return unit  # type: ignore[return-value]


def convert_cell_value(value: Decimal, source: str, target: str) -> Decimal:
    """Convert one stored cell value between canonical units (ADR 0007 E1)."""
    return convert_scale(value, _as_canonical_unit(source), _as_canonical_unit(target))


def compile_lookup(cell: CellMatch) -> tuple[Decimal, CanonicalUnit]:
    return cell.value, _as_canonical_unit(cell.unit)


def compile_difference(end: CellMatch, start: CellMatch) -> tuple[Decimal, CanonicalUnit]:
    start_converted = convert_cell_value(start.value, start.unit, end.unit)
    return end.value - start_converted, _as_canonical_unit(end.unit)


def compile_growth_rate(end: CellMatch, start: CellMatch) -> tuple[Decimal, CanonicalUnit]:
    start_converted = convert_cell_value(start.value, start.unit, end.unit)
    if start_converted == 0:
        raise ZeroDivisionError("growth_rate base period value is zero")
    return (end.value - start_converted) / abs(start_converted), "ratio"


def compile_compare(metric_a: CellMatch, metric_b: CellMatch) -> tuple[Decimal, CanonicalUnit]:
    b_converted = convert_cell_value(metric_b.value, metric_b.unit, metric_a.unit)
    return metric_a.value - b_converted, _as_canonical_unit(metric_a.unit)


def compile_compare_companies(
    company_a: CellMatch, company_b: CellMatch
) -> tuple[Decimal, CanonicalUnit]:
    b_converted = convert_cell_value(company_b.value, company_b.unit, company_a.unit)
    return company_a.value - b_converted, _as_canonical_unit(company_a.unit)


def compile_ratio(numerator: CellMatch, denominator: CellMatch) -> tuple[Decimal, CanonicalUnit]:
    denominator_converted = convert_cell_value(denominator.value, denominator.unit, numerator.unit)
    if denominator_converted == 0:
        raise ZeroDivisionError("ratio denominator is zero")
    return numerator.value / denominator_converted, "ratio"


def compile_average(cells: tuple[CellMatch, ...]) -> tuple[Decimal, CanonicalUnit]:
    target_unit = cells[0].unit
    converted = [convert_cell_value(cell.value, cell.unit, target_unit) for cell in cells]
    return sum(converted, start=Decimal(0)) / len(converted), _as_canonical_unit(target_unit)


def compile_sum(cells: tuple[CellMatch, ...]) -> tuple[Decimal, CanonicalUnit]:
    target_unit = cells[0].unit
    converted = [convert_cell_value(cell.value, cell.unit, target_unit) for cell in cells]
    return sum(converted, start=Decimal(0)), _as_canonical_unit(target_unit)


def compile_rank(cells: tuple[CellMatch, ...], *, top_k: int) -> tuple[Decimal, CanonicalUnit]:
    target_unit = cells[0].unit
    converted = sorted(
        (convert_cell_value(cell.value, cell.unit, target_unit) for cell in cells),
        reverse=True,
    )
    return converted[top_k - 1], _as_canonical_unit(target_unit)
