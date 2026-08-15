"""Tests for the Day 18 locator (ADR 0007 decision D1: never guess)."""

from decimal import Decimal

import pandas as pd

from financial_report_qa.execution.locator import locate
from financial_report_qa.planning.plan_contracts import MetricSelector

TABLE_ID = "tbl_" + "1" * 64
CELL_A = "cell_" + "a" * 64
CELL_B = "cell_" + "b" * 64


def _row(
    *,
    cell_id: str,
    company_code: str = "ACB",
    row_label_raw: str = "Tien mat",
    row_label_canonical: str | None = "cash_and_cash_equivalents",
    value: str,
    unit: str = "VND",
    period: int = 2020,
    period_inferred: bool = False,
) -> dict[str, object]:
    return {
        "table_id": TABLE_ID,
        "cell_id": cell_id,
        "company_code": company_code,
        "row_idx": 1,
        "col_idx": 1,
        "row_label_raw": row_label_raw,
        "row_label_canonical": row_label_canonical,
        "column_label": "Năm 2020",
        "unit": unit,
        "value": Decimal(value),
        "period": pd.array([period], dtype="Int64")[0],
        "period_inferred": period_inferred,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["period"] = frame["period"].astype("Int64")
    return frame


def test_locate_returns_metric_not_found_when_no_row_matches_selector() -> None:
    frame = _frame([_row(cell_id=CELL_A, row_label_canonical="revenue", value="100")])
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "metric_not_found"
    assert result.error_message is not None


def test_locate_returns_period_unresolved_when_metric_exists_at_other_period() -> None:
    frame = _frame([_row(cell_id=CELL_A, value="100", period=2019)])
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "period_unresolved"


def test_locate_returns_single_match() -> None:
    frame = _frame([_row(cell_id=CELL_A, value="100")])
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("100")
    assert result.match.cell_ids == (CELL_A,)
    assert result.match.unit == "VND"
    assert result.match.period == 2020
    assert result.match.period_inferred is False


def test_locate_merges_duplicate_rows_that_agree() -> None:
    """Day 18 plan §1.4: 4.67% of row groups have >=2 physical cells; when they
    agree the evidence must carry every cell_id, not just the first."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="100"),
            _row(cell_id=CELL_B, value="100"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("100")
    assert set(result.match.cell_ids) == {CELL_A, CELL_B}


def test_locate_returns_cell_ambiguous_when_duplicate_rows_conflict() -> None:
    """Day 18 plan §1.4: 93.2% of duplicate-row groups have conflicting values;
    picking the first row would silently produce a wrong answer."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="100"),
            _row(cell_id=CELL_B, value="200"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"
    assert CELL_A in (result.error_message or "")
    assert CELL_B in (result.error_message or "")


def test_locate_treats_same_value_different_unit_as_ambiguous() -> None:
    """A numerically-equal value under a different unit is not the same
    economic value (100 VND != 100 VND_million); it must not be silently
    collapsed into one match."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="100", unit="VND"),
            _row(cell_id=CELL_B, value="100", unit="VND_million"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_uses_raw_text_branch_of_selector() -> None:
    frame = _frame([_row(cell_id=CELL_A, row_label_raw="tiền gửi có kỳ hạn", value="50")])
    result = locate(frame, MetricSelector(raw_text="tiền gửi có kỳ hạn"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("50")


def test_locate_filters_by_company_code_for_compare_companies() -> None:
    frame = _frame(
        [
            _row(cell_id=CELL_A, company_code="ACB", value="100"),
            _row(cell_id=CELL_B, company_code="MBB", value="999"),
        ]
    )
    result = locate(
        frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020, company_code="ACB"
    )
    assert result.match is not None
    assert result.match.value == Decimal("100")
    assert result.match.cell_ids == (CELL_A,)


def test_locate_marks_period_inferred_from_frame() -> None:
    frame = _frame([_row(cell_id=CELL_A, value="900", period_inferred=True)])
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is not None
    assert result.match.period_inferred is True
