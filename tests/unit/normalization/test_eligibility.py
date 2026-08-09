from decimal import Decimal

from financial_report_qa.normalization.eligibility import classify_cell_eligibility
from financial_report_qa.schemas.normalization import NormalizationIssueCode
from financial_report_qa.schemas.tables import CellRecord


def _cell(
    *,
    metric: str | None = "net_revenue",
    period: str | None = "2024",
    value_raw: str = "100",
    value_numeric: Decimal | None = Decimal("100"),
    unit: str | None = "VND_million",
) -> CellRecord:
    return CellRecord(
        cell_id="cell_" + "a" * 64,
        table_id="tbl_" + "b" * 64,
        row_idx=0,
        col_idx=0,
        row_label_raw="Doanh thu thuần",
        row_label_canonical=metric,
        column_label_raw="2024",
        column_label_canonical=period,
        value_raw=value_raw,
        value_numeric=value_numeric,
        period=period,
        unit=unit,
        source_line_start=1,
        source_line_end=1,
        extraction_confidence=1.0,
    )


def _classify(
    cell: CellRecord, issue_codes: tuple[NormalizationIssueCode, ...] = ()
) -> tuple[bool, bool, bool, tuple[str, ...]]:
    result = classify_cell_eligibility(cell, issue_codes)
    return (
        result.searchable,
        result.comparable,
        result.calculable,
        result.blocking_reasons,
    )


def test_complete_monetary_cell_is_comparable_and_calculable() -> None:
    assert _classify(_cell()) == (False, True, True, ())


def test_complete_non_monetary_cell_is_comparable_but_not_monetary_calculable() -> None:
    assert _classify(_cell(unit="percent")) == (False, True, False, ())


def test_missing_unit_keeps_comparison_but_blocks_monetary_calculation() -> None:
    assert _classify(_cell(unit=None)) == (False, True, False, ())


def test_blocking_issue_disables_comparison_and_calculation() -> None:
    assert _classify(_cell(), ("number_ambiguous",)) == (
        False,
        False,
        False,
        ("number_ambiguous",),
    )


def test_unparsed_labeled_raw_cell_remains_searchable() -> None:
    assert _classify(_cell(value_raw="chưa kiểm toán", value_numeric=None, unit=None)) == (
        True,
        False,
        False,
        (),
    )


def test_unlabeled_or_empty_cell_is_not_eligible() -> None:
    assert _classify(_cell(metric=None, value_numeric=None, unit=None)) == (
        False,
        False,
        False,
        (),
    )
    assert _classify(_cell(value_raw="  ", value_numeric=None, unit=None)) == (
        False,
        False,
        False,
        (),
    )


def test_non_blocking_issue_does_not_hide_valid_cell() -> None:
    assert _classify(_cell(), ("metric_unknown",)) == (False, True, True, ())
