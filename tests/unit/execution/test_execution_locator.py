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
    statutory_code: str | None = None,
    column_label: str = "Năm 2020",
) -> dict[str, object]:
    return {
        "table_id": TABLE_ID,
        "cell_id": cell_id,
        "company_code": company_code,
        "row_idx": 1,
        "col_idx": 1,
        "row_label_raw": row_label_raw,
        "row_label_canonical": row_label_canonical,
        "column_label": column_label,
        "unit": unit,
        "value": Decimal(value),
        "period": pd.array([period], dtype="Int64")[0],
        "period_inferred": period_inferred,
        "statutory_code": statutory_code,
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


def test_locate_raw_text_match_is_case_and_whitespace_normalization_tolerant() -> None:
    """The raw-text branch must match corpus labels that differ from the
    selector only in casing or collapsible whitespace (e.g. Day 23 grounding
    fallback labels sourced straight from `row_label_raw`), not require
    byte-for-byte string equality."""
    frame = _frame([_row(cell_id=CELL_A, row_label_raw="  Lãi   TIỀN GỬI ", value="50")])
    result = locate(frame, MetricSelector(raw_text="lãi tiền gửi"), 2020)
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


def test_locate_returns_unit_missing_when_resolved_unit_is_null() -> None:
    """Day 20 plan Sec 1.3 / ADR 0009 decision C1: a cell with no recorded
    unit must be reported as `unit_missing`, not stringified into a
    fabricated 'nan' CellMatch and misclassified as `unit_incompatible`
    downstream."""
    frame = _frame([_row(cell_id=CELL_A, value="100", unit=None)])  # type: ignore[arg-type]
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "unit_missing"
    assert result.error_message is not None


def test_locate_ignores_null_unit_duplicate_when_value_and_known_unit_agree() -> None:
    """Day 21 plan §1.7 / ADR 0010 decision C1: `drop_duplicates(subset=["value",
    "unit"])` counted (X, None) and (X, "VND") as two distinct pairs, reporting
    `cell_ambiguous` where the value is identical (measured OCB case:
    2582236224358.0 None vs 2582236224358.0 VND). When exactly one distinct
    value exists and the unit split is only NULL-vs-known, resolve using the
    known unit -- this is not a real conflict."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="2582236224358", unit=None),  # type: ignore[arg-type]
            _row(cell_id=CELL_B, value="2582236224358", unit="VND"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("2582236224358")
    assert result.match.unit == "VND"
    assert set(result.match.cell_ids) == {CELL_A, CELL_B}


def test_locate_still_ambiguous_when_two_known_units_disagree_alongside_null() -> None:
    """The NULL-unit rescue (previous test) must not swallow real unit
    conflicts: when >=2 *known* units are present, still cell_ambiguous."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="100", unit=None),  # type: ignore[arg-type]
            _row(cell_id=CELL_B, value="100", unit="VND"),
            _row(cell_id="cell_" + "c" * 64, value="100", unit="VND_million"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_marks_period_inferred_from_frame() -> None:
    frame = _frame([_row(cell_id=CELL_A, value="900", period_inferred=True)])
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is not None
    assert result.match.period_inferred is True


def test_locate_canonical_matches_a_decorated_raw_label_in_the_locked_release() -> None:
    """Day 23: the locked release baked `row_label_canonical` in at ingestion
    time, so its 57,466 ordinal/footnote-decorated cells ("1. Hàng tồn kho",
    "Tổng tài sản (1)") are still NULL there and unreachable by canonical
    match alone. Re-deriving the canonical from `row_label_raw` at query
    time recovers them without re-ingesting (which would change
    dataset_fingerprint and invalidate every pinned baseline)."""
    frame = _frame(
        [
            _row(
                cell_id=CELL_A,
                row_label_raw="1. Hàng tồn kho",
                row_label_canonical=None,
                value="700",
            )
        ]
    )
    result = locate(frame, MetricSelector(canonical="inventory"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("700")


def test_locate_canonical_fallback_does_not_match_an_unrelated_raw_label() -> None:
    frame = _frame(
        [
            _row(
                cell_id=CELL_A,
                row_label_raw="1. Một chỉ tiêu không có trong từ điển",
                row_label_canonical=None,
                value="700",
            )
        ]
    )
    result = locate(frame, MetricSelector(canonical="inventory"), 2020)
    assert result.match is None
    assert result.error_code == "metric_not_found"


def test_locate_canonical_column_still_wins_when_already_populated() -> None:
    """The existing canonical column stays authoritative; the raw-label
    fallback only adds rows it never covered."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, row_label_canonical="inventory", value="500"),
            _row(
                cell_id=CELL_B,
                row_label_raw="1. Hàng tồn kho",
                row_label_canonical="inventory",
                value="500",
            ),
        ]
    )
    result = locate(frame, MetricSelector(canonical="inventory"), 2020)
    assert result.error_code is None
    assert result.match is not None
    assert set(result.match.cell_ids) == {CELL_A, CELL_B}


def test_locate_prefers_statutory_row_only_when_explicitly_requested() -> None:
    """A statement line and a note-table line can share a label and disagree.
    Only the statement line carries a Circular 200 code (8,449 of 146,011
    tables have the column at all), so the code identifies which one the
    question means -- resolving a conflict that is otherwise `cell_ambiguous`."""
    frame = _frame(
        [
            _row(
                cell_id=CELL_A,
                row_label_canonical="general_administration_expenses",
                value="1000",
                statutory_code="26",
            ),
            _row(
                cell_id=CELL_B,
                row_label_canonical="general_administration_expenses",
                value="250",
                statutory_code=None,
            ),
        ]
    )
    result = locate(
        frame,
        MetricSelector(canonical="general_administration_expenses"),
        2020,
        prefer_statutory_rows=True,
    )
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("1000")
    assert result.match.cell_ids == (CELL_A,)


def test_locate_keeps_main_statement_and_note_conflict_ambiguous_by_default() -> None:
    """A statutory code identifies the main-statement row but does not prove
    that the question intended that source instead of the conflicting note."""
    frame = _frame(
        [
            _row(
                cell_id=CELL_A,
                row_label_canonical="general_administration_expenses",
                value="1000",
                statutory_code="26",
            ),
            _row(
                cell_id=CELL_B,
                row_label_canonical="general_administration_expenses",
                value="250",
                statutory_code=None,
            ),
        ]
    )

    result = locate(frame, MetricSelector(canonical="general_administration_expenses"), 2020)

    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_stays_ambiguous_when_two_statutory_rows_disagree() -> None:
    """The rule narrows to the statement rows; it never picks among them.
    Two coded rows disagreeing is a real conflict (consolidated vs separate,
    measured at 92.8% of cross-scope groups), not a main-versus-note artifact."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="1000", statutory_code="26"),
            _row(cell_id=CELL_B, value="250", statutory_code="26"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_stays_ambiguous_when_no_row_carries_a_statutory_code() -> None:
    """Note tables have no code column, so a conflict between two note rows
    has no code signal to break it and must keep failing loudly."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="1000", statutory_code=None),
            _row(cell_id=CELL_B, value="250", statutory_code=None),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_uses_the_column_selector_to_separate_same_named_columns() -> None:
    """Real PC1 2025 tax table: one row "Thuế giá trị gia tăng" spans six
    semantically distinct columns. Question 283 asks for the amount *payable*
    at year end (60,792,014,797 under "Số phải nộp cuối năm"), but the closing
    balance column also resolves to 2025, so filtering by period alone leaves
    two conflicting cells and abstains. The column is the missing dimension."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="38137021285", column_label="Số cuối nămVND"),
            _row(cell_id=CELL_B, value="60792014797", column_label="Số phải nộpcuối năm"),
        ]
    )
    result = locate(
        frame,
        MetricSelector(canonical="cash_and_cash_equivalents", column_text="phải nộp cuối năm"),
        2020,
    )
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("60792014797")
    assert result.match.cell_ids == (CELL_B,)


def test_locate_without_a_column_selector_keeps_reporting_the_conflict() -> None:
    """No column named in the question means no basis to choose; the two
    columns still conflict and must stay `cell_ambiguous`, not pick the first."""
    frame = _frame(
        [
            _row(cell_id=CELL_A, value="38137021285", column_label="Số cuối nămVND"),
            _row(cell_id=CELL_B, value="60792014797", column_label="Số phải nộpcuối năm"),
        ]
    )
    result = locate(frame, MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is None
    assert result.error_code == "cell_ambiguous"


def test_locate_reports_metric_not_found_when_the_column_selector_matches_nothing() -> None:
    """A column the table does not have is a typed failure, never a silent
    fallback to whatever column happens to be present."""
    frame = _frame([_row(cell_id=CELL_A, value="100", column_label="Số cuối năm")])
    result = locate(
        frame,
        MetricSelector(canonical="cash_and_cash_equivalents", column_text="số đã thực nộp"),
        2020,
    )
    assert result.match is None
    assert result.error_code == "metric_not_found"


def test_locate_carries_row_index_and_column_provenance() -> None:
    """plan.md §9: a resolved cell must report the exact row it came from, so
    grounding can pin the fact positionally instead of by label string."""
    row = _row(cell_id=CELL_A, value="100")
    row["row_idx"] = 14
    result = locate(_frame([row]), MetricSelector(canonical="cash_and_cash_equivalents"), 2020)
    assert result.match is not None
    assert result.match.row_index == 14
    assert result.match.column_label == "Năm 2020"


def test_locate_resolves_by_row_index_when_selector_is_position_bound() -> None:
    """plan.md §14: once grounding has pinned the row, Pandas does positional
    extraction only -- two rows sharing a label are no longer ambiguous."""
    first = _row(cell_id=CELL_A, value="100")
    first["row_idx"] = 3
    second = _row(cell_id=CELL_B, value="900")
    second["row_idx"] = 14
    selector = MetricSelector(
        canonical="cash_and_cash_equivalents", table_id=TABLE_ID, row_index=14
    )
    result = locate(_frame([first, second]), selector, 2020)
    assert result.error_code is None
    assert result.match is not None
    assert result.match.value == Decimal("900")
    assert result.match.row_index == 14


def test_locate_by_row_index_ignores_the_label_entirely() -> None:
    """The bound row wins even when its label does not match the selector's --
    semantic matching happened at grounding time, not here."""
    row = _row(cell_id=CELL_A, row_label_canonical="revenue", row_label_raw="Doanh thu", value="55")
    row["row_idx"] = 14
    selector = MetricSelector(
        canonical="cash_and_cash_equivalents", table_id=TABLE_ID, row_index=14
    )
    result = locate(_frame([row]), selector, 2020)
    assert result.match is not None
    assert result.match.value == Decimal("55")


def test_locate_by_row_index_reports_metric_not_found_when_position_is_absent() -> None:
    """Positional grounding never falls back to guessing by label."""
    row = _row(cell_id=CELL_A, value="100")
    row["row_idx"] = 3
    selector = MetricSelector(
        canonical="cash_and_cash_equivalents", table_id=TABLE_ID, row_index=14
    )
    result = locate(_frame([row]), selector, 2020)
    assert result.match is None
    assert result.error_code == "metric_not_found"


def test_locate_by_row_index_is_scoped_to_its_own_table() -> None:
    other_table = "tbl_" + "2" * 64
    row = _row(cell_id=CELL_A, value="100")
    row["row_idx"] = 14
    row["table_id"] = other_table
    selector = MetricSelector(
        canonical="cash_and_cash_equivalents", table_id=TABLE_ID, row_index=14
    )
    result = locate(_frame([row]), selector, 2020)
    assert result.match is None
    assert result.error_code == "metric_not_found"
