import pytest

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.periods import has_period_evidence, normalize_period


@pytest.mark.parametrize(
    ("raw", "report_year", "canonical"),
    [
        ("31/12/2024", 2024, "2024-12-31"),
        ("Ngày 01-01-2025", 2024, "2025-01-01"),
        ("31/12/24", 2024, "2024-12-31"),
        ("Quý IV năm 2024", 2024, "2024-Q4"),
        ("Q2/2023", 2024, "2023-Q2"),
        ("Quý I", 2024, "2024-Q1"),
        ("Năm 2024", 2023, "2024"),
        ("Tháng 3", 2024, "2024-03"),
    ],
)
def test_normalize_period_supports_common_report_headers(
    raw: str, report_year: int, canonical: str
) -> None:
    decision = normalize_period(raw, report_year)

    assert decision.value == canonical
    assert decision.issue_code is None


def test_normalize_period_preserves_ambiguous_two_digit_date() -> None:
    decision = normalize_period("01/02/24", report_year=2024)

    assert decision.value is None
    assert decision.issue_code == "period_ambiguous"


@pytest.mark.parametrize("raw", ["31/02/2024", "29/02/2023"])
def test_normalize_period_reports_invalid_calendar_dates(raw: str) -> None:
    decision = normalize_period(raw, report_year=2024)

    assert decision.value is None
    assert decision.issue_code == "period_invalid"


@pytest.mark.parametrize("raw", ["Quý", "Tháng 13"])
def test_normalize_period_reports_incomplete_period_evidence(raw: str) -> None:
    decision = normalize_period(raw, report_year=2024)

    assert decision.value is None
    assert decision.issue_code == "period_incomplete"


@pytest.mark.parametrize(
    ("raw", "report_year", "canonical"),
    [
        ("Năm nay", 2024, "2024"),
        ("Năm trước", 2024, "2023"),
        ("Năm nay VND", 2024, "2024"),
        ("Năm trướcVND", 2024, "2023"),
        ("Năm 2024VND", 2025, "2024"),
        ("Năm 2024\nTriệu đồng", 2025, "2024"),
    ],
)
def test_normalize_period_resolves_relative_and_composite_years(
    raw: str, report_year: int, canonical: str
) -> None:
    assert normalize_period(raw, report_year) == Decision(value=canonical)


def test_non_period_year_phrase_remains_incomplete() -> None:
    assert normalize_period("Năm hết hiệu lực", 2019) == Decision(
        value=None, issue_code="period_incomplete"
    )


@pytest.mark.parametrize(
    ("raw", "report_year", "canonical"),
    [
        ("Năm kết thúc ngày 30/9/2018VND", 2020, "2018-09-30"),
        ("Năm kết thúc 31/12/2015Triệu VND", 2020, "2015-12-31"),
        ("Năm tài chính kết thúc ngày 31 tháng 12 năm 2023Triệu đồng", 2024, "2023"),
        ("Năm 2025 Nước ngoài Triệu VND", 2024, "2025"),
        ("Năm trước Giá trị theo mệnh giá VND", 2024, "2023"),
    ],
)
def test_normalize_period_extracts_year_from_noisy_composite_headers(
    raw: str, report_year: int, canonical: str
) -> None:
    assert normalize_period(raw, report_year) == Decision(value=canonical)


@pytest.mark.parametrize("raw", ["", "Chỉ tiêu", "Mã số", "Doanh thu thuần"])
def test_normalize_period_ignores_non_period_labels(raw: str) -> None:
    decision = normalize_period(raw, report_year=2024)

    assert decision.value is None
    assert decision.issue_code is None


def test_has_period_evidence_is_conservative() -> None:
    assert has_period_evidence("Quý IV năm 2024") is True
    assert has_period_evidence("31/12/2024") is True
    assert has_period_evidence("Doanh thu 2024") is False
    assert has_period_evidence(None) is False


def test_normalize_period_combines_multilevel_header_components() -> None:
    decision = normalize_period("Kỳ báo cáo\nQuý IV\nNăm 2024", report_year=2024)

    assert decision.value == "2024-Q4"
    assert decision.issue_code is None


def test_normalize_period_accepts_balance_sheet_instant_prefix() -> None:
    decision = normalize_period("Tại ngày 31/12/2024", report_year=2024)

    assert decision.value == "2024-12-31"
    assert decision.issue_code is None
