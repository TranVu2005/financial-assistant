import pytest

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.periods import has_period_evidence, normalize_period


@pytest.mark.parametrize(
    ("raw", "report_year", "expected"),
    [
        ("2024", 2024, Decision(value="2024")),
        ("Năm 2023", 2024, Decision(value="2023")),
        ("Quý IV/2024", 2024, Decision(value="2024-Q4")),
        ("Q1 2022", 2024, Decision(value="2022-Q1")),
        ("Quý 2", 2024, Decision(value="2024-Q2")),
        ("31/12/2024", 2024, Decision(value="2024-12-31")),
        ("31-02-2024", 2024, Decision(value=None, issue_code="period_invalid")),
        ("12/11/24", 2024, Decision(value=None, issue_code="period_ambiguous")),
        ("Tháng 12", 2024, Decision(value="2024-12")),
        ("Tháng", 2024, Decision(value=None, issue_code="period_incomplete")),
        ("Chỉ tiêu", 2024, Decision(value=None)),
    ],
)
def test_normalize_period(raw: str, report_year: int, expected: Decision[str]) -> None:
    assert normalize_period(raw, report_year) == expected


def test_period_evidence_rejects_generic_headers() -> None:
    assert has_period_evidence("Giá trị") is False
    assert has_period_evidence("Số tiền") is False
    assert has_period_evidence("2024") is True
    assert has_period_evidence("Tháng 12") is True
