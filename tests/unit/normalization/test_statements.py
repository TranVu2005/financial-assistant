import pytest

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.statements import normalize_statement_type


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bảng cân đối kế toán", "balance_sheet"),
        ("Báo cáo kết quả hoạt động kinh doanh", "income_statement"),
        ("Báo cáo lưu chuyển tiền tệ", "cash_flow_statement"),
        ("Báo cáo thay đổi vốn chủ sở hữu", "equity_changes"),
        ("Thuyết minh báo cáo tài chính", "notes"),
    ],
)
def test_statement_aliases(raw: str, expected: str) -> None:
    assert normalize_statement_type(raw) == Decision(value=expected)


def test_statement_conflict_is_not_guessed() -> None:
    assert normalize_statement_type(
        "Bảng cân đối kế toán / Báo cáo lưu chuyển tiền tệ"
    ) == Decision(value=None, issue_code="statement_conflict")
