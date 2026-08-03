import pytest

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.metrics import normalize_metric


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Doanh thu bán hàng và cung cấp dịch vụ", "revenue"),
        ("Doanh thu thuần về bán hàng và cung cấp dịch vụ", "net_revenue"),
        ("Lợi nhuận kế toán trước thuế", "profit_before_tax"),
        ("Lợi nhuận sau thuế thu nhập doanh nghiệp", "profit_after_tax"),
        ("Tổng cộng tài sản", "total_assets"),
        ("Nợ phải trả", "total_liabilities"),
        ("Vốn chủ sở hữu", "equity"),
        ("Tiền và các khoản tương đương tiền", "cash_and_cash_equivalents"),
        ("Lưu chuyển tiền thuần từ hoạt động kinh doanh", "operating_cash_flow"),
    ],
)
def test_metric_aliases(raw: str, expected: str) -> None:
    assert normalize_metric(raw) == Decision(value=expected)


def test_metric_matching_collapses_unicode_and_whitespace_only() -> None:
    assert normalize_metric("  TỔNG   CỘNG TÀI SẢN ") == Decision(value="total_assets")


def test_unknown_metric_is_auditable() -> None:
    assert normalize_metric("Chỉ tiêu chưa ánh xạ") == Decision(
        value=None, issue_code="metric_unknown"
    )
