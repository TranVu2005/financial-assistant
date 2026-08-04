from financial_report_qa.normalization._shared import Decision, normalized_key, validate_aliases

METRIC_ALIASES = validate_aliases(
    {
        "Doanh thu bán hàng và cung cấp dịch vụ": "revenue",
        "Revenue": "revenue",
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ": "net_revenue",
        "Net revenue": "net_revenue",
        "Lợi nhuận kế toán trước thuế": "profit_before_tax",
        "Profit before tax": "profit_before_tax",
        "Lợi nhuận sau thuế thu nhập doanh nghiệp": "profit_after_tax",
        "Profit after tax": "profit_after_tax",
        "Tổng cộng tài sản": "total_assets",
        "Total assets": "total_assets",
        "Nợ phải trả": "total_liabilities",
        "Total liabilities": "total_liabilities",
        "Vốn chủ sở hữu": "equity",
        "Owners' equity": "equity",
        "Tiền và các khoản tương đương tiền": "cash_and_cash_equivalents",
        "Cash and cash equivalents": "cash_and_cash_equivalents",
        "Lưu chuyển tiền thuần từ hoạt động kinh doanh": "operating_cash_flow",
        "Net cash flows from operating activities": "operating_cash_flow",
        "Lợi nhuận gộp": "gross_profit",
        "Gross profit": "gross_profit",
        "Doanh thu thuần": "net_revenue",
        "Giá vốn hàng bán": "cost_of_goods_sold",
        "Cost of goods sold": "cost_of_goods_sold",
        "Chi phí bán hàng": "selling_expenses",
        "Chi phí quản lý doanh nghiệp": "general_administration_expenses",
        "Lợi nhuận thuần từ hoạt động kinh doanh": "net_operating_profit",
        "Chi phí thuế TNDN hiện hành": "current_income_tax_expense",
        "Chi phí thuế thu nhập doanh nghiệp hiện hành": "current_income_tax_expense",
    }
)

_NON_METRIC_PATTERNS = {
    "stt",
    "số tt",
    "chỉ tiêu",
    "mã số",
    "thuyết minh",
    "ghi chú",
    "tổng cộng",
    "cộng",
    "a. tài sản",
    "b. nợ phải trả",
    "c. vốn chủ sở hữu",
}


def normalize_metric(raw: str | None) -> Decision[str]:
    if raw is None:
        return Decision(value=None)

    key = normalized_key(raw)
    if not key or key in _NON_METRIC_PATTERNS:
        return Decision(value=None)

    canonical = METRIC_ALIASES.get(key)
    if canonical is not None:
        return Decision(value=canonical)

    return Decision(value=None, issue_code="metric_unknown")
