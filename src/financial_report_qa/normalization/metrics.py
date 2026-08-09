"""Canonical financial metrics and conservative Vietnamese/English aliases.

Only metrics reported by a source table belong in ``METRIC_ALIASES``. Ratios and
other calculated values are listed separately in ``DERIVED_METRICS`` so the
normalizer never presents a calculated value as if it came directly from a
source cell.
"""

import re

from financial_report_qa.normalization._shared import (
    Decision,
    normalized_key,
    validate_aliases,
)

INCOME_STATEMENT_ALIASES = {
    "Doanh thu bán hàng và cung cấp dịch vụ": "revenue",
    "Doanh thu bán hàng": "revenue",
    "Doanh thu bán hàng và CCDV": "revenue",
    "Revenue": "revenue",
    "Sales revenue": "revenue",
    "Doanh thu thuần": "net_revenue",
    "Doanh thu thuần về bán hàng và cung cấp dịch vụ": "net_revenue",
    "Doanh thu thuần bán hàng và cung cấp dịch vụ": "net_revenue",
    "Net revenue": "net_revenue",
    "Net sales": "net_revenue",
    "Các khoản giảm trừ doanh thu": "sales_deductions",
    "Các khoản giảm trừ": "sales_deductions",
    "Sales deductions": "sales_deductions",
    "Giá vốn hàng bán": "cost_of_goods_sold",
    "Giá vốn": "cost_of_goods_sold",
    "Cost of goods sold": "cost_of_goods_sold",
    "Cost of sales": "cost_of_goods_sold",
    "Lợi nhuận gộp": "gross_profit",
    "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ": "gross_profit",
    "Lãi gộp": "gross_profit",
    "LN gộp": "gross_profit",
    "Gross profit": "gross_profit",
    "Doanh thu hoạt động tài chính": "financial_income",
    "Doanh thu tài chính": "financial_income",
    "Doanh thu HĐTC": "financial_income",
    "Financial income": "financial_income",
    "Chi phí tài chính": "financial_expenses",
    "Financial expenses": "financial_expenses",
    "Trong đó: Chi phí lãi vay": "interest_expense",
    "Trong đó chi phí lãi vay": "interest_expense",
    "Chi phí lãi vay": "interest_expense",
    "Lãi vay phải trả": "interest_expense",
    "Interest expense": "interest_expense",
    "Chi phí bán hàng": "selling_expenses",
    "Selling expenses": "selling_expenses",
    "Chi phí quản lý doanh nghiệp": "general_administration_expenses",
    "Chi phí quản lý": "general_administration_expenses",
    "Chi phí QLDN": "general_administration_expenses",
    "General and administrative expenses": "general_administration_expenses",
    "General administration expenses": "general_administration_expenses",
    "Chi phí bán hàng và QLDN": "operating_expenses",
    "Chi phí bán hàng và quản lý doanh nghiệp": "operating_expenses",
    "Operating expenses": "operating_expenses",
    "Lợi nhuận thuần từ hoạt động kinh doanh": "operating_profit",
    "Lợi nhuận thuần từ HĐKD": "operating_profit",
    "Lợi nhuận hoạt động kinh doanh": "operating_profit",
    "Operating profit": "operating_profit",
    "Thu nhập khác": "other_income",
    "Other income": "other_income",
    "Chi phí khác": "other_expenses",
    "Other expenses": "other_expenses",
    "Lợi nhuận khác": "other_profit",
    "Other profit": "other_profit",
    "Lợi nhuận kế toán trước thuế": "profit_before_tax",
    "Lợi nhuận trước thuế": "profit_before_tax",
    "Lãi trước thuế": "profit_before_tax",
    "LNTT": "profit_before_tax",
    "Profit before tax": "profit_before_tax",
    "Chi phí thuế thu nhập doanh nghiệp hiện hành": "current_income_tax_expense",
    "Chi phí thuế TNDN hiện hành": "current_income_tax_expense",
    "Current income tax expense": "current_income_tax_expense",
    "Chi phí thuế thu nhập doanh nghiệp hoãn lại": "deferred_income_tax_expense",
    "Chi phí thuế TNDN hoãn lại": "deferred_income_tax_expense",
    "Deferred income tax expense": "deferred_income_tax_expense",
    "Lợi nhuận sau thuế thu nhập doanh nghiệp": "profit_after_tax",
    "Lợi nhuận sau thuế": "profit_after_tax",
    "Lãi sau thuế": "profit_after_tax",
    "LNST": "profit_after_tax",
    "Profit after tax": "profit_after_tax",
    "Net profit after tax": "profit_after_tax",
    "Lợi nhuận sau thuế của cổ đông công ty mẹ": "profit_attributable_to_parent",
    "Lợi nhuận sau thuế thuộc về cổ đông công ty mẹ": "profit_attributable_to_parent",
    "Lợi nhuận thuộc về cổ đông của công ty mẹ": "profit_attributable_to_parent",
    "Profit attributable to owners of the parent": "profit_attributable_to_parent",
    "Lợi ích cổ đông không kiểm soát": "non_controlling_interest",
    "Lợi ích của cổ đông không kiểm soát": "non_controlling_interest",
    "Lợi nhuận sau thuế của cổ đông không kiểm soát": "non_controlling_interest",
    "Non-controlling interests": "non_controlling_interest",
    "EPS cơ bản": "basic_eps",
    "Lãi cơ bản trên cổ phiếu": "basic_eps",
    "Lãi cơ bản trên cổ phiếu (VNĐ)": "basic_eps",
    "Basic earnings per share": "basic_eps",
    "Basic EPS": "basic_eps",
    "EPS pha loãng": "diluted_eps",
    "Lãi suy giảm trên cổ phiếu": "diluted_eps",
    "Lãi pha loãng trên cổ phiếu": "diluted_eps",
    "Diluted earnings per share": "diluted_eps",
    "Diluted EPS": "diluted_eps",
}


BALANCE_SHEET_ALIASES = {
    "Tổng cộng tài sản": "total_assets",
    "Tổng tài sản": "total_assets",
    "Total assets": "total_assets",
    "Tài sản ngắn hạn": "current_assets",
    "Tổng tài sản ngắn hạn": "current_assets",
    "Current assets": "current_assets",
    "Tài sản dài hạn": "non_current_assets",
    "Tổng tài sản dài hạn": "non_current_assets",
    "Tài sản không ngắn hạn": "non_current_assets",
    "Non-current assets": "non_current_assets",
    "Tiền và các khoản tương đương tiền": "cash_and_cash_equivalents",
    "Tiền và tương đương tiền": "cash_and_cash_equivalents",
    "Cash and cash equivalents": "cash_and_cash_equivalents",
    "Các khoản phải thu ngắn hạn": "accounts_receivable",
    "Phải thu ngắn hạn": "accounts_receivable",
    "Các khoản phải thu": "accounts_receivable",
    "Khoản phải thu": "accounts_receivable",
    "Accounts receivable": "accounts_receivable",
    "Short-term receivables": "accounts_receivable",
    "Hàng tồn kho": "inventory",
    "HTK": "inventory",
    "Inventories": "inventory",
    "Inventory": "inventory",
    "Đầu tư tài chính ngắn hạn": "short_term_investments",
    "Các khoản đầu tư tài chính ngắn hạn": "short_term_investments",
    "Short-term investments": "short_term_investments",
    "Tài sản cố định": "fixed_assets",
    "TSCĐ": "fixed_assets",
    "Fixed assets": "fixed_assets",
    "Tài sản cố định hữu hình": "property_plant_equipment",
    "TSCĐ hữu hình": "property_plant_equipment",
    "Property, plant and equipment": "property_plant_equipment",
    "Tài sản cố định vô hình": "intangible_assets",
    "Tài sản vô hình": "intangible_assets",
    "TSCĐ vô hình": "intangible_assets",
    "Intangible assets": "intangible_assets",
    "Chi phí xây dựng cơ bản dở dang": "construction_in_progress",
    "Xây dựng cơ bản dở dang": "construction_in_progress",
    "XDCB dở dang": "construction_in_progress",
    "Construction in progress": "construction_in_progress",
    "Đầu tư tài chính dài hạn": "long_term_investments",
    "Các khoản đầu tư tài chính dài hạn": "long_term_investments",
    "Long-term investments": "long_term_investments",
    "Nợ phải trả": "total_liabilities",
    "Tổng nợ phải trả": "total_liabilities",
    "Tổng nợ": "total_liabilities",
    "Total liabilities": "total_liabilities",
    "Nợ ngắn hạn": "current_liabilities",
    "Tổng nợ ngắn hạn": "current_liabilities",
    "Current liabilities": "current_liabilities",
    "Nợ dài hạn": "non_current_liabilities",
    "Tổng nợ dài hạn": "non_current_liabilities",
    "Nợ phải trả dài hạn": "non_current_liabilities",
    "Non-current liabilities": "non_current_liabilities",
    "Vay và nợ thuê tài chính ngắn hạn": "short_term_debt",
    "Vay ngắn hạn": "short_term_debt",
    "Nợ vay ngắn hạn": "short_term_debt",
    "Short-term borrowings": "short_term_debt",
    "Short-term debt": "short_term_debt",
    "Vay và nợ thuê tài chính dài hạn": "long_term_debt",
    "Vay dài hạn": "long_term_debt",
    "Nợ vay dài hạn": "long_term_debt",
    "Long-term borrowings": "long_term_debt",
    "Long-term debt": "long_term_debt",
    "Tổng nợ vay": "total_debt",
    "Tổng vay và nợ thuê tài chính": "total_debt",
    "Total debt": "total_debt",
    "Phải trả người bán": "accounts_payable",
    "Phải trả người bán ngắn hạn": "accounts_payable",
    "Khoản phải trả": "accounts_payable",
    "Accounts payable": "accounts_payable",
    "Trade payables": "accounts_payable",
    "Vốn chủ sở hữu": "equity",
    "VCSH": "equity",
    "Nguồn vốn chủ sở hữu": "equity",
    "Owners' equity": "equity",
    "Owner's equity": "equity",
    "Shareholders' equity": "equity",
    "Vốn góp của chủ sở hữu": "share_capital",
    "Vốn đầu tư của chủ sở hữu": "share_capital",
    "Vốn cổ phần": "share_capital",
    "Share capital": "share_capital",
    "Lợi nhuận sau thuế chưa phân phối": "retained_earnings",
    "Lợi nhuận chưa phân phối": "retained_earnings",
    "LNST chưa phân phối": "retained_earnings",
    "Retained earnings": "retained_earnings",
    # This is the balance-sheet total, not an alias for total_assets. The two
    # values should reconcile, but preserving their distinct source labels is
    # necessary for provenance and accounting-identity checks.
    "Tổng cộng nguồn vốn": "total_equity_and_liabilities",
    "Tổng nguồn vốn": "total_equity_and_liabilities",
    "Total equity and liabilities": "total_equity_and_liabilities",
}


CASH_FLOW_ALIASES = {
    "Lưu chuyển tiền thuần từ hoạt động kinh doanh": "operating_cash_flow",
    "Lưu chuyển tiền thuần từ HĐKD": "operating_cash_flow",
    "Tiền thuần từ hoạt động kinh doanh": "operating_cash_flow",
    "Net cash flows from operating activities": "operating_cash_flow",
    "Net cash from operating activities": "operating_cash_flow",
    "Lưu chuyển tiền thuần từ hoạt động đầu tư": "investing_cash_flow",
    "Lưu chuyển tiền thuần từ HĐĐT": "investing_cash_flow",
    "Tiền thuần từ hoạt động đầu tư": "investing_cash_flow",
    "Net cash flows from investing activities": "investing_cash_flow",
    "Net cash from investing activities": "investing_cash_flow",
    "Lưu chuyển tiền thuần từ hoạt động tài chính": "financing_cash_flow",
    "Lưu chuyển tiền thuần từ HĐTC": "financing_cash_flow",
    "Tiền thuần từ hoạt động tài chính": "financing_cash_flow",
    "Net cash flows from financing activities": "financing_cash_flow",
    "Net cash from financing activities": "financing_cash_flow",
    "Lưu chuyển tiền thuần trong kỳ": "net_cash_flow",
    "Lưu chuyển tiền và tương đương tiền thuần trong kỳ": "net_cash_flow",
    "Tiền và tương đương tiền tăng giảm trong kỳ": "net_cash_flow",
    "Net increase decrease in cash and cash equivalents": "net_cash_flow",
    "Net cash flow": "net_cash_flow",
    "Tiền và tương đương tiền đầu kỳ": "cash_beginning",
    "Tiền và các khoản tương đương tiền đầu kỳ": "cash_beginning",
    "Cash and cash equivalents at beginning of period": "cash_beginning",
    "Tiền và tương đương tiền cuối kỳ": "cash_ending",
    "Tiền và các khoản tương đương tiền cuối kỳ": "cash_ending",
    "Cash and cash equivalents at end of period": "cash_ending",
    "Khấu hao tài sản cố định": "depreciation",
    "Khấu hao TSCĐ": "depreciation",
    "Khấu hao TSCĐ và BĐSĐT": "depreciation",
    "Depreciation": "depreciation",
    "Chi trả lãi vay": "interest_paid",
    "Tiền lãi vay đã trả": "interest_paid",
    "Lãi vay đã trả": "interest_paid",
    "Interest paid": "interest_paid",
    "Thuế thu nhập doanh nghiệp đã nộp": "income_tax_paid",
    "Thuế TNDN đã nộp": "income_tax_paid",
    "Income tax paid": "income_tax_paid",
    "Tiền chi để mua sắm xây dựng TSCĐ và các tài sản dài hạn khác": "capital_expenditure",
    "Mua sắm xây dựng TSCĐ và các tài sản dài hạn khác": "capital_expenditure",
    "Purchase of fixed assets and other long-term assets": "capital_expenditure",
    "Capital expenditure": "capital_expenditure",
}


METRIC_ALIASES = validate_aliases(
    INCOME_STATEMENT_ALIASES | BALANCE_SHEET_ALIASES | CASH_FLOW_ALIASES
)

SOURCE_METRICS_BY_STATEMENT = {
    "income_statement": frozenset(INCOME_STATEMENT_ALIASES.values()),
    "balance_sheet": frozenset(BALANCE_SHEET_ALIASES.values()),
    "cash_flow_statement": frozenset(CASH_FLOW_ALIASES.values()),
}

# These values must be produced by an auditable calculation layer with links to
# their input cells. They are intentionally excluded from METRIC_ALIASES.
DERIVED_METRICS = frozenset(
    {
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "return_on_equity",
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "debt_ratio",
        "asset_turnover",
        "inventory_turnover",
        "receivable_turnover",
        "interest_coverage",
        "operating_cash_flow_ratio",
        "free_cash_flow",
        "ebit",
        "ebitda",
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
    "năm hiện hành",
    "outstanding shares",
}

_ORDINAL_PREFIX_RE = re.compile(r"^\d+\.\s*(.*)$")
_STRUCTURAL_TEXT_PREFIXES = ("cam kết cho vay không hủy ngang",)


def is_non_metric_label(raw: str | None) -> bool:
    """Return true only for labels that are structurally not source metrics."""
    if raw is None:
        return True
    key = normalized_key(raw)
    if not key or key in _NON_METRIC_PATTERNS:
        return True

    ordinal = _ORDINAL_PREFIX_RE.fullmatch(key)
    if ordinal is None:
        return False
    suffix = ordinal.group(1).strip()
    if not suffix:
        return True
    if suffix in METRIC_ALIASES:
        return False
    if suffix in _NON_METRIC_PATTERNS:
        return True
    return any(suffix.startswith(prefix) for prefix in _STRUCTURAL_TEXT_PREFIXES)


def normalize_metric(raw: str | None) -> Decision[str]:
    if raw is None:
        return Decision(value=None)

    key = normalized_key(raw)
    if is_non_metric_label(raw):
        return Decision(value=None)

    canonical = METRIC_ALIASES.get(key)
    if canonical is not None:
        return Decision(value=canonical)

    return Decision(value=None, issue_code="metric_unknown")
