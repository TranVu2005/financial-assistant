from financial_report_qa.normalization._shared import Decision, normalized_key

_STATEMENT_FAMILIES: dict[str, tuple[str, ...]] = {
    "balance_sheet": (
        "bảng cân đối kế toán",
        "bảng cân đối số phát sinh",
        "balance sheet",
    ),
    "income_statement": (
        "báo cáo kết quả hoạt động kinh doanh",
        "báo cáo kết quả kinh doanh",
        "income statement",
        "profit and loss",
    ),
    "cash_flow_statement": (
        "báo cáo lưu chuyển tiền tệ",
        "lưu chuyển tiền tệ",
        "cash flow statement",
        "statement of cash flows",
    ),
    "equity_changes": (
        "báo cáo thay đổi vốn chủ sở hữu",
        "biến động vốn chủ sở hữu",
        "statement of changes in equity",
    ),
    "notes": (
        "thuyết minh báo cáo tài chính",
        "thuyết minh bctc",
        "notes to the financial statements",
    ),
}

_NARRATIVE_MARKERS = (
    "thể hiện trên",
    "trích từ",
    "bao gồm các khoản",
    "thay đổi trên",
)


def normalize_statement_type(title_raw: str | None) -> Decision[str]:
    if title_raw is None:
        return Decision(value=None)

    key = normalized_key(title_raw)
    if not key:
        return Decision(value=None)

    matched_families: set[str] = set()
    for family, aliases in _STATEMENT_FAMILIES.items():
        for alias in aliases:
            if alias in key:
                matched_families.add(family)
                break

    if not matched_families:
        return Decision(value=None)
    if len(matched_families) > 1:
        if any(marker in key for marker in _NARRATIVE_MARKERS):
            return Decision(value=None)
        if "notes" in matched_families and not key.startswith("thuyết minh"):
            primary = matched_families - {"notes"}
            if len(primary) == 1:
                return Decision(value=next(iter(primary)))
        return Decision(value=None, issue_code="statement_conflict")
    return Decision(value=next(iter(matched_families)))
