import datetime
import re

from financial_report_qa.normalization._shared import Decision, normalized_key

_ROMAN_QUARTERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}

_DATE_2DIGIT_YEAR_RE = re.compile(r"^\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})\b$")
_DATE_4DIGIT_YEAR_RE = re.compile(r"^(?:ngày\s+)?(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$")
_QUARTER_RE = re.compile(
    r"^(?:quý|q)\s*(i{1,3}|iv|[1-4])(?:\s*(?:năm|/|-)?\s*(\d{4}))?$", re.IGNORECASE
)
_YEAR_RE = re.compile(r"^(?:năm\s+)?((?:19|20)\d{2})$", re.IGNORECASE)
_MONTH_ONLY_RE = re.compile(r"^(?:tháng\s+)?(\d{1,2})$", re.IGNORECASE)


def normalize_period(raw: str, report_year: int) -> Decision[str]:
    key = normalized_key(raw)
    if not key:
        return Decision(value=None)

    # Check 2-digit year date first
    m_2yr = _DATE_2DIGIT_YEAR_RE.match(key)
    if m_2yr:
        day_s, month_s, yy_s = m_2yr.groups()
        day, month, yy = int(day_s), int(month_s), int(yy_s)
        if day <= 12 and month <= 12:
            return Decision(value=None, issue_code="period_ambiguous")
        century = (report_year // 100) * 100
        year = century + yy
        try:
            d = datetime.date(year, month, day)
            return Decision(value=d.isoformat())
        except ValueError:
            return Decision(value=None, issue_code="period_invalid")

    # Check 4-digit year date (DD/MM/YYYY or DD-MM-YYYY)
    date_match = _DATE_4DIGIT_YEAR_RE.match(key)
    if date_match:
        day_s, month_s, year_s = date_match.groups()
        day, month, year = int(day_s), int(month_s), int(year_s)
        try:
            d = datetime.date(year, month, day)
            return Decision(value=d.isoformat())
        except ValueError:
            return Decision(value=None, issue_code="period_invalid")

    # Check Quarter
    quarter_match = _QUARTER_RE.match(key)
    if quarter_match:
        q_str, year_str = quarter_match.groups()
        q_num = _ROMAN_QUARTERS[q_str.lower()]
        yr = int(year_str) if year_str else report_year
        return Decision(value=f"{yr}-Q{q_num}")

    # Check Annual (Year)
    year_match = _YEAR_RE.match(key)
    if year_match:
        return Decision(value=year_match.group(1))

    # Check month only (or "tháng X")
    m_month = re.match(r"^(?:tháng\s+)?(\d{1,2})$", key, re.IGNORECASE)
    if m_month:
        m_val = int(m_month.group(1))
        if 1 <= m_val <= 12:
            return Decision(value=f"{report_year}-{m_val:02d}")

    if key.startswith("tháng") or key.startswith("quý") or key.startswith("năm"):
        return Decision(value=None, issue_code="period_incomplete")

    return Decision(value=None)
