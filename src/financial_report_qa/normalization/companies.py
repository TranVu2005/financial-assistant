import re

from financial_report_qa.normalization._shared import Decision, normalized_key
from financial_report_qa.schemas.documents import DocumentRecord

_TICKER_PATTERN = re.compile(
    r"(?:mã ck|mã chứng khoán|stock code|ticker)\s*[:\-\=]?\s*([a-zA-Z0-9]{2,10})",
    re.IGNORECASE,
)


def normalize_company(document: DocumentRecord, title_raw: str | None) -> Decision[str]:
    if title_raw is not None:
        key = normalized_key(title_raw)
        match = _TICKER_PATTERN.search(key)
        if match:
            found_ticker = match.group(1).upper()
            if found_ticker != document.company_code.upper():
                return Decision(value=document.company_code, issue_code="company_conflict")
    return Decision(value=document.company_code)
