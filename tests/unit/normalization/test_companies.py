import pytest

from financial_report_qa.normalization._shared import Decision
from financial_report_qa.normalization.companies import normalize_company
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def _document(company_code: str) -> DocumentRecord:
    digest = "a" * 64
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=f"{company_code}/2024/Consolidated/report.txt",
        company_code=company_code,
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )


@pytest.mark.parametrize(
    ("title", "expected_code"),
    [(None, "VCB"), ("BẢNG CÂN ĐỐI KẾ TOÁN", "VCB"), ("Mã CK: VCB", "VCB")],
)
def test_company_uses_inventory_code(title: str | None, expected_code: str) -> None:
    decision = normalize_company(_document("VCB"), title)
    assert decision == Decision(value=expected_code)


def test_company_reports_explicit_conflicting_ticker_without_overriding_document() -> None:
    assert normalize_company(_document("VCB"), "Mã chứng khoán: ACB") == Decision(
        value="VCB", issue_code="company_conflict"
    )
