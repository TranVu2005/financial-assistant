import pytest

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.normalization._shared import (
    RULESET_VERSION,
    Decision,
    issue_sort_key,
    normalized_key,
    sanitize_selector_text,
    validate_aliases,
)
from financial_report_qa.schemas.normalization import NormalizationIssue


def test_shared_primitives_constants_and_decision() -> None:
    assert RULESET_VERSION == "2026.08.6"
    decision = Decision(value="ok", issue_code=None)
    assert decision.value == "ok"


def test_normalized_key_is_unicode_aware_without_changing_source() -> None:
    raw = "  BÁO   CÁO\u00a0TÀI CHÍNH  "
    assert normalized_key(raw) == "báo cáo tài chính"
    assert raw == "  BÁO   CÁO\u00a0TÀI CHÍNH  "


def test_validate_aliases_rejects_conflicting_normalized_keys() -> None:
    with pytest.raises(NormalizationError, match="conflicting alias"):
        validate_aliases({"Báo cáo": "first", "  BÁO  CÁO ": "second"})


def test_issue_sort_key_orders_none_before_identifiers() -> None:
    def issue(table_id: str | None) -> NormalizationIssue:
        return NormalizationIssue(
            code="metric_unknown",
            doc_id=f"doc_{'a' * 64}",
            table_id=table_id,
            cell_id=None,
            field="metric",
            raw_value="unknown",
        )

    table_issue = issue(table_id=f"tbl_{'a' * 64}")
    document_issue = issue(table_id=None)
    assert sorted((table_issue, document_issue), key=issue_sort_key) == [
        document_issue,
        table_issue,
    ]


def test_sanitize_selector_text_collapses_an_embedded_newline() -> None:
    """A real corpus column header can concatenate two source lines with a
    literal newline (e.g. "31/12/2019" + newline + "VND"); label fields
    forbid control characters outright (the 0x00-0x1f range, which includes
    newline), so such text used verbatim as a row/column label raises a
    ValidationError. Collapsing whitespace -- not stripping it --
    keeps the header readable ("31/12/2019 VND") instead of losing the
    boundary between its parts."""
    assert sanitize_selector_text("31/12/2019\nVND") == "31/12/2019 VND"


def test_sanitize_selector_text_preserves_case() -> None:
    """Unlike normalized_key, this has to survive as a literal row/column
    label value -- quoted back into a query or a citation -- not just a
    comparison key; casefolding it would make it no longer name the row it
    was built from."""
    assert sanitize_selector_text("Doanh Thu Thuần") == "Doanh Thu Thuần"


def test_sanitize_selector_text_strips_and_collapses_ordinary_whitespace() -> None:
    assert sanitize_selector_text("  Tiền   mặt  ") == "Tiền mặt"
