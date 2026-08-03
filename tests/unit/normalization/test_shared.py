import pytest

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.normalization._shared import (
    RULESET_VERSION,
    Decision,
    issue_sort_key,
    normalized_key,
    validate_aliases,
)
from financial_report_qa.schemas.normalization import NormalizationIssue


def test_shared_primitives_constants_and_decision() -> None:
    assert RULESET_VERSION == "2026.08.1"
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
