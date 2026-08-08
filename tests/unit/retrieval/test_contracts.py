import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import RetrievalFilters


def test_filters_are_deduplicated_and_sorted() -> None:
    filters = RetrievalFilters(company_codes=("VIC", "ACB", "VIC"), periods=("2024", "2023"))

    assert filters.company_codes == ("ACB", "VIC")
    assert filters.periods == ("2023", "2024")


def test_filters_reject_blank_values() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("",))
