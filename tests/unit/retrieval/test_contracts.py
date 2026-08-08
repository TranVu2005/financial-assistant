import pytest
from pydantic import ValidationError

from financial_report_qa.retrieval.contracts import RetrievalFilters


def test_filters_reject_noncanonical_order_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("VIC", "ACB", "VIC"), periods=("2024", "2023"))


def test_filters_reject_blank_values() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("",))
