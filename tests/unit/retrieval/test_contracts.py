import pytest
from pydantic import ValidationError

import financial_report_qa.retrieval.contracts as contracts
from financial_report_qa.retrieval.contracts import RetrievalFilters


def test_filters_reject_noncanonical_order_and_duplicates() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("VIC", "ACB", "VIC"), periods=("2024", "2023"))


def test_filters_reject_blank_values() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("",))


def test_metric_label_observation_requires_canonical_text() -> None:
    """Would fail if documents can store a metric observation without identity."""
    assert hasattr(contracts, "MetricLabelObservation")
    observation = contracts.MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần")

    with pytest.raises(ValidationError):
        contracts.MetricLabelObservation(canonical="", raw="Doanh thu thuần")

    assert observation.canonical == "net_revenue"


def test_metric_expansion_records_only_tokens_added_to_query() -> None:
    """Would fail if the retrieval trace cannot explain canonical tokens it added."""
    assert hasattr(contracts, "MetricExpansion")
    expansion = contracts.MetricExpansion(
        alias_tokens=("doanh", "thu", "thuần"),
        canonical_metric="net_revenue",
        added_tokens=("net", "revenue"),
    )

    assert expansion.added_tokens == ("net", "revenue")


def test_table_document_requires_sorted_unique_metric_labels() -> None:
    """Would fail if metric observations can have unstable persisted ordering."""
    labels = (
        contracts.MetricLabelObservation(canonical="total_assets", raw="Tổng tài sản"),
        contracts.MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),
    )
    metadata = contracts.TableMetadata(
        table_id="tbl_" + "a" * 64,
        doc_id="doc_a",
        source_path="a.txt",
        line_start=1,
        line_end=1,
    )

    with pytest.raises(ValidationError, match="metric_labels must be sorted and unique"):
        contracts.TableDocument(
            table_id="tbl_" + "a" * 64,
            doc_id="doc_a",
            text="table text",
            metadata=metadata,
            metric_labels=labels,
        )
