"""Tests for the Day 22 live-retrieval bridge (plan §1/§2 decision B): a raw,
never-before-seen question string -> ranked candidate table_ids, with no
gold-labeled `RetrievalFilters` available. Wires two already-tested pieces
(`parse_query_entities`, `to_retrieval_filters`) into `RetrievalService`."""

from __future__ import annotations

from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.service import RetrievalService

TABLE_ACB = "tbl_" + "a" * 64
TABLE_VIC = "tbl_" + "b" * 64


def _documents() -> tuple[TableDocument, ...]:
    return (
        TableDocument(
            table_id=TABLE_ACB,
            doc_id="doc_a",
            text="company_code: ACB\nperiod: 2023\nDoanh thu thuần | 2023 | 100",
            metadata=TableMetadata(
                table_id=TABLE_ACB,
                doc_id="doc_a",
                company_code="ACB",
                periods=("2023",),
                statement_type="income",
                source_path="a.txt",
                line_start=1,
                line_end=3,
            ),
            metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
        ),
        TableDocument(
            table_id=TABLE_VIC,
            doc_id="doc_b",
            text="company_code: VIC\nperiod: 2023\nDoanh thu thuần | 2023 | 200",
            metadata=TableMetadata(
                table_id=TABLE_VIC,
                doc_id="doc_b",
                company_code="VIC",
                periods=("2023",),
                statement_type="income",
                source_path="b.txt",
                line_start=1,
                line_end=3,
            ),
            metric_labels=(MetricLabelObservation(canonical="net_revenue", raw=None),),
        ),
    )


def _service() -> RetrievalService:
    return RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))


def test_retrieve_candidate_table_ids_narrows_by_parsed_company() -> None:
    """A question naming ACB should retrieve only the ACB table, exactly as if
    a hand-labeled RetrievalFilters(company_codes=("ACB",)) had been given --
    but derived here purely from the raw question text."""
    table_ids = retrieve_candidate_table_ids(
        "Doanh thu thuần của ACB năm 2023 là bao nhiêu?", _service(), k=10
    )
    assert table_ids == (TABLE_ACB,)


def test_retrieve_candidate_table_ids_returns_empty_when_no_eligible_tables() -> None:
    """A recognized company (ACB) with a period no table covers must return
    empty, not fall back to ignoring the period filter."""
    table_ids = retrieve_candidate_table_ids(
        "Doanh thu thuần của ACB năm 2019 là bao nhiêu?", _service(), k=10
    )
    assert table_ids == ()


def test_retrieve_candidate_table_ids_respects_k() -> None:
    table_ids = retrieve_candidate_table_ids(
        "Doanh thu thuần năm 2023 là bao nhiêu?", _service(), k=1
    )
    assert len(table_ids) <= 1
