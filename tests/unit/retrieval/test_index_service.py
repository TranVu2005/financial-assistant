from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.service import RetrievalService


def _documents() -> tuple[TableDocument, ...]:
    table_a = "tbl_" + "a" * 64
    table_b = "tbl_" + "b" * 64
    return (
        TableDocument(
            table_id=table_a,
            doc_id="doc_a",
            text="company_code: ACB\nperiod: 2024\nDoanh thu | 2024 | 100",
            metadata=TableMetadata(
                table_id=table_a,
                doc_id="doc_a",
                company_code="ACB",
                period="2024",
                statement_type="income",
                source_path="a.txt",
                line_start=1,
                line_end=3,
            ),
        ),
        TableDocument(
            table_id=table_b,
            doc_id="doc_b",
            text="company_code: VIC\nperiod: 2024\nDoanh thu | 2024 | 200",
            metadata=TableMetadata(
                table_id=table_b,
                doc_id="doc_b",
                company_code="VIC",
                period="2024",
                statement_type="income",
                source_path="v.txt",
                line_start=1,
                line_end=3,
            ),
        ),
    )


def test_filter_first_never_returns_ineligible_document() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("ACB",)), k=10)

    assert [item.table_id for item in trace.results] == ["tbl_" + "a" * 64]
    assert trace.eligible_count == 1
    assert trace.filter_decisions[0].field == "company_codes"
    assert trace.filter_decisions[0].matched_count_before_intersection == 1


def test_empty_query_tokens_return_empty_without_padding() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_out_of_vocabulary_query_returns_empty_without_zero_score_ranking() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("khongtontaitrongindex", filters=RetrievalFilters(), k=10)

    assert trace.results == ()
