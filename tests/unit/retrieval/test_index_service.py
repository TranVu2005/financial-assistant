from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.index import build_bm25_index
from financial_report_qa.retrieval.service import RetrievalService


def _documents() -> tuple[TableDocument, ...]:
    return (
        TableDocument(
            table_id="ACB.2024.income",
            text="company_code: ACB\nperiod: 2024\nDoanh thu | 2024 | 100",
            metadata=TableMetadata(
                table_id="ACB.2024.income",
                company_code="ACB",
                period="2024",
                statement_type="income",
                source_path="a.txt",
                start_line=1,
                end_line=3,
            ),
        ),
        TableDocument(
            table_id="VIC.2024.income",
            text="company_code: VIC\nperiod: 2024\nDoanh thu | 2024 | 200",
            metadata=TableMetadata(
                table_id="VIC.2024.income",
                company_code="VIC",
                period="2024",
                statement_type="income",
                source_path="v.txt",
                start_line=1,
                end_line=3,
            ),
        ),
    )


def test_filter_first_never_returns_ineligible_document() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("doanh thu", filters=RetrievalFilters(company_codes=("ACB",)), k=10)

    assert [item.table_id for item in trace.results] == ["ACB.2024.income"]
    assert trace.filter_decision.indexed_count == 2
    assert trace.filter_decision.eligible_count == 1
    assert trace.filter_decision.field_decisions[0].field_name == "company_codes"
    assert trace.filter_decision.field_decisions[0].matched_count_before_intersection == 1


def test_empty_query_tokens_return_empty_without_padding() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("", filters=RetrievalFilters(), k=10)

    assert trace.results == ()


def test_out_of_vocabulary_query_returns_empty_without_zero_score_ranking() -> None:
    service = RetrievalService(build_bm25_index(_documents(), dataset_fingerprint="f" * 64))

    trace = service.retrieve("khongtontaitrongindex", filters=RetrievalFilters(), k=10)

    assert trace.results == ()
