from __future__ import annotations

from financial_report_qa.retrieval.contracts import RetrievalFilters, TableDocument, TableMetadata
from financial_report_qa.retrieval.filtering import eligible_positions


def _documents() -> tuple[TableDocument, ...]:
    return (
        TableDocument(
            table_id="tbl_" + "a" * 64,
            doc_id="a",
            text="a",
            metadata=TableMetadata(
                table_id="tbl_" + "a" * 64,
                doc_id="a",
                company_code="ACB",
                periods=("2023",),
                source_path="a.txt",
                line_start=1,
                line_end=1,
            ),
        ),
        TableDocument(
            table_id="tbl_" + "b" * 64,
            doc_id="b",
            text="b",
            metadata=TableMetadata(
                table_id="tbl_" + "b" * 64,
                doc_id="b",
                company_code="ACB",
                periods=("2024",),
                source_path="b.txt",
                line_start=1,
                line_end=1,
            ),
        ),
    )


def test_eligible_positions_ors_periods_and_intersects_company() -> None:
    """Post-filtering would allow the 2024 ACB row to leak into this request."""
    positions, decisions = eligible_positions(
        _documents(), RetrievalFilters(company_codes=("ACB",), periods=("2023", "2025"))
    )
    assert positions == (0,)
    assert [item.eligible_count_after_intersection for item in decisions] == [2, 1]
