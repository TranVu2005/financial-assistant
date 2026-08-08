import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.retrieval.documents import build_table_documents


def test_build_table_documents_is_stable_and_contains_headers(tmp_path) -> None:
    table_id = "tbl_" + "a" * 64
    pq.write_table(
        pa.table(
            {
                "table_id": [table_id],
                "doc_id": ["doc_a"],
                "title_raw": ["Báo cáo kết quả"],
                "statement_type": ["income"],
                "line_start": [10],
                "line_end": [20],
            }
        ),
        tmp_path / "tables.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "doc_id": ["doc_a"],
                "company_code": ["ACB"],
                "report_year": [2024],
                "relative_path": ["a.txt"],
            }
        ),
        tmp_path / "documents.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "table_id": [table_id],
                "row_idx": [0],
                "col_idx": [0],
                "row_label_canonical": ["Doanh thu"],
                "column_label_canonical": ["2024"],
                "value_raw": ["100"],
            }
        ),
        tmp_path / "cells.parquet",
    )

    documents = build_table_documents(
        tmp_path / "documents.parquet", tmp_path / "tables.parquet", tmp_path / "cells.parquet"
    )

    assert [document.table_id for document in documents] == [table_id]
    assert "company_code: ACB" in documents[0].text
    assert "Doanh thu | 2024 | 100" in documents[0].text
