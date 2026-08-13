import hashlib
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.retrieval.contracts import MetricLabelObservation
from financial_report_qa.retrieval.documents import (
    build_table_documents,
    write_table_documents,
)
from financial_report_qa.retrieval.release import ResolvedRetrievalRelease


def _write_release_fixture(tmp_path: Path) -> None:
    table_id = "tbl_" + "a" * 64
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [table_id],
                "doc_id": ["doc_a"],
                "title_raw": ["  Báo cáo\u00a0kết quả  "],
                "statement_type": ["income_statement"],
                "unit_normalized": ["VND_million"],
                "line_start": [10],
                "line_end": [20],
            }
        ),
        tmp_path / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
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
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [table_id, table_id, table_id, table_id],
                "row_idx": [0, 1, 2, 3],
                "col_idx": [0, 0, 0, 0],
                "row_label_canonical": [
                    "net_revenue",
                    "net_revenue",
                    "total_assets",
                    None,
                ],
                "row_label_raw": [
                    "Doanh thu\u00a0thuần",
                    "  Doanh thu thuần ",
                    None,
                    "Dòng trình bày không phải metric",
                ],
                "column_label_canonical": ["2023", "2024", "2024", "2024"],
                "value_raw": ["100", "200", "300", "400"],
                "period": ["2023", "2024", "2024", "2024"],
                "unit": ["VND", "VND_million", "VND_million", "VND_million"],
            }
        ),
        tmp_path / "cells.parquet",
    )


def test_build_table_documents_normalizes_display_fields_and_omits_numbers(tmp_path: Path) -> None:
    _write_release_fixture(tmp_path)

    documents = build_table_documents(
        tmp_path / "documents.parquet", tmp_path / "tables.parquet", tmp_path / "cells.parquet"
    )

    assert [document.table_id for document in documents] == ["tbl_" + "a" * 64]
    assert documents[0].metadata.periods == ("2023", "2024")
    assert documents[0].metric_labels == (
        MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),
        MetricLabelObservation(canonical="total_assets", raw=None),
    )
    assert documents[0].text == (
        "title: Báo cáo kết quả\n"
        "statement: income statement\n"
        "metrics: net revenue | total assets\n"
        "metric aliases: Doanh thu thuần\n"
        "company: ACB\n"
        "periods: 2023 | 2024\n"
        "units: VND | VND million"
    )
    assert "100" not in documents[0].text
    assert "Dòng trình bày không phải metric" not in documents[0].text


def _write_release_fixture_with_group_context(tmp_path: Path) -> str:
    table_id = "tbl_" + "b" * 64
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [table_id],
                "doc_id": ["doc_b"],
                "title_raw": ["Bang can doi"],
                "statement_type": ["balance_sheet"],
                "unit_normalized": ["VND"],
                "line_start": [1],
                "line_end": [5],
            }
        ),
        tmp_path / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "doc_id": ["doc_b"],
                "company_code": ["PVT"],
                "report_year": [2016],
                "relative_path": ["b.txt"],
            }
        ),
        tmp_path / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [table_id, table_id, table_id],
                "row_idx": [0, 1, 1],
                "col_idx": [0, 0, 1],
                "row_label_canonical": [None, "cash", "cash"],
                "row_label_raw": [None, "Tien mat", "Tien mat"],
                "column_label_canonical": [None, "2020", "2020"],
                "value_raw": ["Header", "100", "200"],
                "period": [None, "2020", "2020"],
                "unit": [None, "VND", "VND"],
                "row_group_context_raw": [
                    None,
                    "  Công ty Cổ phần Vận tải  ",
                    "Công ty Cổ phần Vận tải",
                ],
            }
        ),
        tmp_path / "cells.parquet",
    )
    return table_id


def test_build_table_documents_includes_group_context(tmp_path: Path) -> None:
    """Would fail if section-banner context stayed dropped from retrieval input."""
    table_id = _write_release_fixture_with_group_context(tmp_path)

    documents = build_table_documents(
        tmp_path / "documents.parquet", tmp_path / "tables.parquet", tmp_path / "cells.parquet"
    )

    assert [document.table_id for document in documents] == [table_id]
    assert documents[0].metadata.group_contexts == ("Công ty Cổ phần Vận tải",)
    assert "group context: Công ty Cổ phần Vận tải" in documents[0].text
    assert documents[0].text.index("metric aliases:") < documents[0].text.index("group context:")
    assert documents[0].text.index("group context:") < documents[0].text.index("company:")


def test_build_table_documents_tolerates_release_without_group_context_column(
    tmp_path: Path,
) -> None:
    """Would fail if older releases missing the column broke document building."""
    _write_release_fixture(tmp_path)

    documents = build_table_documents(
        tmp_path / "documents.parquet", tmp_path / "tables.parquet", tmp_path / "cells.parquet"
    )

    assert documents[0].metadata.group_contexts == ()
    assert "group context:" not in documents[0].text


def test_write_table_documents_is_deterministic_and_preserves_target_on_count_error(
    tmp_path: Path,
) -> None:
    _write_release_fixture(tmp_path)
    release = cast(
        ResolvedRetrievalRelease,
        SimpleNamespace(release_dir=tmp_path, manifest={"table_count": 1}),
    )
    output_path = tmp_path / "documents.jsonl"

    first = write_table_documents(release, output_path)
    first_bytes = output_path.read_bytes()
    second = write_table_documents(release, output_path)

    assert first == second
    assert first.table_count == 1
    assert first.sha256 == hashlib.sha256(first_bytes).hexdigest()
    assert first_bytes.endswith(b"\n")
    assert unicodedata.normalize("NFKC", first_bytes.decode("utf-8")) == first_bytes.decode("utf-8")

    release.manifest["table_count"] = 2
    with pytest.raises(ValueError, match="table count"):
        write_table_documents(release, output_path)
    assert output_path.read_bytes() == first_bytes
