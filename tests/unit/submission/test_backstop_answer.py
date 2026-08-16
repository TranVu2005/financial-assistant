"""Tests for the Day 23 absolute last-resort tier.

This tier's only job is contract validity (plan.md §2.4 rule 1: a single
missing id fails the *entire* submission ZIP), never correctness -- the
official Dashboard scoring macro-averages over the full 1.012-question set,
so a wrong answer already scores the same 0 credit as a missing one.
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.backstop_answer import build_backstop_item
from financial_report_qa.submission.contracts import RawQuestion

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_OTHER = "tbl_" + "2" * 64
DOC_ID = "doc_" + "a" * 64
DOC_ID_OTHER = "doc_" + "b" * 64


def _document(doc_id: str, company: str, year: int, path: str) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": path,
        "company_code": company,
        "report_year": year,
        "statement_scope": "consolidated",
        "sha256": "0" * 64,
        "file_size_bytes": 10,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }


def _table(table_id: str, doc_id: str) -> dict[str, object]:
    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "source_ordinal": 0,
        "title_raw": "Bang",
        "statement_type": None,
        "unit_raw": "VND",
        "unit_normalized": "vnd",
        "line_start": 5,
        "line_end": 10,
        "row_count": 1,
        "column_count": 2,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _cell(
    cell_id: str, table_id: str, *, row_label_raw: str, value: str, period: str
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": 0,
        "col_idx": 1,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": f"Năm {period}",
        "column_label_canonical": None,
        "value_raw": value,
        "value_numeric": Decimal(value),
        "period": period,
        "unit": "VND",
        "source_line_start": 7,
        "source_line_end": 7,
        "extraction_confidence": 0.9,
    }


def _write_release(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    documents = [
        _document(DOC_ID, "ACB", 2023, "ACB/2023/report.txt"),
        _document(DOC_ID_OTHER, "MBB", 2022, "MBB/2022/report.txt"),
    ]
    tables = [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_OTHER, DOC_ID_OTHER)]
    cells = [
        _cell("cell_a", TABLE_ID, row_label_raw="Doanh thu", value="1000", period="2023"),
        _cell("cell_b", TABLE_ID_OTHER, row_label_raw="Lợi nhuận", value="500", period="2022"),
    ]
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(documents, schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(tables, schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def test_backstop_uses_candidate_table_when_available(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [TABLE_ID], release_dir)

    assert item.id == 1
    assert item.answer == 1000.0
    assert rows[0]["company_code"] == "ACB"


def test_backstop_falls_back_to_whole_corpus_when_no_candidate_tables(tmp_path: Path) -> None:
    """Retrieval returning nothing (42/1.012 real questions) must not
    prevent a valid backstop -- some numeric cell exists somewhere."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=2, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [], release_dir)

    assert item.id == 2
    assert item.answer in (1000.0, 500.0)
    assert rows[0]["value"] in (Decimal("1000"), Decimal("500"))


def test_backstop_item_is_contract_valid_and_replayable(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=3, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [TABLE_ID], release_dir)

    assert item.evidence[0].csv_path == "data/q000003_df1.csv"
    assert item.relevant_docs == ("report",)  # basename of "ACB/2023/report.txt" minus .txt
    assert item.relevant_tables[0].startswith(item.relevant_docs[0])


def test_backstop_pandas_query_replays_to_the_declared_answer(tmp_path: Path) -> None:
    import pandas as pd

    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=4, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [TABLE_ID], release_dir)

    frame = pd.DataFrame(list(rows))
    frame["period"] = frame["period"].astype("Int64")
    result = replay_in_sandbox(item.pandas_query, frame, timeout_seconds=5.0)

    assert result.error_code is None
    assert result.value is not None
    assert float(result.value) == item.answer


def test_backstop_raises_when_release_has_no_numeric_cell_at_all(tmp_path: Path) -> None:
    """The one case this tier genuinely cannot recover from -- an empty
    release. Must fail loudly, not silently fabricate a value."""
    release_dir = tmp_path / "empty_release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([], schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    question = RawQuestion(id=5, question="Câu hỏi không xác định được.")

    with pytest.raises(RuntimeError):
        build_backstop_item(question, [], release_dir)
