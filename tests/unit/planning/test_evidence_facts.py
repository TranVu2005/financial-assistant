"""Tests for plan.md §12 candidate-fact enumeration.

The Evidence-Aware Planner is only as good as the facts it is shown: they
must be real cells read out of the release, each already carrying its value,
unit, period and position, so the planner has nothing left to invent.
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.planning.evidence_facts import enumerate_candidate_facts
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_MBB = "tbl_" + "2" * 64
DOC_ID = "doc_" + "a" * 64
DOC_ID_MBB = "doc_" + "b" * 64


def _document(doc_id: str, company: str) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": f"{company}/2023/report.txt",
        "company_code": company,
        "report_year": 2023,
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
        "title_raw": "Bao cao ket qua kinh doanh",
        "statement_type": "income_statement",
        "unit_raw": "VND",
        "unit_normalized": "vnd",
        "line_start": 1,
        "line_end": 10,
        "row_count": 3,
        "column_count": 3,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _cell(
    character: str,
    *,
    table_id: str = TABLE_ID,
    row_idx: int,
    col_idx: int = 1,
    row_label_raw: str,
    value: str | None,
    period: str | None = "2023",
    unit: str | None = "VND",
) -> dict[str, object]:
    return {
        "cell_id": "cell_" + character * 64,
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": f"Năm {period}" if period else "Năm",
        "column_label_canonical": None,
        "value_raw": value or "-",
        "value_numeric": Decimal(value) if value is not None else None,
        "period": period,
        "unit": unit,
        "source_line_start": row_idx + 1,
        "source_line_end": row_idx + 1,
        "extraction_confidence": 0.9,
    }


def _write_release(tmp_path: Path, cells: list[dict[str, object]]) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [_document(DOC_ID, "ACB"), _document(DOC_ID_MBB, "MBB")], schema=DOCUMENT_SCHEMA
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(
            [_table(TABLE_ID, DOC_ID), _table(TABLE_ID_MBB, DOC_ID_MBB)], schema=TABLE_SCHEMA
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def _candidate(
    *, rank: int, row_idx: int, table_id: str = TABLE_ID, row_label_raw: str
) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{table_id}|row_{row_idx}",
        table_id=table_id,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 / rank,
        snippet=row_label_raw,
        metadata=RowMetadata(table_id=table_id, row_idx=row_idx, row_label_raw=row_label_raw),
    )


def test_enumerate_reads_real_values_for_the_retrieved_rows(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=1, row_label_raw="Doanh thu thuan", value="63075"),
            _cell("b", row_idx=2, row_label_raw="Gia von hang ban", value="41000"),
        ],
    )
    facts = enumerate_candidate_facts(
        release_dir,
        [
            _candidate(rank=1, row_idx=1, row_label_raw="Doanh thu thuan"),
            _candidate(rank=2, row_idx=2, row_label_raw="Gia von hang ban"),
        ],
    )
    assert [fact.fact_id for fact in facts] == ["F1", "F2"]
    assert [fact.row_label for fact in facts] == ["Doanh thu thuan", "Gia von hang ban"]
    assert [fact.raw_value for fact in facts] == [Decimal("63075"), Decimal("41000")]
    assert all(fact.unit == "VND" and fact.period == 2023 for fact in facts)
    assert [fact.row_index for fact in facts] == [1, 2]
    assert all(fact.company_code == "ACB" for fact in facts)


def test_enumerate_orders_facts_by_retrieval_rank(tmp_path: Path) -> None:
    """The planner reads a numbered list top-down; the best-retrieved row
    should be F1, not whichever row happens to sit first in the table."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=1, row_label_raw="Doanh thu thuan", value="63075"),
            _cell("b", row_idx=2, row_label_raw="Gia von hang ban", value="41000"),
        ],
    )
    facts = enumerate_candidate_facts(
        release_dir,
        [
            _candidate(rank=1, row_idx=2, row_label_raw="Gia von hang ban"),
            _candidate(rank=2, row_idx=1, row_label_raw="Doanh thu thuan"),
        ],
    )
    assert [fact.row_label for fact in facts] == ["Gia von hang ban", "Doanh thu thuan"]


def test_enumerate_emits_one_fact_per_period_of_the_same_row(tmp_path: Path) -> None:
    """§12's own example is a growth rate: the same row at two periods has to
    arrive as two separately addressable facts."""
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=1, col_idx=1, row_label_raw="Doanh thu thuan", value="63075"),
            _cell(
                "b",
                row_idx=1,
                col_idx=2,
                row_label_raw="Doanh thu thuan",
                value="60180",
                period="2022",
            ),
        ],
    )
    facts = enumerate_candidate_facts(
        release_dir, [_candidate(rank=1, row_idx=1, row_label_raw="Doanh thu thuan")]
    )
    assert len(facts) == 2
    assert {fact.period for fact in facts} == {2022, 2023}
    assert {fact.raw_value for fact in facts} == {Decimal("63075"), Decimal("60180")}
    assert {fact.row_index for fact in facts} == {1}


def test_enumerate_restricts_to_the_questions_company(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=1, row_label_raw="Doanh thu thuan", value="63075"),
            _cell(
                "b",
                table_id=TABLE_ID_MBB,
                row_idx=1,
                row_label_raw="Doanh thu thuan",
                value="99999",
            ),
        ],
    )
    facts = enumerate_candidate_facts(
        release_dir,
        [
            _candidate(rank=1, row_idx=1, row_label_raw="Doanh thu thuan"),
            _candidate(rank=2, row_idx=1, table_id=TABLE_ID_MBB, row_label_raw="Doanh thu thuan"),
        ],
        company_code="ACB",
    )
    assert [fact.raw_value for fact in facts] == [Decimal("63075")]


def test_enumerate_restricts_to_the_questions_periods_when_given(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=1, col_idx=1, row_label_raw="Doanh thu thuan", value="63075"),
            _cell(
                "b",
                row_idx=1,
                col_idx=2,
                row_label_raw="Doanh thu thuan",
                value="60180",
                period="2022",
            ),
        ],
    )
    facts = enumerate_candidate_facts(
        release_dir,
        [_candidate(rank=1, row_idx=1, row_label_raw="Doanh thu thuan")],
        periods=(2023,),
    )
    assert [fact.period for fact in facts] == [2023]


def test_enumerate_skips_cells_with_no_recorded_unit(tmp_path: Path) -> None:
    """A fact without a unit cannot be arithmetic-checked downstream, and
    showing it to the planner invites an answer nothing can verify."""
    release_dir = _write_release(
        tmp_path,
        [_cell("a", row_idx=1, row_label_raw="Doanh thu thuan", value="63075", unit=None)],
    )
    facts = enumerate_candidate_facts(
        release_dir, [_candidate(rank=1, row_idx=1, row_label_raw="Doanh thu thuan")]
    )
    assert facts == ()


def test_enumerate_caps_the_prompt_budget(tmp_path: Path) -> None:
    cells = [
        _cell(format(index, "x") * 1, row_idx=index, row_label_raw=f"Dong {index}", value="1")
        for index in range(1, 10)
    ]
    for index, cell in enumerate(cells):
        cell["cell_id"] = "cell_" + f"{index:064x}"
    release_dir = _write_release(tmp_path, cells)
    facts = enumerate_candidate_facts(
        release_dir,
        [
            _candidate(rank=index, row_idx=index, row_label_raw=f"Dong {index}")
            for index in range(1, 10)
        ],
        max_facts=3,
    )
    assert [fact.fact_id for fact in facts] == ["F1", "F2", "F3"]


def test_enumerate_returns_nothing_without_retrieved_rows(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path, [_cell("a", row_idx=1, row_label_raw="Doanh thu thuan", value="63075")]
    )
    assert enumerate_candidate_facts(release_dir, []) == ()
