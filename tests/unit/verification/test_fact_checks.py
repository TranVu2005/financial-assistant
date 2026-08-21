"""Tests for plan.md §15 per-fact verification.

Verification stops treating the answer as one opaque number: every
`GroundedFact` behind it is independently re-located in the release --
row/column/unit, not the pre-computed value the compiler already trusted --
and a fact that cannot be re-found, or re-locates to a different value or
unit, blocks the package on its own, before the formula is ever checked.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.verification.fact_checks import verify_fact, verify_facts

TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64


def _document() -> dict[str, object]:
    return {
        "doc_id": DOC_ID,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": "ACB/2023/report.txt",
        "company_code": "ACB",
        "report_year": 2023,
        "statement_scope": "consolidated",
        "sha256": "0" * 64,
        "file_size_bytes": 10,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }


def _table() -> dict[str, object]:
    return {
        "table_id": TABLE_ID,
        "doc_id": DOC_ID,
        "source_ordinal": 0,
        "title_raw": "Bao cao ket qua kinh doanh",
        "statement_type": "income_statement",
        "unit_raw": "VND",
        "unit_normalized": "vnd",
        "line_start": 1,
        "line_end": 10,
        "row_count": 2,
        "column_count": 2,
        "quality_score": 0.9,
        "csv_path": None,
    }


def _cell(
    character: str,
    *,
    row_idx: int,
    row_label_raw: str = "Doanh thu thuan",
    value: str,
    unit: str | None = "VND",
    period: str | None = "2023",
    column_label_raw: str | None = "Năm 2023",
) -> dict[str, object]:
    return {
        "cell_id": "cell_" + character * 64,
        "table_id": TABLE_ID,
        "row_idx": row_idx,
        "col_idx": 1,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": column_label_raw,
        "column_label_canonical": None,
        "value_raw": value,
        "value_numeric": Decimal(value),
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
        pa.Table.from_pylist([_document()], schema=DOCUMENT_SCHEMA),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([_table()], schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def _fact(**overrides: object) -> GroundedFact:
    payload: dict[str, object] = {
        "fact_id": "F1",
        "table_id": TABLE_ID,
        "row_index": 14,
        "row_label": "Doanh thu thuan",
        "column": "Năm 2023",
        "company_code": "ACB",
        "period": 2023,
        "raw_value": Decimal("63075"),
        "unit": "VND",
        "grounding_score": 0.94,
    }
    payload.update(overrides)
    return GroundedFact.model_validate(payload)


def test_verify_fact_passes_when_the_release_agrees(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, [_cell("a", row_idx=14, value="63075")])
    assert verify_fact(_fact(), release_dir) is None


def test_verify_fact_fails_when_the_row_no_longer_exists(tmp_path: Path) -> None:
    """A fact naming a position no live cell occupies -- e.g. a stale fact
    carried past an ingestion re-run -- must block, not silently pass."""
    release_dir = _write_release(tmp_path, [_cell("a", row_idx=3, value="63075")])
    issue = verify_fact(_fact(row_index=14), release_dir)
    assert issue is not None
    assert issue.code == "fact_not_found"


def test_verify_fact_fails_when_the_value_disagrees(tmp_path: Path) -> None:
    """The exact bug class this check exists for: a fact whose row/column
    position is real, but whose recorded value drifted from the release --
    e.g. a `fact_grounding` indexing bug pairing evidence with the wrong
    replay row."""
    release_dir = _write_release(tmp_path, [_cell("a", row_idx=14, value="99999")])
    issue = verify_fact(_fact(), release_dir)
    assert issue is not None
    assert issue.code == "fact_value_mismatch"
    assert "F1" in issue.message


def test_verify_fact_fails_when_the_unit_disagrees(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path, [_cell("a", row_idx=14, value="63075", unit="VND_million")]
    )
    issue = verify_fact(_fact(), release_dir)
    assert issue is not None
    assert issue.code == "fact_value_mismatch"


def test_verify_fact_fails_when_the_period_disagrees(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, [_cell("a", row_idx=14, value="63075", period="2022")])
    issue = verify_fact(_fact(), release_dir)
    assert issue is not None
    assert issue.code == "fact_not_found"


def test_verify_fact_ignores_the_column_when_the_fact_has_none(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path, [_cell("a", row_idx=14, value="63075", column_label_raw="Bất kỳ")]
    )
    assert verify_fact(_fact(column=None), release_dir) is None


def test_verify_fact_requires_the_named_column_when_the_fact_has_one(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [_cell("a", row_idx=14, value="63075", column_label_raw="Năm 2022")],
    )
    issue = verify_fact(_fact(column="Năm 2023"), release_dir)
    assert issue is not None
    assert issue.code == "fact_not_found"


def test_verify_facts_returns_one_issue_per_failing_fact(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=14, value="63075"),
            _cell("b", row_idx=20, value="41000"),
        ],
    )
    facts = (
        _fact(fact_id="F1", row_index=14, raw_value=Decimal("63075")),
        _fact(fact_id="F2", row_index=20, raw_value=Decimal("999")),
    )
    issues = verify_facts(facts, release_dir)
    assert len(issues) == 1
    assert "F2" in issues[0].message


def test_verify_facts_returns_nothing_when_every_fact_agrees(tmp_path: Path) -> None:
    release_dir = _write_release(
        tmp_path,
        [
            _cell("a", row_idx=14, value="63075"),
            _cell("b", row_idx=20, value="41000"),
        ],
    )
    facts = (
        _fact(fact_id="F1", row_index=14, raw_value=Decimal("63075")),
        _fact(fact_id="F2", row_index=20, raw_value=Decimal("41000")),
    )
    assert verify_facts(facts, release_dir) == ()


def test_verify_facts_is_empty_for_no_facts(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path, [_cell("a", row_idx=14, value="63075")])
    assert verify_facts((), release_dir) == ()
