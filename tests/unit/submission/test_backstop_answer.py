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


@pytest.fixture
def release_dir():
    path = Path("data/processed/release_v2_422df141c935")
    if not path.exists():
        pytest.skip("release chưa có sẵn trên máy này")
    return path


@pytest.fixture
def sample_table_ids(release_dir):
    """A real table_id with >= 2 numeric cells (Minor finding 9, 2026-08-21
    final review: "Critical 1 wearing a disguise" -- without the `HAVING`
    filter, this fixture could pick a real singleton-cell table (2,446 of
    130,518 exist), which `build_backstop_item` now correctly refuses to
    answer from, making these two tests flaky depending on which table the
    unfiltered `LIMIT 1` happened to return."""
    import duckdb

    connection = duckdb.connect(":memory:")
    frame = connection.execute(
        "SELECT table_id FROM read_parquet(?) WHERE value_numeric IS NOT NULL "
        "AND period IS NOT NULL GROUP BY table_id HAVING COUNT(*) >= 2 "
        "ORDER BY table_id LIMIT 1",
        [str(release_dir / "cells.parquet")],
    ).fetchdf()
    connection.close()
    return tuple(frame["table_id"].tolist())


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
    cell_id: str,
    table_id: str,
    *,
    row_label_raw: str,
    value: str,
    period: str,
    row_idx: int = 0,
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": row_idx,
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
    # Each table carries >= 2 numeric cells (Critical 1, 2026-08-21 final
    # review): a singleton-cell table is now rejected by
    # `_uniquely_addressable_row` -- its one row's `value` would equal
    # `item.answer`, reproducing the exact hardcode shape compliance checks
    # C1+C2 exist to catch. `row_idx=0` on the primary cell keeps it first
    # in `build_cell_frame`'s `ORDER BY table_id, row_idx, col_idx`, so the
    # existing assertions on which value gets chosen still hold.
    cells = [
        _cell("cell_a", TABLE_ID, row_label_raw="Doanh thu", value="1000", period="2023", row_idx=0),
        _cell(
            "cell_a2", TABLE_ID, row_label_raw="Chi phí", value="200", period="2023", row_idx=1
        ),
        _cell(
            "cell_b", TABLE_ID_OTHER, row_label_raw="Lợi nhuận", value="500", period="2022", row_idx=0
        ),
        _cell(
            "cell_b2",
            TABLE_ID_OTHER,
            row_label_raw="Chi phí khác",
            value="50",
            period="2022",
            row_idx=1,
        ),
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


def _append_rows(
    release_dir: Path,
    *,
    documents: list[dict[str, object]],
    tables: list[dict[str, object]],
    cells: list[dict[str, object]],
) -> None:
    """Append extra documents/tables/cells onto an already-written release
    (used to add a singleton-cell table alongside `_write_release`'s
    normal, >= 2-cell tables, without disturbing the existing fixture)."""
    for name, schema, new_rows in (
        ("documents.parquet", DOCUMENT_SCHEMA, documents),
        ("tables.parquet", TABLE_SCHEMA, tables),
        ("cells.parquet", CELL_SCHEMA, cells),
    ):
        path = release_dir / name
        existing = pq.read_table(path)  # type: ignore[no-untyped-call]
        appended = pa.concat_tables(
            [existing, pa.Table.from_pylist(new_rows, schema=schema)]  # type: ignore[no-untyped-call]
        )
        pq.write_table(appended, path)  # type: ignore[no-untyped-call]


def test_backstop_uses_candidate_table_when_available(tmp_path: Path) -> None:
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=1, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [TABLE_ID], release_dir)

    assert item.id == 1
    assert item.answer == 1000.0
    assert rows[0]["company_code"] == "ACB"


def test_backstop_falls_back_to_whole_corpus_when_no_candidate_tables(tmp_path: Path) -> None:
    """Retrieval returning nothing (42/1.012 real questions) must not
    prevent a valid backstop -- some numeric cell exists somewhere. But the
    arbitrary fallback table must never be reported as relevant (spec §6.1:
    "không được emit bảng tuỳ ý")."""
    release_dir = _write_release(tmp_path)
    question = RawQuestion(id=2, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [], release_dir)

    assert item.id == 2
    assert item.answer in (1000.0, 500.0)
    assert rows[0]["value"] in (Decimal("1000"), Decimal("500"))
    assert item.relevant_docs == ()
    assert item.relevant_tables == ()


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


def test_backstop_rejects_singleton_cell_table_and_falls_through_to_next_candidate(
    tmp_path: Path,
) -> None:
    """Critical 1: a table with exactly 1 numeric cell must never be chosen
    -- its one CSV row's `value` would equal `item.answer`, reproducing the
    exact hardcode shape (`result = df["answer"].iloc[0]`) the whole plan
    exists to eliminate (measured on the real corpus: 2.446/130.518 tables).
    A singleton-cell candidate ranked first must be skipped in favor of the
    next usable candidate, not silently selected.
    """
    release_dir = _write_release(tmp_path)
    singleton_table_id = "tbl_" + "9" * 64
    singleton_doc_id = "doc_" + "9" * 64
    _append_rows(
        release_dir,
        documents=[_document(singleton_doc_id, "SGT", 2023, "SGT/2023/report.txt")],
        tables=[_table(singleton_table_id, singleton_doc_id)],
        cells=[
            _cell(
                "cell_singleton",
                singleton_table_id,
                row_label_raw="Doanh thu",
                value="9999",
                period="2023",
            )
        ],
    )
    question = RawQuestion(id=6, question="Câu hỏi không xác định được.")

    # Singleton table ranked first (would have been chosen under the old
    # first-usable-cell logic); TABLE_ID (>= 2 numeric cells) ranked second.
    item, rows = build_backstop_item(question, [singleton_table_id, TABLE_ID], release_dir)

    assert item.answer != 9999.0, "phải bỏ qua bảng chỉ có 1 ô numeric"
    assert len(rows) >= 2
    assert {row["table_id"] for row in rows} == {TABLE_ID}


def test_backstop_falls_through_to_any_corpus_table_when_all_candidates_are_singletons(
    tmp_path: Path,
) -> None:
    """Critical 2: when every ranked candidate table is unusable (e.g. all
    singleton-cell tables), `build_backstop_item` must NOT raise and abort
    the whole export -- it must fall through to `_any_corpus_table_id`
    (the same floor already used for the `no_candidate_tables` case), same
    as an empty candidate list. `relevant_docs`/`relevant_tables` must stay
    empty for this path, per spec §6.1.
    """
    release_dir = _write_release(tmp_path)
    singleton_table_id = "tbl_" + "8" * 64
    singleton_doc_id = "doc_" + "8" * 64
    # col_idx=0 keeps this cell out of `_any_corpus_table_id`'s own query
    # (`col_idx > 0`) so the fallback deterministically lands on a real,
    # non-singleton table (TABLE_ID / TABLE_ID_OTHER) rather than this one.
    _append_rows(
        release_dir,
        documents=[_document(singleton_doc_id, "SGT", 2023, "SGT/2023/report.txt")],
        tables=[_table(singleton_table_id, singleton_doc_id)],
        cells=[
            {
                **_cell(
                    "cell_singleton2",
                    singleton_table_id,
                    row_label_raw="Doanh thu",
                    value="9999",
                    period="2023",
                ),
                "col_idx": 0,
            }
        ],
    )
    question = RawQuestion(id=7, question="Câu hỏi không xác định được.")

    item, rows = build_backstop_item(question, [singleton_table_id], release_dir)

    assert item.id == 7
    assert len(rows) >= 2
    assert item.relevant_docs == ()
    assert item.relevant_tables == ()


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


def test_backstop_raises_when_any_corpus_fallback_table_is_unusable(tmp_path: Path) -> None:
    """Minor 3 (2026-08-21 final review round 2): exercises the *other*
    RuntimeError site -- not `_any_corpus_table_id`'s own "release is
    completely empty" raise (covered by
    `test_backstop_raises_when_release_has_no_numeric_cell_at_all`), but the
    one in `build_backstop_item` right after it, for when
    `_any_corpus_table_id` *does* find a table satisfying its own >= 2
    numeric-cell guard, yet that table still fails
    `_uniquely_addressable_row`'s stricter usability check.

    Constructed via a `period` mismatch: `_ANY_TABLE_QUERY` only checks the
    *raw* `c.period IS NOT NULL`, but `build_cell_frame` recomputes `period`
    via `TRY_CAST(LEFT(c.period, 4) AS INTEGER)`. A raw `period` value that
    is non-null but not a parseable year (e.g. "??") passes the raw-period
    filter yet resolves to a NULL computed `period` in `build_cell_frame`'s
    output -- so `_uniquely_addressable_row`'s `period.notna()` usable-filter
    rejects every row of that table, and no other table exists in this
    release for the fallback to try instead.
    """
    release_dir = tmp_path / "release_bad_period"
    release_dir.mkdir()
    bad_table_id = "tbl_" + "7" * 64
    bad_doc_id = "doc_" + "7" * 64
    documents = [_document(bad_doc_id, "BAD", 2023, "BAD/2023/report.txt")]
    tables = [_table(bad_table_id, bad_doc_id)]
    cells = [
        _cell("cell_bad1", bad_table_id, row_label_raw="Doanh thu", value="1", period="??", row_idx=0),
        _cell("cell_bad2", bad_table_id, row_label_raw="Chi phí", value="2", period="??", row_idx=1),
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
    question = RawQuestion(id=8, question="Câu hỏi không xác định được.")

    with pytest.raises(RuntimeError, match="bảng dự phòng toàn kho"):
        build_backstop_item(question, [], release_dir)


def test_backstop_emits_full_source_table_not_one_row(release_dir, sample_table_ids) -> None:
    """BI-4: backstop không được tổng hợp dòng nào."""
    from financial_report_qa.submission.backstop_answer import build_backstop_item
    from financial_report_qa.submission.contracts import RawQuestion

    question = RawQuestion(id=1, question="Doanh thu thuần năm 2023?")
    item, rows = build_backstop_item(question, sample_table_ids, release_dir)

    assert len(rows) >= 2, "backstop phải xuất trọn bảng, không phải một dòng"
    assert {"table_id", "row_idx", "col_idx"} <= set(rows[0].keys())


def test_backstop_answer_replays_from_its_own_csv(release_dir, sample_table_ids) -> None:
    """C7: đáp án phải tính được từ chính CSV kèm theo."""
    import pandas as pd

    from financial_report_qa.submission.backstop_answer import build_backstop_item
    from financial_report_qa.submission.compliance import check_item
    from financial_report_qa.submission.contracts import RawQuestion

    question = RawQuestion(id=1, question="Doanh thu thuần năm 2023?")
    item, rows = build_backstop_item(question, sample_table_ids, release_dir)
    frame = pd.DataFrame(list(rows))
    frame["period"] = frame["period"].astype("Int64")

    violations = check_item(item, frame, timeout_seconds=5)
    assert violations == (), f"backstop vẫn vi phạm: {violations}"
