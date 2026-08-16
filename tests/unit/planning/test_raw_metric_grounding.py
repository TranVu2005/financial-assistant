"""Tests for the Day 23 raw-metric grounding fallback.

ADR 0004 rejected an open-vocabulary "fuzzy match row_label_raw" fallback
(Option B) because with no scoping, several near-matching rows makes picking
one a guess. This module is a narrower mechanism scoped to only a question's
own retrieved candidate tables, gated by requiring an unambiguous single
match -- see docs/decisions/0004-metric-locator-strategy.md and
docs/plans/day23-coverage-and-evidence-table.md.
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from financial_report_qa.data.dataset_builder import CELL_SCHEMA
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.raw_metric_grounding import (
    candidate_column_labels,
    candidate_row_labels,
    ground_raw_metric,
    plan_with_raw_grounding_fallback,
)

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64
_KNOWN_TABLES = frozenset({TABLE_A, TABLE_B})


def _cell(
    cell_id: str,
    table_id: str,
    *,
    row_label_raw: str | None,
    col_idx: int = 1,
    value_numeric: str | None = "100",
    column_label_raw: str = "Năm 2020",
) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "table_id": table_id,
        "row_idx": 0,
        "col_idx": col_idx,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": column_label_raw,
        "column_label_canonical": None,
        "value_raw": value_numeric or "",
        "value_numeric": Decimal(value_numeric) if value_numeric is not None else None,
        "period": "2020",
        "unit": "VND",
        "source_line_start": 1,
        "source_line_end": 1,
        "extraction_confidence": 0.9,
    }


def _write_cells(tmp_path: Path, cells: list[dict[str, object]]) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist(cells, schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def test_ground_raw_metric_matches_unique_candidate_label(tmp_path: Path) -> None:
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi")])
    result = ground_raw_metric(
        "Lãi tiền gửi năm 2018 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result == "Lãi tiền gửi"


def test_ground_raw_metric_returns_none_when_no_label_matches(tmp_path: Path) -> None:
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi")])
    result = ground_raw_metric(
        "Doanh thu thuần năm 2018 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result is None


def test_ground_raw_metric_ignores_labels_below_min_token_count(tmp_path: Path) -> None:
    """A 2-token label like 'Số dư' is exactly the shape of the generic
    boilerplate captions measured in the corpus (Số dư cuối năm, Cộng, ...)
    -- too short to trust as a self-sufficient metric name."""
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Số dư")])
    result = ground_raw_metric("Số dư của VJC năm 2020 là bao nhiêu?", [TABLE_A], release_dir)
    assert result is None


def test_ground_raw_metric_returns_none_when_multiple_labels_match(tmp_path: Path) -> None:
    """Never guess: if the question contains two distinct candidate-table row
    labels, picking one over the other would be exactly what ADR 0004 Option
    B was rejected for."""
    release_dir = _write_cells(
        tmp_path,
        [
            _cell("cell_a", TABLE_A, row_label_raw="Tiền gửi của khách hàng"),
            _cell("cell_b", TABLE_A, row_label_raw="Tiền gửi có kỳ hạn"),
        ],
    )
    question = "Tiền gửi của khách hàng và tiền gửi có kỳ hạn của VJC năm 2020 là bao nhiêu?"
    result = ground_raw_metric(question, [TABLE_A], release_dir)
    assert result is None


def test_ground_raw_metric_ignores_non_numeric_cells(tmp_path: Path) -> None:
    """A row-label cell (col_idx == 0) or a placeholder cell with no
    value_numeric must never seed a metric grounding match."""
    release_dir = _write_cells(
        tmp_path,
        [
            _cell("cell_label", TABLE_A, row_label_raw="Lãi tiền gửi", col_idx=0),
            _cell("cell_placeholder", TABLE_A, row_label_raw="Lãi tiền gửi", value_numeric=None),
        ],
    )
    result = ground_raw_metric(
        "Lãi tiền gửi năm 2020 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result is None


def test_ground_raw_metric_is_case_and_diacritic_normalization_tolerant(tmp_path: Path) -> None:
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="lãi   tiền gửi")])
    result = ground_raw_metric(
        "LÃI TIỀN GỬI năm 2020 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result == "lãi   tiền gửi"


def test_ground_raw_metric_scopes_to_given_candidate_table_ids(tmp_path: Path) -> None:
    """A label that only exists in a table outside this question's own
    candidate set must not ground a match -- grounding must not reach into
    the whole corpus."""
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_B, row_label_raw="Lãi tiền gửi")])
    result = ground_raw_metric(
        "Lãi tiền gửi năm 2020 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result is None


def test_ground_raw_metric_strips_leading_and_trailing_whitespace(tmp_path: Path) -> None:
    """The returned label feeds `QueryEntities.metrics`, whose validator
    rejects any value that differs from its own `.strip()` -- corpus labels
    can carry OCR whitespace noise the caller must never have to know about."""
    release_dir = _write_cells(
        tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="  Lãi tiền gửi  ")]
    )
    result = ground_raw_metric(
        "Lãi tiền gửi năm 2020 của VJC là bao nhiêu?", [TABLE_A], release_dir
    )
    assert result == "Lãi tiền gửi"


def test_ground_raw_metric_treats_same_normalized_label_across_tables_as_one_match(
    tmp_path: Path,
) -> None:
    """The same metric name recurring (with different casing) across two
    candidate tables is one distinct metric identity, not an ambiguity."""
    release_dir = _write_cells(
        tmp_path,
        [
            _cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi"),
            _cell("cell_b", TABLE_B, row_label_raw="LÃI TIỀN GỬI"),
        ],
    )
    result = ground_raw_metric(
        "Lãi tiền gửi năm 2020 của VJC là bao nhiêu?", [TABLE_A, TABLE_B], release_dir
    )
    assert result is not None


def test_plan_with_raw_grounding_fallback_never_invoked_when_rule_planner_succeeds(
    tmp_path: Path,
) -> None:
    """Mirrors the existing `route_plan`/LLM-fallback guarantee: a rule plan
    that already succeeds must never be second-guessed by the fallback."""
    release_dir = _write_cells(tmp_path, [])
    entities = parse_query_entities("Tra cứu doanh thu thuần của NVL năm 2023.")

    result, grounded = plan_with_raw_grounding_fallback(
        entities,
        candidate_table_ids=(TABLE_A,),
        known_table_ids=_KNOWN_TABLES,
        release_dir=release_dir,
    )

    assert grounded is False
    assert result.plan is not None
    assert result.plan.metric is not None
    assert result.plan.metric.canonical == "net_revenue"


def test_plan_with_raw_grounding_fallback_resolves_metric_unknown_abstain(
    tmp_path: Path,
) -> None:
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi")])
    entities = parse_query_entities("Lãi tiền gửi của NVL năm 2023 là bao nhiêu?")
    assert entities.ambiguity == ("metric_unknown",)

    result, grounded = plan_with_raw_grounding_fallback(
        entities,
        candidate_table_ids=(TABLE_A,),
        known_table_ids=_KNOWN_TABLES,
        release_dir=release_dir,
    )

    assert grounded is True
    assert result.plan is not None
    assert result.plan.metric is not None
    assert result.plan.metric.raw_text == "Lãi tiền gửi"


def test_plan_with_raw_grounding_fallback_skips_when_other_ambiguity_present(
    tmp_path: Path,
) -> None:
    """Only fires when `metric_unknown` is the *sole* blocker -- a question
    with, e.g., a missing company too must not silently pick a metric while
    still lacking evidence for another required field."""
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi")])
    entities = parse_query_entities("Lãi tiền gửi năm 2023 là bao nhiêu?")
    assert "company_missing" in entities.ambiguity
    assert "metric_unknown" in entities.ambiguity

    result, grounded = plan_with_raw_grounding_fallback(
        entities,
        candidate_table_ids=(TABLE_A,),
        known_table_ids=_KNOWN_TABLES,
        release_dir=release_dir,
    )

    assert grounded is False
    assert result.plan is None
    assert result.abstain_codes == ("entity_ambiguous",)


def test_plan_with_raw_grounding_fallback_returns_original_abstain_when_no_grounding(
    tmp_path: Path,
) -> None:
    release_dir = _write_cells(tmp_path, [])
    entities = parse_query_entities("Lãi tiền gửi của NVL năm 2023 là bao nhiêu?")

    result, grounded = plan_with_raw_grounding_fallback(
        entities,
        candidate_table_ids=(TABLE_A,),
        known_table_ids=_KNOWN_TABLES,
        release_dir=release_dir,
    )

    assert grounded is False
    assert result.plan is None
    assert result.abstain_codes == ("entity_ambiguous",)


def test_candidate_row_labels_lists_real_numeric_row_labels(tmp_path: Path) -> None:
    """Day 25: the LLM cell-grounding tier must choose from labels that
    genuinely carry a numeric value in this question's own candidate
    tables -- never from an arbitrary vocabulary."""
    release_dir = _write_cells(
        tmp_path,
        [
            _cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi"),
            _cell("cell_b", TABLE_A, row_label_raw="Chi phí nhân viên"),
            _cell("cell_label_col", TABLE_A, row_label_raw="Bỏ qua", col_idx=0),
            _cell("cell_no_value", TABLE_A, row_label_raw="Không có số", value_numeric=None),
        ],
    )
    labels = candidate_row_labels(release_dir, [TABLE_A])
    assert "Lãi tiền gửi" in labels
    assert "Chi phí nhân viên" in labels
    assert "Bỏ qua" not in labels
    assert "Không có số" not in labels


def test_candidate_row_labels_is_empty_without_tables(tmp_path: Path) -> None:
    release_dir = _write_cells(tmp_path, [])
    assert candidate_row_labels(release_dir, []) == ()


def test_candidate_row_labels_reads_the_release_once_per_process(tmp_path: Path) -> None:
    """`_candidate_raw_labels` opened a fresh DuckDB connection and re-scanned
    the whole cells file on every call, and `ground_raw_metric` calls it once
    per question. Over the 1,012 official questions that is ~500 full scans of
    6.2M rows, which turned a documented ~2-minute export into a 15+ minute one
    (faulthandler dump caught the process inside `_hardened_connection`).

    Deleting the parquet after the first call proves the second is served from
    the materialized table rather than re-reading the release.
    """
    from financial_report_qa.planning.raw_metric_grounding import _label_connection

    _label_connection.cache_clear()
    release_dir = _write_cells(tmp_path, [_cell("cell_a", TABLE_A, row_label_raw="Lãi tiền gửi")])
    first = candidate_row_labels(release_dir, [TABLE_A])
    assert first

    (release_dir / "cells.parquet").unlink()
    assert candidate_row_labels(release_dir, [TABLE_A]) == first
    _label_connection.cache_clear()


def test_candidate_column_labels_lists_the_columns_of_one_row(tmp_path: Path) -> None:
    """The menu `choose_column_label` indexes into: the real headers carrying a
    number on that row, and only that row -- another row's columns would let
    the model pick a header the chosen row does not actually have."""
    release_dir = _write_cells(
        tmp_path,
        [
            _cell("c1", TABLE_A, row_label_raw="Thuế GTGT", column_label_raw="Số đầu năm"),
            _cell("c2", TABLE_A, row_label_raw="Thuế GTGT", column_label_raw="Số cuối năm"),
            _cell("c3", TABLE_A, row_label_raw="Thuế TNDN", column_label_raw="Số đã nộp"),
        ],
    )
    columns = candidate_column_labels(release_dir, [TABLE_A], "Thuế GTGT")
    assert set(columns) == {"Số đầu năm", "Số cuối năm"}
    assert "Số đã nộp" not in columns


def test_candidate_column_labels_matches_the_row_normalization_tolerantly(
    tmp_path: Path,
) -> None:
    """The row label arrives from `choose_row_label`, i.e. straight out of the
    corpus, so it can differ in casing or spacing from the stored value."""
    release_dir = _write_cells(
        tmp_path,
        [_cell("c1", TABLE_A, row_label_raw="  Thuế   GTGT ", column_label_raw="Số cuối năm")],
    )
    assert candidate_column_labels(release_dir, [TABLE_A], "thuế gtgt") == ("Số cuối năm",)
