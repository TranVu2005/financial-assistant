"""Behavioral contract for the ViFinQA dataset profiling notebook."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "01_dataset_profile.ipynb"


def _notebook() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(NOTEBOOK.read_text(encoding="utf-8")))


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _load_helpers() -> dict[str, Any]:
    notebook = _notebook()
    source = next(
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "profile-helpers" in cell.get("metadata", {}).get("tags", [])
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_parse_report_path_extracts_vifinqa_hierarchy(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = (
        root
        / "AAA"
        / "2015"
        / "AAA_financial_statements_2015_consolidated"
        / "report.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")

    parse_report_path = cast(
        Callable[[Path, Path], dict[str, object]], _load_helpers()["parse_report_path"]
    )
    parsed = parse_report_path(path, root)

    assert parsed["ticker"] == "AAA"
    assert parsed["year"] == 2015
    assert parsed["statement_type"] == "consolidated"
    assert parsed["structure_status"] == "valid"


def test_parse_report_path_keeps_unexpected_depth_as_anomaly(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = root / "AAA" / "2015" / "document" / "extra" / "report.txt"
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")

    parse_report_path = cast(
        Callable[[Path, Path], dict[str, object]], _load_helpers()["parse_report_path"]
    )
    parsed = parse_report_path(path, root)

    assert parsed["structure_status"] == "malformed"
    assert "expected exactly" in cast(str, parsed["structure_issue"])


def test_load_company_map_normalizes_and_flags_rows(tmp_path: Path) -> None:
    source = tmp_path / "code_stock.csv"
    source.write_text(
        "Mã CK,Tên công ty\naaa,Công ty AAA\n,Thiếu mã\n",
        encoding="utf-8",
    )

    load_company_map = cast(
        Callable[[Path], pd.DataFrame], _load_helpers()["load_company_map"]
    )
    frame = load_company_map(source)

    assert frame.loc[0, "ticker"] == "AAA"
    assert frame["is_valid"].tolist() == [True, False]
    assert frame.loc[1, "validation_issue"] == "invalid or missing ticker"


def test_load_company_map_rejects_missing_columns(tmp_path: Path) -> None:
    source = tmp_path / "code_stock.csv"
    source.write_text("ticker,name\nAAA,Công ty AAA\n", encoding="utf-8")
    load_company_map = cast(
        Callable[[Path], pd.DataFrame], _load_helpers()["load_company_map"]
    )

    try:
        load_company_map(source)
    except ValueError as error:
        assert "Mã CK" in str(error)
    else:
        raise AssertionError("missing ViFinQA columns must be rejected")


def test_load_questions_preserves_malformed_and_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        '{"id": 1, "question": "Doanh thu HPG năm 2022?"}\n'
        "{bad json}\n"
        '{"id": 2, "question": ""}\n',
        encoding="utf-8",
    )

    load_questions = cast(Callable[[Path], pd.DataFrame], _load_helpers()["load_questions"])
    frame = load_questions(source)

    assert frame["line_number"].tolist() == [1, 2, 3]
    assert frame["is_valid"].tolist() == [True, False, False]
    assert "JSON" in cast(str, frame.loc[1, "validation_issue"])
    assert frame.loc[2, "validation_issue"] == "question must be a non-empty string"


def test_build_report_inventory_is_sorted_and_keeps_malformed_paths(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    valid = (
        root
        / "VCB"
        / "2022"
        / "VCB_financial_statements_2022_separate"
        / "report.txt"
    )
    malformed = root / "loose.txt"
    valid.parent.mkdir(parents=True)
    valid.write_text("<table>100</table>", encoding="utf-8")
    malformed.write_text("noise", encoding="utf-8")

    build_report_inventory = cast(
        Callable[[Path], pd.DataFrame], _load_helpers()["build_report_inventory"]
    )
    frame = build_report_inventory(root)

    assert set(frame["structure_status"]) == {"valid", "malformed"}
    assert frame["relative_path"].tolist() == sorted(frame["relative_path"].tolist())
    assert frame.loc[frame["ticker"].eq("VCB"), "statement_type"].item() == "separate"


def test_extract_mentioned_tickers_uses_token_boundaries() -> None:
    extract_mentioned_tickers = cast(
        Callable[[str, Sequence[str]], tuple[str, ...]],
        _load_helpers()["extract_mentioned_tickers"],
    )

    result = extract_mentioned_tickers("So sánh HPG và VCB năm 2022", ["VCB", "HP", "HPG"])

    assert result == ("HPG", "VCB")


def test_sample_paths_is_bounded_and_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.txt" for index in range(10)]
    sample_paths = cast(
        Callable[[Sequence[Path], int, int], list[Path]], _load_helpers()["sample_paths"]
    )

    first = sample_paths(paths, 4, 42)
    second = sample_paths(list(reversed(paths)), 4, 42)

    assert first == second
    assert len(first) == 4
    assert len(sample_paths(paths, 40, 42)) == 10


def test_inspect_text_file_reports_utf8_and_table_markers(tmp_path: Path) -> None:
    path = tmp_path / "report.txt"
    path.write_text(
        "Bảng tài chính\n<table><tr><td>100</td></tr></table>",
        encoding="utf-8",
    )
    inspect_text_file = cast(
        Callable[[Path, int], dict[str, object]], _load_helpers()["inspect_text_file"]
    )

    result = inspect_text_file(path, 1_000_000)

    assert result["utf8_valid"] is True
    assert result["has_html_table"] is True
    assert result["line_count"] == 2
    assert result["read_error"] is None


def test_inspect_text_file_records_invalid_utf8_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "invalid.txt"
    path.write_bytes(b"valid-prefix\xff\xfe")
    inspect_text_file = cast(
        Callable[[Path, int], dict[str, object]], _load_helpers()["inspect_text_file"]
    )

    result = inspect_text_file(path, 1_000_000)

    assert result["utf8_valid"] is False
    assert cast(int, result["replacement_char_count"]) >= 1
    assert result["read_error"] is None


def test_inspect_text_file_does_not_reject_utf8_split_at_sample_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "boundary.txt"
    path.write_text("A€B", encoding="utf-8")
    inspect_text_file = cast(
        Callable[[Path, int], dict[str, object]], _load_helpers()["inspect_text_file"]
    )

    result = inspect_text_file(path, 2)

    assert result["truncated"] is True
    assert result["utf8_valid"] is True
    assert result["replacement_char_count"] == 0


def test_inspect_text_file_keeps_encoding_unknown_when_read_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    inspect_text_file = cast(
        Callable[[Path, int], dict[str, object]], _load_helpers()["inspect_text_file"]
    )

    result = inspect_text_file(missing, 1_000)

    assert result["read_error"] is not None
    assert result["utf8_valid"] is None


def test_notebook_contains_required_vifinqa_analysis_sections() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for heading in (
        "Dataset overview",
        "Report coverage",
        "Question analysis",
        "Integrity checks",
        "Report content sample",
        "Readiness summary",
    ):
        assert heading in markdown


def test_notebook_cells_have_unique_stable_ids() -> None:
    notebook = _notebook()
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert all(isinstance(cell_id, str) and cell_id for cell_id in cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def test_notebook_does_not_write_to_data_root() -> None:
    notebook = _notebook()
    code = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    forbidden_fragments = (
        "DATA_ROOT.mkdir",
        "DATA_ROOT.write_",
        "STATEMENTS_ROOT.mkdir",
        "open(DATA_ROOT, 'w'",
        'open(DATA_ROOT, "w"',
        ".to_csv(DATA_ROOT",
        ".to_parquet(DATA_ROOT",
    )

    assert not any(fragment in code for fragment in forbidden_fragments)
