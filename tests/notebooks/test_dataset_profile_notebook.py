"""Behavioral contract for the OCR dataset profiling notebook."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "01_dataset_profile.ipynb"


def _notebook() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(NOTEBOOK.read_text(encoding="utf-8")),
    )


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


def test_parse_report_path_extracts_financial_report_hierarchy(tmp_path: Path) -> None:
    root = tmp_path / "ocr_result"
    path = (
        root / "HTG" / "2022" / "HTG_Baocaotaichinh_2022_Kiemtoan_Hopnhat" / "report_extracted.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")

    parse_report_path = cast(
        Callable[[Path, Path], dict[str, object]],
        _load_helpers()["parse_report_path"],
    )
    parsed = parse_report_path(path, root)

    assert parsed["ticker"] == "HTG"
    assert parsed["year"] == 2022
    assert parsed["scope"] == "consolidated"
    assert parsed["assurance"] == "audited"
    assert parsed["structure_status"] == "valid"


def test_build_inventory_keeps_malformed_paths_as_anomalies(tmp_path: Path) -> None:
    root = tmp_path / "ocr_result"
    valid = root / "FPT" / "2023" / "FPT_Baocaotaichinh_2023" / "valid.txt"
    malformed = root / "loose.txt"
    valid.parent.mkdir(parents=True)
    valid.write_text("<table>100</table>", encoding="utf-8")
    malformed.write_text("noise", encoding="utf-8")

    build_inventory = cast(Callable[[Path], pd.DataFrame], _load_helpers()["build_inventory"])
    frame = build_inventory(root)

    assert len(frame) == 2
    assert set(frame["structure_status"]) == {"valid", "malformed"}
    assert frame["relative_path"].tolist() == sorted(frame["relative_path"].tolist())


def test_sample_paths_is_bounded_and_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.txt" for index in range(10)]
    sample_paths = cast(
        Callable[[Sequence[Path], int, int], list[Path]],
        _load_helpers()["sample_paths"],
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
        Callable[[Path, int], dict[str, object]],
        _load_helpers()["inspect_text_file"],
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
        Callable[[Path, int], dict[str, object]],
        _load_helpers()["inspect_text_file"],
    )
    result = inspect_text_file(path, 1_000_000)

    assert result["utf8_valid"] is False
    assert cast(int, result["replacement_char_count"]) >= 1
    assert result["read_error"] is None


def test_readiness_summary_has_actionable_schema() -> None:
    helpers = _load_helpers()
    inventory = pd.DataFrame(
        [
            {
                "structure_status": "malformed",
                "size_bytes": 0,
                "ticker": None,
                "year": None,
            }
        ]
    )
    content = pd.DataFrame(
        [
            {
                "utf8_valid": False,
                "has_html_table": False,
                "read_error": None,
            }
        ]
    )
    build_readiness_summary = cast(
        Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame],
        helpers["build_readiness_summary"],
    )

    summary = build_readiness_summary(inventory, content)

    assert list(summary.columns) == ["priority", "finding", "evidence", "next_action"]
    assert {"path parsing", "quarantine", "encoding", "table detection"}.issubset(
        set(summary["finding"])
    )


def test_notebook_contains_required_analysis_sections() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        _cell_source(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for heading in (
        "Corpus overview",
        "Coverage",
        "Anomalies",
        "Content quality sample",
        "Recommended next steps",
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
        "open(DATA_ROOT, 'w'",
        'open(DATA_ROOT, "w"',
        ".to_csv(DATA_ROOT",
        ".to_parquet(DATA_ROOT",
    )
    assert not any(fragment in code for fragment in forbidden_fragments)
