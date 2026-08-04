from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from financial_report_qa.ingestion.provenance import ExtractionResult, stable_cell_id
from financial_report_qa.ingestion.table_detector import detect_table_candidates
from financial_report_qa.ingestion.table_extractor import extract_candidates
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import stable_table_id


def extract(tmp_path: Path, source: str) -> ExtractionResult:
    content = source.encode()
    relative = "AAA/2024/AAA_consolidated/source.txt"
    path = tmp_path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    record = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )
    decoded = read_document(tmp_path, record)
    detection = detect_table_candidates(decoded)
    return extract_candidates(decoded, detection)


def test_extracts_rectangular_grid_with_shared_span_placements(tmp_path: Path) -> None:
    source = (
        "Báº¢NG CÃ‚N Äá»I Káº¾ TOÃN\n"
        "<table>\n"
        '<tr><th rowspan="2">Chá»‰ tiÃªu</th><th colspan="2">NÄƒm</th></tr>\n'
        "<tr><th>2024</th><th>2023</th></tr>\n"
        "<tr><td>Doanh thu</td><td>1.000</td><td>900</td></tr>\n"
        "</table>\n"
    )
    result = extract(tmp_path, source)

    assert len(result.tables) == 1
    extracted = result.tables[0]
    assert extracted.table.title_raw == "Báº¢NG CÃ‚N Äá»I Káº¾ TOÃN"
    assert (extracted.table.line_start, extracted.table.line_end) == (2, 6)
    assert (extracted.table.row_count, extracted.table.column_count) == (3, 3)
    assert len(extracted.cells) == 7
    assert len(extracted.placements) == 9
    first_cell_id = extracted.placements[0].cell_id
    assert extracted.placements[3].cell_id == first_cell_id
    assert [cell.value_raw for cell in extracted.cells][-3:] == ["Doanh thu", "1.000", "900"]


def test_decodes_entities_but_keeps_raw_candidate(tmp_path: Path) -> None:
    source = "<table><tr><td>Lá»£i nhuáº­n &amp; thu nháº­p<br>khÃ¡c</td><td>1</td></tr></table>\n"
    result = extract(tmp_path, source)

    assert result.tables[0].cells[0].value_raw == "Lá»£i nhuáº­n & thu nháº­p\nkhÃ¡c"
    assert "&amp;" in result.blocks[0].text


def test_composes_multiline_headers_without_normalizing_values(tmp_path: Path) -> None:
    source = (
        "<table>\n"
        '<tr><th rowspan="2">Chá»‰ tiÃªu</th><th colspan="2">NÄƒm</th></tr>\n'
        "<tr><th>2024</th><th>2023</th></tr>\n"
        "<tr><td>Lá»£i nhuáº­n</td><td>(1.234,50)</td><td>-</td></tr>\n"
        "</table>\n"
    )
    result = extract(tmp_path, source)
    values = {cell.value_raw: cell for cell in result.tables[0].cells}

    assert values["(1.234,50)"].row_label_raw == "Lá»£i nhuáº­n"
    assert values["(1.234,50)"].column_label_raw == "NÄƒm\n2024"
    assert values["(1.234,50)"].value_numeric is None
    assert values["(1.234,50)"].period is None
    assert values["(1.234,50)"].unit is None
    assert (
        values["(1.234,50)"].source_line_start,
        values["(1.234,50)"].source_line_end,
    ) == (4, 4)


@pytest.mark.parametrize(
    ("attribute", "reason"),
    [('rowspan="0"', "invalid_span_value"), ('colspan="100001"', "expansion_limit_exceeded")],
)
def test_invalid_expansion_rejects_whole_candidate_and_continues(
    tmp_path: Path,
    attribute: str,
    reason: str,
) -> None:
    source = (
        f"<table><tr><td {attribute}>bad</td></tr></table>\n"
        "<table><tr><td>good</td><td>1</td></tr></table>\n"
    )
    result = extract(tmp_path, source)

    assert [item.reason for item in result.rejected] == [reason]
    assert len(result.tables) == 1
    assert [cell.value_raw for cell in result.tables[0].cells] == ["good", "1"]


def test_span_collision_is_atomic(tmp_path: Path) -> None:
    source = (
        "<table>"
        '<tr><td rowspan="2">A</td><td>B</td><td rowspan="2">C</td></tr>'
        '<tr><td colspan="2">overlap</td></tr>'
        "</table>\n"
    )
    result = extract(tmp_path, source)

    assert result.tables == ()
    assert [item.reason for item in result.rejected] == ["span_collision"]


def test_ragged_html_rows_use_absent_placements_not_invented_cells(tmp_path: Path) -> None:
    source = "<table><tr><td>A</td><td>1</td></tr><tr><td>B</td></tr></table>\n"
    table = extract(tmp_path, source).tables[0]

    assert (table.table.row_count, table.table.column_count) == (2, 2)
    assert len(table.cells) == 3
    assert len(table.placements) == 3


@pytest.mark.parametrize(
    ("attribute", "reason"),
    [
        ("colspan", "invalid_span_value"),
        ('colspan="' + ("9" * 4_301) + '"', "expansion_limit_exceeded"),
    ],
)
def test_invalid_or_huge_span_rejects_before_grid_expansion(
    tmp_path: Path,
    attribute: str,
    reason: str,
) -> None:
    source = (
        f"<table><tr><td {attribute}>bad</td></tr></table>\n"
        "<table><tr><td>good</td><td>1</td></tr></table>\n"
    )

    result = extract(tmp_path, source)

    assert [item.reason for item in result.rejected] == [reason]
    assert [[cell.value_raw for cell in table.cells] for table in result.tables] == [["good", "1"]]


@pytest.mark.parametrize(
    "prior_line",
    [
        "===== PAGE 1 =====",
        "2024  2023  2022",
        "x" * 201,
    ],
)
def test_title_skips_ineligible_nearest_prior_lines(tmp_path: Path, prior_line: str) -> None:
    source = f"Eligible title\n{prior_line}\n<table><tr><td>A</td><td>1</td></tr></table>\n"

    result = extract(tmp_path, source)

    assert result.tables[0].table.title_raw == "Eligible title"


def test_title_excludes_prior_table_content(tmp_path: Path) -> None:
    source = (
        "<table>\n"
        "<tr><td>Prior table label</td><td>1</td></tr>\n"
        "</table>\n"
        "<table><tr><td>New table</td><td>2</td></tr></table>\n"
    )

    result = extract(tmp_path, source)

    assert len(result.tables) == 2
    assert result.tables[1].table.title_raw is None


def test_html_unit_marker_returns_decoded_cell_text(tmp_path: Path) -> None:
    source = (
        "<table><tr><th>Đơn vị: triệu đồng</th><th>2024</th></tr>"
        "<tr><td>A</td><td>1</td></tr></table>\n"
    )

    result = extract(tmp_path, source)

    assert result.tables[0].table.unit_raw == "Đơn vị: triệu đồng"


def test_extracts_validated_structured_text_with_line_provenance(tmp_path: Path) -> None:
    source = "Metric\t2024\t2023\nRevenue\t1.000\t900\nProfit\t100\t80\n"

    table = extract(tmp_path, source).tables[0]

    assert (table.table.row_count, table.table.column_count) == (3, 3)
    assert [cell.value_raw for cell in table.cells] == [
        "Metric",
        "2024",
        "2023",
        "Revenue",
        "1.000",
        "900",
        "Profit",
        "100",
        "80",
    ]
    assert (table.cells[-1].source_line_start, table.cells[-1].source_line_end) == (3, 3)


def test_merges_compatible_page_continuation_and_drops_repeated_header(
    tmp_path: Path,
) -> None:
    source = (
        "BẢNG KẾT QUẢ KINH DOANH\n"
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
        "<tr><td>Doanh thu</td><td>100</td></tr></table>\n"
        "===== PAGE 2 =====\n"
        "BẢNG KẾT QUẢ KINH DOANH\n"
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
        "<tr><td>Lợi nhuận</td><td>20</td></tr></table>\n"
    )
    result = extract(tmp_path, source)

    assert len(result.tables) == 1
    table = result.tables[0]
    assert (table.table.line_start, table.table.line_end) == (2, 5)
    assert (table.table.row_count, table.table.column_count) == (3, 2)
    assert [cell.value_raw for cell in table.cells] == [
        "Chỉ tiêu",
        "2024",
        "Doanh thu",
        "100",
        "Lợi nhuận",
        "20",
    ]
    assert "continued_across_page" in table.evidence


def test_does_not_merge_different_headers(tmp_path: Path) -> None:
    source = (
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
        "<tr><td>A</td><td>1</td></tr></table>\n"
        "===== PAGE 2 =====\n"
        "<table><tr><th>Mã số</th><th>2023</th></tr>"
        "<tr><td>B</td><td>2</td></tr></table>\n"
    )

    assert len(extract(tmp_path, source).tables) == 2


def test_three_page_merge_rebuilds_ids_rows_and_span_placements(tmp_path: Path) -> None:
    header = (
        '<tr><th rowspan="2">Metric</th><th colspan="2">Year</th></tr>'
        "<tr><th>2024</th><th>2023</th></tr>"
    )
    source = (
        "INCOME STATEMENT\n"
        f"<table>{header}"
        '<tr><td rowspan="2">Revenue</td><td>100</td><td>90</td></tr>'
        '<tr><td colspan="2">Audited</td></tr></table>\n'
        "===== PAGE 2 =====\n"
        "INCOME STATEMENT\n"
        f"<table>{header}<tr><td>Expense</td><td>50</td><td>40</td></tr></table>\n"
        "===== PAGE 3 =====\n"
        "INCOME STATEMENT\n"
        f"<table>{header}<tr><td>Profit</td><td>20</td><td>10</td></tr></table>\n"
    )

    result = extract(tmp_path, source)

    assert len(result.tables) == 1
    extracted = result.tables[0]
    expected_table_id = stable_table_id(result.doc_id, 2, 8)
    assert extracted.table.table_id == expected_table_id
    assert (extracted.table.row_count, extracted.table.column_count) == (6, 3)
    assert [cell.value_raw for cell in extracted.cells].count("Metric") == 1
    cells = {cell.value_raw: cell for cell in extracted.cells}
    row_indices = {
        value: cells[value].row_idx for value in ("Revenue", "Audited", "Expense", "Profit")
    }
    assert row_indices == {
        "Revenue": 2,
        "Audited": 3,
        "Expense": 4,
        "Profit": 5,
    }
    assert cells["Revenue"].cell_id == stable_cell_id(expected_table_id, 2, 0)
    assert cells["Audited"].cell_id == stable_cell_id(expected_table_id, 3, 1)
    assert all(cell.table_id == expected_table_id for cell in extracted.cells)
    placements = {
        (placement.row_idx, placement.col_idx): placement.cell_id
        for placement in extracted.placements
    }
    assert placements[(2, 0)] == placements[(3, 0)] == cells["Revenue"].cell_id
    assert placements[(3, 1)] == placements[(3, 2)] == cells["Audited"].cell_id
    assert placements[(4, 0)] == cells["Expense"].cell_id
    assert placements[(5, 0)] == cells["Profit"].cell_id


def test_does_not_merge_compatible_headers_across_prose(tmp_path: Path) -> None:
    table = (
        "<table><tr><th>Metric</th><th>2024</th></tr><tr><td>Revenue</td><td>100</td></tr></table>"
    )
    source = f"{table}\n===== PAGE 2 =====\nNarrative explanation\n{table}\n"

    assert len(extract(tmp_path, source).tables) == 2


def test_body_unit_marker_is_not_promoted_to_table_metadata(tmp_path: Path) -> None:
    source = (
        "<table><tr><th>Metric</th><th>2024</th></tr>"
        "<tr><td>Đơn vị: kg</td><td>1</td></tr></table>\n"
    )

    result = extract(tmp_path, source)

    assert result.tables[0].table.unit_raw is None
