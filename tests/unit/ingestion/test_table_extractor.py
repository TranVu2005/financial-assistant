from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from financial_report_qa.ingestion.provenance import (
    DetectionResult,
    ExtractionResult,
    stable_cell_id,
)
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


def test_same_line_tables_have_distinct_table_and_cell_ids(tmp_path: Path) -> None:
    source = (
        "<table><tr><td>A</td><td>1</td></tr></table><table><tr><td>B</td><td>2</td></tr></table>\n"
    )

    tables = extract(tmp_path, source).tables

    assert len(tables) == 2
    assert [(table.table.line_start, table.table.line_end) for table in tables] == [(1, 1), (1, 1)]
    assert [table.table.source_ordinal for table in tables] == [0, 1]
    assert len({table.table.table_id for table in tables}) == 2
    assert len({cell.cell_id for table in tables for cell in table.cells}) == 4


def test_td_header_drives_metric_column_selection(tmp_path: Path) -> None:
    source = (
        "<table>"
        "<tr><td>STT</td><td>Chỉ tiêu</td><td>Mã số</td><td>2024</td></tr>"
        "<tr><td>1</td><td>Doanh thu</td><td>01</td><td>100</td></tr>"
        "</table>\n"
    )

    table = extract(tmp_path, source).tables[0]
    values = {cell.value_raw: cell for cell in table.cells}

    assert values["100"].row_label_raw == "Doanh thu"
    assert values["100"].column_label_raw == "2024"
    assert values["1"].row_label_raw == "Doanh thu"


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


def test_section_banner_context_propagates_to_child_rows(tmp_path: Path) -> None:
    source = (
        "<table>"
        "<tr><td>Chỉ tiêu</td><td>2024</td></tr>"
        "<tr><td>A. TÀI SẢN NGẮN HẠN</td><td></td></tr>"
        "<tr><td>Tiền và các khoản tương đương tiền</td><td>100</td></tr>"
        "<tr><td>Hàng tồn kho</td><td>200</td></tr>"
        "<tr><td>B. TÀI SẢN DÀI HẠN</td><td></td></tr>"
        "<tr><td>Tài sản cố định</td><td>300</td></tr>"
        "</table>\n"
    )

    values = {cell.value_raw: cell for cell in extract(tmp_path, source).tables[0].cells}

    assert values["A. TÀI SẢN NGẮN HẠN"].row_group_context_raw is None
    assert values["100"].row_group_context_raw == "A. TÀI SẢN NGẮN HẠN"
    assert values["200"].row_group_context_raw == "A. TÀI SẢN NGẮN HẠN"
    assert values["B. TÀI SẢN DÀI HẠN"].row_group_context_raw == "A. TÀI SẢN NGẮN HẠN"
    assert values["300"].row_group_context_raw == "B. TÀI SẢN DÀI HẠN"


def test_no_section_banner_leaves_group_context_none(tmp_path: Path) -> None:
    source = (
        "<table>"
        "<tr><td>Chỉ tiêu</td><td>2024</td></tr>"
        "<tr><td>Doanh thu</td><td>100</td></tr>"
        "</table>\n"
    )

    table = extract(tmp_path, source).tables[0]

    assert all(cell.row_group_context_raw is None for cell in table.cells)


def test_header_rows_have_no_group_context(tmp_path: Path) -> None:
    source = (
        "<table>"
        '<tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">Năm</th></tr>'
        "<tr><th>2024</th><th>2023</th></tr>"
        "<tr><td>A. TÀI SẢN</td><td></td><td></td></tr>"
        "<tr><td>Tiền</td><td>1</td><td>2</td></tr>"
        "</table>\n"
    )

    table = extract(tmp_path, source).tables[0]
    header_cells = [cell for cell in table.cells if cell.row_idx < 2]

    assert all(cell.row_group_context_raw is None for cell in header_cells)


def test_section_banner_context_tracks_full_nested_path(tmp_path: Path) -> None:
    """Would fail if a data row only saw its immediate parent, not the full ancestor path.

    A banner row's own context reflects whichever path was still open right
    before it appeared (matching the pre-existing single-level semantics,
    where a sibling banner's context is "the previous banner"), not its true
    structural ancestors. Data rows, which is what downstream retrieval and
    analysis actually consume, always get their real, fully nested path.
    """
    source = (
        "<table>"
        "<tr><td>Chỉ tiêu</td><td>2024</td></tr>"
        "<tr><td>A. TÀI SẢN NGẮN HẠN</td><td></td></tr>"
        "<tr><td>I. Tiền và các khoản tương đương tiền</td><td></td></tr>"
        "<tr><td>1. Tiền mặt</td><td></td></tr>"
        "<tr><td>Tại quỹ</td><td>100</td></tr>"
        "<tr><td>II. Đầu tư tài chính</td><td></td></tr>"
        "<tr><td>Chứng khoán</td><td>200</td></tr>"
        "<tr><td>B. TÀI SẢN DÀI HẠN</td><td></td></tr>"
        "<tr><td>Tài sản cố định</td><td>300</td></tr>"
        "</table>\n"
    )

    values = {cell.value_raw: cell for cell in extract(tmp_path, source).tables[0].cells}

    assert values["A. TÀI SẢN NGẮN HẠN"].row_group_context_raw is None
    assert (
        values["I. Tiền và các khoản tương đương tiền"].row_group_context_raw
        == "A. TÀI SẢN NGẮN HẠN"
    )
    assert values["1. Tiền mặt"].row_group_context_raw == (
        "A. TÀI SẢN NGẮN HẠN > I. Tiền và các khoản tương đương tiền"
    )
    # Data row: full 3-level ancestor path (letter > roman > arabic).
    assert values["100"].row_group_context_raw == (
        "A. TÀI SẢN NGẮN HẠN > I. Tiền và các khoản tương đương tiền > 1. Tiền mặt"
    )
    # The next data row, now nested only under the roman-level sibling (II.)
    # that replaced the closed arabic item (1.) while keeping the letter
    # ancestor (A.).
    assert values["200"].row_group_context_raw == ("A. TÀI SẢN NGẮN HẠN > II. Đầu tư tài chính")
    assert values["300"].row_group_context_raw == "B. TÀI SẢN DÀI HẠN"


def test_section_banner_unrecognized_prefix_replaces_sibling_like_before(
    tmp_path: Path,
) -> None:
    """Would fail if an unnumbered banner started nesting under an unrelated ancestor."""
    source = (
        "<table>"
        "<tr><td>Chỉ tiêu</td><td>2024</td></tr>"
        "<tr><td>TIỀN GỬI NGÂN HÀNG</td><td></td></tr>"
        "<tr><td>Ngân hàng A</td><td>100</td></tr>"
        "<tr><td>CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN</td><td></td></tr>"
        "<tr><td>Kỳ hạn 3 tháng</td><td>200</td></tr>"
        "</table>\n"
    )

    values = {cell.value_raw: cell for cell in extract(tmp_path, source).tables[0].cells}

    assert values["TIỀN GỬI NGÂN HÀNG"].row_group_context_raw is None
    assert values["100"].row_group_context_raw == "TIỀN GỬI NGÂN HÀNG"
    assert values["CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN"].row_group_context_raw == "TIỀN GỬI NGÂN HÀNG"
    assert values["200"].row_group_context_raw == "CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN"


def test_header_row_detection_scans_beyond_three_rows(tmp_path: Path) -> None:
    """Would fail if a genuinely 4-row nested header got truncated to 3."""
    source = (
        "<table>"
        '<tr><th rowspan="4">Chỉ tiêu</th><th colspan="4">Năm 2024</th></tr>'
        '<tr><th colspan="2">Quý 1</th><th colspan="2">Quý 2</th></tr>'
        "<tr><th>Tháng 1</th><th>Tháng 2</th><th>Tháng 4</th><th>Tháng 5</th></tr>"
        "<tr><th>VND</th><th>VND</th><th>VND</th><th>VND</th></tr>"
        "<tr><td>Doanh thu</td><td>10</td><td>20</td><td>30</td><td>40</td></tr>"
        "</table>\n"
    )

    table = extract(tmp_path, source).tables[0]
    header_cells = [cell for cell in table.cells if cell.row_idx < 4]
    data_cells = [cell for cell in table.cells if cell.row_idx == 4]

    assert len(header_cells) > 0
    assert all(cell.column_label_raw is None for cell in header_cells)
    assert any(cell.column_label_raw and "VND" in cell.column_label_raw for cell in data_cells)


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


def test_three_page_merge_preserves_ids_rows_and_span_placements(tmp_path: Path) -> None:
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
    first_source_table_id = stable_table_id(result.doc_id, 2, 2, 0)
    second_source_table_id = stable_table_id(result.doc_id, 5, 5, 0)
    third_source_table_id = stable_table_id(result.doc_id, 8, 8, 0)
    assert cells["Revenue"].cell_id == stable_cell_id(first_source_table_id, 2, 0)
    assert cells["Audited"].cell_id == stable_cell_id(first_source_table_id, 3, 1)
    assert cells["Expense"].cell_id == stable_cell_id(second_source_table_id, 2, 0)
    assert cells["Profit"].cell_id == stable_cell_id(third_source_table_id, 2, 0)
    assert all(cell.table_id == expected_table_id for cell in extracted.cells)
    placements = {
        (placement.row_idx, placement.col_idx): placement.cell_id
        for placement in extracted.placements
    }
    assert placements[(2, 0)] == placements[(3, 0)] == cells["Revenue"].cell_id
    assert placements[(3, 1)] == placements[(3, 2)] == cells["Audited"].cell_id
    assert placements[(4, 0)] == cells["Expense"].cell_id
    assert placements[(5, 0)] == cells["Profit"].cell_id


def test_continuation_merge_preserves_source_cell_ids(tmp_path: Path) -> None:
    source = (
        "INCOME STATEMENT\n"
        "<table><tr><th>Metric</th><th>2024</th></tr>"
        "<tr><td>Revenue</td><td>100</td></tr></table>\n"
        "===== PAGE 2 =====\n"
        "INCOME STATEMENT\n"
        "<table><tr><th>Metric</th><th>2024</th></tr>"
        "<tr><td>Profit</td><td>20</td></tr></table>\n"
    )
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
    individual = {}
    for candidate in detection.candidates:
        single = extract_candidates(
            decoded,
            DetectionResult(
                candidates=(candidate,),
                rejected=detection.rejected,
                blocks=detection.blocks,
            ),
        )
        individual.update({cell.value_raw: cell.cell_id for cell in single.tables[0].cells})

    merged = extract_candidates(decoded, detection).tables[0]
    merged_ids = {cell.value_raw: cell.cell_id for cell in merged.cells}

    assert merged_ids["Revenue"] == individual["Revenue"]
    assert merged_ids["Profit"] == individual["Profit"]


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
