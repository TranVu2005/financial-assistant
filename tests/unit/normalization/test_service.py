from decimal import Decimal

from financial_report_qa.ingestion.provenance import (
    CellPlacement,
    ExtractedTable,
    ExtractionResult,
    stable_cell_id,
)
from financial_report_qa.normalization import (
    RULESET_VERSION,
    convert_scale,
    economic_value,
    normalize_extraction,
)
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id


def _extraction_fixture() -> tuple[DocumentRecord, ExtractionResult]:
    digest = "a" * 64
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )
    table_id = stable_table_id(document.doc_id, 3, 6)
    title = "Báo cáo kết quả hoạt động kinh doanh"
    metric = "Doanh thu thuần về bán hàng và cung cấp dịch vụ"
    source_cells = (
        (0, 0, "Chỉ tiêu", None, "Chỉ tiêu", 4),
        (0, 1, "2023", None, "2023", 4),
        (0, 2, "2024", None, "2024", 4),
        (1, 0, metric, metric, "Chỉ tiêu", 5),
        (1, 1, "(1.500)", metric, "2023", 5),
        (1, 2, "2.000", metric, "2024", 5),
    )
    cells = tuple(
        CellRecord(
            cell_id=stable_cell_id(table_id, row_idx, col_idx),
            table_id=table_id,
            row_idx=row_idx,
            col_idx=col_idx,
            row_label_raw=row_label,
            row_label_canonical=None,
            column_label_raw=column_label,
            column_label_canonical=None,
            value_raw=value,
            value_numeric=None,
            period=None,
            unit=None,
            source_line_start=line,
            source_line_end=line,
            extraction_confidence=1.0,
        )
        for row_idx, col_idx, value, row_label, column_label, line in source_cells
    )
    table = ExtractedTable(
        table=TableRecord(
            table_id=table_id,
            doc_id=document.doc_id,
            title_raw=title,
            statement_type=None,
            unit_raw="Đơn vị tính: triệu đồng",
            unit_normalized=None,
            line_start=3,
            line_end=6,
            row_count=2,
            column_count=3,
            quality_score=1.0,
            csv_path=None,
        ),
        cells=cells,
        placements=tuple(
            CellPlacement(row_idx=cell.row_idx, col_idx=cell.col_idx, cell_id=cell.cell_id)
            for cell in cells
        ),
        evidence=("html_table_marker",),
    )
    return document, ExtractionResult(
        doc_id=document.doc_id,
        blocks=(),
        tables=(table,),
        rejected=(),
    )


def test_normalize_extraction_populates_canonical_fields_and_preserves_raw() -> None:
    document, result = _extraction_fixture()

    normalized = normalize_extraction(document, result)

    table = normalized.extraction.tables[0]
    assert table.table.statement_type == "income_statement"
    assert table.table.unit_raw == "Đơn vị tính: triệu đồng"
    assert table.table.unit_normalized == "VND_million"
    values = {cell.value_raw: cell for cell in table.cells}
    assert values["(1.500)"].value_numeric == Decimal("-1500")
    assert values["(1.500)"].unit == "VND_million"
    assert values["(1.500)"].period == "2023"
    assert values["(1.500)"].row_label_canonical == "net_revenue"
    assert values["(1.500)"].row_label_raw == (
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ"
    )
    assert normalized.extraction.blocks == result.blocks
    assert normalized.extraction.rejected == result.rejected
    assert normalized.normalization_fingerprint == normalize_extraction(
        document, result
    ).normalization_fingerprint


def test_public_api_exports() -> None:
    assert RULESET_VERSION == "2026.08.1"
    assert economic_value(Decimal("10"), "VND_thousand") == Decimal("10000")
    assert convert_scale(Decimal("1000"), "VND_thousand", "VND_million") == Decimal("1")
