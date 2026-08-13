from collections.abc import Sequence

import pytest

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.ingestion.provenance import (
    CellPlacement,
    ExtractedTable,
    ExtractionResult,
    stable_cell_id,
)
from financial_report_qa.normalization._shared import RULESET_VERSION
from financial_report_qa.normalization.service import normalize_extraction
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.normalization import NormalizationIssue
from financial_report_qa.schemas.tables import (
    CellRecord,
    TableRecord,
    stable_table_id,
)


def _document(digest: str = "a" * 64, company_code: str = "VCB") -> DocumentRecord:
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=f"{company_code}/2024/Consolidated/report.txt",
        company_code=company_code,
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=120,
        encoding="utf-8",
        inventory_status="ready",
    )


def _cell(
    table_id: str,
    *,
    row_idx: int,
    col_idx: int,
    row_label: str,
    column_label: str,
    value: str,
    row_group_context: str | None = None,
) -> CellRecord:
    return CellRecord(
        cell_id=stable_cell_id(table_id, row_idx, col_idx),
        table_id=table_id,
        row_idx=row_idx,
        col_idx=col_idx,
        row_label_raw=row_label,
        row_label_canonical=None,
        row_group_context_raw=row_group_context,
        column_label_raw=column_label,
        column_label_canonical=None,
        value_raw=value,
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=11 + row_idx,
        source_line_end=11 + row_idx,
        extraction_confidence=1.0,
    )


def _extraction(
    document: DocumentRecord,
    cells: Sequence[CellRecord],
    *,
    title: str = "Báo cáo kết quả hoạt động kinh doanh",
    unit_raw: str | None = "Đơn vị tính: triệu đồng",
) -> ExtractionResult:
    table_id = stable_table_id(document.doc_id, 10, 20)
    if any(cell.table_id != table_id for cell in cells):
        raise AssertionError("test cells must use the fixture table ID")

    table = TableRecord(
        table_id=table_id,
        doc_id=document.doc_id,
        title_raw=title,
        statement_type=None,
        unit_raw=unit_raw,
        unit_normalized=None,
        line_start=10,
        line_end=20,
        row_count=max((cell.row_idx for cell in cells), default=-1) + 1,
        column_count=max((cell.col_idx for cell in cells), default=-1) + 1,
        quality_score=1.0,
        csv_path=None,
    )
    return ExtractionResult(
        doc_id=document.doc_id,
        blocks=(),
        tables=(
            ExtractedTable(
                table=table,
                cells=tuple(cells),
                placements=tuple(
                    CellPlacement(row_idx=cell.row_idx, col_idx=cell.col_idx, cell_id=cell.cell_id)
                    for cell in cells
                ),
                evidence=("unit-test",),
            ),
        ),
        rejected=(),
    )


def _issues_for(
    issues: Sequence[NormalizationIssue], *, field: str, code: str
) -> list[NormalizationIssue]:
    return [issue for issue in issues if issue.field == field and issue.code == code]


def test_normalize_extraction_populates_canonical_fields_and_is_deterministic() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    source_cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Doanh thu thuần",
        column_label="Năm 2024",
        value="1.234.567",
    )
    extraction = _extraction(document, [source_cell])

    first = normalize_extraction(document, extraction)
    second = normalize_extraction(document, extraction)
    normalized_table = first.extraction.tables[0]
    normalized_cell = normalized_table.cells[0]

    assert normalized_table.table.statement_type == "income_statement"
    assert normalized_table.table.unit_normalized == "VND_million"
    assert normalized_cell.row_label_canonical == "net_revenue"
    assert normalized_cell.column_label_canonical == "2024"
    assert normalized_cell.period == "2024"
    assert str(normalized_cell.value_numeric) == "1234567"
    assert normalized_cell.unit == "VND_million"
    assert first.issues == ()
    assert first.ruleset_version == RULESET_VERSION
    assert first.normalization_fingerprint == second.normalization_fingerprint
    assert extraction.tables[0].cells[0] == source_cell


def test_normalize_extraction_rejects_mismatched_document_identity() -> None:
    document = _document("a" * 64)
    extraction = ExtractionResult(
        doc_id=stable_document_id("b" * 64),
        blocks=(),
        tables=(),
        rejected=(),
    )

    with pytest.raises(NormalizationError, match="IDs must match"):
        normalize_extraction(document, extraction)


def test_metric_unknown_is_emitted_once_per_logical_row() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=0,
            col_idx=col_idx,
            row_label="Chỉ tiêu chưa đăng ký",
            column_label=year,
            value=value,
        )
        for col_idx, (year, value) in enumerate((("2024", "100"), ("2023", "90")))
    ]

    normalized = normalize_extraction(document, _extraction(document, cells))
    issues = _issues_for(normalized.issues, field="metric", code="metric_unknown")

    assert len(issues) == 1
    assert issues[0].raw_value == "Chỉ tiêu chưa đăng ký"


def test_row_group_context_resolves_metric_for_bare_child_row() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=0,
            col_idx=col_idx,
            row_label="Ngắn hạn",
            row_group_context="Vay và nợ thuê tài chính",
            column_label=year,
            value=value,
        )
        for col_idx, (year, value) in enumerate((("2024", "100"), ("2023", "90")))
    ]

    normalized = normalize_extraction(document, _extraction(document, cells))
    issues = _issues_for(normalized.issues, field="metric", code="metric_unknown")
    canonical = {
        cell.row_label_canonical for table in normalized.extraction.tables for cell in table.cells
    }

    assert issues == []
    assert canonical == {"short_term_debt"}


def test_period_issue_is_emitted_once_per_logical_column() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=row_idx,
            col_idx=0,
            row_label=row_label,
            column_label="Quý",
            value=value,
        )
        for row_idx, (row_label, value) in enumerate(
            (("Doanh thu thuần", "100"), ("Lợi nhuận sau thuế", "10"))
        )
    ]

    normalized = normalize_extraction(document, _extraction(document, cells))
    issues = _issues_for(normalized.issues, field="period", code="period_incomplete")

    assert len(issues) == 1
    assert issues[0].raw_value == "Quý"


def test_table_unit_issue_is_emitted_once_with_unit_evidence() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=row_idx,
            col_idx=0,
            row_label=row_label,
            column_label="2024",
            value=value,
        )
        for row_idx, (row_label, value) in enumerate(
            (("Doanh thu thuần", "100"), ("Lợi nhuận sau thuế", "10"))
        )
    ]

    normalized = normalize_extraction(
        document,
        _extraction(document, cells, unit_raw="Đơn vị tính: nghìn USD"),
    )
    issues = _issues_for(normalized.issues, field="unit", code="unit_unknown")

    assert len(issues) == 1
    assert issues[0].cell_id is None
    assert issues[0].raw_value == "Đơn vị tính: nghìn USD"


def test_unknown_metric_in_notes_does_not_create_metric_noise() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Nội dung thuyết minh tự do",
        column_label="2024",
        value="100",
    )

    normalized = normalize_extraction(
        document,
        _extraction(
            document,
            [cell],
            title="Thuyết minh báo cáo tài chính",
            unit_raw=None,
        ),
    )

    assert _issues_for(normalized.issues, field="metric", code="metric_unknown") == []


def test_composite_column_unit_is_applied_to_numeric_cell() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Doanh thu thuần",
        column_label="2025Triệu VND",
        value="531.695",
    )

    normalized = normalize_extraction(
        document,
        _extraction(document, [cell], unit_raw=None),
    )
    normalized_cell = normalized.extraction.tables[0].cells[0]

    assert normalized_cell.unit == "VND_million"
    assert _issues_for(normalized.issues, field="unit", code="unit_unknown") == []


def test_composite_header_text_does_not_create_cell_unit_issue() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Mối quan hệ",
        column_label="2024",
        value="Công ty liên kết",
    )

    normalized = normalize_extraction(
        document,
        _extraction(document, [cell], title="Thuyết minh báo cáo tài chính", unit_raw=None),
    )

    assert _issues_for(normalized.issues, field="unit", code="unit_unknown") == []


def test_unit_issue_is_not_emitted_for_missing_or_text_cells() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=0,
            col_idx=0,
            row_label="Doanh thu thuần",
            column_label="2024nghìn USD",
            value="-",
        ),
        _cell(
            table_id,
            row_idx=1,
            col_idx=0,
            row_label="Ghi chú",
            column_label="Mối quan hệ",
            value="Công ty Vinpearl",
        ),
    ]

    normalized = normalize_extraction(document, _extraction(document, cells, unit_raw=None))

    assert all(
        issue.cell_id is None
        for issue in _issues_for(normalized.issues, field="unit", code="unit_unknown")
    )


def test_relative_composite_period_is_resolved_once_per_logical_column() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=row_idx,
            col_idx=0,
            row_label=row_label,
            column_label="Năm nay VND",
            value=value,
        )
        for row_idx, (row_label, value) in enumerate(
            (("Doanh thu thuần", "100"), ("Lợi nhuận sau thuế", "10"))
        )
    ]

    normalized = normalize_extraction(document, _extraction(document, cells, unit_raw=None))
    outputs = normalized.extraction.tables[0].cells

    assert [cell.period for cell in outputs] == ["2024", "2024"]
    assert [cell.unit for cell in outputs] == ["VND", "VND"]
    assert _issues_for(normalized.issues, field="period", code="period_incomplete") == []
    assert _issues_for(normalized.issues, field="unit", code="unit_unknown") == []


@pytest.mark.parametrize(
    ("value", "column", "expected_numeric", "number_issue"),
    [
        ("1.764", "2024Triệu VND", "1764", None),
        ("25.967", "Năm trước", None, "number_ambiguous"),
        ("31.12.2021", "2024", None, None),
        ("4 - 5", "2024", None, None),
        ("50%30%", "2024", None, "number_invalid"),
    ],
)
def test_number_parsing_uses_unit_context_and_preserves_audit_cases(
    value: str,
    column: str,
    expected_numeric: str | None,
    number_issue: str | None,
) -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Doanh thu thuần",
        column_label=column,
        value=value,
    )

    normalized = normalize_extraction(document, _extraction(document, [cell], unit_raw=None))
    output = normalized.extraction.tables[0].cells[0]
    number_issues = _issues_for(normalized.issues, field="number", code=number_issue or "")

    if expected_numeric is None:
        assert output.value_numeric is None
    else:
        assert str(output.value_numeric) == expected_numeric
    if number_issue is None:
        assert number_issues == []
    else:
        assert len(number_issues) == 1


def test_metric_unknown_is_suppressed_for_label_only_header_rows() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="4. Tiền trả nợ gốc vay",
        column_label="CHỈ TIÊU",
        value="4. Tiền trả nợ gốc vay",
    )

    normalized = normalize_extraction(document, _extraction(document, [cell]))

    assert _issues_for(normalized.issues, field="metric", code="metric_unknown") == []
