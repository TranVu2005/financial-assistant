from decimal import Decimal

from financial_report_qa.ingestion.provenance import (
    ExtractedTable,
    ExtractionResult,
    stable_cell_id,
)
from financial_report_qa.normalization.metrics import normalize_metric
from financial_report_qa.normalization.numbers import parse_number
from financial_report_qa.normalization.periods import normalize_period
from financial_report_qa.normalization.service import normalize_extraction
from financial_report_qa.normalization.statements import normalize_statement_type
from financial_report_qa.normalization.units import normalize_unit, resolve_unit
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id


def test_remediate_unit_unknown_false_positives() -> None:
    # Unsupported unit alias
    dec1 = normalize_unit("Triệu VNĐ")
    assert dec1.value == "VND_million"
    assert dec1.issue_code is None

    # Year header treated as unit
    dec2 = normalize_unit("Năm 2024")
    assert dec2.value is None
    assert dec2.issue_code is None


def test_remediate_metric_unknown_false_positives() -> None:
    # Non-metric structural row label
    dec1 = normalize_metric("STT")
    assert dec1.value is None
    assert dec1.issue_code is None

    # Unsupported valid metric alias
    dec2 = normalize_metric("Lợi nhuận gộp")
    assert dec2.value == "gross_profit"
    assert dec2.issue_code is None


def test_remediate_number_invalid_ocr_corruption() -> None:
    # OCR typo O for 0 at end of integer string
    dec1 = parse_number("1,000,00O")
    assert dec1.value == Decimal("1000000")
    assert dec1.issue_code is None

    # Trailing dot from OCR or sentence end
    dec2 = parse_number("123,456.")
    assert dec2.value == Decimal("123456")
    assert dec2.issue_code is None


def test_remediate_number_missing_legitimate_on_non_metric_row() -> None:
    digest = "1" * 64
    doc_id = stable_document_id(digest)
    doc = DocumentRecord(
        doc_id=doc_id,
        repo_id="repo",
        revision="rev",
        relative_path="test.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=100,
        encoding="utf-8",
        inventory_status="ready",
    )
    table_id = stable_table_id(doc_id, 1, 10)
    table_rec = TableRecord(
        table_id=table_id,
        doc_id=doc_id,
        title_raw="Báo cáo kết quả kinh doanh",
        statement_type="income_statement",
        unit_raw="triệu đồng",
        unit_normalized="VND_million",
        line_start=1,
        line_end=10,
        row_count=1,
        column_count=2,
        quality_score=1.0,
        csv_path=None,
    )
    cell_id = stable_cell_id(table_id, 1, 1)
    cell_rec = CellRecord(
        cell_id=cell_id,
        table_id=table_id,
        row_idx=1,
        col_idx=1,
        row_label_raw="Cộng",  # Non-canonical structural line
        row_label_canonical=None,
        column_label_raw="Năm 2024",
        column_label_canonical=None,
        value_raw="-",  # Legitimate missing marker on non-metric row
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=1,
        source_line_end=1,
        extraction_confidence=1.0,
    )
    ext_table = ExtractedTable(
        table=table_rec,
        cells=(cell_rec,),
        placements=(),
        evidence=("html_table_marker",),
    )
    res = ExtractionResult(doc_id=doc_id, blocks=(), tables=(ext_table,), rejected=())

    norm_doc = normalize_extraction(doc, res)
    issue_codes = {issue.code for issue in norm_doc.issues}
    assert "number_missing" not in issue_codes


def test_remediate_unit_conflict_mixed_unit_table() -> None:
    # Cell unit hint (percent) overrides table general unit without conflict
    dec = resolve_unit(cell_hint="percent", column_raw="Tỷ lệ", table_raw="triệu đồng")
    assert dec.value == "percent"
    assert dec.issue_code is None


def test_remediate_statement_conflict_with_ancillary_signals() -> None:
    # Main statement type with footnote/note mention in title should not conflict
    dec = normalize_statement_type("Bảng cân đối kế toán (Thuyết minh BCTC)")
    assert dec.value == "balance_sheet"
    assert dec.issue_code is None


def test_remediate_number_ambiguous_separator() -> None:
    # 4 decimal zeros after dot should be resolved cleanly without ambiguity
    dec = parse_number("10.0000")
    assert dec.value == Decimal("10")
    assert dec.issue_code is None


def test_remediate_period_incomplete_month_only() -> None:
    # Month only header completed using report_year
    dec = normalize_period("Tháng 12", report_year=2024)
    assert dec.value == "2024-12"
    assert dec.issue_code is None


def test_remediate_period_ambiguous_two_digit_year() -> None:
    # 2-digit year resolved unambiguously using report century
    dec = normalize_period("31/12/24", report_year=2024)
    assert dec.value == "2024-12-31"
    assert dec.issue_code is None
