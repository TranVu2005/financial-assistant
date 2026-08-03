from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from financial_report_qa.ingestion.provenance import DecodedDocument
from financial_report_qa.ingestion.table_detector import detect_table_candidates
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def decoded(tmp_path: Path, text: str) -> DecodedDocument:
    content = text.encode()
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
    return read_document(tmp_path, record)


def test_detector_prefers_closed_html_and_preserves_span(tmp_path: Path) -> None:
    source = (
        "Má»Ÿ Ä‘áº§u\n"
        "<table>\n<tr><td>Chá»‰ tiÃªu</td><td>2024</td></tr>\n</table>\n"
        "Sau báº£ng\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.kind == "html"
    assert (candidate.line_start, candidate.line_end) == (2, 4)
    assert candidate.raw_source == source.split("Má»Ÿ Ä‘áº§u\n", 1)[1].split("Sau báº£ng", 1)[0]
    assert candidate.confidence == 1.0
    assert candidate.evidence == ("html_table_marker",)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("<table><tr><td>1</td></tr>", "unclosed_html_table"),
        ("<table><tr><td><table></table></td></tr></table>", "nested_html_table"),
    ],
)
def test_detector_rejects_invalid_html_regions(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == [reason]


def test_detector_rejects_multiline_nested_html_through_outer_close(tmp_path: Path) -> None:
    source = (
        "<table>\n"
        "<table>\n"
        "<tr><td>1</td></tr></table>\n"
        "</table>\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert len(result.rejected) == 1
    rejected = result.rejected[0]
    assert rejected.reason == "nested_html_table"
    assert rejected.raw_source == source
    assert (rejected.line_start, rejected.line_end) == (1, 4)


def test_detector_splits_sibling_tables_on_one_line(tmp_path: Path) -> None:
    source = (
        "<table><tr><td>A</td><td>1</td></tr></table>"
        "<table><tr><td>B</td><td>2</td></tr></table>\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 2
    assert [item.raw_source.count("<table>") for item in result.candidates] == [1, 1]
    assert [(item.line_start, item.line_end) for item in result.candidates] == [(1, 1), (1, 1)]


def test_detector_orders_html_and_text_candidates_by_source_offset(tmp_path: Path) -> None:
    source = (
        "Preface <table><tr><td>A</td><td>1</td></tr></table>\n"
        "Chá»‰ tiÃªu\t2024\t2023\n"
        "Doanh thu\t1.000\t900\n"
        "Lá»£i nhuáº­n\t100\t80\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert [item.kind for item in result.candidates] == ["html", "structured_text"]
    assert [item.ordinal for item in result.candidates] == [0, 1]


def test_fallback_accepts_only_consistent_financial_rows(tmp_path: Path) -> None:
    source = (
        "Chá»‰ tiÃªu\t2024\t2023\n"
        "Doanh thu\t1.000\t900\n"
        "Lá»£i nhuáº­n\t100\t80\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.kind == "structured_text"
    assert candidate.confidence == 0.85
    assert candidate.evidence == (
        "consistent_columns",
        "financial_header",
        "numeric_density",
    )


def test_fallback_rejects_ragged_table_like_rows(tmp_path: Path) -> None:
    source = "Chá»‰ tiÃªu  2024  2023\nDoanh thu  100\nLá»£i nhuáº­n  20  10\n"
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == ["ragged_structured_rows"]


def test_fallback_rejects_rows_with_different_non_empty_column_counts(
    tmp_path: Path,
) -> None:
    source = "Head\t2024\t\nA\t1\t2\nB\t3\t4\n"
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == ["ragged_structured_rows"]


def test_fallback_ignores_explanatory_prose_and_bullets(tmp_path: Path) -> None:
    source = (
        "KÃ­nh gá»­i: á»¦y ban Chá»©ng khoÃ¡n NhÃ  nÆ°á»›c\n\n"
        "- Lá»£i nhuáº­n sau thuáº¿ nÄƒm 2024: 100 Ä‘á»“ng\n"
        "- Lá»£i nhuáº­n sau thuáº¿ nÄƒm 2023: 80 Ä‘á»“ng\n"
        "- NguyÃªn nhÃ¢n: doanh thu giáº£m\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()


def test_fallback_rejects_delimited_rows_without_financial_evidence(tmp_path: Path) -> None:
    source = "TÃªn  PhÃ²ng ban\nAn  Káº¿ toÃ¡n\nBÃ¬nh  Kiá»ƒm toÃ¡n\n"
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == [
        "insufficient_structural_evidence"
    ]


def test_fallback_caps_high_evidence_confidence_at_point_nine(tmp_path: Path) -> None:
    source = (
        "Chá»‰ tiÃªu\t2024\t2023\n"
        "A\t10\t9\n"
        "B\t8\t7\n"
        "C\t6\t5\n"
        "D\t4\t3\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates[0].confidence == 0.9
    assert result.candidates[0].evidence[-1] == "five_or_more_rows"
