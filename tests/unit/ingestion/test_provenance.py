from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from financial_report_qa.ingestion.provenance import (
    CellPlacement,
    DecodedDocument,
    ExtractedTable,
    RejectedCandidate,
    SourceLine,
    TableCandidate,
    TextBlock,
    stable_cell_id,
)
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

DOC_ID = stable_document_id("a" * 64)
TABLE_ID = stable_table_id(DOC_ID, 2, 6)


def test_stable_cell_id_uses_table_and_origin_coordinate() -> None:
    payload = f"{TABLE_ID}\n1\n2".encode()
    expected = f"cell_{hashlib.sha256(payload).hexdigest()}"

    assert stable_cell_id(TABLE_ID, 1, 2) == expected
    assert stable_cell_id(TABLE_ID, 1, 3) != expected


@pytest.mark.parametrize(
    ("table_id", "row", "col"),
    [("bad", 0, 0), (TABLE_ID, -1, 0), (TABLE_ID, 0, -1), (TABLE_ID, True, 0)],
)
def test_stable_cell_id_rejects_invalid_inputs(
    table_id: str,
    row: int,
    col: int,
) -> None:
    with pytest.raises(ValueError):
        stable_cell_id(table_id, row, col)


def test_source_and_candidate_contracts_are_frozen_and_validate_spans() -> None:
    line = SourceLine(number=1, text="Doanh thu", line_ending="\r\n")
    block = TextBlock(kind="paragraph", line_start=1, line_end=1, text="Doanh thu\r\n")
    candidate = TableCandidate(
        ordinal=0,
        kind="html",
        raw_source="<table></table>\r\n",
        line_start=2,
        line_end=2,
        confidence=1.0,
        evidence=("html_table_marker",),
    )

    assert line.text + line.line_ending == "Doanh thu\r\n"
    assert block.text == "Doanh thu\r\n"
    assert candidate.evidence == ("html_table_marker",)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(candidate, "confidence", 0.5)
    with pytest.raises(ValidationError):
        TextBlock(kind="paragraph", line_start=2, line_end=1, text="bad")


def test_rejection_codes_and_placements_are_strict() -> None:
    rejection = RejectedCandidate(
        ordinal=0,
        kind="html",
        raw_source="<table>",
        line_start=4,
        line_end=4,
        reason="unclosed_html_table",
    )
    placement = CellPlacement(row_idx=0, col_idx=1, cell_id="cell_" + "b" * 64)

    assert rejection.reason == "unclosed_html_table"
    assert placement.col_idx == 1
    with pytest.raises(ValidationError):
        RejectedCandidate.model_validate({**rejection.model_dump(), "reason": "unknown"})
    with pytest.raises(ValidationError):
        CellPlacement.model_validate({**placement.model_dump(), "extra": True})


def test_decoded_document_requires_exact_source_reconstruction() -> None:
    document = DocumentRecord(
        doc_id=DOC_ID,
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path="AAA/2024/report/source.txt",
        company_code="AAA",
        report_year=2024,
        statement_scope="other",
        sha256="a" * 64,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )

    with pytest.raises(ValidationError, match="reconstruct"):
        DecodedDocument(
            document=document,
            text="x",
            lines=(SourceLine(number=1, text="y", line_ending=""),),
            blocks=(),
        )


def test_extracted_table_rejects_unknown_cell_reference() -> None:
    table = TableRecord(
        table_id=TABLE_ID,
        doc_id=DOC_ID,
        title_raw=None,
        statement_type=None,
        unit_raw=None,
        unit_normalized=None,
        line_start=2,
        line_end=6,
        row_count=1,
        column_count=1,
        quality_score=1.0,
        csv_path=None,
    )
    cell = CellRecord(
        cell_id=stable_cell_id(TABLE_ID, 0, 0),
        table_id=TABLE_ID,
        row_idx=0,
        col_idx=0,
        row_label_raw=None,
        row_label_canonical=None,
        column_label_raw=None,
        column_label_canonical=None,
        value_raw="1",
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=3,
        source_line_end=3,
        extraction_confidence=1.0,
    )

    with pytest.raises(ValidationError, match="reference source cells"):
        ExtractedTable(
            table=table,
            cells=(cell,),
            placements=(CellPlacement(row_idx=0, col_idx=0, cell_id="cell_" + "f" * 64),),
            evidence=("html_table_marker",),
        )


def test_extracted_table_requires_every_source_cell_and_its_origin() -> None:
    table = TableRecord(
        table_id=TABLE_ID,
        doc_id=DOC_ID,
        title_raw=None,
        statement_type=None,
        unit_raw=None,
        unit_normalized=None,
        line_start=2,
        line_end=6,
        row_count=1,
        column_count=2,
        quality_score=1.0,
        csv_path=None,
    )
    first = CellRecord(
        cell_id=stable_cell_id(TABLE_ID, 0, 0),
        table_id=TABLE_ID,
        row_idx=0,
        col_idx=0,
        row_label_raw=None,
        row_label_canonical=None,
        column_label_raw=None,
        column_label_canonical=None,
        value_raw="1",
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=3,
        source_line_end=3,
        extraction_confidence=1.0,
    )
    second = first.model_copy(
        update={
            "cell_id": stable_cell_id(TABLE_ID, 0, 1),
            "col_idx": 1,
            "value_raw": "2",
        }
    )

    with pytest.raises(ValidationError, match="every source cell"):
        ExtractedTable(
            table=table,
            cells=(first, second),
            placements=(CellPlacement(row_idx=0, col_idx=0, cell_id=first.cell_id),),
            evidence=("html_table_marker",),
        )

    with pytest.raises(ValidationError, match="origin coordinates"):
        ExtractedTable(
            table=table,
            cells=(first,),
            placements=(CellPlacement(row_idx=0, col_idx=1, cell_id=first.cell_id),),
            evidence=("html_table_marker",),
        )
