"""Contract tests for canonical financial tables and cells."""

from __future__ import annotations

from decimal import Decimal
from typing import cast

import pytest
from pydantic import ValidationError

from financial_report_qa.schemas.documents import stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

DOC_ID = stable_document_id("a" * 64)


def valid_table_payload() -> dict[str, object]:
    return {
        "table_id": stable_table_id(DOC_ID, 10, 25),
        "doc_id": DOC_ID,
        "source_ordinal": 0,
        "title_raw": "B\u1ea3ng c\u00e2n \u0111\u1ed1i k\u1ebf to\u00e1n",
        "statement_type": "balance_sheet",
        "unit_raw": "\u0110\u01a1n v\u1ecb: tri\u1ec7u \u0111\u1ed3ng",
        "unit_normalized": "VND_million",
        "line_start": 10,
        "line_end": 25,
        "row_count": 12,
        "column_count": 4,
        "quality_score": 0.95,
        "csv_path": "tables/table-001.csv",
    }


def test_stable_table_id_matches_hand_checked_sha256() -> None:
    result = stable_table_id(DOC_ID, 10, 25)

    assert result == "tbl_02768744497ca8052f2512b555339bc9e892b8e855addcf5f84eb79feaacab60"


def test_stable_table_id_changes_with_document_or_span() -> None:
    base = "tbl_02768744497ca8052f2512b555339bc9e892b8e855addcf5f84eb79feaacab60"

    assert stable_table_id(DOC_ID, 11, 25) != base
    assert stable_table_id(stable_document_id("b" * 64), 10, 25) != base
    assert stable_table_id(DOC_ID, 10, 25, 1) != base


@pytest.mark.parametrize("source_ordinal", [-1, True, 1.5])
def test_stable_table_id_rejects_invalid_source_ordinal(source_ordinal: object) -> None:
    with pytest.raises(ValueError, match="source_ordinal"):
        stable_table_id(DOC_ID, 10, 25, cast(int, source_ordinal))


@pytest.mark.parametrize(
    ("doc_id", "line_start", "line_end"),
    [
        ("invalid", 1, 2),
        (DOC_ID, 0, 2),
        (DOC_ID, 2, 1),
        (DOC_ID, True, 2),
    ],
)
def test_stable_table_id_rejects_invalid_identity_or_span(
    doc_id: str,
    line_start: int,
    line_end: int,
) -> None:
    with pytest.raises(ValueError):
        stable_table_id(doc_id, line_start, line_end)


def test_stable_table_id_rejects_non_string_document_id_with_value_error() -> None:
    with pytest.raises(ValueError, match="doc_id must be a string"):
        stable_table_id(cast(str, None), 1, 1)


def test_table_record_round_trip_preserves_raw_vietnamese_text() -> None:
    payload = valid_table_payload()
    raw_title = "  B\u00e1o c\u00e1o k\u1ebft qu\u1ea3 ho\u1ea1t \u0111\u1ed9ng kinh doanh  "
    payload["title_raw"] = raw_title

    record = TableRecord.model_validate(payload)
    restored = TableRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.title_raw == raw_title


def test_table_record_requires_nullable_raw_fields() -> None:
    payload = valid_table_payload()
    payload.pop("unit_raw")

    with pytest.raises(ValidationError, match="unit_raw"):
        TableRecord.model_validate(payload)


def test_table_record_rejects_mismatched_stable_id() -> None:
    payload = valid_table_payload()
    payload["table_id"] = stable_table_id(DOC_ID, 11, 25)

    with pytest.raises(ValidationError, match="table_id must match"):
        TableRecord.model_validate(payload)


def test_table_record_identity_includes_source_ordinal() -> None:
    payload = valid_table_payload()
    payload["source_ordinal"] = 1

    with pytest.raises(ValidationError, match="source ordinal"):
        TableRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("line_start", 0),
        ("line_start", 26),
        ("line_end", 9),
        ("source_ordinal", -1),
        ("source_ordinal", True),
        ("row_count", -1),
        ("column_count", -1),
        ("quality_score", -0.01),
        ("quality_score", 1.01),
        ("quality_score", True),
        ("quality_score", "0.9"),
    ],
)
def test_table_record_rejects_invalid_shape_or_provenance(
    field: str,
    value: object,
) -> None:
    payload = valid_table_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        TableRecord.model_validate(payload)


def test_table_record_rejects_extra_fields() -> None:
    payload = valid_table_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        TableRecord.model_validate(payload)


@pytest.mark.parametrize(
    "csv_path",
    [
        "",
        "   ",
        "/generated/table.csv",
        "C:/generated/table.csv",
        "c:/generated/table.csv",
        "../escape.csv",
        r"tables\\table.csv",
    ],
)
def test_table_record_rejects_invalid_csv_paths(csv_path: str) -> None:
    payload = valid_table_payload()
    payload["csv_path"] = csv_path

    with pytest.raises(ValidationError, match="csv_path"):
        TableRecord.model_validate(payload)


def test_table_record_accepts_utf8_posix_csv_path_and_explicit_none() -> None:
    payload = valid_table_payload()
    csv_path = "tables/B\u1ea3ng c\u00e2n \u0111\u1ed1i.csv"
    payload["csv_path"] = csv_path

    assert TableRecord.model_validate(payload).csv_path == csv_path

    payload["csv_path"] = None

    assert TableRecord.model_validate(payload).csv_path is None


def valid_cell_payload() -> dict[str, object]:
    return {
        "cell_id": "cell-table-001-r2-c3",
        "table_id": stable_table_id(DOC_ID, 10, 25),
        "row_idx": 2,
        "col_idx": 3,
        "row_label_raw": "  Lợi nhuận sau thuế  ",
        "row_label_canonical": "profit_after_tax",
        "column_label_raw": "Năm 2022",
        "column_label_canonical": "2022",
        "value_raw": "  1.234,50  ",
        "value_numeric": Decimal("1234.50"),
        "period": "2022",
        "unit": "VND_million",
        "source_line_start": 18,
        "source_line_end": 19,
        "extraction_confidence": 0.9,
    }


def test_cell_record_round_trip_preserves_raw_text_and_decimal() -> None:
    record = CellRecord.model_validate(valid_cell_payload())
    restored = CellRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.row_label_raw == "  Lợi nhuận sau thuế  "
    assert restored.value_raw == "  1.234,50  "
    assert restored.value_numeric == Decimal("1234.50")


def test_cell_record_requires_nullable_canonical_fields() -> None:
    payload = valid_cell_payload()
    payload.pop("period")

    with pytest.raises(ValidationError, match="period"):
        CellRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_idx", -1),
        ("col_idx", -1),
        ("source_line_start", 0),
        ("source_line_start", 20),
        ("source_line_end", 17),
        ("extraction_confidence", -0.01),
        ("extraction_confidence", 1.01),
        ("extraction_confidence", True),
        ("extraction_confidence", "0.9"),
        ("table_id", "invalid"),
    ],
)
def test_cell_record_rejects_invalid_coordinates_or_provenance(
    field: str,
    value: object,
) -> None:
    payload = valid_cell_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        CellRecord.model_validate(payload)


def test_cell_record_rejects_extra_fields() -> None:
    payload = valid_cell_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        CellRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("record", "field", "value"),
    [
        (TableRecord.model_validate(valid_table_payload()), "row_count", 99),
        (CellRecord.model_validate(valid_cell_payload()), "row_idx", 99),
    ],
    ids=("table", "cell"),
)
def test_table_and_cell_records_reject_mutation(
    record: TableRecord | CellRecord,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field, value)