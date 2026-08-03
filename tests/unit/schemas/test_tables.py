"""Contract tests for canonical financial tables and cells."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.schemas.documents import stable_document_id
from financial_report_qa.schemas.tables import TableRecord, stable_table_id


DOC_ID = stable_document_id("a" * 64)


def valid_table_payload() -> dict[str, object]:
    return {
        "table_id": stable_table_id(DOC_ID, 10, 25),
        "doc_id": DOC_ID,
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

    assert result == "tbl_32c57ec231bb937a8f18f8e625d660e1a38af5e9fd926b84cae1bcf797e9172c"


def test_stable_table_id_changes_with_document_or_span() -> None:
    base = "tbl_32c57ec231bb937a8f18f8e625d660e1a38af5e9fd926b84cae1bcf797e9172c"

    assert stable_table_id(DOC_ID, 11, 25) != base
    assert stable_table_id(stable_document_id("b" * 64), 10, 25) != base


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("line_start", 0),
        ("line_start", 26),
        ("line_end", 9),
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
