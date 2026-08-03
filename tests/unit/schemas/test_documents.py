"""Contract tests for canonical financial documents."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


SHA256 = "a" * 64


def valid_document_payload() -> dict[str, object]:
    return {
        "doc_id": stable_document_id(SHA256),
        "repo_id": "AIGuruTinix/ViFinQA",
        "revision": "main",
        "relative_path": (
            "financial_statements/AAA/2015/"
            "AAA_financial_statements_2015_consolidated/report.txt"
        ),
        "company_code": "AAA",
        "report_year": 2015,
        "statement_scope": "consolidated",
        "sha256": SHA256,
        "file_size_bytes": 1024,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "notes": (),
    }


def test_stable_document_id_is_content_addressed_and_case_normalized() -> None:
    uppercase_digest = "B" * 64

    first = stable_document_id(uppercase_digest)
    second = stable_document_id(uppercase_digest.lower())

    assert first == second == f"doc_{'b' * 64}"


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "a" * 63, "a" * 65])
def test_stable_document_id_rejects_non_sha256_values(digest: str) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        stable_document_id(digest)


def test_document_record_round_trip_preserves_vietnamese_unicode() -> None:
    payload = valid_document_payload()
    unicode_path = (
        "financial_statements/AAA/2015/"
        "BÃ¡o cÃ¡o tÃ i chÃ­nh há»£p nháº¥t/report.txt"
    )
    payload["relative_path"] = unicode_path

    record = DocumentRecord.model_validate(payload)
    restored = DocumentRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.relative_path == unicode_path


def test_document_record_requires_nullable_encoding_field() -> None:
    payload = valid_document_payload()
    payload.pop("encoding")

    with pytest.raises(ValidationError, match="encoding"):
        DocumentRecord.model_validate(payload)


def test_document_record_allows_explicit_unknown_encoding() -> None:
    payload = valid_document_payload()
    payload["encoding"] = None

    assert DocumentRecord.model_validate(payload).encoding is None


def test_document_record_rejects_mismatched_content_id() -> None:
    payload = valid_document_payload()
    payload["doc_id"] = stable_document_id("b" * 64)

    with pytest.raises(ValidationError, match="doc_id must match sha256"):
        DocumentRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "/absolute/report.txt"),
        ("relative_path", r"AAA\\2015\\report.txt"),
        ("relative_path", "../report.txt"),
        ("company_code", "a"),
        ("report_year", 1899),
        ("file_size_bytes", -1),
        ("inventory_status", "unknown"),
    ],
)
def test_document_record_rejects_invalid_inventory_metadata(
    field: str,
    value: object,
) -> None:
    payload = valid_document_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(payload)


def test_document_record_rejects_extra_fields() -> None:
    payload = valid_document_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        DocumentRecord.model_validate(payload)


def test_document_record_is_frozen() -> None:
    record = DocumentRecord.model_validate(valid_document_payload())

    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, "report_year", 2020)
