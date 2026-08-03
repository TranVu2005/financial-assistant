from __future__ import annotations

import json
from pathlib import Path

import pytest

import financial_report_qa.data.manifests as manifests
from financial_report_qa.data.inventory import InventoryIssue, InventoryResult
from financial_report_qa.data.manifests import write_manifest
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id


def _document(relative_path: str, digest: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative_path,
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=12,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_write_manifest_sorts_all_paths_preserves_unicode_and_round_trips(
    tmp_path: Path,
) -> None:
    document = _document("VCB/2024/Consolidated/Báo cáo.txt", "b" * 64)
    issue = InventoryIssue(
        relative_path="AAA/2023/Separate/bad.txt",
        reason="invalid UTF-8",
        file_size_bytes=2,
        sha256="a" * 64,
    )
    result = InventoryResult(documents=(document,), issues=(issue,))
    path = tmp_path / "nested" / "documents.jsonl"

    write_manifest(result, path)

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert "Báo cáo".encode() in raw
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert [row["relative_path"] for row in rows] == [
        "AAA/2023/Separate/bad.txt",
        "VCB/2024/Consolidated/Báo cáo.txt",
    ]
    assert rows[0]["record_type"] == "issue"
    assert (
        InventoryIssue.model_validate(
            {key: value for key, value in rows[0].items() if key != "record_type"}
        )
        == issue
    )
    assert rows[1]["record_type"] == "document"
    assert DocumentRecord.model_validate(
        {k: v for k, v in rows[1].items() if k != "record_type"}
    ) == document


def test_write_manifest_is_byte_deterministic(tmp_path: Path) -> None:
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )
    path = tmp_path / "documents.jsonl"

    write_manifest(result, path)
    first = path.read_bytes()
    write_manifest(result, path)

    assert path.read_bytes() == first


def test_serialization_failure_preserves_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("previous\n", encoding="utf-8")
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )

    def fail_serialization(record_type: str, model: object) -> str:
        raise TypeError(f"cannot serialize {record_type}: {type(model).__name__}")

    monkeypatch.setattr(manifests, "_serialize_entry", fail_serialization)

    with pytest.raises(TypeError, match="cannot serialize"):
        write_manifest(result, path)

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.iterdir()) == [path]
