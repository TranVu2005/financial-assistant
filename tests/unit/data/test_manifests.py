from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

import financial_report_qa.data.manifests as manifests
from financial_report_qa.core.errors import DatasetBuildError
from financial_report_qa.data.inventory import InventoryIssue, InventoryResult
from financial_report_qa.data.manifests import read_manifest, write_manifest
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


class _FlushFailingStream:
    def __init__(self, path: Path) -> None:
        self.name = str(path)
        self._path = path

    def __enter__(self) -> _FlushFailingStream:
        self._path.write_text("", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        raise OSError("simulated flush failure")


def test_stream_flush_failure_preserves_previous_manifest_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text("previous\n", encoding="utf-8")
    temporary_path = tmp_path / ".documents.jsonl.flush.tmp"
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )

    def failing_named_temporary_file(*args: object, **kwargs: object) -> _FlushFailingStream:
        return _FlushFailingStream(temporary_path)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_named_temporary_file)

    with pytest.raises(OSError, match="simulated flush failure"):
        write_manifest(result, path)

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert not temporary_path.exists()


def test_cleanup_failure_does_not_replace_primary_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "documents.jsonl"
    temporary_path = tmp_path / ".documents.jsonl.flush.tmp"
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )

    def failing_named_temporary_file(*args: object, **kwargs: object) -> _FlushFailingStream:
        return _FlushFailingStream(temporary_path)

    def fail_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_named_temporary_file)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(OSError, match="simulated flush failure"):
        write_manifest(result, path)


def test_read_manifest_returns_models_and_exact_byte_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )
    write_manifest(result, path)

    snapshot = read_manifest(path)

    assert snapshot.inventory == result
    assert snapshot.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "line",
    [
        "{}\n",
        '{"record_type":"unknown"}\n',
        '{"record_type":"document","unexpected":true}\n',
        "not-json\n",
    ],
)
def test_read_manifest_rejects_invalid_rows_with_safe_line_number(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(DatasetBuildError, match="manifest line 1"):
        read_manifest(path)

