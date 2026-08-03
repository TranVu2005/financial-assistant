from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Literal

import pytest

from financial_report_qa.core.errors import (
    InvalidSourceDocumentError,
    SourceReadError,
    SourceSnapshotMismatchError,
    UnsupportedSourceEncodingError,
)
from financial_report_qa.ingestion.txt_reader import read_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id

RELATIVE_PATH = "AAA/2024/AAA_consolidated/Báo_cáo.txt"


def _is_known_windows_symlink_privilege_error(error: OSError) -> bool:
    return sys.platform == "win32" and getattr(error, "winerror", None) == 1314


def write_record(
    root: Path,
    content: bytes,
    *,
    encoding: str = "utf-8",
    status: Literal["ready", "empty", "duplicate", "quarantine"] = "ready",
) -> DocumentRecord:
    path = root / Path(RELATIVE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=RELATIVE_PATH,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding=encoding,
        inventory_status=status,
        notes=(),
    )


def test_read_document_preserves_unicode_and_mixed_line_endings(tmp_path: Path) -> None:
    text = "Dòng một\r\nDòng hai\n\rDòng bốn"
    record = write_record(tmp_path, text.encode())

    result = read_document(tmp_path, record)

    assert result.text == text
    assert [(line.number, line.text, line.line_ending) for line in result.lines] == [
        (1, "Dòng một", "\r\n"),
        (2, "Dòng hai", "\n"),
        (3, "", "\r"),
        (4, "Dòng bốn", ""),
    ]
    assert "".join(line.text + line.line_ending for line in result.lines) == text


def test_read_document_requires_bom_for_utf8_sig(tmp_path: Path) -> None:
    record = write_record(tmp_path, "Báo cáo".encode(), encoding="utf-8-sig")

    with pytest.raises(UnsupportedSourceEncodingError, match=RELATIVE_PATH):
        read_document(tmp_path, record)


def test_read_document_consumes_valid_utf8_bom(tmp_path: Path) -> None:
    content = b"\xef\xbb\xbf" + "Báo cáo".encode()
    record = write_record(tmp_path, content, encoding="utf-8-sig")

    result = read_document(tmp_path, record)

    assert result.text == "Báo cáo"
    assert result.lines[0].text == "Báo cáo"


@pytest.mark.parametrize("encoding", [None, "latin-1"])
def test_read_document_rejects_unapproved_encoding(
    tmp_path: Path,
    encoding: str | None,
) -> None:
    record = write_record(tmp_path, b"source", encoding="utf-8")
    record = record.model_copy(update={"encoding": encoding})

    with pytest.raises(UnsupportedSourceEncodingError, match=RELATIVE_PATH):
        read_document(tmp_path, record)


def test_read_document_wraps_invalid_utf8(tmp_path: Path) -> None:
    record = write_record(tmp_path, b"\xff", encoding="utf-8")

    with pytest.raises(UnsupportedSourceEncodingError, match=RELATIVE_PATH):
        read_document(tmp_path, record)


@pytest.mark.parametrize("changed", [b"changed!", b"x"])
def test_read_document_rejects_changed_bytes(tmp_path: Path, changed: bytes) -> None:
    record = write_record(tmp_path, b"original")
    (tmp_path / Path(RELATIVE_PATH)).write_bytes(changed)

    with pytest.raises(SourceSnapshotMismatchError, match=RELATIVE_PATH):
        read_document(tmp_path, record)


def test_read_document_rejects_non_ready_record(tmp_path: Path) -> None:
    record = write_record(tmp_path, b"same", status="duplicate")

    with pytest.raises(InvalidSourceDocumentError, match="ready"):
        read_document(tmp_path, record)


def test_read_document_segments_page_table_paragraph_and_notes(tmp_path: Path) -> None:
    text = (
        "===== PAGE 1 =====\n"
        "BÁO CÁO TÀI CHÍNH\n\n"
        "<table><tr><td>1</td></tr></table>\n\n"
        "THUYẾT MINH BÁO CÁO TÀI CHÍNH\n"
        "Chi tiết doanh thu\n"
    )
    record = write_record(tmp_path, text.encode())

    result = read_document(tmp_path, record)

    assert [(block.kind, block.line_start, block.line_end) for block in result.blocks] == [
        ("page_marker", 1, 1),
        ("paragraph", 2, 2),
        ("table", 4, 4),
        ("notes", 6, 7),
    ]
    assert result.blocks[-1].text == "THUYẾT MINH BÁO CÁO TÀI CHÍNH\nChi tiết doanh thu\n"


def test_unclosed_table_reserves_remainder_from_fallback(tmp_path: Path) -> None:
    text = "Mở đầu\n\n<table><tr><td>1\nDòng  2024  2023\n"
    record = write_record(tmp_path, text.encode())

    result = read_document(tmp_path, record)

    assert [(block.kind, block.line_start, block.line_end) for block in result.blocks] == [
        ("paragraph", 1, 1),
        ("table", 3, 4),
    ]


def test_read_failure_redacts_absolute_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = write_record(tmp_path, b"source")
    absolute = str((tmp_path / Path(RELATIVE_PATH)).resolve())

    def deny_read(*args: object, **kwargs: object) -> object:
        raise PermissionError(13, "Access denied", absolute)

    monkeypatch.setattr(Path, "open", deny_read)
    with pytest.raises(SourceReadError) as captured:
        read_document(tmp_path, record)

    message = str(captured.value)
    assert RELATIVE_PATH in message
    assert absolute not in message
    assert "Access denied" not in message


def test_missing_source_is_a_safe_read_error(tmp_path: Path) -> None:
    record = write_record(tmp_path, b"source")
    (tmp_path / Path(RELATIVE_PATH)).unlink()

    with pytest.raises(SourceReadError, match=RELATIVE_PATH):
        read_document(tmp_path, record)


def test_read_document_rejects_symlink_that_escapes_snapshot_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot"
    content = b"external source"
    record = write_record(root, content)
    source = root / Path(RELATIVE_PATH)
    source.unlink()
    external = tmp_path / "external.txt"
    external.write_bytes(content)
    try:
        source.symlink_to(external)
    except OSError as error:
        if _is_known_windows_symlink_privilege_error(error):
            pytest.skip("Windows file symlink privilege unavailable (winerror=1314)")
        raise

    with pytest.raises(InvalidSourceDocumentError, match="escapes snapshot root"):
        read_document(root, record)


def test_symlink_regression_does_not_hide_unexpected_os_errors() -> None:
    assert not _is_known_windows_symlink_privilege_error(OSError("unexpected failure"))
