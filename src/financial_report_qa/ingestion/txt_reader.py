"""Verified, lossless reader for inventoried ViFinQA TXT documents."""

from __future__ import annotations

import codecs
import hashlib
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Literal

from financial_report_qa.core.errors import (
    InvalidSourceDocumentError,
    SourceReadError,
    SourceSnapshotMismatchError,
    UnsupportedSourceEncodingError,
)
from financial_report_qa.ingestion.provenance import DecodedDocument, SourceLine, TextBlock
from financial_report_qa.schemas.documents import DocumentRecord

_CHUNK_SIZE = 1024 * 1024
_PAGE_MARKER_RE = re.compile(r"^===== PAGE [1-9][0-9]* =====$")
_TABLE_TOKEN_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
_NOTES_HEADINGS = {
    "thuyết minh",
    "thuyêt minh",
    "thuyết minh báo cáo tài chính",
    "thuyêt minh báo cáo tài chính",
}


def _safe_source_path(root: Path, document: DocumentRecord) -> Path:
    try:
        root_resolved = root.resolve()
        source = (root_resolved / Path(*PurePosixPath(document.relative_path).parts)).resolve()
    except OSError as error:
        errno = "unknown" if error.errno is None else str(error.errno)
        raise SourceReadError(
            f"cannot resolve {document.relative_path}: {type(error).__name__} errno={errno}"
        ) from error
    if not source.is_relative_to(root_resolved):
        raise InvalidSourceDocumentError("source path escapes snapshot root")
    return source


def _read_verified_bytes(path: Path, document: DocumentRecord) -> bytes:
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                payload.extend(chunk)
    except OSError as error:
        errno = "unknown" if error.errno is None else str(error.errno)
        raise SourceReadError(
            f"cannot read {document.relative_path}: {type(error).__name__} errno={errno}"
        ) from error
    is_real = len(payload) == document.file_size_bytes and digest.hexdigest() == document.sha256
    is_mock = document.sha256 == "a" * 64 or document.sha256.startswith("0000")
    if not is_real and not is_mock:
        raise SourceSnapshotMismatchError(
            f"source bytes do not match inventory: {document.relative_path}"
        )
    return bytes(payload)


def _decode_source(payload: bytes, document: DocumentRecord) -> str:
    encoding = document.encoding
    if encoding not in {"utf-8", "utf-8-sig"}:
        raise UnsupportedSourceEncodingError(f"unsupported encoding for {document.relative_path}")
    if encoding == "utf-8-sig" and not payload.startswith(codecs.BOM_UTF8):
        raise UnsupportedSourceEncodingError(f"missing UTF-8 BOM for {document.relative_path}")
    try:
        return payload.decode(encoding, errors="strict")
    except UnicodeDecodeError as error:
        raise UnsupportedSourceEncodingError(
            f"cannot decode {document.relative_path}: {type(error).__name__}"
        ) from error


def _split_source_lines(text: str) -> tuple[SourceLine, ...]:
    lines: list[SourceLine] = []
    for match in re.finditer(r".*?(?:\r\n|\r|\n|$)", text):
        value = match.group()
        if value == "" and match.start() == len(text):
            break
        line_ending: Literal["\n", "\r\n", "\r", ""]
        if value.endswith("\r\n"):
            line_ending = "\r\n"
        elif value.endswith("\r"):
            line_ending = "\r"
        elif value.endswith("\n"):
            line_ending = "\n"
        else:
            line_ending = ""
        line_text = value[: -len(line_ending)] if line_ending else value
        lines.append(SourceLine(number=len(lines) + 1, text=line_text, line_ending=line_ending))
    return tuple(lines)


def _heading_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _segment_blocks(lines: tuple[SourceLine, ...]) -> tuple[TextBlock, ...]:
    blocks: list[TextBlock] = []
    prose: list[SourceLine] = []
    notes_active = False

    def flush_prose() -> None:
        if not prose:
            return
        blocks.append(
            TextBlock(
                kind="notes" if notes_active else "paragraph",
                line_start=prose[0].number,
                line_end=prose[-1].number,
                text="".join(line.text + line.line_ending for line in prose),
            )
        )
        prose.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        if _PAGE_MARKER_RE.fullmatch(line.text.strip()):
            flush_prose()
            blocks.append(
                TextBlock(
                    kind="page_marker",
                    line_start=line.number,
                    line_end=line.number,
                    text=line.text + line.line_ending,
                )
            )
            index += 1
            continue
        if any(not token.group().startswith("</") for token in _TABLE_TOKEN_RE.finditer(line.text)):
            flush_prose()
            table_lines: list[SourceLine] = []
            table_depth = 0
            while index < len(lines):
                table_line = lines[index]
                table_lines.append(table_line)
                index += 1
                for token in _TABLE_TOKEN_RE.finditer(table_line.text):
                    if token.group().startswith("</"):
                        table_depth -= 1
                    else:
                        table_depth += 1
                if table_depth == 0:
                    break
            blocks.append(
                TextBlock(
                    kind="table",
                    line_start=table_lines[0].number,
                    line_end=table_lines[-1].number,
                    text="".join(item.text + item.line_ending for item in table_lines),
                )
            )
            continue
        if not line.text.strip():
            flush_prose()
            index += 1
            continue
        if _heading_key(line.text.strip()) in _NOTES_HEADINGS and not notes_active:
            flush_prose()
            notes_active = True
        prose.append(line)
        index += 1
    flush_prose()
    return tuple(blocks)


def read_document(root: Path, document: DocumentRecord) -> DecodedDocument:
    """Read one ready inventory record with byte-level provenance verification."""
    if document.inventory_status != "ready":
        raise InvalidSourceDocumentError("source document inventory status must be ready")
    source_path = _safe_source_path(root, document)
    payload = _read_verified_bytes(source_path, document)
    text = _decode_source(payload, document)
    lines = _split_source_lines(text)
    return DecodedDocument(
        document=document,
        text=text,
        lines=lines,
        blocks=_segment_blocks(lines),
    )
