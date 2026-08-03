"""Deterministic inventory for immutable ViFinQA TXT snapshots."""

from __future__ import annotations

import codecs
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.schemas.documents import (
    DocumentRecord,
    Sha256Digest,
    stable_document_id,
)

_CHUNK_SIZE = 1024 * 1024
_UTF8_BOM = codecs.BOM_UTF8


class InventoryIssue(BaseModel):
    """A discovered path that cannot become a canonical document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    file_size_bytes: int | None = Field(default=None, ge=0)
    sha256: Sha256Digest | None = None


class InventoryResult(BaseModel):
    """All canonical documents and rejected paths from one snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: tuple[DocumentRecord, ...]
    issues: tuple[InventoryIssue, ...]


class _PathMetadata(NamedTuple):
    relative_path: str
    company_code: str
    report_year: int
    statement_scope: Literal["consolidated", "separate", "aggregated", "other"]


@dataclass(frozen=True)
class _FileInspection:
    file_size_bytes: int
    sha256: str
    encoding: str | None
    decode_error: str | None


def _inspect_file(path: Path) -> _FileInspection:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        prefix = stream.read(len(_UTF8_BOM))
        digest.update(prefix)
        size += len(prefix)
        encoding = "utf-8-sig" if prefix.startswith(_UTF8_BOM) else "utf-8"
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        decode_error = None
        try:
            decoder.decode(prefix)
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as error:
            decode_error = str(error)
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
                size += len(chunk)
    return _FileInspection(size, digest.hexdigest(), encoding, decode_error)


def _path_key(value: str) -> tuple[str, str]:
    return value.casefold(), value


def _parse_vifinqa_path(path: Path, root: Path) -> _PathMetadata:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("path must be inside the inventory root") from error
    parts = relative.parts
    if len(parts) != 4:
        raise ValueError("expected exactly ticker/year/document/file hierarchy")
    ticker_raw, year_raw, document_name, filename = parts
    if not (2 <= len(ticker_raw) <= 10 and ticker_raw.isascii() and ticker_raw.isalnum()):
        raise ValueError("invalid ticker directory")
    if not (len(year_raw) == 4 and year_raw.isascii() and year_raw.isdecimal()):
        raise ValueError("invalid year directory")
    year = int(year_raw)
    if not 1900 <= year <= 2100:
        raise ValueError("invalid year directory")
    if Path(filename).suffix.casefold() != ".txt":
        raise ValueError("expected a .TXT file")
    normalized_name = document_name.casefold()
    scope: Literal["consolidated", "separate", "aggregated", "other"]
    if "consolidated" in normalized_name:
        scope = "consolidated"
    elif "separate" in normalized_name:
        scope = "separate"
    elif "aggregated" in normalized_name:
        scope = "aggregated"
    else:
        scope = "other"
    return _PathMetadata(relative.as_posix(), ticker_raw.upper(), year, scope)


def build_inventory(
    root: Path,
    *,
    repo_id: str,
    revision: str,
) -> InventoryResult:
    if not root.is_dir():
        raise FileNotFoundError(f"inventory root is not a directory: {root}")
    if not repo_id.strip():
        raise ValueError("repo_id must not be empty")
    if not revision.strip():
        raise ValueError("revision must not be empty")

    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".txt"
        ),
        key=lambda path: _path_key(path.relative_to(root).as_posix()),
    )
    documents: list[DocumentRecord] = []
    issues: list[InventoryIssue] = []
    primary_by_digest: dict[str, str] = {}

    for path in paths:
        relative_path = path.relative_to(root).as_posix()
        try:
            inspection = _inspect_file(path)
        except OSError as error:
            issues.append(
                InventoryIssue(relative_path=relative_path, reason=f"read failure: {error}")
            )
            continue
        try:
            metadata = _parse_vifinqa_path(path, root)
        except ValueError as error:
            issues.append(
                InventoryIssue(
                    relative_path=relative_path,
                    reason=str(error),
                    file_size_bytes=inspection.file_size_bytes,
                    sha256=inspection.sha256,
                )
            )
            continue
        if inspection.decode_error is not None:
            issues.append(
                InventoryIssue(
                    relative_path=relative_path,
                    reason=f"invalid UTF-8: {inspection.decode_error}",
                    file_size_bytes=inspection.file_size_bytes,
                    sha256=inspection.sha256,
                )
            )
            continue

        status: Literal["ready", "empty", "duplicate", "quarantine"]
        if inspection.file_size_bytes == 0:
            status = "empty"
            notes: tuple[str, ...] = ()
        elif inspection.sha256 in primary_by_digest:
            status = "duplicate"
            notes = (f"duplicate_of={primary_by_digest[inspection.sha256]}",)
        else:
            status = "ready"
            notes = ()
            primary_by_digest[inspection.sha256] = metadata.relative_path
        documents.append(
            DocumentRecord(
                doc_id=stable_document_id(inspection.sha256),
                repo_id=repo_id,
                revision=revision,
                relative_path=metadata.relative_path,
                company_code=metadata.company_code,
                report_year=metadata.report_year,
                statement_scope=metadata.statement_scope,
                sha256=inspection.sha256,
                file_size_bytes=inspection.file_size_bytes,
                encoding=inspection.encoding,
                inventory_status=status,
                notes=notes,
            )
        )
    return InventoryResult(documents=tuple(documents), issues=tuple(issues))
