"""Deterministic inventory for immutable ViFinQA TXT snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from financial_report_qa.schemas.documents import DocumentRecord, Sha256Digest


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
