"""Canonical contract for immutable financial-report documents."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOC_ID_PATTERN = r"^doc_[0-9a-f]{64}$"

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DocumentId = Annotated[str, StringConstraints(pattern=_DOC_ID_PATTERN)]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
CompanyCode = Annotated[str, StringConstraints(pattern=r"^[A-Z0-9]{2,10}$")]


def stable_document_id(sha256: str) -> str:
    """Return the canonical content-addressed ID for a SHA-256 digest."""
    normalized = sha256.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    return f"doc_{normalized}"


class DocumentRecord(BaseModel):
    """Immutable inventory metadata for one source financial report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: DocumentId
    repo_id: NonEmptyString
    revision: NonEmptyString
    relative_path: str
    company_code: CompanyCode
    report_year: int = Field(strict=True, ge=1900, le=2100)
    statement_scope: Literal["consolidated", "separate", "aggregated", "other"]
    sha256: Sha256Digest
    file_size_bytes: int = Field(strict=True, ge=0)
    encoding: NonEmptyString | None
    inventory_status: Literal["ready", "empty", "duplicate", "quarantine"]
    notes: tuple[str, ...] = ()

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Require a non-empty safe POSIX path relative to the dataset root."""
        path = PurePosixPath(value)
        if not value or value != value.strip():
            raise ValueError("relative_path must be non-empty without outer whitespace")
        if path.is_absolute() or "\\" in value or ".." in path.parts or not path.parts:
            raise ValueError("relative_path must be a safe POSIX relative path")
        return value

    @model_validator(mode="after")
    def validate_content_id(self) -> Self:
        """Keep the declared ID consistent with the source content digest."""
        if self.doc_id != stable_document_id(self.sha256):
            raise ValueError("doc_id must match sha256")
        return self
