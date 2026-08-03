"""Canonical contracts for extracted financial tables and cells."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

_DOC_ID_RE = re.compile(r"^doc_[0-9a-f]{64}$")
_TABLE_ID_PATTERN = r"^tbl_[0-9a-f]{64}$"

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DocumentId = Annotated[str, StringConstraints(pattern=r"^doc_[0-9a-f]{64}$")]
TableId = Annotated[str, StringConstraints(pattern=_TABLE_ID_PATTERN)]


def _validate_line_span(line_start: int, line_end: int) -> None:
    if (
        isinstance(line_start, bool)
        or isinstance(line_end, bool)
        or not isinstance(line_start, int)
        or not isinstance(line_end, int)
        or line_start < 1
        or line_end < line_start
    ):
        raise ValueError("source lines must be one-based and start must not exceed end")


def stable_table_id(doc_id: str, line_start: int, line_end: int) -> str:
    """Return a deterministic ID for a table at a source-line span."""
    if _DOC_ID_RE.fullmatch(doc_id) is None:
        raise ValueError("doc_id must be a canonical document ID")
    _validate_line_span(line_start, line_end)
    payload = f"{doc_id}\n{line_start}\n{line_end}".encode("utf-8")
    return f"tbl_{hashlib.sha256(payload).hexdigest()}"


class TableRecord(BaseModel):
    """Immutable metadata and provenance for one extracted table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    table_id: TableId
    doc_id: DocumentId
    title_raw: str | None
    statement_type: NonEmptyString | None
    unit_raw: str | None
    unit_normalized: NonEmptyString | None
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    row_count: int = Field(strict=True, ge=0)
    column_count: int = Field(strict=True, ge=0)
    quality_score: float = Field(ge=0, le=1)
    csv_path: str | None

    @model_validator(mode="after")
    def validate_identity_and_span(self) -> Self:
        """Require valid provenance and an ID derived from that provenance."""
        _validate_line_span(self.line_start, self.line_end)
        expected_id = stable_table_id(self.doc_id, self.line_start, self.line_end)
        if self.table_id != expected_id:
            raise ValueError("table_id must match doc_id and source-line span")
        return self
