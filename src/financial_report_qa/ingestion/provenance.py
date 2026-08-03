"""Immutable contracts for deterministic source ingestion."""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_report_qa.schemas.documents import DocumentRecord
from financial_report_qa.schemas.tables import CellRecord, TableRecord

BlockKind = Literal["paragraph", "table", "notes", "page_marker"]
CandidateKind = Literal["html", "structured_text"]
RejectionCode = Literal[
    "unclosed_html_table",
    "nested_html_table",
    "unsupported_html_structure",
    "invalid_span_value",
    "span_collision",
    "expansion_limit_exceeded",
    "ragged_structured_rows",
    "insufficient_structural_evidence",
    "empty_extracted_table",
]

_TABLE_ID_RE = re.compile(r"^tbl_[0-9a-f]{64}$")
_DOC_ID_PATTERN = r"^doc_[0-9a-f]{64}$"
NonEmptyString = Annotated[str, Field(min_length=1)]


def _validate_span(line_start: int, line_end: int) -> None:
    if line_end < line_start:
        raise ValueError("line_start must not exceed line_end")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceLine(_FrozenModel):
    number: int = Field(strict=True, ge=1)
    text: str
    line_ending: Literal["\n", "\r\n", "\r", ""]


class TextBlock(_FrozenModel):
    kind: BlockKind
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    text: NonEmptyString

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        _validate_span(self.line_start, self.line_end)
        return self


class DecodedDocument(_FrozenModel):
    document: DocumentRecord
    text: str
    lines: tuple[SourceLine, ...]
    blocks: tuple[TextBlock, ...]

    @model_validator(mode="after")
    def validate_source_map(self) -> Self:
        expected_numbers = tuple(range(1, len(self.lines) + 1))
        if tuple(line.number for line in self.lines) != expected_numbers:
            raise ValueError("source line numbers must be contiguous and one-based")
        if "".join(line.text + line.line_ending for line in self.lines) != self.text:
            raise ValueError("source lines must reconstruct decoded text")
        previous_end = 0
        for block in self.blocks:
            if block.line_end > len(self.lines) or block.line_start <= previous_end:
                raise ValueError("blocks must be ordered, non-overlapping, and in range")
            previous_end = block.line_end
        return self


class TableCandidate(_FrozenModel):
    ordinal: int = Field(strict=True, ge=0)
    kind: CandidateKind
    raw_source: NonEmptyString
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    confidence: float = Field(strict=True, ge=0, le=1)
    evidence: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        _validate_span(self.line_start, self.line_end)
        return self


class RejectedCandidate(_FrozenModel):
    ordinal: int = Field(strict=True, ge=0)
    kind: CandidateKind
    raw_source: NonEmptyString
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    reason: RejectionCode

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        _validate_span(self.line_start, self.line_end)
        return self


class DetectionResult(_FrozenModel):
    candidates: tuple[TableCandidate, ...]
    rejected: tuple[RejectedCandidate, ...]
    blocks: tuple[TextBlock, ...]


class CellPlacement(_FrozenModel):
    row_idx: int = Field(strict=True, ge=0)
    col_idx: int = Field(strict=True, ge=0)
    cell_id: str = Field(pattern=r"^cell_[0-9a-f]{64}$")


class ExtractedTable(_FrozenModel):
    table: TableRecord
    cells: tuple[CellRecord, ...]
    placements: tuple[CellPlacement, ...]
    evidence: tuple[NonEmptyString, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        cell_ids = {cell.cell_id for cell in self.cells}
        if len(cell_ids) != len(self.cells):
            raise ValueError("source cell IDs must be unique")
        if any(cell.table_id != self.table.table_id for cell in self.cells):
            raise ValueError("source cells must belong to the canonical table")
        coordinates = {(item.row_idx, item.col_idx) for item in self.placements}
        if len(coordinates) != len(self.placements):
            raise ValueError("placement coordinates must be unique")
        if any(item.cell_id not in cell_ids for item in self.placements):
            raise ValueError("placements must reference source cells")
        if any(
            item.row_idx >= self.table.row_count or item.col_idx >= self.table.column_count
            for item in self.placements
        ):
            raise ValueError("placements must remain inside the table grid")
        return self


class ExtractionResult(_FrozenModel):
    doc_id: str = Field(pattern=_DOC_ID_PATTERN)
    blocks: tuple[TextBlock, ...]
    tables: tuple[ExtractedTable, ...]
    rejected: tuple[RejectedCandidate, ...]

    @model_validator(mode="after")
    def validate_document_identity(self) -> Self:
        if any(item.table.doc_id != self.doc_id for item in self.tables):
            raise ValueError("tables must belong to the extraction document")
        return self


def stable_cell_id(table_id: str, origin_row: int, origin_col: int) -> str:
    """Return a deterministic ID for one original source table cell."""
    if not isinstance(table_id, str) or _TABLE_ID_RE.fullmatch(table_id) is None:
        raise ValueError("table_id must be a canonical table ID")
    for name, value in (("origin_row", origin_row), ("origin_col", origin_col)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = f"{table_id}\n{origin_row}\n{origin_col}".encode()
    return f"cell_{hashlib.sha256(payload).hexdigest()}"
