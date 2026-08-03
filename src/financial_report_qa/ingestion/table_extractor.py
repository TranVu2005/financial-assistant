"""Atomic, provenance-preserving extraction of detected financial tables."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

from financial_report_qa.ingestion.provenance import (
    CellPlacement,
    DecodedDocument,
    DetectionResult,
    ExtractedTable,
    ExtractionResult,
    RejectedCandidate,
    RejectionCode,
    TableCandidate,
    stable_cell_id,
)
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

_EXPANSION_LIMIT = 100_000
_NUMERIC_VALUE_RE = re.compile(r"\(?[+-]?[0-9][0-9., ]*%?\)?$")
_TAB_SPLIT_RE = re.compile(r"\t+")
_SPACE_SPLIT_RE = re.compile(r" {2,}")
_PAGE_MARKER_RE = re.compile(r"^===== PAGE [1-9][0-9]* =====$")
_UNIT_MARKERS = (
    "đơn vị tính",
    "đơn vị",
    "\u00c4\u2018\u00c6\u00a1n v\u00e1\u00bb\u2039",
    "\u00c4\u2018\u00c6\u00a1n v\u00e1\u00bb\u2039 t\u00c3\u00adnh",
    "\u00c4\u2018vt",
)


class _ExtractionFailure(Exception):
    def __init__(self, reason: RejectionCode) -> None:
        self.reason = reason


@dataclass(frozen=True)
class _RawCell:
    text: str
    line_start: int
    line_end: int
    rowspan: int
    colspan: int
    is_header: bool


@dataclass(frozen=True)
class _RawTable:
    rows: tuple[tuple[_RawCell, ...], ...]


@dataclass
class _OpenCell:
    tag: str
    line_start: int
    rowspan: int
    colspan: int
    parts: list[str]


class _ViFinQATableParser(HTMLParser):
    """Parse one detector-validated HTML table while retaining source lines."""

    def __init__(self, candidate: TableCandidate) -> None:
        super().__init__(convert_charrefs=True)
        self._candidate = candidate
        self._rows: list[list[_RawCell]] = []
        self._current_row: list[_RawCell] | None = None
        self._current_cell: _OpenCell | None = None
        self._in_table = False
        self._seen_table = False
        self._closed_table = False
        self._failure: _ExtractionFailure | None = None

    def _line_number(self) -> int:
        return self._candidate.line_start - 1 + self.getpos()[0]

    def _reject(self, reason: RejectionCode) -> None:
        if self._failure is None:
            self._failure = _ExtractionFailure(reason)

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        attributes = dict(attrs)
        if name not in attributes:
            return 1
        value = attributes[name]
        if value is None or re.fullmatch(r"[1-9][0-9]*", value) is None:
            reason: RejectionCode = "invalid_span_value"
            raise _ExtractionFailure(reason)
        if len(value) > len(str(_EXPANSION_LIMIT)):
            raise _ExtractionFailure("expansion_limit_exceeded")
        span = int(value)
        if span > _EXPANSION_LIMIT:
            raise _ExtractionFailure("expansion_limit_exceeded")
        return span

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if self._failure is not None:
            return
        if tag == "table":
            if self._seen_table:
                self._reject("nested_html_table")
                return
            self._seen_table = True
            self._in_table = True
            return
        if not self._in_table:
            self._reject("unsupported_html_structure")
            return
        if tag == "tr":
            if self._current_row is not None or self._current_cell is not None:
                self._reject("unsupported_html_structure")
                return
            self._current_row = []
            return
        if tag in {"td", "th"}:
            if self._current_row is None or self._current_cell is not None:
                self._reject("unsupported_html_structure")
                return
            try:
                rowspan = self._span(attrs, "rowspan")
                colspan = self._span(attrs, "colspan")
            except _ExtractionFailure as error:
                self._reject(error.reason)
                return
            self._current_cell = _OpenCell(
                tag=tag,
                line_start=self._line_number(),
                rowspan=rowspan,
                colspan=colspan,
                parts=[],
            )
            return
        if tag == "br" and self._current_cell is not None:
            self._current_cell.parts.append("\n")
            return
        if self._current_cell is None:
            self._reject("unsupported_html_structure")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self._failure is not None:
            return
        if tag == "table":
            if (
                not self._in_table
                or self._current_row is not None
                or self._current_cell is not None
            ):
                self._reject("unsupported_html_structure")
                return
            self._in_table = False
            self._closed_table = True
            return
        if not self._in_table:
            self._reject("unsupported_html_structure")
            return
        if tag in {"td", "th"}:
            if self._current_cell is None or self._current_cell.tag != tag:
                self._reject("unsupported_html_structure")
                return
            self._current_row = self._current_row or []
            self._current_row.append(
                _RawCell(
                    text="".join(self._current_cell.parts).strip(),
                    line_start=self._current_cell.line_start,
                    line_end=self._line_number(),
                    rowspan=self._current_cell.rowspan,
                    colspan=self._current_cell.colspan,
                    is_header=tag == "th",
                )
            )
            self._current_cell = None
            return
        if tag == "tr":
            if self._current_row is None or self._current_cell is not None or not self._current_row:
                self._reject("unsupported_html_structure")
                return
            self._rows.append(self._current_row)
            self._current_row = None
            return
        if self._current_cell is None:
            self._reject("unsupported_html_structure")

    def handle_data(self, data: str) -> None:
        if self._failure is not None:
            return
        if self._current_cell is not None:
            self._current_cell.parts.append(data)
        elif data.strip() and self._in_table:
            self._reject("unsupported_html_structure")

    def result(self) -> _RawTable:
        if self._failure is not None:
            raise self._failure
        if not self._closed_table or self._in_table or self._current_row is not None:
            raise _ExtractionFailure("unsupported_html_structure")
        if not self._rows:
            raise _ExtractionFailure("empty_extracted_table")
        return _RawTable(rows=tuple(tuple(row) for row in self._rows))


def _parse_html(candidate: TableCandidate) -> _RawTable:
    parser = _ViFinQATableParser(candidate)
    parser.feed(candidate.raw_source)
    parser.close()
    return parser.result()


def _parse_structured(document: DecodedDocument, candidate: TableCandidate) -> _RawTable:
    rows: list[tuple[_RawCell, ...]] = []
    delimiter: str | None = None
    for line in document.lines[candidate.line_start - 1 : candidate.line_end]:
        if "\t" in line.text:
            row_delimiter = "tab"
            cells = _TAB_SPLIT_RE.split(line.text)
        elif _SPACE_SPLIT_RE.search(line.text):
            row_delimiter = "spaces"
            cells = _SPACE_SPLIT_RE.split(line.text)
        else:
            raise _ExtractionFailure("ragged_structured_rows")
        if delimiter is None:
            delimiter = row_delimiter
        elif delimiter != row_delimiter:
            raise _ExtractionFailure("ragged_structured_rows")
        rows.append(
            tuple(
                _RawCell(
                    text=value.strip(),
                    line_start=line.number,
                    line_end=line.number,
                    rowspan=1,
                    colspan=1,
                    is_header=False,
                )
                for value in cells
            )
        )
    if not rows or not any(rows):
        raise _ExtractionFailure("empty_extracted_table")
    if len({len(row) for row in rows}) != 1:
        raise _ExtractionFailure("ragged_structured_rows")
    return _RawTable(rows=tuple(rows))


def _is_numeric_looking(value: str) -> bool:
    stripped = value.strip()
    return stripped == "-" or _NUMERIC_VALUE_RE.fullmatch(stripped) is not None


def _header_row_count(
    candidate: TableCandidate,
    grid: dict[tuple[int, int], int],
    raw_cells: list[_RawCell],
    row_count: int,
    column_count: int,
) -> int:
    if candidate.kind == "structured_text":
        return 1 if "financial_header" in candidate.evidence else 0
    count = 0
    for row_idx in range(min(3, row_count)):
        cell_tokens = [
            grid[(row_idx, col_idx)]
            for col_idx in range(column_count)
            if (row_idx, col_idx) in grid
        ]
        if not cell_tokens:
            break
        all_headers = all(raw_cells[token].is_header for token in cell_tokens)
        first_token = grid.get((row_idx, 0))
        non_first = [raw_cells[token].text for token in cell_tokens if token != first_token]
        mostly_non_numeric = bool(non_first) and sum(
            _is_numeric_looking(value) for value in non_first
        ) * 2 < len(non_first)
        if not all_headers and not mostly_non_numeric:
            break
        count += 1
    return count


def _metadata_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _contains_unit_marker(value: str) -> bool:
    normalized = _metadata_key(value)
    return any(_metadata_key(marker) in normalized for marker in _UNIT_MARKERS)


def _is_predominantly_numeric(value: str) -> bool:
    compact = [character for character in value if not character.isspace()]
    if not compact:
        return False
    numeric = sum(character.isdigit() or character in ".,()-+%" for character in compact)
    return numeric * 2 >= len(compact)


def _nearby_metadata(
    document: DecodedDocument,
    candidate: TableCandidate,
    raw_cells: list[_RawCell],
) -> tuple[str | None, str | None]:
    prior_lines = document.lines[max(0, candidate.line_start - 4) : candidate.line_start - 1]
    excluded_lines = {
        line_number
        for block in document.blocks
        if block.kind in {"table", "page_marker"}
        for line_number in range(block.line_start, block.line_end + 1)
    }
    title_raw = next(
        (
            line.text.strip()
            for line in reversed(prior_lines)
            if (
                line.number not in excluded_lines
                and len(line.text) <= 200
                and not _PAGE_MARKER_RE.fullmatch(line.text)
                and not _is_predominantly_numeric(line.text)
                and line.text.strip()
            )
        ),
        None,
    )
    unit_raw = next(
        (cell.text for cell in raw_cells if _contains_unit_marker(cell.text)),
        None,
    )
    if unit_raw is None:
        unit_raw = next(
            (
                line.text.strip()
                for line in prior_lines
                if line.number not in excluded_lines and _contains_unit_marker(line.text)
            ),
            None,
        )
    return title_raw, unit_raw


def _materialize_table(
    document: DecodedDocument,
    candidate: TableCandidate,
    rows: tuple[tuple[_RawCell, ...], ...],
) -> ExtractedTable:
    grid: dict[tuple[int, int], int] = {}
    raw_cells: list[_RawCell] = []
    origins: list[tuple[int, int]] = []
    for row_idx, row in enumerate(rows):
        next_col = 0
        for raw_cell in row:
            while (row_idx, next_col) in grid:
                next_col += 1
            placement_count = raw_cell.rowspan * raw_cell.colspan
            if placement_count > _EXPANSION_LIMIT - len(grid):
                raise _ExtractionFailure("expansion_limit_exceeded")
            coordinates = [
                (covered_row, covered_col)
                for covered_row in range(row_idx, row_idx + raw_cell.rowspan)
                for covered_col in range(next_col, next_col + raw_cell.colspan)
            ]
            if any(coordinate in grid for coordinate in coordinates):
                raise _ExtractionFailure("span_collision")
            token = len(raw_cells)
            raw_cells.append(raw_cell)
            origins.append((row_idx, next_col))
            for coordinate in coordinates:
                grid[coordinate] = token
            next_col += raw_cell.colspan
    if not grid:
        raise _ExtractionFailure("empty_extracted_table")

    row_count = max(row_idx for row_idx, _ in grid) + 1
    column_count = max(col_idx for _, col_idx in grid) + 1
    title_raw, unit_raw = _nearby_metadata(document, candidate, raw_cells)
    table_id = stable_table_id(document.document.doc_id, candidate.line_start, candidate.line_end)
    table = TableRecord(
        table_id=table_id,
        doc_id=document.document.doc_id,
        title_raw=title_raw,
        statement_type=None,
        unit_raw=unit_raw,
        unit_normalized=None,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        row_count=row_count,
        column_count=column_count,
        quality_score=candidate.confidence,
        csv_path=None,
    )
    header_rows = _header_row_count(candidate, grid, raw_cells, row_count, column_count)
    column_headers: list[str | None] = []
    for col_idx in range(column_count):
        values: list[str] = []
        for row_idx in range(header_rows):
            header_token = grid.get((row_idx, col_idx))
            if header_token is None:
                continue
            value = raw_cells[header_token].text
            if value and value not in values:
                values.append(value)
        column_headers.append("\n".join(values) or None)

    cells: list[CellRecord] = []
    cell_ids: list[str] = []
    for token, raw_cell in enumerate(raw_cells):
        origin_row, origin_col = origins[token]
        cell_id = stable_cell_id(table_id, origin_row, origin_col)
        cell_ids.append(cell_id)
        is_header = origin_row < header_rows
        row_label = None
        if not is_header:
            first_token = next(
                (
                    grid[(origin_row, col_idx)]
                    for col_idx in range(column_count)
                    if (origin_row, col_idx) in grid
                ),
                None,
            )
            row_label = raw_cells[first_token].text if first_token is not None else None
        cells.append(
            CellRecord(
                cell_id=cell_id,
                table_id=table_id,
                row_idx=origin_row,
                col_idx=origin_col,
                row_label_raw=row_label,
                row_label_canonical=None,
                column_label_raw=None if is_header else column_headers[origin_col],
                column_label_canonical=None,
                value_raw=raw_cell.text,
                value_numeric=None,
                period=None,
                unit=None,
                source_line_start=raw_cell.line_start,
                source_line_end=raw_cell.line_end,
                extraction_confidence=candidate.confidence,
            )
        )
    placements = tuple(
        CellPlacement(row_idx=row_idx, col_idx=col_idx, cell_id=cell_ids[token])
        for (row_idx, col_idx), token in sorted(grid.items())
    )
    return ExtractedTable(
        table=table,
        cells=tuple(cells),
        placements=placements,
        evidence=candidate.evidence,
    )


def _rejected(candidate: TableCandidate, reason: RejectionCode) -> RejectedCandidate:
    return RejectedCandidate(
        ordinal=candidate.ordinal,
        kind=candidate.kind,
        raw_source=candidate.raw_source,
        line_start=candidate.line_start,
        line_end=candidate.line_end,
        reason=reason,
    )


def extract_candidates(document: DecodedDocument, detection: DetectionResult) -> ExtractionResult:
    """Extract independently materialized tables, retaining candidate-level rejections."""
    tables: list[ExtractedTable] = []
    rejected = list(detection.rejected)
    for candidate in sorted(detection.candidates, key=lambda item: (item.line_start, item.ordinal)):
        try:
            raw_table = (
                _parse_html(candidate)
                if candidate.kind == "html"
                else _parse_structured(document, candidate)
            )
            tables.append(_materialize_table(document, candidate, raw_table.rows))
        except _ExtractionFailure as error:
            rejected.append(_rejected(candidate, error.reason))
    return ExtractionResult(
        doc_id=document.document.doc_id,
        blocks=detection.blocks,
        tables=tuple(tables),
        rejected=tuple(sorted(rejected, key=lambda item: (item.line_start, item.ordinal))),
    )
