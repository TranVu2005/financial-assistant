"""Deterministic HTML and structured-text table candidate detection."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TypeAlias

from financial_report_qa.ingestion.provenance import (
    DecodedDocument,
    DetectionResult,
    RejectedCandidate,
    TableCandidate,
)

_TAB_SPLIT_RE = re.compile(r"\t+")
_SPACE_SPLIT_RE = re.compile(r" {2,}")
_HEADER_SIGNALS = ("mÃ£ sá»‘", "chá»‰ tiÃªu", "thuyáº¿t minh", "nÄƒm", "ká»³", "Ä‘Æ¡n vá»‹", "Ä‘vt")
_TABLE_TOKEN_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
_PAGE_MARKER_RE = re.compile(r"^===== PAGE [1-9][0-9]* =====$")
_LIST_PREFIX_RE = re.compile(r"(?:[-â€¢*] |[0-9]+\. )")
DetectionItem: TypeAlias = TableCandidate | RejectedCandidate
DetectionEvent: TypeAlias = tuple[int, DetectionItem]


def _split_structured_row(text: str) -> tuple[str, tuple[str, ...]] | None:
    if "\t" in text:
        cells = tuple(part.strip() for part in _TAB_SPLIT_RE.split(text))
        return "tab", cells
    if _SPACE_SPLIT_RE.search(text):
        cells = tuple(part.strip() for part in _SPACE_SPLIT_RE.split(text))
        return "spaces", cells
    return None


def _is_numeric_looking(value: str) -> bool:
    stripped = value.strip()
    if stripped == "-":
        return True
    return bool(re.fullmatch(r"\(?[+-]?[0-9][0-9., ]*%?\)?", stripped))


def _line_offsets(document: DecodedDocument) -> tuple[int, ...]:
    offsets = [0]
    for line in document.lines:
        offsets.append(offsets[-1] + len(line.text) + len(line.line_ending))
    return tuple(offsets)


def _line_number_for_offset(document: DecodedDocument, offset: int) -> int:
    position = 0
    for line in document.lines:
        position += len(line.text) + len(line.line_ending)
        if offset < position:
            return line.number
    return document.lines[-1].number


def _extend_to_line_ending(text: str, end: int) -> int:
    if text.startswith("\r\n", end):
        return end + 2
    if end < len(text) and text[end] in "\r\n":
        return end + 1
    return end


def _html_events(document: DecodedDocument) -> list[DetectionEvent]:
    events: list[DetectionEvent] = []
    depth = 0
    region_start = 0
    nested = False
    for token in _TABLE_TOKEN_RE.finditer(document.text):
        is_close = token.group().startswith("</")
        if not is_close:
            if depth == 0:
                region_start = token.start()
                nested = False
            else:
                nested = True
            depth += 1
            continue
        if depth == 0:
            continue
        depth -= 1
        if depth != 0:
            continue
        source_end = _extend_to_line_ending(document.text, token.end())
        raw_source = document.text[region_start:source_end]
        line_start = _line_number_for_offset(document, region_start)
        line_end = _line_number_for_offset(document, source_end - 1)
        if nested:
            item: DetectionItem = RejectedCandidate(
                ordinal=0,
                kind="html",
                raw_source=raw_source,
                line_start=line_start,
                line_end=line_end,
                reason="nested_html_table",
            )
        else:
            item = TableCandidate(
                ordinal=0,
                kind="html",
                raw_source=raw_source,
                line_start=line_start,
                line_end=line_end,
                confidence=1.0,
                evidence=("html_table_marker",),
            )
        events.append((region_start, item))
    if depth:
        events.append(
            (
                region_start,
                RejectedCandidate(
                    ordinal=0,
                    kind="html",
                    raw_source=document.text[region_start:],
                    line_start=_line_number_for_offset(document, region_start),
                    line_end=document.lines[-1].number,
                    reason="unclosed_html_table",
                ),
            )
        )
    return events


def _is_list_line(text: str, cells: tuple[str, ...]) -> bool:
    return bool(_LIST_PREFIX_RE.match(text)) and not any(
        _is_numeric_looking(cell) for cell in cells[1:]
    )


def _structured_event(
    document: DecodedDocument,
    start_index: int,
    end_index: int,
    rows: list[tuple[str, tuple[str, ...]]],
) -> DetectionEvent:
    start_line = document.lines[start_index]
    end_line = document.lines[end_index]
    raw_source = "".join(
        line.text + line.line_ending for line in document.lines[start_index : end_index + 1]
    )
    delimiters = {delimiter for delimiter, _ in rows}
    counts = {len(cells) for _, cells in rows}
    non_empty_counts = {sum(cell != "" for cell in cells) for _, cells in rows}
    if len(delimiters) != 1 or len(counts) != 1 or len(non_empty_counts) != 1:
        item: DetectionItem = RejectedCandidate(
            ordinal=0,
            kind="structured_text",
            raw_source=raw_source,
            line_start=start_line.number,
            line_end=end_line.number,
            reason="ragged_structured_rows",
        )
        return _line_offsets(document)[start_index], item

    column_count = counts.pop()
    numeric_rows = sum(any(_is_numeric_looking(cell) for cell in cells[1:]) for _, cells in rows)
    numeric_cells = sum(
        _is_numeric_looking(cell) for _, cells in rows for cell in cells[1:]
    )
    value_cells = len(rows) * (column_count - 1)
    density = Decimal(numeric_cells) / Decimal(value_cells)
    normalized_rows = " ".join(cell for _, cells in rows for cell in cells).casefold()
    header = any(signal.casefold() in normalized_rows for signal in _HEADER_SIGNALS)
    has_density = density >= Decimal("0.5")
    if not 2 <= column_count <= 20 or numeric_rows < 2 or not (header or has_density):
        item = RejectedCandidate(
            ordinal=0,
            kind="structured_text",
            raw_source=raw_source,
            line_start=start_line.number,
            line_end=end_line.number,
            reason="insufficient_structural_evidence",
        )
        return _line_offsets(document)[start_index], item

    evidence = ["consistent_columns"]
    if header:
        evidence.append("financial_header")
    if has_density:
        evidence.append("numeric_density")
    if len(rows) >= 5:
        evidence.append("five_or_more_rows")
    confidence = Decimal("0.75")
    confidence += Decimal("0.05") if header else Decimal()
    confidence += Decimal("0.05") if has_density else Decimal()
    confidence += Decimal("0.05") if len(rows) >= 5 else Decimal()
    item = TableCandidate(
        ordinal=0,
        kind="structured_text",
        raw_source=raw_source,
        line_start=start_line.number,
        line_end=end_line.number,
        confidence=float(min(confidence, Decimal("0.90"))),
        evidence=tuple(evidence),
    )
    return _line_offsets(document)[start_index], item


def _structured_events(document: DecodedDocument) -> list[DetectionEvent]:
    events: list[DetectionEvent] = []
    for block in document.blocks:
        if block.kind not in {"paragraph", "notes"}:
            continue
        index = block.line_start - 1
        stop = block.line_end
        while index < stop:
            line = document.lines[index]
            split = _split_structured_row(line.text)
            if (
                not line.text.strip()
                or len(line.text) > 200
                or _PAGE_MARKER_RE.fullmatch(line.text)
                or split is None
                or _is_list_line(line.text, split[1])
            ):
                index += 1
                continue
            start_index = index
            rows = [split]
            index += 1
            while index < stop:
                next_line = document.lines[index]
                next_split = _split_structured_row(next_line.text)
                if (
                    not next_line.text.strip()
                    or len(next_line.text) > 200
                    or _PAGE_MARKER_RE.fullmatch(next_line.text)
                    or next_split is None
                    or _is_list_line(next_line.text, next_split[1])
                ):
                    break
                rows.append(next_split)
                index += 1
            if len(rows) >= 3:
                events.append(_structured_event(document, start_index, index - 1, rows))
            if index == start_index:
                index += 1
    return events


def _with_ordinals(events: list[DetectionEvent]) -> tuple[
    tuple[TableCandidate, ...], tuple[RejectedCandidate, ...]
]:
    candidates: list[TableCandidate] = []
    rejected: list[RejectedCandidate] = []
    for ordinal, (_, item) in enumerate(sorted(events, key=lambda event: event[0])):
        if isinstance(item, TableCandidate):
            candidates.append(item.model_copy(update={"ordinal": ordinal}))
        else:
            rejected.append(item.model_copy(update={"ordinal": ordinal}))
    return tuple(candidates), tuple(rejected)


def detect_table_candidates(document: DecodedDocument) -> DetectionResult:
    """Detect validated HTML tables before conservative structured-text candidates."""
    candidates, rejected = _with_ordinals(_html_events(document) + _structured_events(document))
    return DetectionResult(candidates=candidates, rejected=rejected, blocks=document.blocks)
