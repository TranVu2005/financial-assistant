# ViFinQA TXT Ingestion Task 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic, provenance-preserving ViFinQA TXT ingestion covering lossless reading, HTML-first detection, conservative fallback, rectangular extraction, multi-line headers, and table continuations.

**Architecture:** A verified reader converts one ready `DocumentRecord` into immutable source lines and blocks. Separate detector and extractor modules turn those blocks into auditable candidates, rejected candidates, canonical `TableRecord`/`CellRecord` values, and logical-grid placements without applying financial normalization.

**Tech Stack:** Python 3.11, Pydantic 2, standard-library `html.parser`, pytest 8, Ruff, mypy strict, uv.

## Global Constraints

- Support only the current ViFinQA `<ticker>/<year>/<document>/<file>.txt` snapshot.
- Accept only `DocumentRecord.inventory_status == "ready"` with `encoding` exactly `utf-8` or `utf-8-sig`.
- Verify source byte size and SHA-256 before parsing; never mutate source files.
- Preserve decoded Unicode, line endings, raw table/cell text, and one-based inclusive provenance.
- Detect HTML tables first; run conservative structured-text fallback only outside HTML regions.
- Do not perform numeric, metric, company, period, statement-type, or unit normalization.
- Expand `rowspan`/`colspan` through placements that reference one source `CellRecord`.
- Reject a candidate atomically on invalid structure; continue processing later candidates.
- Exclude timestamps, absolute paths, randomness, locale-dependent parsing, and unordered output.
- Add no runtime dependency.

---

## File Map

| File | Responsibility |
|---|---|
| `src/financial_report_qa/core/errors.py` | Typed document-level ingestion failures |
| `src/financial_report_qa/ingestion/provenance.py` | Immutable contracts, span rules, stable cell IDs |
| `src/financial_report_qa/ingestion/txt_reader.py` | Integrity verification, lossless decoding, block segmentation |
| `src/financial_report_qa/ingestion/table_detector.py` | HTML-first candidates and conservative text fallback |
| `src/financial_report_qa/ingestion/table_extractor.py` | HTML/text parsing, grid placement, headers, continuations, orchestration |
| `src/financial_report_qa/ingestion/__init__.py` | Approved public exports |
| `tests/unit/ingestion/test_provenance.py` | Contract and stable-ID tests |
| `tests/unit/ingestion/test_txt_reader.py` | Reader, integrity, line, and block tests |
| `tests/unit/ingestion/test_table_detector.py` | Candidate, confidence, and rejection tests |
| `tests/unit/ingestion/test_table_extractor.py` | Span, grid, label, continuation, and atomicity tests |
| `tests/golden/extraction/fixtures/*.txt` | Small synthetic ViFinQA-shaped sources |
| `tests/golden/extraction/expected/*.json` | Hand-reviewed semantic golden outputs |
| `tests/golden/extraction/test_txt_extraction.py` | End-to-end deterministic fixture tests |
| `scripts/smoke_ingestion.py` | Optional local full-snapshot smoke validation |
| `docs/development.md` | Local ViFinQA smoke command and expected report |

---

### Task 1: Immutable ingestion contracts and stable provenance IDs

**Files:**

- Modify: `src/financial_report_qa/core/errors.py`
- Create: `src/financial_report_qa/ingestion/provenance.py`
- Create: `tests/unit/ingestion/test_provenance.py`

**Interfaces:**

- Consumes: `DocumentRecord`, `TableRecord`, and `CellRecord` from `financial_report_qa.schemas`.
- Produces: `SourceLine`, `TextBlock`, `DecodedDocument`, `TableCandidate`, `RejectedCandidate`, `DetectionResult`, `CellPlacement`, `ExtractedTable`, `ExtractionResult`, and `stable_cell_id(table_id: str, origin_row: int, origin_col: int) -> str`.
- Produces errors: `SourceIngestionError`, `InvalidSourceDocumentError`, `UnsupportedSourceEncodingError`, `SourceSnapshotMismatchError`, and `SourceReadError`.

- [ ] **Step 1: Write failing contract and stable-ID tests**

Create `tests/unit/ingestion/test_provenance.py` with these behaviors:

```python
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from financial_report_qa.ingestion.provenance import (
    CellPlacement,
    DecodedDocument,
    ExtractedTable,
    RejectedCandidate,
    SourceLine,
    TableCandidate,
    TextBlock,
    stable_cell_id,
)
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id


DOC_ID = stable_document_id("a" * 64)
TABLE_ID = stable_table_id(DOC_ID, 2, 6)


def test_stable_cell_id_uses_table_and_origin_coordinate() -> None:
    payload = f"{TABLE_ID}\n1\n2".encode()
    expected = f"cell_{hashlib.sha256(payload).hexdigest()}"

    assert stable_cell_id(TABLE_ID, 1, 2) == expected
    assert stable_cell_id(TABLE_ID, 1, 3) != expected


@pytest.mark.parametrize(
    ("table_id", "row", "col"),
    [("bad", 0, 0), (TABLE_ID, -1, 0), (TABLE_ID, 0, -1), (TABLE_ID, True, 0)],
)
def test_stable_cell_id_rejects_invalid_inputs(
    table_id: str,
    row: int,
    col: int,
) -> None:
    with pytest.raises(ValueError):
        stable_cell_id(table_id, row, col)


def test_source_and_candidate_contracts_are_frozen_and_validate_spans() -> None:
    line = SourceLine(number=1, text="Doanh thu", line_ending="\r\n")
    block = TextBlock(kind="paragraph", line_start=1, line_end=1, text="Doanh thu\r\n")
    candidate = TableCandidate(
        ordinal=0,
        kind="html",
        raw_source="<table></table>\r\n",
        line_start=2,
        line_end=2,
        confidence=1.0,
        evidence=("html_table_marker",),
    )

    assert line.text + line.line_ending == "Doanh thu\r\n"
    assert block.text == "Doanh thu\r\n"
    assert candidate.evidence == ("html_table_marker",)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(candidate, "confidence", 0.5)
    with pytest.raises(ValidationError):
        TextBlock(kind="paragraph", line_start=2, line_end=1, text="bad")


def test_rejection_codes_and_placements_are_strict() -> None:
    rejection = RejectedCandidate(
        ordinal=0,
        kind="html",
        raw_source="<table>",
        line_start=4,
        line_end=4,
        reason="unclosed_html_table",
    )
    placement = CellPlacement(row_idx=0, col_idx=1, cell_id="cell_" + "b" * 64)

    assert rejection.reason == "unclosed_html_table"
    assert placement.col_idx == 1
    with pytest.raises(ValidationError):
        RejectedCandidate.model_validate({**rejection.model_dump(), "reason": "unknown"})
    with pytest.raises(ValidationError):
        CellPlacement.model_validate({**placement.model_dump(), "extra": True})


def test_decoded_document_requires_exact_source_reconstruction() -> None:
    document = DocumentRecord(
        doc_id=DOC_ID,
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path="AAA/2024/report/source.txt",
        company_code="AAA",
        report_year=2024,
        statement_scope="other",
        sha256="a" * 64,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )

    with pytest.raises(ValidationError, match="reconstruct"):
        DecodedDocument(
            document=document,
            text="x",
            lines=(SourceLine(number=1, text="y", line_ending=""),),
            blocks=(),
        )


def test_extracted_table_rejects_unknown_cell_reference() -> None:
    table = TableRecord(
        table_id=TABLE_ID,
        doc_id=DOC_ID,
        title_raw=None,
        statement_type=None,
        unit_raw=None,
        unit_normalized=None,
        line_start=2,
        line_end=6,
        row_count=1,
        column_count=1,
        quality_score=1.0,
        csv_path=None,
    )
    cell = CellRecord(
        cell_id=stable_cell_id(TABLE_ID, 0, 0),
        table_id=TABLE_ID,
        row_idx=0,
        col_idx=0,
        row_label_raw=None,
        row_label_canonical=None,
        column_label_raw=None,
        column_label_canonical=None,
        value_raw="1",
        value_numeric=None,
        period=None,
        unit=None,
        source_line_start=3,
        source_line_end=3,
        extraction_confidence=1.0,
    )

    with pytest.raises(ValidationError, match="reference source cells"):
        ExtractedTable(
            table=table,
            cells=(cell,),
            placements=(CellPlacement(row_idx=0, col_idx=0, cell_id="cell_" + "f" * 64),),
            evidence=("html_table_marker",),
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_provenance.py
```

Expected: collection fails because `financial_report_qa.ingestion.provenance` does not exist.

- [ ] **Step 3: Add the typed error hierarchy**

Append to `src/financial_report_qa/core/errors.py`:

```python
class SourceIngestionError(FinancialReportQAError):
    """Base class for deterministic source-ingestion failures."""


class InvalidSourceDocumentError(SourceIngestionError):
    """The inventory record cannot be consumed by ingestion."""


class UnsupportedSourceEncodingError(SourceIngestionError):
    """The inventory-approved source encoding is unsupported or inconsistent."""


class SourceSnapshotMismatchError(SourceIngestionError):
    """The current source bytes differ from the immutable inventory record."""


class SourceReadError(SourceIngestionError):
    """The verified relative source path could not be read."""
```

- [ ] **Step 4: Implement strict immutable provenance contracts**

Create `src/financial_report_qa/ingestion/provenance.py`. Use a shared frozen model base,
strict integer fields, one span validator, and these exact public shapes:

```python
from __future__ import annotations

import hashlib
import re
from typing import Literal, Self

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
    text: str

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_start must not exceed line_end")
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
    raw_source: str
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    confidence: float = Field(strict=True, ge=0, le=1)
    evidence: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_start must not exceed line_end")
        return self


class RejectedCandidate(_FrozenModel):
    ordinal: int = Field(strict=True, ge=0)
    kind: CandidateKind
    raw_source: str
    line_start: int = Field(strict=True, ge=1)
    line_end: int = Field(strict=True, ge=1)
    reason: RejectionCode

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.line_end < self.line_start:
            raise ValueError("line_start must not exceed line_end")
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
    evidence: tuple[str, ...] = Field(min_length=1)

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
    doc_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    blocks: tuple[TextBlock, ...]
    tables: tuple[ExtractedTable, ...]
    rejected: tuple[RejectedCandidate, ...]

    @model_validator(mode="after")
    def validate_document_identity(self) -> Self:
        if any(item.table.doc_id != self.doc_id for item in self.tables):
            raise ValueError("tables must belong to the extraction document")
        return self


def stable_cell_id(table_id: str, origin_row: int, origin_col: int) -> str:
    if not isinstance(table_id, str) or _TABLE_ID_RE.fullmatch(table_id) is None:
        raise ValueError("table_id must be a canonical table ID")
    for name, value in (("origin_row", origin_row), ("origin_col", origin_col)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    payload = f"{table_id}\n{origin_row}\n{origin_col}".encode()
    return f"cell_{hashlib.sha256(payload).hexdigest()}"
```

Factor the repeated span check into one private helper and preserve the public fields and
exact validation behavior above. Set `min_length=1` on candidate raw source,
block text, rejection raw source, and every evidence code. Extend the contract tests with
one invalid `DecodedDocument` reconstruction and one placement referencing an unknown
cell ID so these cross-model validators fail before reader/extractor work begins.

- [ ] **Step 5: Run tests and static checks; verify GREEN**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_provenance.py
uv run --frozen --no-sync ruff check src/financial_report_qa/core/errors.py src/financial_report_qa/ingestion/provenance.py tests/unit/ingestion/test_provenance.py
uv run --frozen --no-sync mypy src/financial_report_qa/core/errors.py src/financial_report_qa/ingestion/provenance.py tests/unit/ingestion/test_provenance.py
```

Expected: all commands pass.

- [ ] **Step 6: Commit the contracts**

```powershell
git add src/financial_report_qa/core/errors.py src/financial_report_qa/ingestion/provenance.py tests/unit/ingestion/test_provenance.py
git commit -m "feat: define ViFinQA ingestion provenance contracts"
```

---

### Task 2: Verified lossless TXT reader and block segmentation

**Files:**

- Create: `src/financial_report_qa/ingestion/txt_reader.py`
- Create: `tests/unit/ingestion/test_txt_reader.py`

**Interfaces:**

- Consumes: `DocumentRecord`, Task 1 reader contracts, and Task 1 typed errors.
- Produces: `read_document(root: Path, document: DocumentRecord) -> DecodedDocument`.
- Guarantees: exact decoded-text reconstruction, safe relative-path resolution, source verification, and deterministic blocks.

- [ ] **Step 1: Add reusable test helpers and failing integrity tests**

Create `tests/unit/ingestion/test_txt_reader.py`. The helper must construct the record from
the exact bytes written, preventing fixture drift:

```python
from __future__ import annotations

import hashlib
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
```

- [ ] **Step 2: Add failing segmentation and safe-error tests**

Extend the same file:

```python
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
```

- [ ] **Step 3: Run reader tests and verify RED**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_txt_reader.py
```

Expected: collection fails because `txt_reader.py` does not exist.

- [ ] **Step 4: Implement verified binary reading and exact line splitting**

Create `src/financial_report_qa/ingestion/txt_reader.py` with these private boundaries:

```python
_CHUNK_SIZE = 1024 * 1024
_PAGE_MARKER_RE = re.compile(r"^===== PAGE [1-9][0-9]* =====$")
_NOTES_HEADINGS = {
    "thuyết minh",
    "thuyêt minh",
    "thuyết minh báo cáo tài chính",
    "thuyêt minh báo cáo tài chính",
}


def _safe_source_path(root: Path, document: DocumentRecord) -> Path:
    try:
        root_resolved = root.resolve()
        source = (
            root_resolved / Path(*PurePosixPath(document.relative_path).parts)
        ).resolve()
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
    if len(payload) != document.file_size_bytes or digest.hexdigest() != document.sha256:
        raise SourceSnapshotMismatchError(
            f"source bytes do not match inventory: {document.relative_path}"
        )
    return bytes(payload)
```

Decode `utf-8` strictly. For `utf-8-sig`, first require `payload.startswith(codecs.BOM_UTF8)`
and then decode strictly. Reject `None` and every other encoding with
`UnsupportedSourceEncodingError`. Convert `UnicodeDecodeError` to the same typed error
without including codec byte dumps or absolute paths.

Implement `_split_source_lines(text: str) -> tuple[SourceLine, ...]` with a cursor over
`re.finditer(r".*?(?:\r\n|\r|\n|$)", text)`. Stop before the terminal zero-length match;
separate exactly one of `\r\n`, `\r`, `\n`, or `""` from each matched line.

- [ ] **Step 5: Implement deterministic block segmentation**

Implement `_segment_blocks(lines: tuple[SourceLine, ...]) -> tuple[TextBlock, ...]` as one
source-order state machine:

1. Flush an accumulated prose block before every page or table boundary.
2. Emit a page-marker line as its own block.
3. When a case-insensitive `<table` appears, collect through the first case-insensitive
   `</table>`; if absent, collect through EOF.
4. Blank lines flush prose and are not standalone blocks.
5. A notes heading switches later prose blocks to `notes`; page and table kinds remain
   unchanged.
6. Construct block text by concatenating the exact participating source lines.

For notes matching, use exactly:

```python
def _heading_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
```

Compare the whole stripped heading key against `_NOTES_HEADINGS`; never apply this
temporary matching form to stored text.

Implement `read_document` in this order: validate `ready`, resolve safely, read and verify,
decode, split lines, segment blocks, construct `DecodedDocument`.

- [ ] **Step 6: Run reader tests and quality checks; verify GREEN**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_provenance.py tests/unit/ingestion/test_txt_reader.py
uv run --frozen --no-sync ruff check src/financial_report_qa/ingestion tests/unit/ingestion
uv run --frozen --no-sync mypy src/financial_report_qa/ingestion tests/unit/ingestion
```

Expected: all commands pass.

- [ ] **Step 7: Commit the reader**

```powershell
git add src/financial_report_qa/ingestion/txt_reader.py tests/unit/ingestion/test_txt_reader.py
git commit -m "feat: read ViFinQA TXT with exact provenance"
```

---

### Task 3: HTML-first detection and conservative structured-text fallback

**Files:**

- Create: `src/financial_report_qa/ingestion/table_detector.py`
- Create: `tests/unit/ingestion/test_table_detector.py`

**Interfaces:**

- Consumes: `DecodedDocument` and its Task 2 blocks.
- Produces: `detect_table_candidates(document: DecodedDocument) -> DetectionResult`.
- Produces evidence: `html_table_marker`, `consistent_columns`, `financial_header`, `numeric_density`, and `five_or_more_rows`.

- [ ] **Step 1: Write failing HTML candidate and rejection tests**

Create a local helper that writes a valid record and calls `read_document`:

```python
def decoded(tmp_path: Path, text: str) -> DecodedDocument:
    content = text.encode()
    relative = "AAA/2024/AAA_consolidated/source.txt"
    path = tmp_path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    record = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )
    return read_document(tmp_path, record)
```

Import `hashlib`, `Path`, `DecodedDocument`, `read_document`, `DocumentRecord`, and
`stable_document_id`, then add:

```python
def test_detector_prefers_closed_html_and_preserves_span(tmp_path: Path) -> None:
    source = (
        "Mở đầu\n"
        "<table>\n<tr><td>Chỉ tiêu</td><td>2024</td></tr>\n</table>\n"
        "Sau bảng\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.kind == "html"
    assert (candidate.line_start, candidate.line_end) == (2, 4)
    assert candidate.raw_source == source.split("Mở đầu\n", 1)[1].split("Sau bảng", 1)[0]
    assert candidate.confidence == 1.0
    assert candidate.evidence == ("html_table_marker",)


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("<table><tr><td>1</td></tr>", "unclosed_html_table"),
        ("<table><tr><td><table></table></td></tr></table>", "nested_html_table"),
    ],
)
def test_detector_rejects_invalid_html_regions(
    tmp_path: Path,
    source: str,
    reason: str,
) -> None:
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == [reason]


def test_detector_splits_sibling_tables_on_one_line(tmp_path: Path) -> None:
    source = (
        "<table><tr><td>A</td><td>1</td></tr></table>"
        "<table><tr><td>B</td><td>2</td></tr></table>\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 2
    assert [item.raw_source.count("<table>") for item in result.candidates] == [1, 1]
    assert [(item.line_start, item.line_end) for item in result.candidates] == [(1, 1), (1, 1)]
```

- [ ] **Step 2: Write failing fallback threshold tests**

Add exact structured and prose cases:

```python
def test_fallback_accepts_only_consistent_financial_rows(tmp_path: Path) -> None:
    source = (
        "Chỉ tiêu\t2024\t2023\n"
        "Doanh thu\t1.000\t900\n"
        "Lợi nhuận\t100\t80\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.kind == "structured_text"
    assert candidate.confidence == 0.85
    assert candidate.evidence == (
        "consistent_columns",
        "financial_header",
        "numeric_density",
    )


def test_fallback_rejects_ragged_table_like_rows(tmp_path: Path) -> None:
    source = "Chỉ tiêu  2024  2023\nDoanh thu  100\nLợi nhuận  20  10\n"
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == ["ragged_structured_rows"]


def test_fallback_ignores_explanatory_prose_and_bullets(tmp_path: Path) -> None:
    source = (
        "Kính gửi: Ủy ban Chứng khoán Nhà nước\n\n"
        "- Lợi nhuận sau thuế năm 2024: 100 đồng\n"
        "- Lợi nhuận sau thuế năm 2023: 80 đồng\n"
        "- Nguyên nhân: doanh thu giảm\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()


def test_fallback_rejects_delimited_rows_without_financial_evidence(tmp_path: Path) -> None:
    source = "Tên  Phòng ban\nAn  Kế toán\nBình  Kiểm toán\n"
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates == ()
    assert [item.reason for item in result.rejected] == [
        "insufficient_structural_evidence"
    ]


def test_fallback_caps_high_evidence_confidence_at_point_nine(tmp_path: Path) -> None:
    source = (
        "Chỉ tiêu\t2024\t2023\n"
        "A\t10\t9\n"
        "B\t8\t7\n"
        "C\t6\t5\n"
        "D\t4\t3\n"
    )
    result = detect_table_candidates(decoded(tmp_path, source))

    assert result.candidates[0].confidence == 0.9
    assert result.candidates[0].evidence[-1] == "five_or_more_rows"
```

- [ ] **Step 3: Run detector tests and verify RED**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_table_detector.py
```

Expected: collection fails because `table_detector.py` does not exist.

- [ ] **Step 4: Implement HTML-region validation**

Create `src/financial_report_qa/ingestion/table_detector.py` with:

```python
_OPEN_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_CLOSE_TABLE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_TAB_SPLIT_RE = re.compile(r"\t+")
_SPACE_SPLIT_RE = re.compile(r" {2,}")
_HEADER_SIGNALS = ("mã số", "chỉ tiêu", "thuyết minh", "năm", "kỳ", "đơn vị", "đvt")


_TABLE_TOKEN_RE = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
```

Scan each `table` block token-by-token with a depth counter and source offsets. A depth-0
opening starts a region; an opening at depth 1 marks that outer region as nested; a close
returning to depth 0 ends the region. Emit each non-nested sibling region as its own HTML
candidate, even when two tables share one source line. Reject an outer nested region as
`nested_html_table`; reject an unmatched opening through block end as
`unclosed_html_table`. Convert substring offsets to one-based document lines by counting
line terminators before each boundary. Assign ordinals in global source order across
accepted and rejected items; preserve the exact table substring and inclusive line span.

- [ ] **Step 5: Implement exact fallback rules and confidence**

Use these private functions:

```python
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
```

For every non-HTML `paragraph` or `notes` block, examine consecutive non-empty lines.
Only treat a run as table-like when at least three lines use a supported delimiter.
Reject it as `ragged_structured_rows` when delimiter classes or non-empty column counts
differ. Otherwise require 2–20 columns, at least two numeric rows outside column zero,
and either a header signal or numeric density of at least 0.5. Reject table-like runs
that miss those final evidence rules as `insufficient_structural_evidence`; ignore normal
prose that never forms a three-line delimited run.

Build evidence in this order:

1. `consistent_columns`;
2. `financial_header` when present;
3. `numeric_density` when density is at least 0.5;
4. `five_or_more_rows` when applicable.

Calculate confidence with `Decimal` or exact additions, then cast to float:
`0.75 + header*0.05 + density*0.05 + five_rows*0.05`, capped at `0.90`.
Reject page markers and any line over 200 characters. Treat `- `, `• `, `* `, or a decimal
prefix such as `1. ` as a list only when the line does not split into at least two columns
with numeric evidence outside column zero; this keeps legitimate numbered financial rows.

- [ ] **Step 6: Run focused and regression checks; verify GREEN**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_provenance.py tests/unit/ingestion/test_txt_reader.py tests/unit/ingestion/test_table_detector.py
uv run --frozen --no-sync ruff check src/financial_report_qa/ingestion tests/unit/ingestion
uv run --frozen --no-sync mypy src/financial_report_qa/ingestion tests/unit/ingestion
```

Expected: all commands pass.

- [ ] **Step 7: Commit detection**

```powershell
git add src/financial_report_qa/ingestion/table_detector.py tests/unit/ingestion/test_table_detector.py
git commit -m "feat: detect ViFinQA table candidates"
```

---

### Task 4: Atomic table extraction, span expansion, and raw labels

**Files:**

- Create: `src/financial_report_qa/ingestion/table_extractor.py`
- Create: `tests/unit/ingestion/test_table_extractor.py`

**Interfaces:**

- Consumes: `DecodedDocument`, `DetectionResult`, `TableCandidate`, canonical schemas, and Task 1 provenance contracts.
- Produces: `extract_candidates(document: DecodedDocument, detection: DetectionResult) -> ExtractionResult` for independent candidates.
- Internal boundary: `_materialize_table(document, candidate, rows) -> ExtractedTable`, where `rows` contain source cells with text, line span, row span, column span, and header flag.

- [ ] **Step 1: Write failing HTML/entity/span tests**

Create `tests/unit/ingestion/test_table_extractor.py` with a self-contained helper:

```python
def extract(tmp_path: Path, source: str) -> ExtractionResult:
    content = source.encode()
    relative = "AAA/2024/AAA_consolidated/source.txt"
    path = tmp_path / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    record = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )
    decoded = read_document(tmp_path, record)
    detection = detect_table_candidates(decoded)
    return extract_candidates(decoded, detection)
```

Import `hashlib`, `Path`, `ExtractionResult`, the three pipeline functions,
`DocumentRecord`, and `stable_document_id`, then add:

```python
def test_extracts_rectangular_grid_with_shared_span_placements(tmp_path: Path) -> None:
    source = (
        "BẢNG CÂN ĐỐI KẾ TOÁN\n"
        "<table>\n"
        '<tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">Năm</th></tr>\n'
        "<tr><th>2024</th><th>2023</th></tr>\n"
        "<tr><td>Doanh thu</td><td>1.000</td><td>900</td></tr>\n"
        "</table>\n"
    )
    result = extract(tmp_path, source)

    assert len(result.tables) == 1
    extracted = result.tables[0]
    assert extracted.table.title_raw == "BẢNG CÂN ĐỐI KẾ TOÁN"
    assert (extracted.table.line_start, extracted.table.line_end) == (2, 6)
    assert (extracted.table.row_count, extracted.table.column_count) == (3, 3)
    assert len(extracted.cells) == 7
    assert len(extracted.placements) == 9
    first_cell_id = extracted.placements[0].cell_id
    assert extracted.placements[3].cell_id == first_cell_id
    assert [cell.value_raw for cell in extracted.cells][-3:] == ["Doanh thu", "1.000", "900"]


def test_decodes_entities_but_keeps_raw_candidate(tmp_path: Path) -> None:
    source = "<table><tr><td>Lợi nhuận &amp; thu nhập<br>khác</td><td>1</td></tr></table>\n"
    result = extract(tmp_path, source)

    assert result.tables[0].cells[0].value_raw == "Lợi nhuận & thu nhập\nkhác"
    assert "&amp;" in result.blocks[0].text
```

- [ ] **Step 2: Write failing provenance, header, and atomic-rejection tests**

Add:

```python
def test_composes_multiline_headers_without_normalizing_values(tmp_path: Path) -> None:
    source = (
        "<table>\n"
        '<tr><th rowspan="2">Chỉ tiêu</th><th colspan="2">Năm</th></tr>\n'
        "<tr><th>2024</th><th>2023</th></tr>\n"
        "<tr><td>Lợi nhuận</td><td>(1.234,50)</td><td>-</td></tr>\n"
        "</table>\n"
    )
    result = extract(tmp_path, source)
    values = {cell.value_raw: cell for cell in result.tables[0].cells}

    assert values["(1.234,50)"].row_label_raw == "Lợi nhuận"
    assert values["(1.234,50)"].column_label_raw == "Năm\n2024"
    assert values["(1.234,50)"].value_numeric is None
    assert values["(1.234,50)"].period is None
    assert values["(1.234,50)"].unit is None
    assert (values["(1.234,50)"].source_line_start, values["(1.234,50)"].source_line_end) == (4, 4)


@pytest.mark.parametrize(
    ("attribute", "reason"),
    [('rowspan="0"', "invalid_span_value"), ('colspan="100001"', "expansion_limit_exceeded")],
)
def test_invalid_expansion_rejects_whole_candidate_and_continues(
    tmp_path: Path,
    attribute: str,
    reason: str,
) -> None:
    source = (
        f"<table><tr><td {attribute}>bad</td></tr></table>\n"
        "<table><tr><td>good</td><td>1</td></tr></table>\n"
    )
    result = extract(tmp_path, source)

    assert [item.reason for item in result.rejected] == [reason]
    assert len(result.tables) == 1
    assert [cell.value_raw for cell in result.tables[0].cells] == ["good", "1"]


def test_span_collision_is_atomic(tmp_path: Path) -> None:
    source = (
        "<table>"
        '<tr><td rowspan="2">A</td><td>B</td><td rowspan="2">C</td></tr>'
        '<tr><td colspan="2">overlap</td></tr>'
        "</table>\n"
    )
    result = extract(tmp_path, source)

    assert result.tables == ()
    assert [item.reason for item in result.rejected] == ["span_collision"]


def test_ragged_html_rows_use_absent_placements_not_invented_cells(tmp_path: Path) -> None:
    source = (
        "<table><tr><td>A</td><td>1</td></tr>"
        "<tr><td>B</td></tr></table>\n"
    )
    table = extract(tmp_path, source).tables[0]

    assert (table.table.row_count, table.table.column_count) == (2, 2)
    assert len(table.cells) == 3
    assert len(table.placements) == 3
```

- [ ] **Step 3: Run extractor tests and verify RED**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_table_extractor.py
```

Expected: collection fails because `table_extractor.py` does not exist.

- [ ] **Step 4: Implement source-aware HTML and structured-row parsing**

In `table_extractor.py`, define private immutable data carriers:

```python
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
```

Create `_ViFinQATableParser(HTMLParser)` with `convert_charrefs=True`. It must:

- reject nested `table` tags;
- require cells inside rows and rows inside the table;
- parse positive decimal `rowspan`/`colspan`, defaulting each to 1;
- record opening-tag line and closing-tag line plus `candidate.line_start - 1`;
- append `\n` for `br` inside a cell;
- collect decoded text from inline tags;
- trim only outer cell whitespace when closing a cell;
- reject unfinished rows/cells and tables with no cells as deterministic reason codes.

Implement structured candidates by splitting with the delimiter already validated by the
detector. Each source cell has `rowspan=colspan=1`, `is_header=False`, and both provenance
lines equal its source line.

- [ ] **Step 5: Implement collision-safe grid expansion and materialization**

Use a dictionary keyed by `(row_idx, col_idx)` while scanning source rows. For each source
cell, select the first free column, verify every requested span coordinate is free, and
insert the same source-cell token into all covered coordinates. Reject on collision or
when the placement count would exceed `100_000`.

After expansion:

1. derive maximum row and column indexes;
2. treat missing coordinates as absent placements, not empty invented cells;
3. create `TableRecord` from candidate span, title/unit discovery, dimensions, confidence,
   and `stable_table_id`;
4. create exactly one `CellRecord` for each source-cell token using its origin coordinate
   and `stable_cell_id`;
5. create ordered placements for every occupied logical coordinate;
6. set all canonical/numeric/period/unit fields to `None` and `csv_path=None`.

Derive the header band as the maximal first three rows. Include an HTML row when every
populated placement refers to a `th` source cell, including a header cell carried into the
row by `rowspan`, or when fewer than half of populated non-first cells are numeric-looking.
For structured text, also include the first row when it contains a detector header signal.
Compose distinct non-empty header texts per column top-to-bottom with `\n`. Set every data
cell's `row_label_raw` to the first populated cell text in its row and `column_label_raw`
to the composed header for its column. Header cells may have `None` raw labels.

Find `title_raw` from the nearest eligible line within three lines before the candidate.
Find `unit_raw` only from original text containing `đơn vị`, `đơn vị tính`, or `đvt`.
Do not derive `statement_type`.

- [ ] **Step 6: Preserve candidate atomicity in `extract_candidates`**

Process candidates in source order. Convert known parser/layout failures into
`RejectedCandidate` using the original ordinal, kind, raw source, and span. Append detector
rejections and extraction rejections in `(line_start, ordinal)` order. Do not catch
programming errors or document-level failures.

Return `ExtractionResult` with the document ID, original blocks, successful independent
tables, and ordered rejections. Repeated calls with equal inputs must return equal models.

- [ ] **Step 7: Run focused and regression checks; verify GREEN**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data
uv run --frozen --no-sync ruff check src/financial_report_qa/ingestion tests/unit/ingestion
uv run --frozen --no-sync mypy src/financial_report_qa/ingestion tests/unit/ingestion
```

Expected: all commands pass.

- [ ] **Step 8: Commit atomic extraction**

```powershell
git add src/financial_report_qa/ingestion/table_extractor.py tests/unit/ingestion/test_table_extractor.py
git commit -m "feat: extract provenance-preserving ViFinQA tables"
```

---

### Task 5: Continuations, public orchestration, golden fixtures, and corpus smoke

**Files:**

- Modify: `src/financial_report_qa/ingestion/table_extractor.py`
- Modify: `src/financial_report_qa/ingestion/__init__.py`
- Modify: `tests/unit/ingestion/test_table_extractor.py`
- Create: `tests/golden/extraction/fixtures/unicode_continuation.txt`
- Create: `tests/golden/extraction/fixtures/structured_fallback.txt`
- Create: `tests/golden/extraction/fixtures/explanatory_letter.txt`
- Create: `tests/golden/extraction/expected/unicode_continuation.json`
- Create: `tests/golden/extraction/expected/structured_fallback.json`
- Create: `tests/golden/extraction/expected/explanatory_letter.json`
- Create: `tests/golden/extraction/test_txt_extraction.py`
- Create: `scripts/smoke_ingestion.py`
- Modify: `docs/development.md`

**Interfaces:**

- Consumes: all Task 1–4 modules and Day 2 `build_inventory`.
- Produces: `extract_document(root: Path, document: DocumentRecord) -> ExtractionResult`.
- Exports: `DecodedDocument`, `DetectionResult`, `ExtractionResult`, `extract_document`, `read_document`, `detect_table_candidates`, `extract_candidates`, and `stable_cell_id`.

- [ ] **Step 1: Write failing continuation tests**

Extend `tests/unit/ingestion/test_table_extractor.py`:

```python
def test_merges_compatible_page_continuation_and_drops_repeated_header(tmp_path: Path) -> None:
    source = (
        "BẢNG KẾT QUẢ KINH DOANH\n"
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
        "<tr><td>Doanh thu</td><td>100</td></tr></table>\n"
        "===== PAGE 2 =====\n"
        "BẢNG KẾT QUẢ KINH DOANH\n"
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr>"
        "<tr><td>Lợi nhuận</td><td>20</td></tr></table>\n"
    )
    result = extract(tmp_path, source)

    assert len(result.tables) == 1
    table = result.tables[0]
    assert (table.table.line_start, table.table.line_end) == (2, 5)
    assert (table.table.row_count, table.table.column_count) == (3, 2)
    assert [cell.value_raw for cell in table.cells] == [
        "Chỉ tiêu", "2024", "Doanh thu", "100", "Lợi nhuận", "20"
    ]
    assert "continued_across_page" in table.evidence


def test_does_not_merge_different_headers(tmp_path: Path) -> None:
    source = (
        "<table><tr><th>Chỉ tiêu</th><th>2024</th></tr><tr><td>A</td><td>1</td></tr></table>\n"
        "===== PAGE 2 =====\n"
        "<table><tr><th>Mã số</th><th>2023</th></tr><tr><td>B</td><td>2</td></tr></table>\n"
    )

    assert len(extract(tmp_path, source).tables) == 2
```

- [ ] **Step 2: Run continuation tests and verify RED**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion/test_table_extractor.py -k "continuation or different_headers"
```

Expected: the compatible case returns two tables instead of one.

- [ ] **Step 3: Implement conservative continuation merging**

After independent extraction, compare adjacent tables only. Merge when all conditions
hold: separator blocks contain only blank source lines, one page marker, and optionally
one repeated title; line distance is at most 20; column counts match; and header
fingerprints match after NFKC, case-folding, and whitespace collapse.

Rebuild a merged table rather than mutating either frozen input:

- span from the first candidate start through the second candidate end;
- retain the first header band once;
- omit the second header band's cells and placements;
- append second-table data rows and re-index their row coordinates;
- recalculate `table_id`, all `cell_id` values, and placements from final coordinates;
- use the minimum input confidence;
- append `continued_across_page` to evidence once.

Repeat left-to-right so a three-page table can merge transitively. Never merge across
prose other than the repeated title allowed above.

- [ ] **Step 4: Add the public orchestration function and exports**

Implement in `table_extractor.py`:

```python
def extract_document(root: Path, document: DocumentRecord) -> ExtractionResult:
    decoded = read_document(root, document)
    detection = detect_table_candidates(decoded)
    return extract_candidates(decoded, detection)
```

Update `ingestion/__init__.py` to import and expose only these names through `__all__`:

```python
__all__ = (
    "DecodedDocument",
    "DetectionResult",
    "ExtractionResult",
    "detect_table_candidates",
    "extract_candidates",
    "extract_document",
    "read_document",
    "stable_cell_id",
)
```

- [ ] **Step 5: Write synthetic fixtures and hand-reviewed semantic goldens**

Create `unicode_continuation.txt` with exactly:

```text
Đơn vị tính: triệu đồng
BẢNG KẾT QUẢ KINH DOANH
<table><tr><th>Chỉ tiêu</th><th>2024</th></tr><tr><td>Doanh thu thuần</td><td>(1.234,50)</td></tr></table>
===== PAGE 2 =====
BẢNG KẾT QUẢ KINH DOANH
<table><tr><th>Chỉ tiêu</th><th>2024</th></tr><tr><td>Lợi nhuận</td><td>20</td></tr></table>
```

Create `structured_fallback.txt` with exactly:

```text
Chỉ tiêu	2024	2023
Doanh thu	1.000	900
Lợi nhuận	100	80
```

Create `explanatory_letter.txt` with:

```text
===== PAGE 1 =====
Kính gửi: Ủy ban Chứng khoán Nhà nước

- Lợi nhuận sau thuế năm 2024: 100 đồng
- Lợi nhuận sau thuế năm 2023: 80 đồng
- Nguyên nhân: doanh thu giảm
```

The three expected JSON files contain only stable semantic projections, not environment
paths or repeated hash assertions. `unicode_continuation.json` is exactly:

```json
{
  "table_count": 1,
  "rejection_codes": [],
  "tables": [
    {
      "line_start": 3,
      "line_end": 6,
      "row_count": 3,
      "column_count": 2,
      "values": ["Chỉ tiêu", "2024", "Doanh thu thuần", "(1.234,50)", "Lợi nhuận", "20"],
      "evidence": ["html_table_marker", "continued_across_page"]
    }
  ]
}
```

`structured_fallback.json` is exactly:

```json
{
  "table_count": 1,
  "rejection_codes": [],
  "tables": [
    {
      "line_start": 1,
      "line_end": 3,
      "row_count": 3,
      "column_count": 3,
      "values": ["Chỉ tiêu", "2024", "2023", "Doanh thu", "1.000", "900", "Lợi nhuận", "100", "80"],
      "evidence": ["consistent_columns", "financial_header", "numeric_density"]
    }
  ]
}
```

`explanatory_letter.json` is exactly:

```json
{"table_count": 0, "rejection_codes": [], "tables": []}
```

- [ ] **Step 6: Write the failing end-to-end golden test**

Create `tests/golden/extraction/test_txt_extraction.py` with this complete harness and
projection:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from financial_report_qa.ingestion import ExtractionResult, extract_document
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id

CASES = (
    "unicode_continuation",
    "structured_fallback",
    "explanatory_letter",
)


def semantic_projection(result: ExtractionResult) -> dict[str, object]:
    return {
        "table_count": len(result.tables),
        "rejection_codes": [item.reason for item in result.rejected],
        "tables": [
            {
                "line_start": item.table.line_start,
                "line_end": item.table.line_end,
                "row_count": item.table.row_count,
                "column_count": item.table.column_count,
                "values": [cell.value_raw for cell in item.cells],
                "evidence": list(item.evidence),
            }
            for item in result.tables
        ],
    }


@pytest.mark.parametrize("case_name", CASES)
def test_fixture_matches_hand_reviewed_golden_and_is_deterministic(
    tmp_path: Path,
    case_name: str,
) -> None:
    base = Path(__file__).parent
    content = (base / "fixtures" / f"{case_name}.txt").read_bytes()
    relative = f"AAA/2024/AAA_consolidated/{case_name}.txt"
    source = tmp_path / Path(relative)
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="abc123",
        relative_path=relative,
        company_code="AAA",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=len(content),
        encoding="utf-8",
        inventory_status="ready",
        notes=(),
    )
    expected = json.loads(
        (base / "expected" / f"{case_name}.json").read_text(encoding="utf-8")
    )

    first = extract_document(tmp_path, document)
    second = extract_document(tmp_path, document)

    assert semantic_projection(first) == expected
    assert second == first
```

- [ ] **Step 7: Run golden tests and fix only contract mismatches**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/golden/extraction/test_txt_extraction.py
```

Expected: all three exact hand-reviewed goldens pass and repeated results are equal. Do
not generate or update expected JSON from implementation output.

- [ ] **Step 8: Add a local corpus smoke command**

Create `scripts/smoke_ingestion.py` with:

```python
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.core.errors import FinancialReportQAError
from financial_report_qa.data.inventory import build_inventory
from financial_report_qa.ingestion import extract_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test ViFinQA TXT ingestion.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repeat-sample", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat_sample < 0:
        print("error: --repeat-sample must be non-negative", file=sys.stderr)
        return 2
    try:
        inventory = build_inventory(
            args.root,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        ready = tuple(
            item for item in inventory.documents if item.inventory_status == "ready"
        )
        table_count = 0
        cell_count = 0
        placement_count = 0
        rejection_count = 0
        html_free_count = 0
        html_free_with_tables = 0
        for index, document in enumerate(ready):
            result = extract_document(args.root, document)
            if index < args.repeat_sample:
                repeated = extract_document(args.root, document)
                if repeated != result:
                    raise ValueError(
                        f"non-deterministic extraction: {document.relative_path}"
                    )
            table_count += len(result.tables)
            cell_count += sum(len(item.cells) for item in result.tables)
            placement_count += sum(len(item.placements) for item in result.tables)
            rejection_count += len(result.rejected)
            html_free = all(block.kind != "table" for block in result.blocks)
            if html_free:
                html_free_count += 1
                if result.tables:
                    html_free_with_tables += 1
    except (FinancialReportQAError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Discovered:            {len(inventory.documents) + len(inventory.issues)}")
    print(f"Ready:                 {len(ready)}")
    print(f"Tables:                {table_count}")
    print(f"Cells:                 {cell_count}")
    print(f"Placements:            {placement_count}")
    print(f"Rejections:            {rejection_count}")
    print(f"HTML-free documents:   {html_free_count}")
    print(f"HTML-free with tables: {html_free_with_tables}")
    print(f"Repeated sample:       {min(len(ready), args.repeat_sample)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Document this command in `docs/development.md`:

```powershell
if ([string]::IsNullOrWhiteSpace($env:VIFINQA_REVISION) -or $env:VIFINQA_REVISION -eq "main") { throw "Set VIFINQA_REVISION to the immutable revision printed by the downloader" }
uv run --frozen --no-sync python scripts/smoke_ingestion.py --root data/raw/financial_statements --repo-id tinixai/ViFinQA --revision $env:VIFINQA_REVISION
```

State that the smoke command is local-data-only, does not write to `data/raw`, and is not
part of the hermetic unit-test gate.

- [ ] **Step 9: Run the complete Task 3 quality gate**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data
uv run --frozen --no-sync ruff check src/financial_report_qa/ingestion scripts/smoke_ingestion.py tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync mypy src/financial_report_qa/ingestion scripts/smoke_ingestion.py tests/unit/ingestion tests/golden/extraction
git diff --check
```

Expected: every test and static check passes; `git diff --check` produces no output.

- [ ] **Step 10: Run the local ViFinQA smoke check**

Set `VIFINQA_REVISION` to the full immutable revision printed by the downloader. Fail the
step if the value is empty or equals `main`, then run:

```powershell
if ([string]::IsNullOrWhiteSpace($env:VIFINQA_REVISION) -or $env:VIFINQA_REVISION -eq "main") { throw "VIFINQA_REVISION must be an immutable revision" }
uv run --frozen --no-sync python scripts/smoke_ingestion.py --root data/raw/financial_statements --repo-id tinixai/ViFinQA --revision $env:VIFINQA_REVISION
```

Expected: exit code 0; 1,973 discovered TXT paths for the current local snapshot; repeated
sample equality; no source mutation. Confirm the eight known HTML-free explanatory letters
produce zero extracted tables unless a future snapshot changes their bytes.

- [ ] **Step 11: Commit orchestration, fixtures, smoke validation, and docs**

```powershell
git add src/financial_report_qa/ingestion/__init__.py src/financial_report_qa/ingestion/table_extractor.py tests/unit/ingestion/test_table_extractor.py tests/golden/extraction scripts/smoke_ingestion.py docs/development.md
git commit -m "feat: complete deterministic ViFinQA TXT ingestion"
```

---

## Final Verification

After all five task commits, run from the repository root:

```powershell
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git status --short
```

Expected: the full test suite passes; Ruff and mypy report no issues. `git status --short`
may list only the user's pre-existing notebook, `plan.md`, and notebook-test changes; Task 3
implementation files must be committed.
