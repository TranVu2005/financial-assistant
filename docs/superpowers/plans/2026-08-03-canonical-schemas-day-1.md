# Canonical Schemas Day 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Day 1 canonical Pydantic contracts for financial documents, tables, and cells with deterministic document/table IDs and line-level provenance.

**Architecture:** Keep schemas as immutable, I/O-free value objects inside `financial_report_qa.schemas`. Document IDs are content-addressed from SHA-256; table IDs are derived from the canonical document ID and a one-based inclusive source-line span. Raw text remains unchanged while identifiers and canonical metadata are validated at module boundaries.

**Tech Stack:** Python 3.11, Pydantic 2, pytest, Ruff, mypy strict, uv.

## Global Constraints

- Python must remain `>=3.11,<3.12`; do not change dependencies or `uv.lock`.
- Run every quality command through `uv run --frozen --no-sync`.
- Create only `DocumentRecord`, `TableRecord`, `CellRecord`, `stable_document_id()`, and `stable_table_id()`.
- Do not add `ExtractionResult`, `NormalizedDocument`, filesystem I/O, TXT parsing, normalization, or Parquet writing.
- Every model uses Pydantic 2 with `extra="forbid"` and `frozen=True`.
- All source-line spans are one-based, inclusive, and require `start <= end`.
- Preserve raw strings and Vietnamese Unicode exactly; strip only identifiers and canonical metadata where whitespace has no meaning.
- Nullable fields named in the contract are required unless the spec explicitly gives a default.
- Work in an isolated worktree created with `superpowers:using-git-worktrees` before implementation.

---

## File Map

| File | Responsibility |
|---|---|
| `src/financial_report_qa/schemas/documents.py` | `DocumentRecord`, SHA-256 validation, relative-path validation, and `stable_document_id()` |
| `src/financial_report_qa/schemas/tables.py` | `TableRecord`, `CellRecord`, line-span validation, and `stable_table_id()` |
| `src/financial_report_qa/schemas/__init__.py` | The five approved public exports |
| `tests/unit/schemas/test_documents.py` | Document ID/model contract tests and package export test |
| `tests/unit/schemas/test_tables.py` | Table/cell ID, provenance, validation, and JSON round-trip tests |

---

### Task 1: Content-addressed document contract

**Files:**

- Create: `src/financial_report_qa/schemas/documents.py`
- Create: `tests/unit/schemas/test_documents.py`

**Interfaces:**

- Consumes: a precomputed SHA-256 digest and immutable inventory metadata.
- Produces: `stable_document_id(sha256: str) -> str` and `DocumentRecord`.

- [ ] **Step 1: Write failing tests for stable document IDs**

Create `tests/unit/schemas/test_documents.py`:

```python
"""Contract tests for canonical financial documents."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from financial_report_qa.schemas.documents import stable_document_id


SHA256 = "a" * 64


def valid_document_payload() -> dict[str, object]:
    return {
        "doc_id": stable_document_id(SHA256),
        "repo_id": "AIGuruTinix/ViFinQA",
        "revision": "main",
        "relative_path": (
            "financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/report.txt"
        ),
        "company_code": "AAA",
        "report_year": 2015,
        "statement_scope": "consolidated",
        "sha256": SHA256,
        "file_size_bytes": 1024,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "notes": (),
    }


def test_stable_document_id_is_content_addressed_and_case_normalized() -> None:
    uppercase_digest = "B" * 64

    first = stable_document_id(uppercase_digest)
    second = stable_document_id(uppercase_digest.lower())

    assert first == second == f"doc_{'b' * 64}"


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "a" * 63, "a" * 65])
def test_stable_document_id_rejects_non_sha256_values(digest: str) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        stable_document_id(digest)


def test_stable_document_id_rejects_non_string_input_with_value_error() -> None:
    with pytest.raises(ValueError, match="sha256 must be a string"):
        stable_document_id(cast(str, None))
```

- [ ] **Step 2: Run the ID tests and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q \
  tests/unit/schemas/test_documents.py::test_stable_document_id_is_content_addressed_and_case_normalized \
  tests/unit/schemas/test_documents.py::test_stable_document_id_rejects_non_sha256_values
```

Expected: collection fails with `ModuleNotFoundError: No module named 'financial_report_qa.schemas.documents'`.

- [ ] **Step 3: Implement the minimal stable document ID helper**

Create `src/financial_report_qa/schemas/documents.py`:

```python
"""Canonical contract for immutable financial-report documents."""

from __future__ import annotations

import re

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def stable_document_id(sha256: str) -> str:
    """Return the canonical content-addressed ID for a SHA-256 digest."""
    if not isinstance(sha256, str):
        raise ValueError("sha256 must be a string")
    normalized = sha256.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
    return f"doc_{normalized}"
```

- [ ] **Step 4: Run the ID tests and verify GREEN**

Run the Step 2 command again.

Expected: all six parametrized cases and the deterministic-ID test pass.

- [ ] **Step 5: Add failing tests for the document model contract**

Update the import and append the model tests to
`tests/unit/schemas/test_documents.py`:

```python
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
```

```python
def test_document_record_round_trip_preserves_vietnamese_unicode() -> None:
    payload = valid_document_payload()
    unicode_path = "financial_statements/AAA/2015/Báo cáo tài chính hợp nhất/report.txt"
    payload["relative_path"] = unicode_path

    record = DocumentRecord.model_validate(payload)
    restored = DocumentRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.relative_path == unicode_path


def test_document_record_requires_nullable_encoding_field() -> None:
    payload = valid_document_payload()
    payload.pop("encoding")

    with pytest.raises(ValidationError, match="encoding"):
        DocumentRecord.model_validate(payload)


def test_document_record_allows_explicit_unknown_encoding() -> None:
    payload = valid_document_payload()
    payload["encoding"] = None

    assert DocumentRecord.model_validate(payload).encoding is None


def test_document_record_rejects_mismatched_content_id() -> None:
    payload = valid_document_payload()
    payload["doc_id"] = stable_document_id("b" * 64)

    with pytest.raises(ValidationError, match="doc_id must match sha256"):
        DocumentRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "/absolute/report.txt"),
        ("relative_path", r"AAA\\2015\\report.txt"),
        ("relative_path", "../report.txt"),
        ("company_code", "a"),
        ("report_year", 1899),
        ("file_size_bytes", -1),
        ("inventory_status", "unknown"),
    ],
)
def test_document_record_rejects_invalid_inventory_metadata(
    field: str,
    value: object,
) -> None:
    payload = valid_document_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(payload)


def test_document_record_rejects_extra_fields() -> None:
    payload = valid_document_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        DocumentRecord.model_validate(payload)


def test_document_record_is_frozen() -> None:
    record = DocumentRecord.model_validate(valid_document_payload())

    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, "report_year", 2020)
```

- [ ] **Step 6: Run the document model tests and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_documents.py
```

Expected: collection fails because `DocumentRecord` is not defined.

- [ ] **Step 7: Implement the document model and cross-field validation**

Replace `src/financial_report_qa/schemas/documents.py` with:

```python
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
    if not isinstance(sha256, str):
        raise ValueError("sha256 must be a string")
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
```

- [ ] **Step 8: Run the document tests and verify GREEN**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_documents.py
```

Expected: all document tests pass.

- [ ] **Step 9: Commit the document contract**

```bash
git add src/financial_report_qa/schemas/documents.py tests/unit/schemas/test_documents.py
git commit -m "feat: define canonical document schema"
```

---

### Task 2: Provenance-preserving table contract

**Files:**

- Create: `src/financial_report_qa/schemas/tables.py`
- Create: `tests/unit/schemas/test_tables.py`

**Interfaces:**

- Consumes: a canonical `doc_id` and a one-based inclusive line span.
- Produces: `stable_table_id(doc_id: str, line_start: int, line_end: int) -> str` and `TableRecord`.

- [ ] **Step 1: Write failing tests for stable table IDs**

Create `tests/unit/schemas/test_tables.py`:

```python
"""Contract tests for canonical financial tables and cells."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from financial_report_qa.schemas.documents import stable_document_id
from financial_report_qa.schemas.tables import stable_table_id


DOC_ID = stable_document_id("a" * 64)


def valid_table_payload() -> dict[str, object]:
    return {
        "table_id": stable_table_id(DOC_ID, 10, 25),
        "doc_id": DOC_ID,
        "title_raw": "Bảng cân đối kế toán",
        "statement_type": "balance_sheet",
        "unit_raw": "Đơn vị: triệu đồng",
        "unit_normalized": "VND_million",
        "line_start": 10,
        "line_end": 25,
        "row_count": 12,
        "column_count": 4,
        "quality_score": 0.95,
        "csv_path": "tables/table-001.csv",
    }


def test_stable_table_id_matches_hand_checked_sha256() -> None:
    result = stable_table_id(DOC_ID, 10, 25)

    assert result == "tbl_32c57ec231bb937a8f18f8e625d660e1a38af5e9fd926b84cae1bcf797e9172c"


def test_stable_table_id_changes_with_document_or_span() -> None:
    base = "tbl_32c57ec231bb937a8f18f8e625d660e1a38af5e9fd926b84cae1bcf797e9172c"

    assert stable_table_id(DOC_ID, 11, 25) != base
    assert stable_table_id(stable_document_id("b" * 64), 10, 25) != base


@pytest.mark.parametrize(
    ("doc_id", "line_start", "line_end"),
    [
        ("invalid", 1, 2),
        (DOC_ID, 0, 2),
        (DOC_ID, 2, 1),
        (DOC_ID, True, 2),
    ],
)
def test_stable_table_id_rejects_invalid_identity_or_span(
    doc_id: str,
    line_start: int,
    line_end: int,
) -> None:
    with pytest.raises(ValueError):
        stable_table_id(doc_id, line_start, line_end)


def test_stable_table_id_rejects_non_string_document_id_with_value_error() -> None:
    with pytest.raises(ValueError, match="doc_id must be a string"):
        stable_table_id(cast(str, None), 1, 1)
```

- [ ] **Step 2: Run the table ID tests and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q \
  tests/unit/schemas/test_tables.py::test_stable_table_id_matches_hand_checked_sha256 \
  tests/unit/schemas/test_tables.py::test_stable_table_id_changes_with_document_or_span \
  tests/unit/schemas/test_tables.py::test_stable_table_id_rejects_invalid_identity_or_span
```

Expected: collection fails with `ModuleNotFoundError: No module named 'financial_report_qa.schemas.tables'`.

- [ ] **Step 3: Implement the stable table ID helper**

Create `src/financial_report_qa/schemas/tables.py`:

```python
"""Canonical contracts for extracted financial tables and cells."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

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
    if not isinstance(doc_id, str):
        raise ValueError("doc_id must be a string")
    if _DOC_ID_RE.fullmatch(doc_id) is None:
        raise ValueError("doc_id must be a canonical document ID")
    _validate_line_span(line_start, line_end)
    payload = f"{doc_id}\n{line_start}\n{line_end}".encode()
    return f"tbl_{hashlib.sha256(payload).hexdigest()}"
```

- [ ] **Step 4: Run the table ID tests and verify GREEN**

Run the Step 2 command again.

Expected: the deterministic and invalid-input tests pass.

- [ ] **Step 5: Add failing tests for the table record**

Update the import and append the table model tests to
`tests/unit/schemas/test_tables.py`:

```python
from financial_report_qa.schemas.tables import TableRecord, stable_table_id
```

```python
def test_table_record_round_trip_preserves_raw_vietnamese_text() -> None:
    payload = valid_table_payload()
    raw_title = "  Báo cáo kết quả hoạt động kinh doanh  "
    payload["title_raw"] = raw_title

    record = TableRecord.model_validate(payload)
    restored = TableRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.title_raw == raw_title


def test_table_record_requires_nullable_raw_fields() -> None:
    payload = valid_table_payload()
    payload.pop("unit_raw")

    with pytest.raises(ValidationError, match="unit_raw"):
        TableRecord.model_validate(payload)


def test_table_record_rejects_mismatched_stable_id() -> None:
    payload = valid_table_payload()
    payload["table_id"] = stable_table_id(DOC_ID, 11, 25)

    with pytest.raises(ValidationError, match="table_id must match"):
        TableRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("line_start", 0),
        ("line_start", 26),
        ("line_end", 9),
        ("row_count", -1),
        ("column_count", -1),
        ("quality_score", -0.01),
        ("quality_score", 1.01),
        ("quality_score", True),
        ("quality_score", "0.9"),
    ],
)
def test_table_record_rejects_invalid_shape_or_provenance(
    field: str,
    value: object,
) -> None:
    payload = valid_table_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        TableRecord.model_validate(payload)


def test_table_record_rejects_extra_fields() -> None:
    payload = valid_table_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        TableRecord.model_validate(payload)


@pytest.mark.parametrize(
    "csv_path",
    [
        "",
        "   ",
        "/generated/table.csv",
        "C:/generated/table.csv",
        "c:/generated/table.csv",
        "../escape.csv",
        r"tables\\table.csv",
    ],
)
def test_table_record_rejects_invalid_csv_paths(csv_path: str) -> None:
    payload = valid_table_payload()
    payload["csv_path"] = csv_path

    with pytest.raises(ValidationError, match="csv_path"):
        TableRecord.model_validate(payload)


def test_table_record_accepts_utf8_posix_csv_path_and_explicit_none() -> None:
    payload = valid_table_payload()
    csv_path = "tables/B\u1ea3ng c\u00e2n \u0111\u1ed1i.csv"
    payload["csv_path"] = csv_path

    assert TableRecord.model_validate(payload).csv_path == csv_path

    payload["csv_path"] = None

    assert TableRecord.model_validate(payload).csv_path is None
```

- [ ] **Step 6: Run the table record tests and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_tables.py
```

Expected: collection fails because `TableRecord` is not defined.

- [ ] **Step 7: Implement the table model**

Append to `src/financial_report_qa/schemas/tables.py`:

```python
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
    quality_score: float = Field(strict=True, ge=0, le=1)
    csv_path: str | None

    @field_validator("csv_path")
    @classmethod
    def validate_csv_path(cls, value: str | None) -> str | None:
        """Require generated artifact paths to be safe POSIX-relative paths."""
        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            not value
            or value != value.strip()
            or path.is_absolute()
            or PureWindowsPath(value).drive
            or "\\" in value
            or ".." in path.parts
            or not path.parts
        ):
            raise ValueError("csv_path must be a safe POSIX relative path")
        return value

    @model_validator(mode="after")
    def validate_identity_and_span(self) -> Self:
        """Require valid provenance and an ID derived from that provenance."""
        _validate_line_span(self.line_start, self.line_end)
        expected_id = stable_table_id(self.doc_id, self.line_start, self.line_end)
        if self.table_id != expected_id:
            raise ValueError("table_id must match doc_id and source-line span")
        return self
```

- [ ] **Step 8: Run the table tests and verify GREEN**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_tables.py
```

Expected: all table tests pass.

- [ ] **Step 9: Commit the table contract**

```bash
git add src/financial_report_qa/schemas/tables.py tests/unit/schemas/test_tables.py
git commit -m "feat: define canonical table schema"
```

---

### Task 3: Immutable cell contract with line provenance

**Files:**

- Modify: `src/financial_report_qa/schemas/tables.py`
- Modify: `tests/unit/schemas/test_tables.py`

**Interfaces:**

- Consumes: canonical `table_id`, zero-based row/column indexes, raw/canonical cell values, and source lines.
- Produces: `CellRecord`.

- [ ] **Step 1: Add failing cell round-trip and provenance tests**

Add the Decimal import and update the schema import at the top of
`tests/unit/schemas/test_tables.py`:

```python
from decimal import Decimal

from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id
```

Append:

```python
def valid_cell_payload() -> dict[str, object]:
    return {
        "cell_id": "cell-table-001-r2-c3",
        "table_id": stable_table_id(DOC_ID, 10, 25),
        "row_idx": 2,
        "col_idx": 3,
        "row_label_raw": "  Lợi nhuận sau thuế  ",
        "row_label_canonical": "profit_after_tax",
        "column_label_raw": "Năm 2022",
        "column_label_canonical": "2022",
        "value_raw": "  1.234,50  ",
        "value_numeric": Decimal("1234.50"),
        "period": "2022",
        "unit": "VND_million",
        "source_line_start": 18,
        "source_line_end": 19,
        "extraction_confidence": 0.9,
    }


def test_cell_record_round_trip_preserves_raw_text_and_decimal() -> None:
    record = CellRecord.model_validate(valid_cell_payload())
    restored = CellRecord.model_validate_json(record.model_dump_json())

    assert restored == record
    assert restored.row_label_raw == "  Lợi nhuận sau thuế  "
    assert restored.value_raw == "  1.234,50  "
    assert restored.value_numeric == Decimal("1234.50")


def test_cell_record_requires_nullable_canonical_fields() -> None:
    payload = valid_cell_payload()
    payload.pop("period")

    with pytest.raises(ValidationError, match="period"):
        CellRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_idx", -1),
        ("col_idx", -1),
        ("source_line_start", 0),
        ("source_line_start", 20),
        ("source_line_end", 17),
        ("extraction_confidence", -0.01),
        ("extraction_confidence", 1.01),
        ("extraction_confidence", True),
        ("extraction_confidence", "0.9"),
        ("table_id", "invalid"),
    ],
)
def test_cell_record_rejects_invalid_coordinates_or_provenance(
    field: str,
    value: object,
) -> None:
    payload = valid_cell_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        CellRecord.model_validate(payload)


def test_cell_record_rejects_extra_fields() -> None:
    payload = valid_cell_payload()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        CellRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("record", "field", "value"),
    [
        (TableRecord.model_validate(valid_table_payload()), "row_count", 99),
        (CellRecord.model_validate(valid_cell_payload()), "row_idx", 99),
    ],
    ids=("table", "cell"),
)
def test_table_and_cell_records_reject_mutation(
    record: TableRecord | CellRecord,
    field: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field, value)
```

- [ ] **Step 2: Run the cell tests and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_tables.py -k cell
```

Expected: collection fails because `CellRecord` is not defined.

- [ ] **Step 3: Implement the cell model**

Add `Decimal` to the imports in `src/financial_report_qa/schemas/tables.py`:

```python
from decimal import Decimal
```

Append:

```python
class CellRecord(BaseModel):
    """Immutable raw/canonical value and provenance for one table cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: NonEmptyString
    table_id: TableId
    row_idx: int = Field(strict=True, ge=0)
    col_idx: int = Field(strict=True, ge=0)
    row_label_raw: str | None
    row_label_canonical: NonEmptyString | None
    column_label_raw: str | None
    column_label_canonical: NonEmptyString | None
    value_raw: str
    value_numeric: Decimal | None
    period: NonEmptyString | None
    unit: NonEmptyString | None
    source_line_start: int = Field(strict=True, ge=1)
    source_line_end: int = Field(strict=True, ge=1)
    extraction_confidence: float = Field(strict=True, ge=0, le=1)

    @model_validator(mode="after")
    def validate_source_span(self) -> Self:
        """Reject reversed or zero-based line provenance."""
        _validate_line_span(self.source_line_start, self.source_line_end)
        return self
```

- [ ] **Step 4: Run the cell tests and verify GREEN**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_tables.py -k cell
```

Expected: all cell tests pass.

- [ ] **Step 5: Run all table/cell contract tests**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_tables.py
```

Expected: all table and cell tests pass together.

- [ ] **Step 6: Commit the cell contract**

```bash
git add src/financial_report_qa/schemas/tables.py tests/unit/schemas/test_tables.py
git commit -m "feat: define provenance-preserving cell schema"
```

---

### Task 4: Public schema exports and Day 1 quality gate

**Files:**

- Modify: `src/financial_report_qa/schemas/__init__.py`
- Modify: `tests/unit/schemas/test_documents.py`

**Interfaces:**

- Consumes: `DocumentRecord`, `TableRecord`, `CellRecord`, `stable_document_id()`, and `stable_table_id()` from Tasks 1–3.
- Produces: the stable package-level imports `financial_report_qa.schemas.<name>` and `__all__`.

- [ ] **Step 1: Add the failing package export test**

Append to `tests/unit/schemas/test_documents.py`:

```python
def test_schema_package_exports_only_approved_day_one_interfaces() -> None:
    from financial_report_qa import schemas

    expected = (
        "CellRecord",
        "DocumentRecord",
        "TableRecord",
        "stable_document_id",
        "stable_table_id",
    )

    assert schemas.__all__ == expected
    assert all(getattr(schemas, name) is not None for name in expected)
```

- [ ] **Step 2: Run the export test and verify RED**

Run:

```bash
uv run --frozen --no-sync pytest -q \
  tests/unit/schemas/test_documents.py::test_schema_package_exports_only_approved_day_one_interfaces
```

Expected: FAIL because `financial_report_qa.schemas` has no `__all__` or public record exports.

- [ ] **Step 3: Publish the approved interfaces**

Replace `src/financial_report_qa/schemas/__init__.py` with:

```python
"""Stable Pydantic contracts shared across product modules."""

from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.tables import CellRecord, TableRecord, stable_table_id

__all__ = (
    "CellRecord",
    "DocumentRecord",
    "TableRecord",
    "stable_document_id",
    "stable_table_id",
)
```

- [ ] **Step 4: Run the export test and verify GREEN**

Run the Step 2 command again.

Expected: PASS.

- [ ] **Step 5: Run the complete Day 1 schema test suite**

Run:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas
```

Expected: all schema contract tests pass with zero failures.

- [ ] **Step 6: Run Ruff on implementation and tests**

Run:

```bash
uv run --frozen --no-sync ruff check \
  src/financial_report_qa/schemas tests/unit/schemas
```

Expected: `All checks passed!`

- [ ] **Step 7: Run strict type checking**

Run:

```bash
uv run --frozen --no-sync mypy \
  src/financial_report_qa/schemas tests/unit/schemas
```

Expected: `Success: no issues found`.

- [ ] **Step 8: Run the full repository regression suite**

Run:

```bash
uv run --frozen --no-sync pytest -q
```

Expected: all repository tests pass. If an unrelated existing failure appears, capture its exact test name and output instead of weakening the schema tests.

- [ ] **Step 9: Verify scope and whitespace**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; implementation changes are limited to the schema modules, schema unit tests, and approved design/plan documentation. Do not stage unrelated notebook or dataset changes.

- [ ] **Step 10: Commit public exports and the completed plan state**

```bash
git add \
  src/financial_report_qa/schemas/__init__.py \
  tests/unit/schemas/test_documents.py
git commit -m "feat: publish canonical schema contracts"
```

## Day 1 Definition of Done

- [ ] `DocumentRecord`, `TableRecord`, and `CellRecord` reject missing, invalid, and extra fields according to the approved contract.
- [ ] Stable document/table IDs are deterministic and reproduce across repeated calls.
- [ ] Document IDs collapse identical content regardless of path.
- [ ] Table and cell provenance is one-based, inclusive, and validated.
- [ ] Vietnamese Unicode, raw whitespace, and `Decimal` values survive JSON round trips.
- [ ] The schema package exposes exactly the five approved interfaces.
- [ ] Schema pytest, Ruff, mypy strict, and the full repository pytest suite pass.
- [ ] No Task 2–4 behavior or unrelated working-tree changes are included.
