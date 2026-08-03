# Canonical Schemas Day 1 Design

**Date:** 2026-08-03

**Status:** Approved

## Goal

Define the smallest stable Pydantic contract required by Day 1: canonical document,
table, and cell records with deterministic document/table identifiers and source-line
provenance. The contracts must preserve Vietnamese Unicode and remain independent of
inventory, ingestion, normalization, and retrieval implementations.

## Scope

Day 1 creates only:

- `DocumentRecord`
- `TableRecord`
- `CellRecord`
- `stable_document_id()`
- `stable_table_id()`

`ExtractionResult`, `NormalizedDocument`, file I/O, TXT parsing, normalization, and
Parquet writing are deferred to later tasks.

## File Boundaries

- `src/financial_report_qa/schemas/documents.py` owns the document contract and
  content-derived document ID.
- `src/financial_report_qa/schemas/tables.py` owns table/cell contracts, source-line
  validation, and table ID derivation.
- `src/financial_report_qa/schemas/__init__.py` exports the five approved public names.
- `tests/unit/schemas/test_documents.py` tests the document contract and ID behavior.
- `tests/unit/schemas/test_tables.py` tests table/cell contracts, provenance, IDs, and
  serialization.

No schema module performs filesystem or network I/O.

## Model Policy

All records use Pydantic 2 with `extra="forbid"`, `frozen=True`, and stripped string
fields where whitespace has no semantic meaning. JSON round trips use
`model_dump_json()` and `model_validate_json()`. Pydantic validation failures use
`ValidationError`; stable-ID helpers raise `ValueError` for malformed input.

## Document Contract

`DocumentRecord` contains:

| Field | Type and rule |
|---|---|
| `doc_id` | `doc_` plus a lowercase 64-character SHA-256 digest |
| `repo_id` | non-empty string |
| `revision` | non-empty immutable dataset revision |
| `relative_path` | non-empty POSIX-style relative path; Unicode is preserved |
| `company_code` | uppercase alphanumeric code, 2–10 characters |
| `report_year` | integer from 1900 through 2100 |
| `statement_scope` | `consolidated`, `separate`, `aggregated`, or `other` |
| `sha256` | lowercase 64-character SHA-256 digest |
| `file_size_bytes` | integer greater than or equal to zero |
| `encoding` | non-empty string or `None`; the field itself is required |
| `inventory_status` | `ready`, `empty`, `duplicate`, or `quarantine` |
| `notes` | immutable tuple of strings; defaults to an empty tuple |

`stable_document_id(sha256: str) -> str` validates and lowercases the digest, then
returns `doc_<digest>`. Identical content therefore receives the same `doc_id`
regardless of path.

## Table Contract

`TableRecord` contains:

| Field | Type and rule |
|---|---|
| `table_id` | `tbl_` plus a lowercase 64-character SHA-256 digest |
| `doc_id` | valid canonical document ID |
| `title_raw` | original title or `None` |
| `statement_type` | non-empty detected type or `None` |
| `unit_raw` | original unit text or `None` |
| `unit_normalized` | canonical unit or `None` |
| `line_start`, `line_end` | one-based inclusive lines; start must not exceed end |
| `row_count`, `column_count` | non-negative integers |
| `quality_score` | number in the closed interval `[0, 1]` |
| `csv_path` | relative generated artifact path or `None` |

`stable_table_id(doc_id: str, line_start: int, line_end: int) -> str` validates the
document ID and line span, hashes the unambiguous UTF-8 payload
`<doc_id>\n<line_start>\n<line_end>`, and returns `tbl_<digest>`. The same table
location in the same content therefore keeps its ID across runs.

## Cell Contract

`CellRecord` contains:

| Field | Type and rule |
|---|---|
| `cell_id` | non-empty stable identifier supplied by the later extraction task |
| `table_id` | valid canonical table ID |
| `row_idx`, `col_idx` | zero-based non-negative indexes |
| `row_label_raw`, `column_label_raw` | original labels or `None` |
| `row_label_canonical`, `column_label_canonical` | normalized labels or `None` |
| `value_raw` | required original cell text, including empty text when truly empty |
| `value_numeric` | `Decimal` or `None` |
| `period` | canonical period string or `None` |
| `unit` | canonical unit string or `None` |
| `source_line_start`, `source_line_end` | one-based inclusive source span |
| `extraction_confidence` | number in the closed interval `[0, 1]` |

Raw fields are never overwritten by canonical values. Cell ID derivation is deferred
because the extraction design has not fixed how merged and continuation cells are
addressed.

## Data Flow

Task 2 hashes each TXT file and constructs `DocumentRecord`. Task 3 uses the document
ID and source-line spans to construct `TableRecord` and `CellRecord`. Later tasks may
serialize records to JSON or flatten them to Parquet without importing business logic
into `schemas`.

## Test Strategy

Tests are written before implementation and prove:

1. document IDs are deterministic, case-normalized, and content-addressed;
2. table IDs are deterministic and change when the document or line span changes;
3. malformed hashes, IDs, years, indexes, confidence values, and source spans fail;
4. nullable required fields remain required rather than silently defaulting;
5. unexpected fields are rejected;
6. Vietnamese Unicode and decimals survive JSON round trips;
7. frozen records reject mutation;
8. the package exports exactly the approved Day 1 interfaces.

The Day 1 gate is:

```bash
uv run --frozen --no-sync pytest -q tests/unit/schemas
uv run --frozen --no-sync ruff check src/financial_report_qa/schemas tests/unit/schemas
uv run --frozen --no-sync mypy src/financial_report_qa/schemas tests/unit/schemas
```

## Completion Criteria

Day 1 is complete when all three quality commands pass, JSON round trips preserve
Unicode and decimals, stable IDs reproduce across repeated calls, and the change set
contains only the schema package, its unit tests, and approved documentation.
