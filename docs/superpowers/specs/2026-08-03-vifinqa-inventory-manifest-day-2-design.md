# ViFinQA Inventory and Manifest Day 2 Design

**Date:** 2026-08-03

**Status:** Approved

## Goal

Build a deterministic, read-only inventory for the current ViFinQA financial-statement
snapshot and publish an atomic JSONL manifest without weakening the canonical
`DocumentRecord` contract created on Day 1.

## Scope

Day 2 supports only the current ViFinQA statement hierarchy:

```text
<ticker>/<year>/<document>/<file>.txt
```

The implementation inventories `.txt` files, derives ViFinQA path metadata, hashes file
content, checks supported text encodings, classifies empty and duplicate content, records
files that cannot become canonical documents, writes an atomic JSONL manifest, and exposes
the workflow through the product CLI.

Day 2 does not parse report contents, extract tables, normalize financial values, mutate
source files, support arbitrary dataset layouts, or introduce probabilistic encoding
detection.

## Architecture

The inventory has two output layers:

1. Canonical files become immutable `DocumentRecord` values from
   `financial_report_qa.schemas.documents`.
2. Files whose path, metadata, encoding, or I/O state prevents a valid `DocumentRecord`
   become immutable `InventoryIssue` values owned by the data module.

`InventoryResult` contains both collections. This avoids inventing fallback tickers or
years while ensuring every discovered `.txt` path is represented in the manifest.

## File Boundaries

- `src/financial_report_qa/data/inventory.py` owns discovery, ViFinQA path parsing,
  streaming SHA-256 calculation, deterministic encoding checks, duplicate classification,
  and `InventoryIssue`/`InventoryResult`.
- `src/financial_report_qa/data/manifests.py` owns canonical JSONL serialization and atomic
  replacement of the destination manifest.
- `src/financial_report_qa/cli.py` adds the `inventory-data` dispatcher command.
- `tests/unit/data/test_inventory.py` verifies inventory behavior using synthetic files.
- `tests/unit/data/test_manifests.py` verifies deterministic and atomic manifest writing.
- `tests/unit/test_cli.py` verifies dispatch and operator-facing failures.

The existing exploratory notebook remains unchanged and is not imported by production
code. Production path parsing reproduces its approved ViFinQA rules.

## Public Interfaces

```python
class InventoryIssue(BaseModel):
    relative_path: str
    reason: str
    file_size_bytes: int | None
    sha256: str | None


class InventoryResult(BaseModel):
    documents: tuple[DocumentRecord, ...]
    issues: tuple[InventoryIssue, ...]


def build_inventory(
    root: Path,
    *,
    repo_id: str,
    revision: str,
) -> InventoryResult: ...


def write_manifest(result: InventoryResult, path: Path) -> None: ...
```

Both inventory models use Pydantic 2 with `extra="forbid"` and `frozen=True`.
`build_inventory` raises `FileNotFoundError` when `root` does not exist or is not a
directory. Expected per-file defects are captured as `InventoryIssue`; an inability to
enumerate the root is a workflow-level failure.

## Discovery and Path Rules

Discovery recursively selects regular files whose suffix case-folds to `.txt`. Paths are
converted to POSIX form and sorted by `(relative_path.casefold(), relative_path)` before
processing. The source tree is opened only for reading.

A canonical ViFinQA path must contain exactly four components. The first component is a
2–10 character alphanumeric ticker, normalized to uppercase. The second is a four-digit
year in the inclusive range 1900–2100. Scope comes from the third component after
case-folding: `consolidated`, `separate`, and `aggregated` are recognized in that order;
all other document names map to `other`.

A path outside these rules is emitted as an `InventoryIssue` with its original safe POSIX
relative path. It is never forced into `DocumentRecord`.

## Hashing, Encoding, and Status Rules

Each regular file is streamed in fixed-size binary chunks. The same byte stream supplies
the exact byte count and lowercase SHA-256 digest. Files are not loaded fully into memory.

Encoding checks are deterministic:

- a UTF-8 BOM produces `encoding="utf-8-sig"`;
- otherwise, valid UTF-8 produces `encoding="utf-8"`;
- invalid UTF-8 produces an `InventoryIssue` and is quarantined from canonical documents.

No fallback single-byte codec is attempted because it could accept corrupted data without
evidence. An I/O failure becomes an `InventoryIssue` containing whichever size or digest
fields were available before the failure.

Canonical status precedence is:

1. zero bytes: `empty`;
2. first non-empty occurrence of a SHA-256 digest in sorted path order: `ready`;
3. later non-empty occurrences of that digest: `duplicate`.

Empty files are always `empty`, even though their digests match. Duplicate records include
the primary record's relative path in `notes`. `doc_id` remains content-addressed, so
duplicate records intentionally share the same `doc_id`.

## Manifest Contract

The manifest is UTF-8 JSONL with one object per discovered path in deterministic path
order. Each object contains `record_type` (`document` or `issue`) plus the complete model
payload. JSON uses stable key ordering, compact separators, Unicode characters without
ASCII escaping, and a final newline. Volatile values such as timestamps and absolute paths
are excluded, so identical snapshot bytes and parameters produce byte-identical output.

`write_manifest` creates a temporary file in the destination directory, flushes and closes
it, then replaces the destination with `Path.replace`. If serialization or writing fails,
the previous manifest remains intact and the temporary file is removed when possible.

## CLI Contract

The product command is:

```text
financial-report-qa inventory-data \
  --root data/raw/ViFinQA/financial_statements \
  --repo-id <dataset-repository> \
  --revision <immutable-revision> \
  --manifest data/manifests/documents.jsonl
```

All four values are explicit except the manifest, which defaults to
`data/manifests/documents.jsonl`. `repo-id` and `revision` must be non-empty; callers should
pass the resolved immutable revision recorded by the download workflow. Success prints
document, ready, empty, duplicate, and issue counts plus the manifest path. Expected
validation and filesystem failures return exit code 2 with a concise stderr message and no
traceback.

## Data Flow

```text
ViFinQA snapshot
  -> deterministic .txt discovery
  -> path validation + streaming hash + UTF-8 check
  -> DocumentRecord or InventoryIssue
  -> duplicate classification
  -> InventoryResult
  -> atomic deterministic JSONL manifest
```

The resulting `DocumentRecord` values are the only Day 2 records consumed by Day 3 TXT
ingestion. Issues remain auditable but cannot enter extraction.

## Test Strategy

Tests are written before implementation and prove:

1. valid Unicode ViFinQA paths produce correct ticker, year, scope, size, digest, encoding,
   and stable document ID;
2. malformed depth, ticker, year, and non-UTF-8 content produce explicit issues;
3. empty files remain `empty` and do not become duplicates;
4. duplicate ownership follows deterministic relative-path order and records the primary;
5. repeated inventory runs return equal immutable models without modifying source files;
6. mixed-case `.txt` suffixes are discovered while non-TXT files are ignored;
7. manifest bytes are deterministic, preserve Unicode, end with a newline, and deserialize
   into the original models;
8. a simulated write failure preserves the previous manifest;
9. CLI dispatch forwards arguments, prints counts on success, and reports expected failures
   without a traceback.

The Day 2 quality gate is:

```bash
uv run --frozen --no-sync pytest -q tests/unit/data tests/unit/test_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data src/financial_report_qa/cli.py tests/unit/data tests/unit/test_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/data src/financial_report_qa/cli.py tests/unit/data tests/unit/test_cli.py
```

## Completion Criteria

Day 2 is complete when the quality gate passes, a synthetic smoke snapshot produces a
complete manifest, a second run produces byte-identical manifest content, malformed files
remain visible as issues, and the change set contains no mutation of raw source data or
unrelated project files.
