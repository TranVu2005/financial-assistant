# Source Table Occurrence Audit Design

## Goal

Publish a deterministic source-table occurrence ledger that accounts for every one of the
146,246 inline HTML `<table>` tags in the ViFinQA snapshot, without treating malformed OCR
as parsed financial data.

## Scope

- Keep `tables.parquet` as the canonical, parseable-table release.
- Add `source-table-occurrences.parquet` with one row per raw HTML table occurrence.
- Record duplicate-path and rejected-HTML occurrences as auditable states.
- Preserve the existing canonical table and cell contracts.
- Make count reconciliation explicit in `manifest.json` and CLI output.

Out of scope:

- Inventing cells, dimensions, or normalized values for malformed HTML.
- Changing raw source files.
- Changing the semantic definition of a canonical parsed table.

## Data model

`source-table-occurrences.parquet` has one row per source occurrence and includes:

- `source_table_id`: stable SHA-256 ID derived from source path, source SHA-256, and line span;
- source document path, document SHA-256, and inclusive line span;
- `status`: `canonical`, `rejected`, or `duplicate`;
- `canonical_table_id`: populated for an occurrence represented by a canonical table;
- `rejection_code`: populated only for rejected occurrences;
- `duplicate_of_relative_path`: populated only for duplicate occurrences.

Two continuation source occurrences may point to the same `canonical_table_id`. This preserves
the source count while retaining the existing logical-table merge policy.

## Processing

1. Inventory still marks identical content paths as `duplicate`; canonical document processing
   still consumes only `ready` records.
2. For each ready document, detector output creates an occurrence for every HTML candidate or
   detector rejection. Extraction maps accepted candidates to canonical tables and maps failed
   candidates to `rejected` occurrences.
3. Each duplicate document reuses the deterministic occurrence layout of its declared primary
   document, binds it to the duplicate path and digest, and emits `duplicate` occurrences.
4. The builder writes canonical Parquet artifacts plus the occurrence ledger atomically.
5. The release manifest records `source_table_occurrence_count`, counts by status, and requires
   their sum to equal the source count.

For the current immutable snapshot, expected reconciliation is:

```text
source table occurrences: 146246
canonical tables:         146011
rejected occurrences:        231
duplicate occurrences:         3
canonical occurrences:    146012
```

The 146,012 canonical occurrences map to 146,011 canonical tables because one continuation
pair intentionally maps to a single canonical table.

## Error handling

- A duplicate record must declare an existing primary path with the same digest.
- Each source table occurrence must have exactly one status-consistent mapping.
- The builder fails closed if the source occurrence ledger cannot reconcile to raw HTML count.
- Rejected occurrences remain data, not build exceptions.

## Tests

- Unit: every accepted candidate creates one canonical occurrence.
- Unit: a rejected candidate creates one rejected occurrence with its reason.
- Unit: a duplicate document creates duplicate occurrences without a duplicate canonical table.
- Unit: continuation pair creates two occurrences mapped to one canonical table.
- Integration: release manifest reconciles source occurrences, canonical tables, rejections, and
  duplicates.
- Regression: normalization preserves canonical table count and IDs; headerless row-zero numeric
  cells are normalized; period labels do not emit `unit_unknown`.

## Acceptance criteria

- Current snapshot yields `source_table_occurrence_count == 146246`.
- Canonical `table_count` remains auditable and is no longer confused with source occurrence
  count.
- Every excluded source table has a deterministic reason or duplicate linkage.
