# Task 2 report

Date: 2026-08-04
Status: DONE
Task: Derive duplicate source-table occurrence rows and emit the source occurrence artifact.

## Scope completed

- Implemented `build_duplicate_source_table_occurrences(...)` in `src/financial_report_qa/data/dataset_builder.py`.
- Extended `FlattenedDataset` with `source_table_occurrences`.
- Wired ready-document occurrence generation into `build_dataset(...)` using Task 1 APIs.
- Derived duplicate occurrence rows from primary layouts via `duplicate_of=...` manifest notes.
- Added deterministic occurrence sorting, fingerprint inclusion, Parquet output, manifest counts, and dataset-level validation for:
  - status membership/sum
  - canonical ID presence only on `canonical`
  - duplicate metadata presence only on `duplicate`
  - global `source_table_id` uniqueness
- Rejected malformed duplicate notes, missing primary layouts, and SHA mismatches.

## Files changed

- `src/financial_report_qa/data/dataset_builder.py`
  - added duplicate occurrence mapper
  - added canonical occurrence mapping helper for ready docs
  - added occurrence validation/sorting helpers
  - emitted `source_table_occurrences.parquet`
  - included occurrence rows in fingerprint + manifest counts
- `tests/unit/data/test_dataset_builder.py`
  - added duplicate-row unit test
  - added dataset artifact/manifest duplicate-path unit test

## Tests run

Environment:

- Python: `D:\GitHub\financial-assistant\.venv\Scripts\python.exe`
- `PYTHONPATH=D:\GitHub\financial-assistant\.worktrees\source-table-occurrence-audit\src`

Executed:

1. RED check
   - `python -m pytest tests/unit/data/test_dataset_builder.py::test_duplicate_rows_reuse_layout_but_not_canonical_data -v`
   - Result: FAIL as expected before implementation (`ImportError: cannot import name 'build_duplicate_source_table_occurrences'`)

2. Focused green check
   - `python -m pytest tests/unit/data/test_dataset_builder.py::test_duplicate_rows_reuse_layout_but_not_canonical_data -v`
   - Result: PASS

3. Task 2 verification command
   - `python -m pytest tests/unit/data/test_dataset_builder.py -v`
   - Result: PASS
   - Summary: `7 passed`

## Commit

- Commit: `c2aeb5b7b4d4690e4c2f03a515361581b3215914`
- Message: `feat: emit source table occurrence audit`

## Concerns

- No known functional issues within Task 2 scope.
- Broader release-contract/E2E assertions remain intentionally deferred to Task 3.
- Full snapshot reconciliation remains intentionally deferred to Task 5.

## Notes

- Git staging/commit initially failed because the worktree was not marked as a Git `safe.directory`; this was resolved so the requested Task 2 commit could be created.
- `.superpowers` artifacts and plan workspace files were intentionally left uncommitted.
