# Canonical Schemas Day 1 — Final Fix Report

## Scope and files

Implemented the complete final-review fix wave, limited to the approved schema
modules, schema unit tests, implementation plan, and this required report:

- `src/financial_report_qa/schemas/documents.py`
- `src/financial_report_qa/schemas/tables.py`
- `tests/unit/schemas/test_documents.py`
- `tests/unit/schemas/test_tables.py`
- `docs/superpowers/plans/2026-08-03-canonical-schemas-day-1.md`
- `.superpowers/sdd/2026-08-03-canonical-schemas-day-1/final-fix-report.md`

Changes:

- `TableRecord.csv_path` now accepts only a non-empty, outer-whitespace-free,
  POSIX relative path without backslashes or traversal components, while retaining
  explicit `None`.
- Stable-ID helpers now reject non-string digest/document-ID input with `ValueError`.
- Added a single parametrized frozen-mutation test for both `TableRecord` and
  `CellRecord`.
- Updated the approved plan's matching test and implementation snippets.

## RED

Command:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_documents.py::test_stable_document_id_rejects_non_string_input_with_value_error tests/unit/schemas/test_tables.py::test_stable_table_id_rejects_non_string_document_id_with_value_error tests/unit/schemas/test_tables.py::test_table_record_rejects_invalid_csv_paths tests/unit/schemas/test_tables.py::test_table_record_accepts_utf8_posix_csv_path_and_explicit_none tests/unit/schemas/test_tables.py::test_table_and_cell_records_reject_mutation
```

Output (exit 1):

```text
FFFFFFF...                                                               [100%]
7 failed, 3 passed in 0.34s
```

The expected failures were: `AttributeError` for `stable_document_id(None)`,
`TypeError` for `stable_table_id(None, 1, 1)`, and all five invalid `csv_path`
values being accepted. The valid POSIX path, explicit `None`, and existing frozen
behavior already passed.

## GREEN

Command:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_documents.py::test_stable_document_id_rejects_non_string_input_with_value_error tests/unit/schemas/test_tables.py::test_stable_table_id_rejects_non_string_document_id_with_value_error tests/unit/schemas/test_tables.py::test_table_record_rejects_invalid_csv_paths tests/unit/schemas/test_tables.py::test_table_record_accepts_utf8_posix_csv_path_and_explicit_none tests/unit/schemas/test_tables.py::test_table_and_cell_records_reject_mutation
```

Output (exit 0):

```text
..........                                                               [100%]
10 passed in 0.27s
```

## Final gates

```powershell
uv run --frozen --no-sync pytest -q tests/unit/schemas
```

```text
..............................................................           [100%]
62 passed in 0.47s
```

```powershell
uv run --frozen --no-sync ruff check src/financial_report_qa/schemas tests/unit/schemas
```

```text
All checks passed!
```

```powershell
uv run --frozen --no-sync mypy --strict src/financial_report_qa/schemas tests/unit/schemas
```

```text
Success: no issues found in 5 source files
```

```powershell
uv run --frozen --no-sync pytest -q
```

```text
........................................................................ [ 80%]
.................                                                        [100%]
89 passed in 1.11s
```

## Scope check and self-review

`git diff --check` exited 0 with no whitespace errors. `git status --short`
contained only the five implementation/plan files above before this report was
created; the report is the explicitly required sixth file.

Reviewed the final diff for all findings: the `csv_path` validator runs only for
non-`None` values and rejects each required unsafe form; runtime type guards run
before string operations and preserve the promised `ValueError`; and the one
parametrized mutation test exercises both frozen model types. No production changes
were needed for the frozen behavior.

## Concerns

None for the requested scope. Git commands emit a benign sandbox warning about an
unreadable global ignore file (`C:\Users\Admin\.config\git\ignore`); it did not
affect the diff, status, tests, linting, or type checking.
