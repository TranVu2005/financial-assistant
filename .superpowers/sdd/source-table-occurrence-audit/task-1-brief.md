# Task 1 brief

Read `docs/superpowers/plans/2026-08-04-source-table-occurrence-audit.md`, Task 1, first. It is the source of truth.

Implement Task 1 only. Add `SOURCE_TABLE_OCCURRENCE_SCHEMA` and `build_source_table_occurrences(document, detection, extraction, canonical_table_ids)` in `src/financial_report_qa/data/dataset_builder.py`; add focused tests in `tests/unit/data/test_dataset_builder.py`. The candidate key is `(ordinal, line_start, line_end)`. Audit only `kind == "html"`. Source ID is deterministic SHA-256 over path, digest, ordinal, and line span. An HTML candidate must become canonical or rejected; otherwise raise `ValueError`. Continuations may map to one table ID. Do not modify normalization, release writing, or documentation.

Run the focused test and report results. Commit only Task 1 source/test changes.
