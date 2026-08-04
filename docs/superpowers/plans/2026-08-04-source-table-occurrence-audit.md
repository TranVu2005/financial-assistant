
# Source Table Occurrence Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Account for all 146,246 raw HTML table occurrences while preserving the 146,011 canonical tables, and fix the two normalization false positives.

**Architecture:** Keep `tables.parquet` canonical. Add `source_table_occurrences.parquet`: one row per raw HTML table occurrence with status `canonical`, `rejected`, or `duplicate`. Continuation occurrences may map to a shared canonical table ID. Build this alongside the existing release and fingerprint it.

**Tech Stack:** Python, PyArrow/Parquet, pytest.

## Global Constraints

- Canonical `tables.parquet` remains 146,011 rows for the immutable snapshot.
- Occurrence reconciliation is total 146,246 = 146,012 canonical + 231 rejected + 3 duplicate.
- `source_table_id` is SHA-256 of relative path, source SHA-256, ordinal, line start, and line end.
- Duplicate rows never enter canonical tables, cells, or retrieval data.
- No dependencies added.

---

## File Structure

- `src/financial_report_qa/data/dataset_builder.py`: schema, ready/duplicate mapping, output, fingerprint, manifest counts.
- `src/financial_report_qa/normalization/service.py`: first-row value eligibility.
- `src/financial_report_qa/normalization/units.py`: unit-evidence gate for column headers.
- `tests/unit/data/test_dataset_builder.py`: occurrence tests.
- `tests/integration/test_pipeline_e2e.py`: artifact and reproducibility.
- `tests/unit/normalization/test_service.py`, `tests/unit/normalization/test_units.py`: regression tests.
- `data/raw/README.md`: correct stale table count.

### Task 1: Model source occurrences from ready documents

**Files:**
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Test: `tests/unit/data/test_dataset_builder.py`

**Interfaces:**
- Produces `SOURCE_TABLE_OCCURRENCE_SCHEMA`.
- Produces `build_source_table_occurrences(document, detection, extraction, canonical_table_ids) -> list[dict[str, object]]`.
- Key every candidate by `(ordinal, line_start, line_end)`.

- [ ] **Step 1: Write a failing accepted/rejected/continuation test**

```python
rows = build_source_table_occurrences(
    decoded_document, detection_with_two_html_candidates,
    extraction_with_one_merged_table_and_one_html_rejection,
    {(1, 10, 20): "table-1", (2, 21, 30): "table-1"},
)
assert [row["status"] for row in rows] == ["canonical", "canonical", "rejected"]
assert [row["canonical_table_id"] for row in rows] == ["table-1", "table-1", None]
assert rows[2]["rejection_code"] == "unsupported_html_structure"
assert len({row["source_table_id"] for row in rows}) == 3
```

- [ ] **Step 2: Verify red**

Run: `pytest tests/unit/data/test_dataset_builder.py::test_build_source_table_occurrences_tracks_rejection_and_continuation -v`

Expected: FAIL because the schema and function do not exist.

- [ ] **Step 3: Add schema and deterministic identity**

```python
SOURCE_TABLE_OCCURRENCE_SCHEMA = pa.schema([
    pa.field("source_table_id", pa.string(), nullable=False),
    pa.field("doc_id", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("ordinal", pa.int32(), nullable=False),
    pa.field("line_start", pa.int32(), nullable=False),
    pa.field("line_end", pa.int32(), nullable=False),
    pa.field("status", pa.string(), nullable=False),
    pa.field("canonical_table_id", pa.string()),
    pa.field("rejection_code", pa.string()),
    pa.field("duplicate_of_relative_path", pa.string()),
])
def _source_table_id(*, relative_path, source_sha256, ordinal, line_start, line_end):
    value = f"{relative_path}|{source_sha256}|{ordinal}|{line_start}|{line_end}"
    return sha256(value.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Implement mapping**

Filter to HTML candidates only. Build a rejected-key map from `extraction.rejected`. Emit rejected rows with null canonical ID and the reason. Emit canonical rows from `canonical_table_ids`. Raise `ValueError` if an HTML candidate has no outcome. Reject duplicate source IDs.

- [ ] **Step 5: Verify green and commit**

Run: `pytest tests/unit/data/test_dataset_builder.py::test_build_source_table_occurrences_tracks_rejection_and_continuation -v`

Expected: PASS.

Commit: `feat: model source table occurrences`.

### Task 2: Derive duplicate rows and emit the artifact

**Files:**
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Test: `tests/unit/data/test_dataset_builder.py`

**Interfaces:**
- Produces `build_duplicate_source_table_occurrences(duplicate_document, primary_rows, duplicate_of_relative_path)`.
- Extends `FlattenedDataset` with `source_table_occurrences`.

- [ ] **Step 1: Write a failing duplicate test**

```python
rows = build_duplicate_source_table_occurrences(duplicate_doc, primary_rows, "SSH/2024/primary.txt")
assert {row["status"] for row in rows} == {"duplicate"}
assert all(row["canonical_table_id"] is None for row in rows)
assert all(row["rejection_code"] is None for row in rows)
assert {row["duplicate_of_relative_path"] for row in rows} == {"SSH/2024/primary.txt"}
```

- [ ] **Step 2: Verify red**

Run: `pytest tests/unit/data/test_dataset_builder.py::test_duplicate_rows_reuse_layout_but_not_canonical_data -v`

Expected: FAIL because the duplicate mapper does not exist.

- [ ] **Step 3: Implement duplicate derivation**

Retain primary ordinal and line span, substitute duplicate provenance, recalculate ID, set status `duplicate`, set canonical ID/rejection to null, and store the primary relative path. In `build_dataset`, cache ready layouts by path, parse the existing `duplicate_of=` note, and reject malformed notes, missing primary layouts, or SHA mismatches.

- [ ] **Step 4: Wire artifact build**

Add occurrence rows to `FlattenedDataset`, fingerprint payload, temporary release output, and manifest:

```json
"source_table_occurrence_counts": {
  "total": 146246, "canonical": 146012, "rejected": 231, "duplicate": 3
}
```

Derive values from output rows. Validate status sum, canonical ID presence only for canonical rows, and global ID uniqueness.

- [ ] **Step 5: Verify green and commit**

Run: `pytest tests/unit/data/test_dataset_builder.py -v`

Expected: PASS.

Commit: `feat: emit source table occurrence audit`.

### Task 3: Lock the release contract and correct documentation

**Files:**
- Modify: `tests/integration/test_pipeline_e2e.py`
- Modify: `data/raw/README.md`

- [ ] **Step 1: Extend the E2E test**

```python
occurrences = pq.read_table(release_path / "source_table_occurrences.parquet")
assert occurrences.schema == SOURCE_TABLE_OCCURRENCE_SCHEMA
assert occurrences.num_rows == 2
assert set(occurrences.column("status").to_pylist()) == {"canonical"}
assert manifest["source_table_occurrence_counts"] == {
    "total": 2, "canonical": 2, "rejected": 0, "duplicate": 0,
}
```

Also include this file in byte-for-byte release reproducibility checks.

- [ ] **Step 2: Verify the E2E contract**

Run: `pytest tests/integration/test_pipeline_e2e.py -v`

Expected: PASS because Task 2 has already emitted the artifact.

- [ ] **Step 3: Update README**

Replace `143,815` with: `146,246 raw HTML table occurrences in 1,973 source files; canonical build emits 146,011 parseable tables and records all source occurrences in source_table_occurrences.parquet.`

- [ ] **Step 4: Commit**

Commit: `test: verify source occurrence release contract`.

### Task 4: Correct normalization false positives

**Files:**
- Modify: `src/financial_report_qa/normalization/service.py`
- Modify: `src/financial_report_qa/normalization/units.py`
- Test: `tests/unit/normalization/test_service.py`
- Test: `tests/unit/normalization/test_units.py`

- [ ] **Step 1: Write first-row regression**

```python
result = normalize_extraction(headerless_first_row_extraction)
assert len(result.tables) == 1
assert any(cell.row_idx == 0 and cell.value_numeric == Decimal("125") for cell in result.cells)
```

- [ ] **Step 2: Verify red, then change only the predicate**

Run: `pytest tests/unit/normalization/test_service.py::test_normalize_extraction_keeps_values_from_first_row_without_headers -v`

Expected: FAIL because `is_value_candidate` requires `cell.row_idx > 0`.

Change:
```python
return bool(row_label and cell.raw_text.strip() and cell.row_idx > 0)
```
to:
```python
return bool(row_label and cell.raw_text.strip())
```

- [ ] **Step 3: Write year-header regression**

```python
decision, issues = resolve_unit(table_raw=None, column_raw="2024", cell_raw=None)
assert decision.unit_normalized is None
assert issues == []
```

- [ ] **Step 4: Verify red, then gate column parsing**

Run: `pytest tests/unit/normalization/test_units.py::test_resolve_unit_ignores_year_column_labels_without_unit_markers -v`

Expected: FAIL because `normalize_unit("2024")` creates unknown-unit noise.

Add private `_has_unit_evidence(value)` using:
```python
re.compile(r"(?i)(vnd|vnÃ„â€˜|Ã„â€˜Ã¡Â»â€œng|dong|nghÃƒÂ¬n|ngÃƒÂ n|ngan|triÃ¡Â»â€¡u|trieu|tÃ¡Â»Â·|ty|%|phÃ¡ÂºÂ§n trÃ„Æ’m|lan|lÃ¡ÂºÂ§n)")
```
Call `normalize_unit(column_raw)` only when this predicate is true. Preserve table/cell unit parsing and explicit unknown-unit errors.

- [ ] **Step 5: Verify green and commit**

Run: `pytest tests/unit/normalization/test_service.py tests/unit/normalization/test_units.py -v`

Expected: PASS; table IDs and canonical table count are unchanged.

Commit: `fix: reduce normalization false positives`.

### Task 5: Build and reconcile the immutable snapshot

**Files:**
- Verify: newly generated release directory and `manifest.json`.

- [ ] **Step 1: Run targeted suite**

Run: `pytest tests/unit/data tests/unit/normalization tests/integration/test_pipeline_e2e.py -v`

Expected: PASS.

- [ ] **Step 2: Build a new release**

Run: `python -m financial_report_qa.data.dataset_builder --help`, then run the documented builder command in `data/raw/README.md` with a new output root; do not replace the existing release.

Expected: release includes `source_table_occurrences.parquet`.

- [ ] **Step 3: Reconcile emitted counts**

Use PyArrow to count statuses, distinct non-null canonical IDs, and `tables.parquet` rows.

Expected: total 146246; canonical 146012; rejected 231; duplicate 3; distinct canonical IDs 146011; tables 146011.

- [ ] **Step 4: Run final suite**

Run: `pytest -q`

Expected: all executable tests pass; record any Windows symlink skip as environment-only. Do not add generated releases unless repository convention already tracks them.

## Plan Self-Review

- Coverage: Tasks 1Ã¢â‚¬â€œ2 implement canonical, rejected, duplicate, and continuation rows; Task 3 verifies release/documentation; Task 4 fixes normalization only; Task 5 validates the complete reconciliation.
- Type consistency: `SOURCE_TABLE_OCCURRENCE_SCHEMA`, `build_source_table_occurrences`, `build_duplicate_source_table_occurrences`, `source_table_occurrences`, and `source_table_occurrence_counts` are used consistently.
- Placeholder scan: no deferred markers or unspecified test work remain.




