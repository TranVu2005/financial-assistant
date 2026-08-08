# Complete Day 7 Week-1 Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work directly in the current checkout as requested; do not create a worktree and do not overwrite unrelated dirty changes.

**Goal:** Close Day 7 with a real, reproducible `dataset-pilot-v1` decision on the current ViFinQA corpus: 60 frozen pilot documents, expert-reviewed main-table annotations, 100% automated accepted-cell provenance, 30/30 manual cell provenance, at least 85% usable main tables, deterministic reports, and a content-addressed release lock.

**Architecture:** Keep the canonical release and raw TXT snapshot immutable. First align the Week-1 annotation/provenance implementation with its written contract, then prepare a deterministic 20-company × 3-document pilot, generate a review worksheet without treating extractor suggestions as ground truth, finalize human annotations, sample and review 30 cells, and evaluate into attempt-specific directories. Only a passing result is copied byte-for-byte to the canonical report path and bound to `dataset-pilot-v1`.

**Tech Stack:** Python 3.11, Pydantic 2, PyArrow/Parquet, Pandas, orjson, pytest, Ruff, mypy, PowerShell, ViFinQA TXT ingestion/normalization.

## Global Constraints

- Execute in `D:\GitHub\financial-assistant` on the current checkout; no Git worktree.
- Preserve all existing user/uncommitted changes. Stage or commit only files explicitly listed by the completed task.
- Immutable snapshot: `data/raw/financial_statements`. Never edit a TXT source file.
- Immutable manifest: `data/manifests/documents.jsonl` with SHA-256 `924d165211c63bbfc718b790f217ec356f80236e21fa0d8aa2acb497e186a5cf`.
- Initial release: `data/processed/release_v2_37a61be7aeba` with dataset fingerprint `37a61be7aebae1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Initial release facts: 1,971 documents; 146,011 canonical tables; 6,199,661 cells; 6,675,057 placements; 92,822 retained normalization issues.
- Annotation schema version remains `1` and sampling version remains `week1-pilot-v1`.
- Gold table annotations come from reviewing raw TXT. Extracted tables may suggest candidates, but may not silently define ground truth.
- Any manual cell marked `false` is evidence of a real defect. Never change it to `true` to make the gate pass.
- Do not start retrieval work until the canonical Day-7 `gate-result.json` has `"passed": true`.
- A code/data remediation publishes a new content-addressed release. Never mutate an existing `release_v2_*` directory.

## Pass Criteria

Day 7 is complete only when all checks are true:

| Check | Required value |
|---|---:|
| Pilot documents | exactly 60 |
| Company coverage | exactly 20 companies × 3 documents |
| Expected tables | at least 30 for each of balance sheet, income statement, cash flow |
| Overall usability | `usable * 100 >= annotated * 85` |
| Automated provenance | accepted cells > 0 and valid cells = accepted cells |
| Manual provenance | exactly 30 sampled cells and 30 `verified=true` |
| Eligible strata | every stratum with at least 10 annotations has usability ≥ 70% |
| Implementation quality | full pytest, Ruff, mypy and `git diff --check` pass |
| Reproducibility | repeated evaluation emits byte-identical JSON/Markdown/CSV |

## Critical Path and Estimate

| Phase | Tasks | Estimate |
|---|---|---:|
| Harden annotation/provenance contracts | 1–3 | 4–6 hours |
| Add the review worksheet workflow | 4 | 2–3 hours |
| Review 60 documents and finalize tables | 5 | 5–10 hours |
| Review 30 cells and run the first gate | 6–7 | 1–2 hours |
| Remediation after a failed real gate | 7 | 2–8 hours per iteration |
| Freeze evidence and close the milestone | 8 | 1 hour |

Expected best case is roughly 13–22 hours. Treat “Day 7” as the milestone name, not a promise
that the remaining expert review and any corpus-driven remediation fit into one calendar day.

## File Map

| Path | Responsibility |
|---|---|
| `src/financial_report_qa/evaluation/week1_contracts.py` | Canonical annotation types, stable IDs, period parsing, CSV/JSON contracts |
| `src/financial_report_qa/evaluation/week1_annotations.py` | Load/finalize annotations and reject invalid pilot coverage before sampling |
| `src/financial_report_qa/evaluation/week1_provenance.py` | Strict snapshot verification, re-extraction comparison, cell audit generation |
| `src/financial_report_qa/evaluation/week1_cli.py` | `prepare`, `prepare-review`, `finalize-tables`, `sample-cells` and `evaluate` workflows |
| `tests/unit/evaluation/test_week1_contracts.py` | Period, ID and CSV contract tests |
| `tests/unit/evaluation/test_week1_annotations.py` | Annotation bundle and review-finalization tests |
| `tests/unit/evaluation/test_week1_provenance.py` | Strict source/re-extraction provenance tests |
| `tests/integration/evaluation/test_week1_cli.py` | Complete CLI lifecycle |
| `data/qa/week1_pilot/` | Frozen pilot metadata and human-reviewed annotations |
| `data/interim/week1_gate_review/37a61be7aeba/` | Generated review worksheet; rebuildable and not committed |
| `data/interim/week1_gate_attempts/37a61be7aeba/` | Failed/passing attempt reports; rebuildable |
| `data/interim/week1_gate/37a61be7aeba/` | Canonical passing report only |
| `data/qa/week1_pilot/dataset-pilot-v1.json` | Small release lock binding alias, fingerprints and passing report |
| `README.md` and `plan.md` | Operator commands and final Day-7 status |

---

### Task 1: Align the expected-table contract before human labeling

**Files:**

- Modify: `src/financial_report_qa/evaluation/week1_contracts.py`
- Modify: `src/financial_report_qa/evaluation/week1_cli.py`
- Modify: `src/financial_report_qa/evaluation/week1_evaluator.py`
- Modify: `tests/unit/evaluation/test_week1_contracts.py`
- Modify: `tests/integration/evaluation/test_week1_cli.py`

**Interfaces:**

- Produces: `parse_expected_periods(raw: str) -> tuple[str, ...]`.
- Produces: `ExpectedTable` that accepts an empty period tuple, requires sorted unique periods, and requires its `annotation_id` to equal the result of `stable_annotation_id` for the same source identity.
- Consumers: both `sample-cells` and `evaluate` use `parse_expected_periods`; neither implements its own delimiter logic.

- [ ] **Step 1: Record the current green baseline**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation tests/integration/evaluation/test_week1_cli.py
```

Expected: 62 tests pass. Save the terminal result in the task notes; this is the before-state, not the real-corpus Day-7 pass.

- [ ] **Step 2: Write failing delimiter, empty-period and derived-ID tests**

Add these cases to `tests/unit/evaluation/test_week1_contracts.py`:

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ()),
        ("2023", ("2023",)),
        ("2023|2024", ("2023", "2024")),
    ],
)
def test_parse_expected_periods_uses_pipe_contract(
    raw: str, expected: tuple[str, ...]
) -> None:
    assert parse_expected_periods(raw) == expected


@pytest.mark.parametrize("raw", ["2024|2023", "2023|2023", "2023;2024", " 2023"])
def test_parse_expected_periods_rejects_noncanonical_values(raw: str) -> None:
    with pytest.raises(Week1GateInputError):
        parse_expected_periods(raw)


def test_expected_table_rejects_annotation_id_not_derived_from_source_identity() -> None:
    payload = valid_expected_table_payload()
    payload["annotation_id"] = "ann_" + "0" * 64
    with pytest.raises(ValidationError, match="annotation_id"):
        ExpectedTable.model_validate(payload)
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_contracts.py
```

Expected: FAIL because period parsing is duplicated in CLI/evaluator, empty periods are rejected, and `ExpectedTable` does not validate its derived ID.

- [ ] **Step 3: Implement the single canonical parser and ID validation**

Implement in `week1_contracts.py`:

```python
def parse_expected_periods(raw: str) -> tuple[str, ...]:
    if raw == "":
        return ()
    if raw != raw.strip() or ";" in raw:
        raise Week1GateInputError("expected_periods must use canonical pipe separators")
    periods = tuple(raw.split("|"))
    if any(not period for period in periods) or periods != tuple(sorted(set(periods))):
        raise Week1GateInputError("expected_periods must be sorted and duplicate-free")
    return periods
```

Change `ExpectedTable.validate_periods` to allow `()` while retaining sorted/unique validation. Add an `after` model validator that recomputes `stable_annotation_id(doc_id, line_start, line_end, statement_type)` and rejects any mismatch.

Replace both existing `split(";")` implementations with `parse_expected_periods(r["expected_periods"])`.

- [ ] **Step 4: Update fixtures to use stable annotation IDs**

Replace fixture IDs such as `exp_001` and `ann_000` with calls to `stable_annotation_id`. Do not weaken the production validator to preserve synthetic test shortcuts.

- [ ] **Step 5: Run focused verification**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_contracts.py tests/unit/evaluation/test_week1_evaluator.py tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_contracts.py src/financial_report_qa/evaluation/week1_cli.py src/financial_report_qa/evaluation/week1_evaluator.py tests/unit/evaluation/test_week1_contracts.py tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_contracts.py src/financial_report_qa/evaluation/week1_cli.py src/financial_report_qa/evaluation/week1_evaluator.py tests/unit/evaluation/test_week1_contracts.py tests/integration/evaluation/test_week1_cli.py
```

Expected: all commands exit 0.

---

### Task 2: Fail closed on invalid or under-covered annotation bundles

**Files:**

- Create: `src/financial_report_qa/evaluation/week1_annotations.py`
- Create: `tests/unit/evaluation/test_week1_annotations.py`
- Modify: `src/financial_report_qa/evaluation/week1_cli.py`
- Modify: `src/financial_report_qa/evaluation/week1_evaluator.py`
- Modify: `tests/integration/evaluation/test_week1_cli.py`

**Interfaces:**

- Produces: frozen `AnnotationBundle(metadata, pilot_documents, expected_tables)`.
- Produces: `load_annotation_bundle(dataset: GateDataset, annotation_dir: Path, *, require_expected_tables: bool) -> AnnotationBundle`.
- Consumers: `sample_cells_workflow` and `evaluate_week1_gate`.

- [ ] **Step 1: Write failing bundle-integrity tests**

Cover each condition independently:

```python
@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_pilot_doc",
        "wrong_company_count",
        "wrong_documents_per_company",
        "expected_doc_outside_pilot",
        "expected_path_mismatch",
        "overlapping_same_statement_annotations",
        "fewer_than_30_balance_sheets",
        "fewer_than_30_income_statements",
        "fewer_than_30_cash_flows",
    ],
)
def test_load_annotation_bundle_rejects_invalid_gate_input(
    tmp_path: Path, mutation: str
) -> None:
    dataset, annotation_dir = complete_annotation_fixture(tmp_path)
    mutate_annotation_fixture(annotation_dir, mutation)
    with pytest.raises(Week1GateInputError):
        load_annotation_bundle(dataset, annotation_dir, require_expected_tables=True)
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_annotations.py
```

Expected: FAIL because `week1_annotations.py` does not exist.

- [ ] **Step 2: Implement one shared loader**

The loader must:

1. validate `pilot-metadata.json` against release and manifest fingerprints;
2. verify `pilot-documents.csv` SHA-256 against metadata;
3. require exactly 60 unique documents, 20 unique companies and exactly 3 documents per company;
4. require every pilot row to equal the released `DocumentRecord`;
5. parse expected periods through `parse_expected_periods`;
6. require every expected table to reference a selected document with exact `relative_path`;
7. reject duplicate annotation IDs and overlapping annotations of the same statement family in one document;
8. when `require_expected_tables=True`, require at least 30 annotations for each of the three statement types.

Keep this validation separate from table matching: malformed annotation input exits 2; a valid annotation for a missing/broken extracted table remains quality data and exits 1 after evaluation.

- [ ] **Step 3: Route sampling and evaluation through the shared loader**

Remove duplicated metadata/CSV parsing from `week1_cli.py` and `week1_evaluator.py`. Preserve public CLI arguments and exit codes.

- [ ] **Step 4: Verify invalid coverage stops before creating `cell-audit.csv`**

Add an integration assertion:

```python
assert week1_main(sample_cells_args) == 2
assert not (annotation_dir / "cell-audit.csv").exists()
```

- [ ] **Step 5: Run the focused gate**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_annotations.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_evaluator.py tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation tests/unit/evaluation tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation tests/unit/evaluation tests/integration/evaluation/test_week1_cli.py
```

Expected: all commands exit 0.

---

### Task 3: Make accepted-cell provenance match the immutable-ingestion contract

**Files:**

- Modify: `src/financial_report_qa/evaluation/week1_provenance.py`
- Modify: `src/financial_report_qa/core/errors.py` only if an existing typed error cannot express the failure
- Modify: `tests/unit/evaluation/test_week1_provenance.py`
- Modify: `tests/integration/evaluation/test_week1_cli.py`

**Interfaces:**

- Consumes: `GateDataset`, `data/raw/financial_statements`, expected tables and deterministic matches.
- Produces: `CellAudit.verified=True` only when source bytes are inventory-valid and the released canonical cell equals a fresh extraction+normalization result.
- Preserves the existing `generate_cell_audits` arguments and `tuple[CellAudit, ...]` return type.

- [ ] **Step 1: Write failing provenance regression tests**

Add a frozen `_ProvenanceCase` test helper containing `dataset`, `corpus_dir`,
`expected_tables`, `matched_tables` and `document`. Then add these concrete tests:

```python
def test_generate_cell_audits_fails_on_source_hash_mismatch(tmp_path: Path) -> None:
    case = _write_provenance_case(tmp_path)
    source_path = case.corpus_dir / case.document.relative_path
    source_path.write_bytes(source_path.read_bytes() + b"tampered")
    with pytest.raises(Week1GateSourceError):
        generate_cell_audits(
            case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
        )


def test_generate_cell_audits_fails_on_invalid_utf8_instead_of_replacing(
    tmp_path: Path,
) -> None:
    case = _write_provenance_case(tmp_path, source_bytes=b"\xff", encoding="utf-8")
    with pytest.raises(Week1GateSourceError):
        generate_cell_audits(
            case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
        )


def test_generate_cell_audits_marks_canonical_cell_drift_invalid(tmp_path: Path) -> None:
    case = _write_provenance_case(
        tmp_path, source_value_raw="100", released_value_raw="999"
    )
    audits = generate_cell_audits(
        case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
    )
    assert audits
    assert all(audit.verified is False for audit in audits)


def test_generate_cell_audits_accepts_exact_reextraction(tmp_path: Path) -> None:
    case = _write_provenance_case(tmp_path)
    audits = generate_cell_audits(
        case.dataset, case.corpus_dir, case.expected_tables, case.matched_tables
    )
    assert audits
    assert all(audit.verified is True for audit in audits)
```

The hash/encoding failures must raise `Week1GateSourceError`. Cell drift must emit `invalid_provenance` so it appears in the Pareto and fails both the table and 100% accepted-cell checks.
Add a fifth test with two matched tables from the same document and a monkeypatched
`read_document`; assert its call count is exactly one.

- [ ] **Step 2: Replace permissive file reading**

Remove:

```python
doc_path.read_text(encoding="utf-8", errors="replace")
```

Use `read_document(corpus_dir, document)` so safe-path, file-size, SHA-256, UTF-8/UTF-8-SIG and line-ending contracts are enforced.

- [ ] **Step 3: Re-extract and normalize once per matched document**

For each matched document, compute:

```python
decoded = read_document(corpus_dir, document)
detection = detect_table_candidates(decoded)
extraction = extract_candidates(decoded, detection)
normalized = normalize_extraction(document, extraction)
fresh_cells = {
    cell.cell_id: cell
    for table in normalized.extraction.tables
    for cell in table.cells
}
```

Cache the result by `doc_id`. Compare every accepted release cell with `fresh_cells[cell_id]` using the complete Pydantic model, including raw labels/value, numeric value, period, unit and source spans.

- [ ] **Step 4: Preserve bounded source excerpts**

Build excerpts from `DecodedDocument.lines` using one-based inclusive spans. Keep a 500-code-point cap only in the manual sample; automated equality must use the full source/cell model.

- [ ] **Step 5: Verify provenance and ingestion regressions**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_provenance.py tests/unit/ingestion tests/unit/normalization tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_provenance.py tests/unit/evaluation/test_week1_provenance.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_provenance.py tests/unit/evaluation/test_week1_provenance.py
```

Expected: all commands exit 0.

---

### Task 4: Add a review worksheet without turning model output into gold truth

**Files:**

- Modify: `src/financial_report_qa/evaluation/week1_annotations.py`
- Modify: `src/financial_report_qa/evaluation/week1_cli.py`
- Modify: `tests/unit/evaluation/test_week1_annotations.py`
- Modify: `tests/integration/evaluation/test_week1_cli.py`
- Modify: `README.md`

**Interfaces:**

- Produces: `prepare-review` command writing `table-review.csv` under the requested interim path.
- Produces: `finalize-tables` command atomically replacing only the header-only `expected-tables.csv` template.
- Review columns: `include,doc_id,relative_path,company_code,report_year,table_id,line_start,line_end,title_raw,source_excerpt,statement_type,row_count,column_count,unit_normalized,expected_periods,notes`.

- [ ] **Step 1: Write failing review/finalization tests**

Tests must prove:

- suggestions include only pilot documents and remain stably sorted;
- suggestions are advisory and retain raw line excerpts;
- reviewers may append a missing-table row with an empty `table_id`;
- only lowercase `true` rows become expected tables;
- `annotation_id` is computed, never typed by the reviewer;
- finalization refuses unknown docs, invalid spans, invalid statement types/units/periods and a non-empty canonical output;
- the output has exact `EXPECTED_TABLE_COLUMNS` and LF-terminated UTF-8.

- [ ] **Step 2: Generate bounded suggestions**

Prepopulate rows from released tables whose normalized statement type is one of:

```python
MAIN_STATEMENTS = (
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
)
```

Include the raw title and a bounded source excerpt for navigation. Set `include` empty; do not pre-approve any row.

- [ ] **Step 3: Finalize reviewed rows**

For each row with `include=true`:

1. discard advisory `table_id`, `title_raw` and `source_excerpt`;
2. parse/validate the manually confirmed fields;
3. compute `stable_annotation_id`;
4. sort by `(doc_id, line_start, line_end, statement_type)`;
5. validate through `load_annotation_bundle`;
6. atomically replace only the original header-only `expected-tables.csv`.

- [ ] **Step 4: Add CLI integration coverage**

The full synthetic lifecycle becomes:

```text
prepare -> prepare-review -> human-review fixture -> finalize-tables
        -> sample-cells -> human-cell fixture -> evaluate
```

- [ ] **Step 5: Run focused verification**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_annotations.py tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_annotations.py src/financial_report_qa/evaluation/week1_cli.py tests/unit/evaluation/test_week1_annotations.py tests/integration/evaluation/test_week1_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_annotations.py src/financial_report_qa/evaluation/week1_cli.py tests/unit/evaluation/test_week1_annotations.py tests/integration/evaluation/test_week1_cli.py
```

Expected: all commands exit 0.

---

### Task 5: Prepare and complete the real 60-document table review

**Files:**

- Generate: `data/qa/week1_pilot/pilot-metadata.json`
- Generate: `data/qa/week1_pilot/pilot-documents.csv`
- Generate then finalize: `data/qa/week1_pilot/expected-tables.csv`
- Generate locally: `data/interim/week1_gate_review/37a61be7aeba/table-review.csv`

**Interfaces:**

- Consumes: the fixed manifest, release and raw corpus from Global Constraints.
- Produces: at least 90 independently reviewed main-table annotations, with at least 30 of each statement type.

- [ ] **Step 1: Set exact PowerShell paths and verify identity**

```powershell
$manifestPath = "data/manifests/documents.jsonl"
$releasePath = "data/processed/release_v2_37a61be7aeba"
$corpusPath = "data/raw/financial_statements"
$annotationPath = "data/qa/week1_pilot"
$reviewPath = "data/interim/week1_gate_review/37a61be7aeba/table-review.csv"

(Get-FileHash -Algorithm SHA256 $manifestPath).Hash.ToLowerInvariant()
Get-Content -Raw "$releasePath/manifest.json"
```

Expected: manifest hash `924d165211c63bbfc718b790f217ec356f80236e21fa0d8aa2acb497e186a5cf` and dataset fingerprint `37a61be7aebae1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.

- [ ] **Step 2: Prepare the deterministic pilot**

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate prepare `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --annotation-root $annotationPath
```

Expected: exit 0; `pilot-documents.csv` has 60 data rows and `pilot-metadata.json` reports 20 companies.

- [ ] **Step 3: Generate the advisory review worksheet**

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate prepare-review `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --corpus-dir $corpusPath `
  --annotation-dir $annotationPath `
  --output-path $reviewPath
```

Expected: exit 0 and a UTF-8 CSV sorted by document/source line.

- [ ] **Step 4: Review every selected raw TXT document**

For each row in `pilot-documents.csv`:

1. join `data/raw/financial_statements` with the row's `relative_path` and open that TXT file;
2. locate the main balance sheet, income statement and cash-flow statement;
3. confirm one-based inclusive `line_start`/`line_end` against the raw file;
4. confirm logical dimensions after rowspan/colspan expansion and continuation merge;
5. set `unit_normalized` to empty, `VND`, `VND_thousand`, `VND_million`, `VND_billion`, `percent` or `ratio` only when supported by source;
6. write periods as empty or sorted pipe-separated canonical values, for example `2023|2024`;
7. mark a suggested row `include=true` only after confirmation;
8. append a new row with empty `table_id` if a true main table is absent from suggestions;
9. use `notes` only for source anomalies, never to override scoring.

Do not merely accept all suggestions. The raw document is the source of truth.

- [ ] **Step 5: Finalize canonical expected tables**

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate finalize-tables `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --annotation-dir $annotationPath `
  --review-path $reviewPath
```

Expected: exit 0; summary reports at least 30 rows for each main statement type and no invalid annotation.

- [ ] **Step 6: Review the final CSV before sampling**

Check:

```powershell
$rows = Import-Csv "$annotationPath/expected-tables.csv"
$rows.Count
$rows | Group-Object statement_type | Select-Object Name,Count
$rows | Group-Object annotation_id | Where-Object Count -gt 1
```

Expected: total ≥ 90; each statement count ≥ 30; no duplicate annotation IDs.

---

### Task 6: Generate and manually verify the 30-cell audit

**Files:**

- Generate and then modify only review fields: `data/qa/week1_pilot/cell-audit.csv`

**Interfaces:**

- Immutable columns: every column except `verified` and `review_notes`.
- Reviewer output: exactly 30 lowercase `true`/`false` values.

- [ ] **Step 1: Generate the deterministic sample**

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate sample-cells `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --corpus-dir $corpusPath `
  --annotation-dir $annotationPath
```

Expected: exit 0; `cell-audit.csv` has exactly 30 rows, unique `cell_id` values and blank `verified`.

- [ ] **Step 2: Review each cell against the raw source**

For every row:

1. open `relative_path` at `source_line_start..source_line_end`;
2. verify `value_raw` occurs in the stated source span;
3. verify row/column labels identify the same logical cell;
4. verify the cell belongs to the stated main table and not an adjacent table;
5. set `verified=true` only when provenance is correct;
6. otherwise set `verified=false` and record a concrete reason such as `wrong_span`, `wrong_table`, `raw_value_mismatch` or `header_mapping_error`.

Do not edit `cell_id`, source fields, labels, numeric value, period or unit.

- [ ] **Step 3: Validate manual completion**

```powershell
$audits = Import-Csv "$annotationPath/cell-audit.csv"
$audits.Count
$audits | Group-Object verified | Select-Object Name,Count
$audits | Group-Object cell_id | Where-Object Count -gt 1
```

Expected for a passing release: 30 rows, one `true` group of count 30, and no duplicate cell IDs. If any row is `false`, continue to Task 7 without changing the label.

---

### Task 7: Evaluate, triage by Pareto, and remediate without contaminating gold labels

**Files:**

- Generate: `data/interim/week1_gate_attempts/37a61be7aeba/attempt-01/`
- Modify code/tests only for source-proven defects.
- If code changes affect the dataset: generate a new immutable `data/processed/release_v2_*` and a new fingerprint-bound annotation directory.

**Interfaces:**

- Exit 0: valid input and every gate check passes.
- Exit 1: valid annotations but quality thresholds fail.
- Exit 2: invalid workflow, annotation, release identity, source integrity or I/O.

- [ ] **Step 1: Run the first real evaluation**

```powershell
$attemptPath = "data/interim/week1_gate_attempts/37a61be7aeba/attempt-01"
uv run --frozen --no-sync financial-report-qa week1-gate evaluate `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --corpus-dir $corpusPath `
  --annotation-dir $annotationPath `
  --output-dir $attemptPath
$LASTEXITCODE
```

- [ ] **Step 2: Inspect all gate checks and the error Pareto**

```powershell
$gate = Get-Content -Raw "$attemptPath/gate-result.json" | ConvertFrom-Json
$gate.checks | Format-Table name,passed,numerator,denominator,threshold_percent
Import-Csv "$attemptPath/pareto-errors.csv" | Select-Object -First 10
```

Never diagnose from aggregate `issue_count` alone; use matched-table failures and raw examples.

- [ ] **Step 3: Classify each failure before changing anything**

| Failure class | Action |
|---|---|
| Wrong human span/shape/type/unit/period | Re-check raw TXT and correct the annotation review row; regenerate canonical annotations and cell sample |
| `missing_table` with a valid gold table | Add a failing detector/extractor golden test, fix extraction, rebuild release |
| `invalid_provenance` or manual `false` | Add a failing reader/extractor/provenance regression, fix code, rebuild release |
| `no_numeric_value` on OCR-corrupt numeric data | Preserve issue or improve parser only with source-backed tests; never invent a number |
| Statement/period/unit mismatch caused by normalization | Add a labeled regression, fix conservative rule, rebuild release |
| Fewer than 30 per statement | Complete missing raw-source annotations; do not lower the threshold |

- [ ] **Step 4: Apply every code fix with a failing test first**

Run the failing test before implementation and record the failure reason. After the minimal fix, run the focused module tests and the full pre-retrieval gate:

```powershell
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

- [ ] **Step 5: Rebuild instead of mutating the old release**

Only when production ingestion/normalization code changed:

```powershell
uv run --frozen --no-sync python -m financial_report_qa.cli.build_dataset `
  --snapshot-root data/raw/financial_statements `
  --manifest-path data/manifests/documents.jsonl `
  --processed-root data/processed `
  --schema-version 2
```

Record the new release path/fingerprint, then derive the next annotation path:

```powershell
$newFingerprint = (Get-Content -Raw "$newReleasePath/manifest.json" | ConvertFrom-Json).dataset_fingerprint
$newPrefix = $newFingerprint.Substring(0, 12)
$newAnnotationPath = "data/qa/week1_pilot_$newPrefix"
```

Run `prepare` against `$newAnnotationPath`. Migrate human table annotations only if the 60 selected `doc_id` values and raw spans are identical; regenerate stable IDs and `cell-audit.csv`. Never edit metadata to pretend an old annotation belongs to a new fingerprint.

- [ ] **Step 6: Repeat with attempt-specific output directories**

Use `attempt-02`, `attempt-03` and so on. Do not overwrite evidence from an earlier failed run. Stop changing code when the gate passes; do not optimize issue counts unrelated to failed checks.

---

### Task 8: Publish the canonical pass and close Day 7

**Files:**

- Generate: `data/interim/week1_gate/37a61be7aeba/gate-result.json`
- Generate: `data/interim/week1_gate/37a61be7aeba/gate-report.md`
- Generate: `data/interim/week1_gate/37a61be7aeba/pareto-errors.csv`
- Create: `data/qa/week1_pilot/dataset-pilot-v1.json`
- Create: `src/financial_report_qa/evaluation/week1_release.py`
- Create: `tests/unit/evaluation/test_week1_release.py`
- Modify: `README.md`
- Modify: `plan.md`

**Interfaces:**

- Produces: `publish_release_lock(release_path: Path, gate_result_path: Path, output_path: Path) -> ReleaseLock`.
- Produces: one immutable release-lock JSON with this typed contract:

```python
class ReleaseLock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alias: Literal["dataset-pilot-v1"]
    sampling_version: Literal["week1-pilot-v1"]
    dataset_fingerprint: str
    source_manifest_sha256: str
    release_path: str
    gate_result_path: str
    evaluation_inputs_sha256: str
```

If remediation changed the release, use the actual passing fingerprint/path in every field.

- [ ] **Step 1: Publish into a fresh canonical directory**

```powershell
$canonicalReportPath = "data/interim/week1_gate/37a61be7aeba"
uv run --frozen --no-sync financial-report-qa week1-gate evaluate `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --corpus-dir $corpusPath `
  --annotation-dir $annotationPath `
  --output-dir $canonicalReportPath
if ($LASTEXITCODE -ne 0) { throw "Day 7 gate did not pass" }
```

- [ ] **Step 2: Prove deterministic report bytes**

Run the same evaluation into `data/interim/week1_gate_replay/37a61be7aeba`, then compare hashes:

```powershell
$replayPath = "data/interim/week1_gate_replay/37a61be7aeba"
uv run --frozen --no-sync financial-report-qa week1-gate evaluate `
  --manifest-path $manifestPath `
  --release-path $releasePath `
  --corpus-dir $corpusPath `
  --annotation-dir $annotationPath `
  --output-dir $replayPath

Get-ChildItem $canonicalReportPath -File | Sort-Object Name | Get-FileHash -Algorithm SHA256
Get-ChildItem $replayPath -File | Sort-Object Name | Get-FileHash -Algorithm SHA256
```

Expected: corresponding files have identical SHA-256 values.

- [ ] **Step 3: Create and validate the release lock**

Write failing tests proving the publisher rejects a failed gate, fingerprint mismatch, unsafe
path and an existing non-identical lock. Implement the publisher so it reads
`evaluation_inputs_sha256` from the passing result rather than accepting a user-entered hash.
Expose it as `week1-gate lock-release`, then run:

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate lock-release `
  --release-path $releasePath `
  --gate-result-path "$canonicalReportPath/gate-result.json" `
  --output-path "$annotationPath/dataset-pilot-v1.json"
```

Expected: exit 0; the lock resolves only safe relative paths and agrees with both release and
gate fingerprints.

- [ ] **Step 4: Run the final quality gate**

```powershell
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Update project status**

In `plan.md`, mark the four Day-7 bullets complete only after the canonical report exists and passes. In `README.md`, record the exact prepare/review/finalize/sample/evaluate commands and explain that retrieval work consumes the release lock, not an arbitrary `data/processed` directory.

- [ ] **Step 6: Commit only the approved Day-7 scope**

```powershell
git add src/financial_report_qa/evaluation src/financial_report_qa/core/errors.py tests/unit/evaluation tests/integration/evaluation/test_week1_cli.py README.md plan.md docs/superpowers/plans/2026-08-07-complete-day-7-quality-gate.md data/qa/week1_pilot
git status --short
git commit -m "feat: close week one dataset quality gate"
```

Before committing, remove unrelated paths from the index. Do not stage `data/raw`, `data/processed`, `data/interim` or unrelated normalization artifacts.

## Final Evidence to Report

The completion handoff must include:

- passing release path and full dataset fingerprint;
- source manifest SHA-256;
- exact six gate checks with numerators/denominators;
- top ten Pareto rows, even if the final list is empty;
- 30/30 manual audit result;
- canonical report paths and SHA-256 comparison against replay;
- pytest/Ruff/mypy/`git diff --check` outputs;
- any remaining known normalization issues, clearly separated from Day-7 gate failures.

## Self-Review

- Spec coverage: release identity, deterministic 60-document selection, annotation integrity, ≥30 per statement, ≥85% usability, 100% automated provenance, 30/30 manual provenance, ≥70% eligible strata, Pareto triage, immutable rebuild and deterministic publication are all assigned to tasks.
- Placeholder scan: runtime values that may change after remediation are explicitly sourced from command output; no `TBD`/`TODO` instructions remain.
- Interface consistency: Tasks 2, 4, 5 and 6 use `AnnotationBundle` and `parse_expected_periods` from Tasks 1–2; both sampling and evaluation use the same loader; all real-corpus commands use the current CLI argument names.
