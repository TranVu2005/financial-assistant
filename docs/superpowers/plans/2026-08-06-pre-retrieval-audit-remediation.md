# Pre-Retrieval Pipeline Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a green, provenance-safe pipeline from inventory/TXT ingestion through canonical Parquet generation, with one truthful schema-v2 contract immediately before retrieval.

**Architecture:** Treat source occurrences and source cells as immutable identities, while canonical tables may merge page continuations through placements. Centralize dataset schema versioning, keep normalization conservative under ambiguity, and validate every persisted foreign-key/provenance relationship before publishing a content-addressed release.

**Tech Stack:** Python 3.11, Pydantic 2, PyArrow/Parquet, orjson, pytest, Ruff, mypy, uv.

## Global Constraints

- Preserve the current uncommitted normalization work; do not reset, checkout, stash, or overwrite it.
- Before implementation, create an isolated worktree only after the current WIP is committed or explicitly transplanted with user approval.
- Every production behavior change starts with a failing regression test and an observed expected failure.
- Preserve `value_raw`, raw labels, source line spans, source occurrence IDs, and original source `cell_id` values.
- Reject ambiguous number/period/unit evidence instead of guessing.
- Dataset schema v2 always includes `placements.parquet` and `placement_count`.
- A content-addressed release directory is immutable; never delete and replace an existing release.
- Corpus claims require the pinned local TXT snapshot; the committed manifest alone is not corpus access.

---

## File Structure

- Modify `src/financial_report_qa/data/dataset_builder.py` to own the schema-v2 constant, validate persisted relationships, and publish releases immutably.
- Modify `src/financial_report_qa/cli/build_dataset.py` to consume the same schema constant and report every v2 count.
- Modify `src/financial_report_qa/ingestion/table_detector.py` so ordinals are local to a source line span rather than globally shiftable.
- Modify `src/financial_report_qa/ingestion/table_extractor.py` so continuation merging retains original source-cell IDs.
- Modify `src/financial_report_qa/ingestion/provenance.py` to enforce cell/placement coverage and bounds.
- Modify normalization modules under `src/financial_report_qa/normalization/` only for the proven failures listed below.
- Update corresponding unit, golden, integration, and Week-1 release fixtures.
- Update `scripts/smoke_ingestion.py` and release documentation only where output/contracts change.

### Task 1: Lock the Dataset Schema-v2 Boundary

**Files:**
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Modify: `src/financial_report_qa/cli/build_dataset.py`
- Modify: `tests/unit/data/test_dataset_builder.py`
- Modify: `tests/integration/test_pipeline_e2e.py`
- Modify: `tests/integration/evaluation/test_week1_cli.py`

**Interfaces:**
- Produce: `DATASET_SCHEMA_VERSION: Final = "2"`.
- Constrain: `DatasetBuildConfig.schema_version` to the supported version.
- Produce: CLI JSON field `placement_count`.
- Require: `documents.parquet`, `tables.parquet`, `cells.parquet`, `placements.parquet`, `issues.parquet`, `source_table_occurrences.parquet`, and `manifest.json` in every v2 fixture/release.

- [ ] **Step 1: Add failing schema-version tests**

```python
def test_dataset_config_rejects_mislabeled_schema() -> None:
    with pytest.raises(ValidationError):
        DatasetBuildConfig(
            snapshot_root=Path("snapshot"),
            manifest_path=Path("documents.jsonl"),
            processed_root=Path("processed"),
            schema_version="1",
        )
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run --frozen --no-sync pytest tests/unit/data/test_dataset_builder.py -k mislabeled -v`

Expected: FAIL because arbitrary non-empty schema versions are currently accepted.

- [ ] **Step 3: Centralize the supported schema version**

```python
from typing import Final, Literal

DATASET_SCHEMA_VERSION: Final = "2"

class DatasetBuildConfig(BaseModel):
    schema_version: Literal["2"] = DATASET_SCHEMA_VERSION
```

Use `DATASET_SCHEMA_VERSION` as the CLI default/choice and add `placement_count` to CLI JSON output.

- [ ] **Step 4: Upgrade all v2 integration fixtures**

Write `placements.parquet` with `PLACEMENT_SCHEMA`, add `placement_count` to each fixture manifest, and add `placements.parquet` to the byte-identical E2E filename list.

- [ ] **Step 5: Run schema/CLI/E2E tests**

Run: `uv run --frozen --no-sync pytest tests/unit/data/test_dataset_builder.py tests/integration/test_pipeline_e2e.py tests/integration/evaluation/test_week1_cli.py -v`

Expected: PASS, including the three Week-1 tests currently failing on missing `placements.parquet`.

- [ ] **Step 6: Commit**

```text
fix(data): enforce the canonical schema-v2 release contract
```

### Task 2: Preserve Stable Source Identities Across Detection and Continuation

**Files:**
- Modify: `src/financial_report_qa/ingestion/table_detector.py`
- Modify: `src/financial_report_qa/ingestion/table_extractor.py`
- Modify: `src/financial_report_qa/ingestion/provenance.py`
- Modify: `tests/unit/ingestion/test_table_detector.py`
- Modify: `tests/unit/ingestion/test_table_extractor.py`
- Modify: `tests/unit/ingestion/test_provenance.py`
- Modify: `tests/golden/extraction/expected/unicode_continuation.json`
- Modify: `tests/golden/extraction/test_txt_extraction.py`

**Interfaces:**
- `TableCandidate.ordinal` is the zero-based occurrence index within the same `(kind, line_start, line_end)` source span.
- A merged continuation gets a canonical `table_id`, but each retained source cell keeps its pre-merge `cell_id`.
- `CellRecord.table_id`, canonical row/column coordinates, and placements may be updated for the merged table without changing source-cell identity.

- [ ] **Step 1: Add a failing ordinal-stability test**

Construct two same-line HTML events plus unrelated earlier events. Assert the same-line ordinals remain `(0, 1)` regardless of unrelated detection events.

- [ ] **Step 2: Add a failing continuation identity test**

```python
first_ids = {cell.value_raw: cell.cell_id for cell in extract(tmp_path, first_page).tables[0].cells}
second_ids = {cell.value_raw: cell.cell_id for cell in extract(tmp_path, second_page).tables[0].cells}
merged = extract(tmp_path, first_page + separator + second_page).tables[0]
merged_ids = {cell.value_raw: cell.cell_id for cell in merged.cells}
assert merged_ids["Revenue"] == first_ids["Revenue"]
assert merged_ids["Profit"] == second_ids["Profit"]
```

- [ ] **Step 3: Run both tests and verify RED**

Expected: ordinal test exposes global enumeration; continuation test exposes regenerated IDs at `_merge_pair`.

- [ ] **Step 4: Implement local ordinals and retained cell IDs**

Assign ordinals with a counter keyed by `(item.kind, item.line_start, item.line_end)`. In `_merge_pair`, map `(source_number, old_cell_id)` to `old_cell_id`; update the canonical `table_id` and coordinates only.

- [ ] **Step 5: Strengthen provenance validation**

Require every canonical cell to have at least one placement, require every cell origin to be within table bounds, and require the cell origin coordinate to reference that cell. Keep rowspan/colspan represented by additional placements.

- [ ] **Step 6: Update golden continuation output and run ingestion tests**

Run: `uv run --frozen --no-sync pytest tests/unit/ingestion tests/golden/extraction -v`

Expected: PASS; IDs remain deterministic across repeat extraction and page merges.

- [ ] **Step 7: Commit**

```text
fix(ingestion): preserve source identities through continuation merges
```

### Task 3: Stop Ambiguous Numeric Coercion and Correct Eligibility

**Files:**
- Modify: `src/financial_report_qa/normalization/numbers.py`
- Modify: `src/financial_report_qa/normalization/eligibility.py`
- Modify: `tests/unit/normalization/test_numbers.py`
- Modify: `tests/unit/normalization/test_eligibility.py`
- Modify: `tests/regression/normalization/test_false_positive_remediations.py`

**Interfaces:**
- `parse_number("0,123")`, `parse_number("0.123")`, `parse_number("1,234")`, and `parse_number("1.234")` return `number_ambiguous` without a numeric value.
- `comparable` means metric + period + numeric value + no blocking issue.
- `calculable` additionally requires a supported monetary unit.

- [ ] **Step 1: Keep the current four failing number cases as RED evidence**

Run: `uv run --frozen --no-sync pytest tests/unit/normalization/test_numbers.py -k single_three_digit -v`

Expected: 4 FAIL; current values are `123` or `1234`.

- [ ] **Step 2: Add boundary cases**

Assert `1.234.567`, `1,234,567`, `12,34`, and `12.34` retain their documented behavior, while a single three-digit suffix remains ambiguous.

- [ ] **Step 3: Make single three-digit separators conservative**

In `_resolve_single_separator`, return `None` for exactly two groups when the right group has length three. Keep repeated valid grouping separators accepted.

- [ ] **Step 4: Correct eligibility algebra**

```python
comparable = is_comparable_base and not has_blocking
calculable = comparable and is_monetary
```

Sort/deduplicate `blocking_reasons` so output is deterministic for any input collection.

- [ ] **Step 5: Run normalization number/eligibility tests**

Run: `uv run --frozen --no-sync pytest tests/unit/normalization/test_numbers.py tests/unit/normalization/test_eligibility.py tests/regression/normalization/test_false_positive_remediations.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```text
fix(normalization): reject ambiguous numbers and correct eligibility
```

### Task 4: Normalize Period and Unit Evidence at Logical Scope

**Files:**
- Modify: `src/financial_report_qa/normalization/periods.py`
- Modify: `src/financial_report_qa/normalization/units.py`
- Modify: `src/financial_report_qa/normalization/service.py`
- Modify: `tests/unit/normalization/test_periods.py`
- Modify: `tests/unit/normalization/test_units.py`
- Modify: `tests/unit/normalization/test_service.py`

**Interfaces:**
- Parse multi-level headers such as `Kỳ báo cáo\nQuý IV\nNăm 2024` as `2024-Q4`.
- Parse balance-sheet instant headers such as `Tại ngày 31/12/2024` as `2024-12-31`.
- Resolve evidence precedence as `cell > column > table`; lower-priority monetary defaults do not conflict with a higher-priority explicit scale.
- Emit one period issue per logical column and one table-unit issue per table, using the actual unit evidence as `raw_value`.

- [ ] **Step 1: Run the current period/unit/service failures as RED**

Run: `uv run --frozen --no-sync pytest tests/unit/normalization/test_periods.py tests/unit/normalization/test_units.py tests/unit/normalization/test_service.py -v`

Expected: 5 relevant FAIL across multi-level period, instant prefix, unit precedence, and issue cardinality.

- [ ] **Step 2: Parse explicit components, not arbitrary substrings**

Normalize line breaks, remove only approved labels (`kỳ báo cáo`, `tại ngày`), then combine one quarter token with one four-digit year. Reject contradictory years/quarters as `period_ambiguous`.

- [ ] **Step 3: Implement evidence precedence**

Resolve each evidence level independently. Return the highest-priority valid unit. Return `unit_conflict` only when contradictory evidence exists at the same priority level or when non-monetary semantics conflict explicitly.

- [ ] **Step 4: Move issue creation out of the per-cell loop**

Choose the lowest `(row_idx, cell_id)` cell as the anchor for one logical-column period issue. Emit table unit failures once with `cell_id=None` and `raw_value=table_rec.unit_raw`. Emit cell/column unit issues only for cell/column-specific evidence.

- [ ] **Step 5: Run focused tests**

Run: `uv run --frozen --no-sync pytest tests/unit/normalization/test_periods.py tests/unit/normalization/test_units.py tests/unit/normalization/test_service.py -v`

Expected: PASS with stable issue ordering and fingerprinting.

- [ ] **Step 6: Commit**

```text
fix(normalization): scope period and unit decisions to their evidence
```

### Task 5: Finish Registry, Ruleset, and Static Quality

**Files:**
- Modify: `src/financial_report_qa/normalization/companies.py`
- Add: `src/financial_report_qa/normalization/company_registry.csv`
- Modify: `src/financial_report_qa/normalization/metrics.py`
- Modify: `tests/unit/normalization/test_shared.py`
- Modify: `tests/unit/normalization/test_companies.py`
- Modify: `pyproject.toml` only if an explicit Hatch include rule is needed after wheel inspection.

**Interfaces:**
- Registry loading remains strict UTF-8 and packaged in both wheel and sdist.
- `RULESET_VERSION` test matches the version shipped by the implementation.
- Ruff and mypy pass without suppressing the registry loader.

- [ ] **Step 1: Add a wheel-resource smoke test**

Build into a temporary output directory, install the wheel into an isolated temporary target, and assert `company_name_for_code("STB")` returns the canonical UTF-8 name.

- [ ] **Step 2: Fix the Traversable API typing**

Use `resource.open("r", encoding="utf-8")` without the unsupported `newline` argument, or use `importlib.resources.as_file` plus built-in `open(..., newline="")` when newline control is required.

- [ ] **Step 3: Align ruleset and formatting**

Update `tests/unit/normalization/test_shared.py` to the approved current ruleset version, organize the metrics import, and run Ruff formatting on touched Python files.

- [ ] **Step 4: Run static and packaging checks**

```text
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync ruff format --check src tests
uv run --frozen --no-sync mypy
uv build --no-build-isolation
```

Expected: all exit 0; the wheel and sdist contain `financial_report_qa/normalization/company_registry.csv`.

- [ ] **Step 5: Commit**

```text
fix(normalization): finalize registry and static quality
```

### Task 6: Publish Content-Addressed Releases Without Destructive Replacement

**Files:**
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Modify: `tests/unit/data/test_dataset_builder.py`
- Modify: `tests/integration/test_pipeline_e2e.py`

**Interfaces:**
- If the computed release path does not exist, atomically rename the completed temporary directory.
- If it exists and its manifest fingerprint/schema/counts match, return it without rewriting bytes.
- If it exists but is incomplete or inconsistent, raise `DatasetBuildError`; do not delete it.

- [ ] **Step 1: Add idempotency and corruption RED tests**

Build twice into the same processed root and assert every file's modification time and SHA-256 remain unchanged. Corrupt an existing manifest, rebuild, and assert `DatasetBuildError` while the corrupt directory remains untouched.

- [ ] **Step 2: Run tests and verify RED**

Expected: current implementation deletes `release_dir` with `shutil.rmtree` and republishes it.

- [ ] **Step 3: Add release verification before reuse**

Validate schema version, dataset fingerprint, source manifest SHA-256, all row counts, required filenames, and Parquet schemas. Reuse only a fully matching release.

- [ ] **Step 4: Handle publish races without deletion**

On rename collision, verify the winner and discard only this process's temporary directory. Never remove the content-addressed destination.

- [ ] **Step 5: Run dataset builder/E2E tests**

Run: `uv run --frozen --no-sync pytest tests/unit/data/test_dataset_builder.py tests/integration/test_pipeline_e2e.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```text
fix(data): make release publication immutable and idempotent
```

### Task 7: Rebuild and Audit the Pre-Retrieval Corpus Boundary

**Files:**
- Verify: `data/manifests/documents.jsonl`
- Generate locally: a new schema-v2 release under `data/processed/`
- Generate locally: normalization audit reports under the configured artifact directory
- Modify tests/docs only when the verified corpus exposes a reproducible defect.

**Interfaces:**
- Corpus revision must be explicitly identified as immutable.
- Release manifest counts equal Parquet row counts.
- Every table/cell/placement/issue/source-occurrence foreign key resolves.
- Repeated builds have the same dataset fingerprint and byte-identical Parquet output.

- [ ] **Step 1: Verify corpus availability and revision**

Confirm all 1,971 ready documents from the committed manifest exist under the named snapshot root and match size/SHA-256. Do not treat manifest-only validation as a smoke pass.

- [ ] **Step 2: Run ingestion smoke twice**

Run: `uv run --frozen --no-sync python scripts/smoke_ingestion.py --root data/raw/ocr_annual_financials --repo-id tinixai/ViFinQA --revision 60 --repeat-sample 10`

Before accepting the result, verify that revision `60` names an immutable snapshot; otherwise stop and regenerate the inventory manifest with the resolved commit identifier.

Expected: both runs report identical document/table/cell/placement/rejection counts.

- [ ] **Step 3: Build schema-v2 release twice in separate output roots**

Compare manifest fingerprints and SHA-256 for every persisted file.

- [ ] **Step 4: Re-run normalization audit**

Use the reviewed sample/labels against the new ruleset. Record table/cell/placement counts, per-code false-positive rates, and any ID migration caused by schema v2. Do not reuse the old `release_v1_f481...` quality claim as evidence for schema v2.

- [ ] **Step 5: Run the full verification gate**

```text
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync ruff format --check .
uv run --frozen --no-sync mypy
git diff --check
```

Expected: 0 failures; a Windows symlink test may remain an explicitly reported environmental skip.

- [ ] **Step 6: Request code review and commit evidence**

Review provenance invariants, schema migration, normalization ambiguity handling, and corpus metrics before integrating.

## Plan Self-Review

- Spec coverage: schema v2, placements, stable source identity, normalization correctness, registry packaging, immutable publication, corpus verification, and full quality gate are covered.
- Placeholder scan: every implementation task has an exact test, behavior, command, and expected result.
- Type consistency: `DATASET_SCHEMA_VERSION`, `placement_count`, local source ordinals, retained source `cell_id`, and schema-v2 filenames are used consistently across producers and consumers.
- Scope: retrieval, planning, execution, UI, and model work remain excluded.
