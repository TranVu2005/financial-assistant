# Week 1 Gate Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the correctness and coverage gaps found in the Day 7 Week 1 quality-gate review.

**Architecture:** Preserve the current `financial_report_qa.evaluation` modules and repair behavior at their existing boundaries. Bind every phase to one immutable release, separate automated provenance from the 30-row manual audit, calculate all approved checks centrally, and exercise the real product CLI end to end.

**Tech Stack:** Python 3.11+, Pydantic, PyArrow, pytest, Hypothesis, Ruff, mypy, uv.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-03-week-1-quality-gate-day-7-design.md` exactly.
- Use TDD: add one focused failing test before each implementation change.
- Preserve deterministic UTF-8/LF serialization and never emit absolute paths in artifacts.
- Do not modify ingestion, normalization, dataset-builder behavior, raw data, Parquet releases, notebooks, `plan.md`, or unrelated dirty files.
- Exit codes remain `0` for success/pass, `1` for a valid failed gate, and `2` for invalid input or workflow state.

---

### Task 1: Bind annotations to the immutable release

**Files:**
- Modify: `src/financial_report_qa/evaluation/week1_evaluator.py`
- Test: `tests/unit/evaluation/test_week1_evaluator.py`

**Interfaces:**
- Consumes: `PilotMetadata`, the loaded release manifest, and source-manifest identity.
- Produces: early `Week1GateInputError` on either fingerprint mismatch.

- [ ] **Step 1: Add failing identity tests**

```python
def test_evaluate_rejects_dataset_fingerprint_mismatch(gate_case) -> None:
    gate_case.metadata["dataset_fingerprint"] = "0" * 64
    gate_case.write_metadata()
    with pytest.raises(Week1GateInputError, match="dataset fingerprint mismatch"):
        evaluate_gate(**gate_case.evaluate_args)


def test_evaluate_rejects_source_manifest_fingerprint_mismatch(gate_case) -> None:
    gate_case.metadata["source_manifest_sha256"] = "f" * 64
    gate_case.write_metadata()
    with pytest.raises(Week1GateInputError, match="source manifest fingerprint mismatch"):
        evaluate_gate(**gate_case.evaluate_args)
```

- [ ] **Step 2: Prove both tests fail before implementation**

Run: `uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_evaluator.py -k "fingerprint_mismatch"`

Expected: both tests fail because evaluation currently accepts mismatched metadata.

- [ ] **Step 3: Validate identity immediately after loading inputs**

```python
if metadata.dataset_fingerprint != dataset.manifest.dataset_fingerprint:
    raise Week1GateInputError("dataset fingerprint mismatch")
if metadata.source_manifest_sha256 != dataset.manifest.source_manifest_sha256:
    raise Week1GateInputError("source manifest fingerprint mismatch")
```

Perform these checks before reading or publishing audit results.

- [ ] **Step 4: Run the focused tests and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_evaluator.py -k "fingerprint"
git add src/financial_report_qa/evaluation/week1_evaluator.py tests/unit/evaluation/test_week1_evaluator.py
git commit -m "fix: bind week one annotations to release identity"
```

---

### Task 2: Make table matching conform to the approved rules

**Files:**
- Modify: `src/financial_report_qa/evaluation/week1_matching.py`
- Test: `tests/unit/evaluation/test_week1_matching.py`

**Interfaces:**
- Consumes: expected and observed tables grouped by `doc_id`.
- Produces: deterministic one-to-one assignments ranked by exact span, overlap, boundary distance, and `table_id`.

- [ ] **Step 1: Add failing ambiguity and one-to-one tests**

```python
def test_matcher_prefers_best_span_even_when_candidates_overlap() -> None:
    annotation = _annotation(line_start=10, line_end=19)
    weaker = _table(table_id="tbl_a", line_start=10, line_end=17)
    exact = _table(table_id="tbl_z", line_start=10, line_end=19)
    assert match_tables((annotation,), (weaker, exact))[0].table.table_id == "tbl_z"


def test_matcher_never_reuses_one_observed_table() -> None:
    matches = match_tables(
        (_annotation(annotation_id="ann_a"), _annotation(annotation_id="ann_b")),
        (_table(table_id="tbl_only"),),
    )
    assert sum(match.table is not None for match in matches) == 1
```

Add a deterministic tie test where input order is reversed and output is unchanged.

- [ ] **Step 2: Run tests and confirm the current matcher fails at least one case**

Run: `uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_matching.py`

- [ ] **Step 3: Implement the exact deterministic assignment**

Use eligible span-overlap pairs only. Score assignments with this tuple, in order:

```python
(
    int(observed.line_start == expected.line_start and observed.line_end == expected.line_end),
    overlap_length / expected_span_length,
    -abs(observed.line_start - expected.line_start)
    - abs(observed.line_end - expected.line_end),
)
```

Maximize total assignment score without reusing `table_id`; break equal optima by the lexicographically smallest ordered tuple of assigned table IDs. Do not use statement type to override the approved span ranking; a wrong type remains a diagnostic `statement_mismatch`. Remove the stray unused table-ID comprehension.

- [ ] **Step 4: Run matching tests and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_matching.py
git add src/financial_report_qa/evaluation/week1_matching.py tests/unit/evaluation/test_week1_matching.py
git commit -m "fix: enforce deterministic one-to-one table matching"
```

---

### Task 3: Implement deterministic 30-cell manual audit workflow

**Files:**
- Modify: `src/financial_report_qa/evaluation/week1_sampling.py`
- Modify: `src/financial_report_qa/evaluation/week1_provenance.py`
- Modify: `src/financial_report_qa/evaluation/week1_evaluator.py`
- Modify: `src/financial_report_qa/evaluation/week1_cli.py`
- Test: `tests/unit/evaluation/test_week1_provenance.py`
- Test: `tests/unit/evaluation/test_week1_evaluator.py`
- Test: `tests/integration/evaluation/test_week1_cli.py`

**Interfaces:**
- Produces: `sample_audit_cells(...) -> tuple[CellAudit, ...]` containing exactly 30 deterministic rows with `verified=None` and blank notes.
- Consumes during evaluation: the completed `cell-audit.csv`; only `verified` and `review_notes` may differ from the regenerated sample.

- [ ] **Step 1: Add failing sampling tests**

```python
def test_sample_is_deterministic_stratified_and_table_capped(candidates) -> None:
    selected = select_audit_cells(candidates, sample_size=30, max_per_table=2)
    reversed_selected = select_audit_cells(tuple(reversed(candidates)), 30, 2)
    assert selected == reversed_selected
    assert len({item.cell_id for item in selected}) == 30
    assert max(Counter(item.table_id for item in selected).values()) <= 2
```

Also assert fewer than 30 eligible cells raises `Week1GateInputError`.

- [ ] **Step 2: Implement round-robin cell selection and safe excerpts**

Bucket by `(company_code, report_year, statement_type)`, rank buckets and cells with `SAMPLING_VERSION`, and cycle buckets until 30 unique cells are selected. Build excerpts from inclusive source lines and cap them at 500 Unicode code points using `text[:497] + "..."`.

- [ ] **Step 3: Add the missing `sample-cells` CLI phase**

```python
sample_parser = subparsers.add_parser("sample-cells")
add_common_arguments(sample_parser)
```

The handler validates release identity and expected tables, runs automated provenance, writes `cell-audit.csv` atomically, and refuses to overwrite an existing file.

- [ ] **Step 4: Make final evaluation consume manual evidence**

Regenerate the deterministic sample, require exactly the same 30 immutable rows, and validate `verified` is lowercase `true` or `false`. Do not auto-generate `verified=True` rows during `evaluate`.

- [ ] **Step 5: Run focused tests and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_provenance.py tests/unit/evaluation/test_week1_evaluator.py tests/integration/evaluation/test_week1_cli.py -k "sample or audit"
git add src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_provenance.py src/financial_report_qa/evaluation/week1_evaluator.py src/financial_report_qa/evaluation/week1_cli.py tests/unit/evaluation/test_week1_provenance.py tests/unit/evaluation/test_week1_evaluator.py tests/integration/evaluation/test_week1_cli.py
git commit -m "feat: add deterministic manual cell audit phase"
```

---

### Task 4: Enforce all approved gate checks and canonical hashes

**Files:**
- Modify: `src/financial_report_qa/evaluation/week1_evaluator.py`
- Modify: `src/financial_report_qa/evaluation/week1_contracts.py`
- Test: `tests/unit/evaluation/test_week1_evaluator.py`

**Interfaces:**
- Produces checks: `pilot_document_count`, `statement_type_coverage`, `overall_table_usability`, `accepted_cell_provenance`, `manual_cell_audit`, and `eligible_strata_usability`.
- Produces populated statement and eligible-stratum metrics plus content hashes of both annotation CSVs.

- [ ] **Step 1: Add failing boundary tests**

```python
@pytest.mark.parametrize(
    ("usable", "annotated", "threshold", "passed"),
    [(85, 100, 85, True), (84, 100, 85, False), (7, 10, 70, True), (6, 10, 70, False)],
)
def test_percentage_checks_use_integer_arithmetic(usable, annotated, threshold, passed):
    assert percentage_passes(usable, annotated, threshold) is passed
```

Add cases for fewer than 30 annotations in any statement type, zero accepted cells, 29/30 manual rows, one manual `false`, a nine-table stratum excluded, and a failing ten-table stratum included.

- [ ] **Step 2: Replace incomplete and extra checks**

Remove the undocumented `table_matching_rate >= 90%` pass/fail check. Calculate the six approved checks with integer comparisons; require `accepted_cells > 0`, `provenance_valid_cells == accepted_cells`, exactly 30 audit rows, and all 30 verified.

- [ ] **Step 3: Populate statement and stratum metrics**

Group expected-table assessments by statement type and by `(company_code, report_year, statement_type)`. Include only strata with `annotated >= 10` in the stratum gate, ordered lexicographically, and apply `usable * 100 >= annotated * 70`.

- [ ] **Step 4: Hash canonical file bytes**

```python
expected_tables_sha256 = hashlib.sha256(expected_tables_path.read_bytes()).hexdigest()
cell_audit_sha256 = hashlib.sha256(cell_audit_path.read_bytes()).hexdigest()
```

Add a regression test that changing only `verified` changes `cell_audit_sha256`.

- [ ] **Step 5: Emit the specified report name**

Rename `evaluation_report.md` to `gate-report.md` and update tests. The output set must be exactly `gate-result.json`, `gate-report.md`, and `pareto-errors.csv`.

- [ ] **Step 6: Run focused tests and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_evaluator.py
git add src/financial_report_qa/evaluation/week1_evaluator.py src/financial_report_qa/evaluation/week1_contracts.py tests/unit/evaluation/test_week1_evaluator.py
git commit -m "fix: enforce complete week one gate criteria"
```

---

### Task 5: Replace the false-positive CLI integration coverage

**Files:**
- Modify: `tests/integration/evaluation/test_week1_cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `README.md`
- Modify: `docs/development.md`

**Interfaces:**
- Exercises: `financial-report-qa week1-gate prepare|sample-cells|evaluate` through the real dispatcher.

- [ ] **Step 1: Remove the dead argv list and direct `prepare_pilot()` call**

Call `cli_main([...])` for every phase and assert return codes plus generated files. Do not import helpers from another test module; move shared fixture helpers into `tests/integration/evaluation/conftest.py` if needed.

- [ ] **Step 2: Add a complete workflow test**

```python
assert cli_main(["week1-gate", "prepare", *common_args]) == 0
fill_expected_tables(annotation_root)
assert cli_main(["week1-gate", "sample-cells", *common_args]) == 0
mark_all_audits_verified(annotation_root)
assert cli_main(["week1-gate", "evaluate", *common_args, "--report-root", str(report_root)]) == 0
```

Assert `prepare` creates the templates, `sample-cells` creates exactly 30 blank audit rows, `evaluate` publishes the three exact artifacts, and a second evaluation is byte-identical.

- [ ] **Step 3: Add exit-code tests**

Assert one valid threshold failure returns `1`; metadata/release mismatch and malformed manual audit return `2`; neither case mutates annotations or an existing report.

- [ ] **Step 4: Update operator commands**

Document the three-phase sequence and the two manual edits: complete `expected-tables.csv`, then set all 30 `verified` values in `cell-audit.csv`.

- [ ] **Step 5: Run CLI tests and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/test_cli.py tests/integration/evaluation/test_week1_cli.py
git add tests/integration/evaluation/test_week1_cli.py tests/unit/test_cli.py README.md docs/development.md
git commit -m "test: cover the real week one gate CLI workflow"
```

---

### Task 6: Full regression and scope verification

**Files:**
- Verify only; modify a source file only if a failing check exposes an in-scope defect.

**Interfaces:**
- Produces: evidence that Day 7 behavior matches the approved specification without regressing prior pipeline work.

- [ ] **Step 1: Run Day 7 tests**

```powershell
$env:UV_CACHE_DIR='D:\GitHub\financial-assistant\.cache\uv'
uv run --frozen --no-sync pytest -q tests/unit/evaluation tests/integration/evaluation
```

- [ ] **Step 2: Run broader regression tests**

```powershell
uv run --frozen --no-sync pytest -q tests/unit tests/integration tests/golden
```

- [ ] **Step 3: Run static checks**

```powershell
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync ruff format --check src tests
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation tests/unit/evaluation tests/integration/evaluation
git diff --check
```

- [ ] **Step 4: Verify the CLI surface**

```powershell
uv run --frozen --no-sync financial-report-qa week1-gate --help
```

Expected: help lists `prepare`, `sample-cells`, and `evaluate`.

- [ ] **Step 5: Inspect scope before the final commit**

```powershell
git -c safe.directory=D:/GitHub/financial-assistant status --short
git -c safe.directory=D:/GitHub/financial-assistant diff --stat
```

Confirm no unrelated pre-existing dirty file is staged. Use `superpowers:verification-before-completion` before claiming completion.

---

## Review Coverage

- [ ] Release/annotation identity mismatch is rejected before evaluation.
- [ ] Matching is deterministic, one-to-one, and span-ranked.
- [ ] `sample-cells` exists and produces exactly 30 reviewable rows.
- [ ] Final evaluation consumes, rather than fabricates, manual evidence.
- [ ] All approved corpus, statement, provenance, manual, and stratum gates are enforced.
- [ ] Annotation hashes cover canonical CSV bytes.
- [ ] The report is named `gate-report.md`.
- [ ] Integration tests invoke the real CLI dispatcher for all three phases.
- [ ] Existing unrelated dirty files remain untouched.
