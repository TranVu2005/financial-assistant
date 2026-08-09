# Normalization v2 Labeled Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove sample-proven false-positive normalization issues from release `7fc5d5d57bf6` while preserving genuine OCR/ambiguity issues, canonical table IDs, cells, placements, and raw provenance.

**Architecture:** Keep field normalizers deterministic and conservative. Make unit/period/number interpretation context-aware through explicit pure-function interfaces, tighten company and statement evidence before the service emits issues, and replay the same frozen sample after rebuilding. The immutable release is never overwritten; a new ruleset version produces a new content-addressed release.

**Tech Stack:** Python 3.11, Pydantic 2, PyArrow/Parquet, pytest, Ruff, mypy, uv.

## Global Constraints

- Execute directly in `D:\GitHub\financial-assistant`; do not create a worktree, per user instruction.
- Preserve unrelated dirty-worktree changes and stage only files named by the current task.
- Use release `data/processed/release_v2_7fc5d5d57bf6` as the immutable before release.
- Use `data/qa/normalization_issue_sample_v2_7fc5d5d57bf6_capped.parquet` as the frozen replay sample.
- Use `data/qa/normalization_issue_labels_v2_7fc5d5d57bf6.csv` as the label source after Task 1 calibration.
- Treat the existing labels as AI-assisted draft labels until Task 1 completes; do not call them human ground truth beforehand.
- Keep schema version `2`, canonical table count `146011`, and the exact before/after canonical table-ID set.
- Do not change ingestion detection, extraction, continuation merge, source occurrences, placements, or raw text.
- Do not suppress an issue merely to reduce counts; every removed issue must map to a reviewed `false_positive` sample and a regression test.
- Preserve every reviewed `true_issue` as unresolved unless stronger source evidence is documented during label calibration.
- Do not add fuzzy matching or an ML/LLM classifier to runtime normalization.
- Add no runtime dependency.
- Run each behavior through red-green-refactor TDD and commit only after the focused tests pass.
- Generated QA samples, labels, releases, and comparison reports remain local and are not staged by default.

## Baseline to Preserve

The current AI-assisted baseline contains 1,018 labeled samples:

| Issue code | Sample | True issue | False positive |
|---|---:|---:|---:|
| `company_conflict` | 200 | 0 | 200 |
| `metric_unknown` | 200 | 2 | 198 |
| `number_ambiguous` | 100 | 13 | 87 |
| `number_invalid` | 200 | 163 | 37 |
| `period_incomplete` | 100 | 1 | 99 |
| `statement_conflict` | 18 | 0 | 18 |
| `unit_unknown` | 200 | 0 | 200 |

The acceptance gate is based on corrected Task 1 labels, not on forcing these draft counts to remain unchanged.

---

### Task 1: Calibrate and freeze the AI-assisted labels

**Files:**
- Review: `data/qa/normalization_issue_review_v2_7fc5d5d57bf6_labeled.csv`
- Modify locally: `data/qa/normalization_issue_labels_v2_7fc5d5d57bf6.csv`
- Regenerate locally: `artifacts/normalization-audit/v2_reviewed_baseline/baseline.json`
- Regenerate locally: `artifacts/normalization-audit/v2_reviewed_baseline/baseline.md`

**Interfaces:**
- Consumes: frozen sample IDs and full source context from the labeled review CSV.
- Produces: one valid `LabelRecord(sample_id, label, cause_code, reviewer_note)` for every one of the 1,018 sample IDs.

- [ ] **Step 1: Review every high-risk automated decision**

Filter the review CSV and inspect `raw_value`, table/row/column context, and `source_excerpt` for these exact sets:

```text
label=true_issue                                      # 179 rows
cause_code=unsupported_metric_alias                   # 162 rows
issue_code=company_conflict                           # 200 rows
issue_code=statement_conflict                         # 18 rows
```

Apply these semantic rules:

```text
unsupported_metric_alias -> false_positive only when reviewer_note names an
existing canonical metric using the note format `canonical_metric=net_revenue`.

Readable but unsupported source concepts with no existing canonical metric
remain true_issue; do not map them to a broader metric.

company_conflict -> true_issue only for explicit issuer evidence such as
"Mã CK:", "ticker", or "stock code" that contradicts document.company_code.

statement_conflict -> false_positive for narrative sentences mentioning more
than one statement; preserve true_issue for a concise heading that genuinely
names multiple primary statements.
```

- [ ] **Step 2: Spot-check homogeneous false-positive strata**

Review at least 20 rows from each of these strata, sorted by `raw_value`, `company_code`, and `report_year`:

```text
unit_unknown/non_value_cell
unit_unknown/unsupported_unit_alias
period_incomplete/period_missing_year
number_ambiguous/separator_ambiguity
number_invalid/non_value_cell
metric_unknown/non_metric_row
```

If one reviewed row contradicts the heuristic, review the entire matching `(issue_code, cause_code, normalized raw_value)` stratum and correct all affected labels.

- [ ] **Step 3: Export the corrected four-column labels file**

Keep exactly this header and exactly one row per frozen sample ID:

```csv
sample_id,label,cause_code,reviewer_note
```

Allowed labels remain `true_issue`, `false_positive`, and `uncertain`. Every `false_positive/unsupported_metric_alias` note must contain the `canonical_metric=` prefix followed by a value already present in `METRIC_ALIASES.values()`; for example, `canonical_metric=net_revenue`.

- [ ] **Step 4: Validate the corrected baseline**

Run:

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  normalization-audit baseline `
  --sample data/qa/normalization_issue_sample_v2_7fc5d5d57bf6_capped.parquet `
  --labels data/qa/normalization_issue_labels_v2_7fc5d5d57bf6.csv `
  --output-dir artifacts/normalization-audit/v2_reviewed_baseline
```

Expected: command exits `0`; every issue has `unlabeled_count = 0`; total `sample_count = 1018`; conclusive coverage is at least `0.90` for every issue code selected for remediation.

- [ ] **Step 5: Record the immutable label checksum**

Run:

```powershell
Get-FileHash `
  data/qa/normalization_issue_labels_v2_7fc5d5d57bf6.csv `
  -Algorithm SHA256
```

Copy the resulting SHA-256 into the execution notes. Do not stage generated labels unless the user explicitly changes the repository data policy.

---

### Task 2: Unify unit evidence, extraction, and cell-level issue eligibility

**Files:**
- Modify: `src/financial_report_qa/normalization/units.py:1-169`
- Modify: `src/financial_report_qa/normalization/service.py:137-235`
- Test: `tests/unit/normalization/test_units.py`
- Test: `tests/unit/normalization/test_service.py`
- Test: `tests/regression/normalization/test_false_positive_remediations.py`

**Interfaces:**
- Consumes: `normalized_key(raw)`, current `CanonicalUnit`, and existing `resolve_unit(cell_hint, column_raw, table_raw)` precedence.
- Produces: `strip_unit_context(raw: str) -> str`, `is_monetary_unit(unit: CanonicalUnit | None) -> bool`, and consistent `normalize_unit`/`has_unit_evidence` behavior for composite headers.

- [ ] **Step 1: Add failing composite-unit and false-substring tests**

Add to `tests/unit/normalization/test_units.py`:

```python
@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("2020VND", "VND"),
        ("2025Triệu VND", "VND_million"),
        ("1/1/2017\nGiá gốc VND", "VND"),
        ("Ngàn VNDNăm trước", "VND_thousand"),
        ("31/12/2024\nVND", "VND"),
    ],
)
def test_normalize_unit_extracts_unit_from_composite_headers(
    raw: str, canonical: str
) -> None:
    assert normalize_unit(raw) == Decision(value=canonical)


@pytest.mark.parametrize("raw", ["Công ty Vinpearl", "Công ty liên kết", "Mối quan hệ"])
def test_unit_evidence_requires_a_real_unit_token(raw: str) -> None:
    assert has_unit_evidence(raw) is False
    assert normalize_unit(raw) == Decision(value=None)
```

Import `Decision` from `_shared` in the test file.

- [ ] **Step 2: Run the focused unit tests and verify RED**

Run:

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_units.py
```

Expected: composite-header cases fail and `Công ty Vinpearl` exposes the unbounded `ty` evidence match.

- [ ] **Step 3: Implement one shared unit-token matcher**

In `units.py`, replace the broad substring regex with ordered phrase patterns. Match longer scale phrases before `VND`, allow adjacency to digits and known period words, and never accept bare ASCII `ty` inside a word.

Provide these exact public helpers:

```python
def strip_unit_context(raw: str) -> str:
    """Remove recognized unit phrases while preserving all non-unit text."""


def is_monetary_unit(unit: CanonicalUnit | None) -> bool:
    return unit in _MONETARY_UNITS
```

Make `has_unit_evidence(raw)` and `normalize_unit(raw)` consume the same ordered matcher so evidence cannot be true while normalization ignores a recognized alias. If multiple aliases are nested, select the longest phrase (`triệu VND` before `VND`). Preserve `unit_unknown` for explicit but unsupported units such as `Đơn vị tính: nghìn USD`.

- [ ] **Step 4: Add failing service tests for missing and numeric cells**

Add to `tests/unit/normalization/test_service.py`:

```python
def test_unit_issue_is_not_emitted_for_missing_or_text_cells() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cells = [
        _cell(
            table_id,
            row_idx=0,
            col_idx=0,
            row_label="Doanh thu thuần",
            column_label="2024VND",
            value="-",
        ),
        _cell(
            table_id,
            row_idx=1,
            col_idx=0,
            row_label="Ghi chú",
            column_label="Mối quan hệ",
            value="Công ty Vinpearl",
        ),
    ]
    normalized = normalize_extraction(
        document, _extraction(document, cells, unit_raw=None)
    )
    assert _issues_for(normalized.issues, field="unit", code="unit_unknown") == []


def test_composite_column_unit_is_applied_to_numeric_cell() -> None:
    document = _document()
    table_id = stable_table_id(document.doc_id, 10, 20)
    cell = _cell(
        table_id,
        row_idx=0,
        col_idx=0,
        row_label="Doanh thu thuần",
        column_label="2025Triệu VND",
        value="531.695",
    )
    normalized = normalize_extraction(
        document, _extraction(document, [cell], unit_raw=None)
    )
    output = normalized.extraction.tables[0].cells[0]
    assert output.unit == "VND_million"
    assert _issues_for(normalized.issues, field="unit", code="unit_unknown") == []
```

- [ ] **Step 5: Run the focused service tests and verify RED**

Run:

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_service.py `
  -k "unit_issue_is_not_emitted or composite_column_unit"
```

Expected: at least the composite-unit assertion fails before the implementation is connected.

- [ ] **Step 6: Gate unit resolution on a parsed numeric value**

In `normalize_extraction`, cache column unit decisions by raw column label. Resolve unit context for numeric candidates, but emit a cell-level `unit_unknown` only when all conditions hold:

```python
num_dec is not None
and num_dec.value is not None
and unit_dec.value is None
and unit_dec.issue_code == "unit_unknown"
```

Missing markers, narrative strings, dates, and ranges must not produce cell-level unit issues. Keep the existing one-per-table issue for an explicit unsupported `table.unit_raw`.

- [ ] **Step 7: Run unit and service tests and verify GREEN**

Run:

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_units.py `
  tests/unit/normalization/test_service.py `
  tests/regression/normalization/test_false_positive_remediations.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the unit remediation**

```powershell
git add `
  src/financial_report_qa/normalization/units.py `
  src/financial_report_qa/normalization/service.py `
  tests/unit/normalization/test_units.py `
  tests/unit/normalization/test_service.py `
  tests/regression/normalization/test_false_positive_remediations.py
git commit -m "fix(normalization): resolve composite units without cell noise"
```

---

### Task 3: Restrict company and statement evidence to auditable headings

**Files:**
- Modify: `src/financial_report_qa/normalization/companies.py:156-224`
- Modify: `src/financial_report_qa/normalization/statements.py:34-58`
- Modify: `src/financial_report_qa/normalization/service.py:64-107`
- Test: `tests/unit/normalization/test_companies.py`
- Test: `tests/unit/normalization/test_statements.py`
- Test: `tests/unit/normalization/test_service.py`

**Interfaces:**
- Consumes: document inventory company code and raw table title.
- Produces: unchanged general-purpose `resolve_company_code(raw)` plus stricter table-title validation inside `normalize_company`; `normalize_statement_type` returns no issue for narrative references.

- [ ] **Step 1: Add failing company-title evidence tests**

Add to `tests/unit/normalization/test_companies.py`:

```python
@pytest.mark.parametrize("title", ["VND", "CONTENTS", "Trang", "Shares"])
def test_table_layout_titles_are_not_company_evidence(title: str) -> None:
    assert normalize_company(_document("HBC"), title).issue_code is None


def test_only_explicit_conflicting_ticker_is_a_table_company_conflict() -> None:
    decision = normalize_company(_document("VCB"), "Mã CK: MBB")
    assert decision.value == "VCB"
    assert decision.issue_code == "company_conflict"
```

Update the existing inventory-authority conflict assertion to use `Mã CK: MBB`; retain all `resolve_company_code` tests, including open-set `TCB` behavior.

- [ ] **Step 2: Add failing narrative-statement tests**

Add to `tests/unit/normalization/test_statements.py`:

```python
@pytest.mark.parametrize(
    "raw",
    [
        (
            "Tiền và các khoản tương đương tiền thể hiện trên báo cáo lưu chuyển "
            "tiền tệ bao gồm các khoản trên bảng cân đối kế toán sau đây"
        ),
        (
            "Thông tin tài chính được trích từ bảng cân đối kế toán và báo cáo "
            "kết quả hoạt động kinh doanh của các công ty liên kết"
        ),
    ],
)
def test_narrative_statement_references_are_not_headings(raw: str) -> None:
    assert normalize_statement_type(raw) == Decision(value=None)
```

Keep `Bảng cân đối kế toán / Báo cáo lưu chuyển tiền tệ` as a genuine `statement_conflict` regression.

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_companies.py `
  tests/unit/normalization/test_statements.py
```

Expected: generic table-title and narrative-reference cases fail.

- [ ] **Step 4: Separate generic company resolution from table-title validation**

Keep `_company_evidence_codes` and `resolve_company_code` for explicit user/query normalization. In `normalize_company`, treat only `_explicit_tickers(title_raw)` as contradictory table-title evidence. Do not use a bare ticker, contained registry name, unit, or layout heading from an arbitrary table title to contradict immutable document inventory.

- [ ] **Step 5: Add a statement-heading gate**

In `statements.py`, recognize narrative markers before family conflict logic:

```python
_NARRATIVE_MARKERS = (
    "thể hiện trên",
    "trích từ",
    "bao gồm các khoản",
    "thay đổi trên",
)
```

If a title contains a narrative marker and more than one statement family, return `Decision(value=None)` without an issue. Continue returning `statement_conflict` for concise headings that deliberately combine families. Preserve exact single-family heading behavior.

- [ ] **Step 6: Verify service-level issue suppression**

Add a service test with title `VND` and company `HBC`, plus a narrative statement-reference title. Assert no `company_conflict` or `statement_conflict` is emitted and raw table title remains unchanged.

- [ ] **Step 7: Run tests and verify GREEN**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_companies.py `
  tests/unit/normalization/test_statements.py `
  tests/unit/normalization/test_service.py
```

- [ ] **Step 8: Commit the evidence-scope fix**

```powershell
git add `
  src/financial_report_qa/normalization/companies.py `
  src/financial_report_qa/normalization/statements.py `
  src/financial_report_qa/normalization/service.py `
  tests/unit/normalization/test_companies.py `
  tests/unit/normalization/test_statements.py `
  tests/unit/normalization/test_service.py
git commit -m "fix(normalization): scope company and statement evidence"
```

---

### Task 4: Resolve relative and composite periods without hiding genuine incomplete periods

**Files:**
- Modify: `src/financial_report_qa/normalization/periods.py:1-103`
- Consume: `src/financial_report_qa/normalization/units.py::strip_unit_context`
- Test: `tests/unit/normalization/test_periods.py`
- Test: `tests/unit/normalization/test_service.py`

**Interfaces:**
- Consumes: `normalize_period(raw: str, report_year: int)` and recognized unit stripping.
- Produces: the same function signature with deterministic relative-year and embedded-year resolution.

- [ ] **Step 1: Replace the old incomplete-period expectations with sample-backed tests**

Add to `tests/unit/normalization/test_periods.py`:

```python
@pytest.mark.parametrize(
    ("raw", "report_year", "canonical"),
    [
        ("Năm nay", 2024, "2024"),
        ("Năm trước", 2024, "2023"),
        ("Năm nay VND", 2024, "2024"),
        ("Năm trướcVND", 2024, "2023"),
        ("Năm 2024VND", 2025, "2024"),
        ("Năm 2024\nTriệu đồng", 2025, "2024"),
    ],
)
def test_normalize_period_resolves_relative_and_composite_years(
    raw: str, report_year: int, canonical: str
) -> None:
    assert normalize_period(raw, report_year) == Decision(value=canonical)


def test_non_period_year_phrase_remains_incomplete() -> None:
    assert normalize_period("Năm hết hiệu lực", 2019) == Decision(
        value=None, issue_code="period_incomplete"
    )
```

Remove `Năm nay` from the existing incomplete-period parameter list and import `Decision`.

- [ ] **Step 2: Run period tests and verify RED**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_periods.py
```

- [ ] **Step 3: Normalize period text before matching**

At the start of `normalize_period`, assign `stripped = strip_unit_context(raw)` and then `key = normalized_key(stripped)`. Resolve exact relative-year phrases before generic `startswith("năm")` handling:

```python
if key in {"năm nay", "năm hiện hành"}:
    return Decision(value=str(report_year))
if key in {"năm trước", "năm trước đó"}:
    return Decision(value=str(report_year - 1))
```

Search for an explicit standalone `(19|20)\d{2}` after unit stripping when the header starts with `năm`. Do not extract a year from arbitrary narrative labels such as `Doanh thu 2024`.

- [ ] **Step 4: Add a service regression for one issue per logical column**

Create two cells sharing `column_label="Năm nay VND"`. Assert both receive period `2024`, both receive unit `VND`, and no `period_incomplete` or `unit_unknown` issue is emitted.

- [ ] **Step 5: Run period, unit, and service tests and verify GREEN**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_periods.py `
  tests/unit/normalization/test_units.py `
  tests/unit/normalization/test_service.py
```

- [ ] **Step 6: Commit period resolution**

```powershell
git add `
  src/financial_report_qa/normalization/periods.py `
  tests/unit/normalization/test_periods.py `
  tests/unit/normalization/test_service.py
git commit -m "fix(normalization): resolve relative and composite periods"
```

---

### Task 5: Make number parsing context-aware while preserving genuine ambiguity and OCR errors

**Files:**
- Modify: `src/financial_report_qa/normalization/numbers.py:1-188`
- Modify: `src/financial_report_qa/normalization/service.py:160-235`
- Consume: `src/financial_report_qa/normalization/units.py::is_monetary_unit`
- Test: `tests/unit/normalization/test_numbers.py`
- Test: `tests/unit/normalization/test_service.py`
- Test: `tests/regression/normalization/test_false_positive_remediations.py`

**Interfaces:**
- Consumes: parsed column/table unit context from Task 2.
- Produces: `NumberContext = Literal["unknown", "monetary", "percent"]` and `parse_number(raw: str, *, context: NumberContext = "unknown") -> NumberDecision`.

- [ ] **Step 1: Add failing candidate-filter tests**

Extend `tests/unit/normalization/test_numbers.py`:

```python
@pytest.mark.parametrize("raw", ["4 - 5", "10 - 39", "31.12.2021"])
def test_numeric_candidate_excludes_ranges_and_dates(raw: str) -> None:
    assert is_numeric_candidate(raw) is False


def test_numeric_candidate_keeps_malformed_merged_percent_for_audit() -> None:
    assert is_numeric_candidate("50%30%") is True
    assert parse_number("50%30%").issue_code == "number_invalid"
```

The second test protects reviewed OCR/merged-cell `true_issue` records from disappearing.

- [ ] **Step 2: Add failing separator-context tests**

```python
def test_monetary_context_resolves_single_three_digit_group() -> None:
    assert parse_number("1.764", context="monetary").value == Decimal("1764")


def test_percent_context_resolves_decimal_comma() -> None:
    decision = parse_number("99,999%", context="percent")
    assert decision.value == Decimal("99.999")
    assert decision.unit_hint == "percent"


def test_unknown_context_preserves_separator_ambiguity() -> None:
    assert parse_number("25.967", context="unknown").issue_code == "number_ambiguous"
```

- [ ] **Step 3: Run number tests and verify RED**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_numbers.py
```

- [ ] **Step 4: Implement explicit number context**

Add `NumberContext = Literal["unknown", "monetary", "percent"]` and change the function signature to `parse_number(raw: str, *, context: NumberContext = "unknown") -> NumberDecision`.

Keep the default conservative behavior unchanged. In the single-separator branch, use this exact context gate:

```python
allow_single_group = is_negative or had_trailing_separator or context == "monetary"
```

When `unit_hint == "percent"` or `context == "percent"`, interpret one comma or dot followed by one to three digits as the decimal part. Continue rejecting malformed grouping and merged values.

Update `is_numeric_candidate` to reject full-string date and numeric-range patterns before its character-class check. Do not reject repeated `%`; those are auditable invalid numeric candidates.

- [ ] **Step 5: Add service-level context tests**

Add three cases to `tests/unit/normalization/test_service.py`:

```text
value=1.764, column=2024Triệu VND -> value_numeric=1764, no number_ambiguous
value=25.967, column=Năm trước, no unit -> number_ambiguous remains
value=31.12.2021 or 4 - 5 -> no number_invalid because neither is a number candidate
```

Assert `50%30%` still emits `number_invalid`.

- [ ] **Step 6: Resolve provisional unit before number parsing in the service**

For each value candidate:

1. Resolve column/table unit with `cell_hint=None`.
2. Derive `NumberContext`: `percent`, `monetary`, or `unknown`.
3. Parse the raw value with that context.
4. If `NumberDecision.unit_hint` is present, resolve unit again so cell `%` remains the most specific evidence.
5. Emit number and unit issues using the final decisions.

Do not change raw values or source IDs.

- [ ] **Step 7: Run number, unit, service, and eligibility tests**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_numbers.py `
  tests/unit/normalization/test_units.py `
  tests/unit/normalization/test_service.py `
  tests/unit/normalization/test_eligibility.py `
  tests/regression/normalization/test_false_positive_remediations.py
```

Expected: contextual false positives resolve; unknown-context ambiguity remains blocking and searchable according to existing eligibility contracts.

- [ ] **Step 8: Commit contextual number parsing**

```powershell
git add `
  src/financial_report_qa/normalization/numbers.py `
  src/financial_report_qa/normalization/service.py `
  tests/unit/normalization/test_numbers.py `
  tests/unit/normalization/test_service.py `
  tests/regression/normalization/test_false_positive_remediations.py
git commit -m "fix(normalization): resolve separators from financial context"
```

---

### Task 6: Remove metric noise without inventing canonical semantics

**Files:**
- Modify: `src/financial_report_qa/normalization/metrics.py:1-310`
- Modify: `src/financial_report_qa/normalization/service.py:108-137`
- Test: `tests/unit/normalization/test_metrics.py`
- Test: `tests/unit/normalization/test_service.py`
- Test: `tests/regression/normalization/test_false_positive_remediations.py`

**Interfaces:**
- Consumes: reviewed Task 1 rows and corrected statement classification from Task 3.
- Produces: exact aliases only for existing source metrics and `is_non_metric_label(raw: str | None) -> bool` for structural/narrative rows.

- [ ] **Step 1: Add a pure structural-label predicate**

Expose:

```python
def is_non_metric_label(raw: str | None) -> bool:
    """Return true only for labels that are structurally not source metrics."""
```

Add parameterized tests for reviewed structural cases:

```python
@pytest.mark.parametrize(
    "raw",
    [
        "Năm hiện hành",
        "2.",
        "5.",
        "3. Cam kết cho vay không hủy ngang",
        "6. Outstanding shares",
    ],
)
def test_reviewed_structural_rows_are_not_metrics(raw: str) -> None:
    assert is_non_metric_label(raw) is True
    assert normalize_metric(raw).issue_code is None
```

Do not use a blanket `^\d+\.` suppression for labels containing an actual supported metric. Strip a leading ordinal only to test the remaining text against structural headings and registered aliases.

- [ ] **Step 2: Add reviewed exact aliases only**

From the calibrated label file, collect rows satisfying both conditions:

```text
label=false_positive
cause_code=unsupported_metric_alias
```

Require `reviewer_note` to contain the `canonical_metric=` prefix. Add each exact reviewed raw label to the relevant statement alias dictionary only when the suffix after that prefix already belongs to `SOURCE_METRICS_BY_STATEMENT`. Reject broader substitutions; for example, do not map `Tài sản dài hạn khác` to `non_current_assets`, and do not map `Nguyên giá` without a context-independent canonical metric.

For every accepted pair, add a parameterized test containing the sample ID prefix as the pytest case ID:

```python
@pytest.mark.parametrize(
    ("raw", "canonical"),
    REVIEWED_ALIAS_CASES,
    ids=REVIEWED_ALIAS_SAMPLE_IDS,
)
def test_reviewed_metric_aliases_are_exact(raw: str, canonical: str) -> None:
    assert normalize_metric(raw) == Decision(value=canonical)
```

- [ ] **Step 3: Run metric tests and verify RED**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_metrics.py
```

- [ ] **Step 4: Implement structural gating and aliases**

Make `normalize_metric` return no issue for `is_non_metric_label(raw)`. Keep exact alias lookup and retain `metric_unknown` for readable source concepts that lack a reviewed canonical metric. Do not add fuzzy matching, containment matching, or derived metrics to `METRIC_ALIASES`.

- [ ] **Step 5: Tighten service emission to primary statement tables**

Emit `metric_unknown` only when all conditions hold:

```text
statement_type in {income_statement, balance_sheet, cash_flow_statement}
row has at least one numeric candidate value cell
row label is not structural
```

Equity-change, notes, segment, off-balance-sheet, narrative, and non-statement tables retain searchable raw labels without metric-noise issues unless a future canonical taxonomy explicitly covers them.

- [ ] **Step 6: Add service regressions for reviewed contexts**

Create rows representing `Năm hiện hành`, off-balance-sheet `3. Cam kết cho vay không hủy ngang`, and one reviewed supported alias in a primary statement. Assert the first two emit no metric issue, while the supported alias receives its canonical value.

- [ ] **Step 7: Run metric, statement, service, and eligibility tests**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q `
  tests/unit/normalization/test_metrics.py `
  tests/unit/normalization/test_statements.py `
  tests/unit/normalization/test_service.py `
  tests/unit/normalization/test_eligibility.py `
  tests/regression/normalization/test_false_positive_remediations.py
```

- [ ] **Step 8: Commit metric-noise remediation**

```powershell
git add `
  src/financial_report_qa/normalization/metrics.py `
  src/financial_report_qa/normalization/service.py `
  tests/unit/normalization/test_metrics.py `
  tests/unit/normalization/test_service.py `
  tests/regression/normalization/test_false_positive_remediations.py
git commit -m "fix(normalization): gate metric issues by reviewed semantics"
```

---

### Task 7: Version the ruleset and run the local verification gate

**Files:**
- Modify: `src/financial_report_qa/normalization/_shared.py:10`
- Test: `tests/unit/normalization/test_shared.py`
- Verify: all source and test files changed in Tasks 2-6

**Interfaces:**
- Consumes: completed behavior changes.
- Produces: ruleset version `2026.08.6` in every normalized document fingerprint.

- [ ] **Step 1: Add a failing ruleset-version assertion**

Update `tests/unit/normalization/test_shared.py`:

```python
def test_shared_primitives_constants_and_decision() -> None:
    assert RULESET_VERSION == "2026.08.6"
```

- [ ] **Step 2: Run the assertion and verify RED**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  pytest -q tests/unit/normalization/test_shared.py
```

- [ ] **Step 3: Bump the ruleset version**

Set:

```python
RULESET_VERSION = "2026.08.6"
```

- [ ] **Step 4: Run the complete verification gate**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync pytest -q
uv --cache-dir .cache/uv run --frozen --no-sync ruff check src tests
uv --cache-dir .cache/uv run --frozen --no-sync ruff format --check src tests
uv --cache-dir .cache/uv run --frozen --no-sync mypy src/financial_report_qa
git diff --check
```

Expected: tests pass except the documented Windows symlink privilege skip; Ruff, formatting, mypy, and diff checks pass.

- [ ] **Step 5: Build the wheel and verify registry packaging**

```powershell
uv --cache-dir .cache/uv build `
  --no-build-isolation `
  --out-dir .cache/audit-dist
```

Expected: wheel builds successfully and contains `financial_report_qa/normalization/company_registry.csv`.

- [ ] **Step 6: Commit the ruleset version**

```powershell
git add `
  src/financial_report_qa/normalization/_shared.py `
  tests/unit/normalization/test_shared.py
git commit -m "chore(normalization): version reviewed v2 rules"
```

---

### Task 8: Rebuild the immutable release and replay the frozen audit

**Files:**
- Read: `data/manifests/documents.jsonl`
- Read: `data/raw/financial_statements/**`
- Generate locally: one new content-addressed release directory under `data/processed/`
- Generate locally: `artifacts/normalization-audit/v2_remediated/comparison.json`
- Generate locally: `artifacts/normalization-audit/v2_remediated/comparison.md`

**Interfaces:**
- Consumes: schema v2 dataset builder, ruleset `2026.08.6`, frozen sample, and calibrated labels.
- Produces: a new immutable release and a passing before/after quality gate.

- [ ] **Step 1: Build the complete corpus**

Run from the repository root:

```powershell
$buildJson = uv --cache-dir .cache/uv run --frozen --no-sync `
  python src/financial_report_qa/cli/build_dataset.py `
  --snapshot-root data/raw/financial_statements `
  --manifest-path data/manifests/documents.jsonl `
  --processed-root data/processed | Out-String

$buildResult = $buildJson | ConvertFrom-Json
if ($buildResult.status -ne "success") { throw "Dataset build failed" }
$afterRelease = $buildResult.release_path
$afterRelease
```

Expected invariants in `$buildResult`:

```text
document_count = 1971
table_count = 146011
cell_count = 6199661
placement_count = 6675057
```

The build may require roughly 13 GB RAM. Do not delete or overwrite the before release.

- [ ] **Step 2: Replay the exact frozen sample**

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  normalization-audit compare `
  --before data/processed/release_v2_7fc5d5d57bf6 `
  --after $afterRelease `
  --sample data/qa/normalization_issue_sample_v2_7fc5d5d57bf6_capped.parquet `
  --labels data/qa/normalization_issue_labels_v2_7fc5d5d57bf6.csv `
  --output-dir artifacts/normalization-audit/v2_remediated
```

Expected: command exits `0` and publishes both reports.

- [ ] **Step 3: Assert comparison invariants explicitly**

```powershell
$comparison = Get-Content `
  artifacts/normalization-audit/v2_remediated/comparison.json `
  -Raw | ConvertFrom-Json

if (-not $comparison.passed) { throw ($comparison.errors -join "; ") }
if ($comparison.before_table_count -ne 146011) { throw "Before table count changed" }
if ($comparison.after_table_count -ne 146011) { throw "After table count changed" }
if ([decimal]$comparison.coverage -lt [decimal]0.90) { throw "Coverage below 0.90" }
```

The comparison implementation separately rejects any changed canonical table-ID set or changed sampled source context.

- [ ] **Step 4: Confirm per-code false-positive rates**

Inspect `comparison.md`. Every remediated issue code must have conclusive coverage at least `0.90` and false-positive rate at most `0.05`. Reviewed true OCR and unknown-context ambiguity samples may remain present.

- [ ] **Step 5: Run a deterministic ingestion smoke check**

Because normalization changes must not mutate extraction outputs, run:

```powershell
uv --cache-dir .cache/uv run --frozen --no-sync `
  python scripts/smoke_ingestion.py `
  --root data/raw/financial_statements `
  --repo-id tinixai/ViFinQA `
  --revision 60 `
  --repeat-sample 10
```

Expected: discovered/ready/table/cell/placement counts remain deterministic and repeated samples compare equal. If revision `60` is not an immutable resolved commit, stop and regenerate the inventory manifest with the immutable revision before claiming corpus verification.

- [ ] **Step 6: Record final evidence without staging generated data**

Record in the handoff:

```text
before/after fingerprints
source manifest SHA-256
document/table/cell/placement counts
issue counts by code before and after
per-code sample coverage and false-positive rate
test/Ruff/format/mypy/build/smoke results
Windows-only skips
```

Do not stage `data/qa`, `data/processed`, or `artifacts/normalization-audit` unless the user explicitly requests those generated artifacts in Git.

## Final Acceptance Checklist

- [ ] All 1,018 frozen sample IDs have valid calibrated labels.
- [ ] Every removed issue maps to a reviewed false-positive cause and a regression test.
- [ ] Reviewed true OCR corruption, unsupported semantics, and unknown-context ambiguity remain unresolved.
- [ ] `RULESET_VERSION == "2026.08.6"`.
- [ ] Full pytest passes with only documented environment skips.
- [ ] Ruff check, Ruff format check, mypy, and `git diff --check` pass.
- [ ] Wheel build includes the company registry.
- [ ] New release has 1,971 documents, 146,011 tables, 6,199,661 cells, and 6,675,057 placements.
- [ ] Before/after canonical table-ID sets are identical.
- [ ] Frozen sample source context is unchanged.
- [ ] Overall and per-code false-positive rates are at most 5% with at least 90% conclusive coverage.
- [ ] Before release remains readable and unmodified.
