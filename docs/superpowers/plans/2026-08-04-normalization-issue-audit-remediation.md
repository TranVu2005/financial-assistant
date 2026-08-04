# Normalization Issue Audit and Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Audit normalization issues deterministically, remediate proven false positives, and enforce a false-positive rate of at most 5% without changing 146,011 canonical tables or their IDs.

**Architecture:** Add a normalization-audit module and CLI under evaluation for sampling, labels, metrics, reports, and release comparison. Put conservative evidence gates in the normalization layer so period, unit, metric and number parsers run only on eligible inputs. Add an explicit searchable/comparable/calculable policy so downstream consumers never treat unresolved values as safe calculations.

**Tech Stack:** Python 3.11, Pydantic, PyArrow/Parquet, CSV/JSON, pytest.

## Global Constraints

- Conclusive label coverage is at least 90% for every remediated issue code.
- False-positive rate is at most 5% per remediated code; denominator is true_issue plus false_positive.
- uncertain remains visible and is excluded only from that denominator.
- tables.parquet remains exactly 146,011 rows and its full table-ID set is unchanged.
- Never remove tables or cells to reduce issue counts.
- Preserve raw labels and raw values even when canonical fields remain null.
- An input without period/unit/metric/number evidence produces no issue for that field; malformed evidence still produces the existing explicit issue.
- Legitimate missing markers remain numeric nulls and are not emitted as parse failures.
- Downstream calculations require numeric value, period, unit and no blocking conflict/ambiguity issue.
- Every rule change maps to reviewed sample IDs, a cause_code, and a regression test.
- Sampling is independent of Parquet row order and deterministic from release fingerprint and issue identity.
- No ML/LLM runtime classifier and no new dependency.

---

## File Structure

- Create src/financial_report_qa/evaluation/normalization_audit.py for schemas, sampling, labels, metrics and gates.
- Create src/financial_report_qa/evaluation/normalization_audit_cli.py for sample, baseline and compare commands.
- Modify pyproject.toml to expose normalization-audit.
- Create configs/normalization_audit.yaml with the fixed seed, per-code limits and stratum cap.
- Create tests/unit/evaluation/test_normalization_audit.py.
- Create tests/integration/evaluation/test_normalization_audit_cli.py.
- Modify only sample-justified modules under src/financial_report_qa/normalization and matching tests.
- Create src/financial_report_qa/normalization/eligibility.py and tests/unit/normalization/test_eligibility.py for downstream usage policy.
- Create docs/normalization-issue-audit.md.
- Generated outputs stay under data/qa and artifacts/normalization-audit.

### Task 1: Deterministic stratified sampler

**Files:**
- Create: src/financial_report_qa/evaluation/normalization_audit.py
- Create: tests/unit/evaluation/test_normalization_audit.py

**Interfaces:**
- Produce SAMPLE_SCHEMA.
- Produce AuditSamplingConfig(issue_limits: dict[str, int], max_per_stratum: int, seed: str).
- Produce build_issue_sample(release_path: Path, release_fingerprint: str, config: AuditSamplingConfig) -> pa.Table.
- sample_id hashes release fingerprint, issue code, document/table/cell IDs, field and raw value.
- selection_rank hashes seed and sample_id.
- stratum_key includes issue code, company, year, statement type and normalized raw value.

- [ ] Step 1: Write a failing order-independence test.

~~~python
def test_sample_is_independent_of_input_order(tmp_path):
    a = build_fixture_release(tmp_path / "a", reverse=False)
    b = build_fixture_release(tmp_path / "b", reverse=True)
    config = AuditSamplingConfig(
        issue_limits={"unit_unknown": 3, "metric_unknown": 2},
        max_per_stratum=1,
        seed="normalization-audit-v1",
    )
    left = build_issue_sample(a, "release-1", config)
    right = build_issue_sample(b, "release-1", config)
    assert left.schema == SAMPLE_SCHEMA
    assert left.to_pylist() == right.to_pylist()
~~~

- [ ] Step 2: Run pytest tests/unit/evaluation/test_normalization_audit.py::test_sample_is_independent_of_input_order -v.

Expected: FAIL because the module does not exist.

- [ ] Step 3: Implement the fixed schema and joins.

Read manifest.json, documents.parquet, tables.parquet, cells.parquet and issues.parquet. Include all fields required by the approved spec. Reject a fingerprint mismatch, duplicate sample_id, or unresolved document/table/cell context. Hash-rank within strata, cap each stratum, cap each issue, then sort by issue_code, selection_rank and sample_id.

- [ ] Step 4: Add a skewed-distribution test proving max_per_stratum is respected and all rows of an issue rarer than its limit are retained.

- [ ] Step 5: Run pytest tests/unit/evaluation/test_normalization_audit.py -v.

Expected: PASS.

- [ ] Step 6: Commit with message feat: add deterministic normalization issue sampler.

### Task 2: Human labels and baseline metrics

**Files:**
- Modify: src/financial_report_qa/evaluation/normalization_audit.py
- Modify: tests/unit/evaluation/test_normalization_audit.py

**Interfaces:**
- Produce LabelRecord(sample_id, label, cause_code, reviewer_note).
- Produce load_and_validate_labels(sample: pa.Table, labels_path: Path) -> tuple[LabelRecord, ...].
- Produce evaluate_labels(sample, labels) -> dict[str, IssueAuditMetrics].
- Metrics fields are sample_count, true_issue_count, false_positive_count, uncertain_count, unlabeled_count, conclusive_coverage, false_positive_rate and cause_counts.

- [ ] Step 1: Write parameterized failing tests for unknown sample ID, duplicate sample ID, invalid label and invalid cause code.

~~~python
with pytest.raises(ValueError, match="unknown sample_id"):
    load_and_validate_labels(sample, labels_with_unknown_id)
~~~

Use exactly true_issue, false_positive and uncertain plus the 14 cause codes from the spec. Missing label rows are allowed and counted as unlabeled.

- [ ] Step 2: Run pytest tests/unit/evaluation/test_normalization_audit.py -k labels -v.

Expected: FAIL because label interfaces are absent.

- [ ] Step 3: Implement validation and exact Decimal metrics.

~~~python
conclusive = true_issue_count + false_positive_count
coverage = Decimal(conclusive) / Decimal(sample_count)
false_positive_rate = Decimal(false_positive_count) / Decimal(conclusive) if conclusive else None
~~~

- [ ] Step 4: Add a test with 4 samples: one true issue, one false positive, one uncertain and one unlabeled. Assert coverage 0.5, false-positive rate 0.5, uncertain 1 and unlabeled 1.

- [ ] Step 5: Run the unit file and commit with message feat: validate normalization audit labels.

### Task 3: Audit CLI and reproducible baseline report

**Files:**
- Create: src/financial_report_qa/evaluation/normalization_audit_cli.py
- Modify: pyproject.toml
- Create: configs/normalization_audit.yaml
- Create: tests/integration/evaluation/test_normalization_audit_cli.py
- Create: docs/normalization-issue-audit.md

**Interfaces:**
- normalization-audit sample --release PATH --output PATH --config PATH.
- normalization-audit baseline --sample PATH --labels PATH --output-dir PATH.
- Produce deterministic baseline.json and baseline.md.

- [ ] Step 1: Write a failing integration test that runs sample twice and asserts byte-identical Parquet, then runs baseline and checks release_fingerprint, metrics_by_issue and both report files.

- [ ] Step 2: Run pytest tests/integration/evaluation/test_normalization_audit_cli.py -v.

Expected: FAIL because the CLI is absent.

- [ ] Step 3: Implement argparse commands, parent-directory creation, fixed-schema Parquet writing, sorted JSON keys and issue-code-ordered Markdown. Refuse to overwrite an existing sample when its embedded fingerprint differs.

Create configs/normalization_audit.yaml with seed normalization-audit-v1, max_per_stratum 5, limits of 200 each for unit_unknown, metric_unknown, number_invalid and number_missing, and limits of 100 each for unit_conflict, statement_conflict, number_ambiguous, period_incomplete and period_ambiguous. Rare codes retain all available rows below their limit.

- [ ] Step 4: Add this pyproject entry.

~~~toml
normalization-audit = "financial_report_qa.evaluation.normalization_audit_cli:main"
~~~

- [ ] Step 5: Document exact sample generation, CSV labeling, baseline validation and source-context inspection commands. State that generated samples, labels and reports are not committed by default.

- [ ] Step 6: Run the integration test and commit with message feat: add normalization audit CLI.

### Task 4: Add conservative evidence gates to normalization

**Files:**
- Modify: src/financial_report_qa/normalization/periods.py
- Modify: src/financial_report_qa/normalization/units.py
- Modify: src/financial_report_qa/normalization/numbers.py
- Modify: src/financial_report_qa/normalization/service.py
- Test: tests/unit/normalization/test_periods.py
- Test: tests/unit/normalization/test_units.py
- Test: tests/unit/normalization/test_numbers.py
- Test: tests/unit/normalization/test_service.py

**Interfaces:**
- Produce has_period_evidence(raw: str | None) -> bool.
- Produce has_unit_evidence(raw: str | None) -> bool.
- Produce is_numeric_candidate(raw: str) -> bool and is_missing_number(raw: str) -> bool.
- Preserve normalize_period, normalize_unit, resolve_unit, parse_number and normalize_extraction signatures.
- service.py calls a parser only after its evidence predicate succeeds.

- [ ] Step 1: Write failing evidence-predicate tests.

~~~python
def test_period_evidence_rejects_generic_headers():
    assert has_period_evidence("Giá trị") is False
    assert has_period_evidence("Số tiền") is False
    assert has_period_evidence("2024") is True
    assert has_period_evidence("Tháng 12") is True


def test_number_candidate_rejects_text_but_keeps_malformed_numeric_input():
    assert is_numeric_candidate("Thuyết minh") is False
    assert is_numeric_candidate("1.50.0") is True
    assert is_missing_number("—") is True


def test_unit_evidence_rejects_year_header():
    assert has_unit_evidence("2024") is False
    assert has_unit_evidence("Đơn vị: triệu đồng") is True
~~~

- [ ] Step 2: Run the focused predicate tests.

Run: pytest tests/unit/normalization/test_periods.py tests/unit/normalization/test_numbers.py tests/unit/normalization/test_units.py -v.

Expected: FAIL because the public evidence predicates do not all exist.

- [ ] Step 3: Implement the predicates.

Period evidence accepts four-digit years, dates, quarter tokens and month tokens, including malformed strings beginning with explicit năm/quý/tháng markers so they still produce period_incomplete. Unit evidence accepts VND/đồng/nghìn/triệu/tỷ/percent/ratio tokens. Numeric evidence accepts digits with spaces, signs, parentheses, dots, commas or percent; missing markers are handled separately. Generic text returns false.

- [ ] Step 4: Write failing service-level issue-emission tests.

~~~python
def test_generic_column_header_emits_no_period_issue(normalization_fixture):
    result = normalize_extraction(*normalization_fixture(column_label="Giá trị"))
    assert not any(issue.field == "period" for issue in result.issues)


def test_missing_marker_stays_null_without_number_issue(normalization_fixture):
    result = normalize_extraction(*normalization_fixture(value_raw="-"))
    cell = result.extraction.tables[0].cells[1]
    assert cell.value_numeric is None
    assert not any(issue.field == "number" for issue in result.issues)


def test_malformed_numeric_candidate_keeps_number_invalid(normalization_fixture):
    result = normalize_extraction(*normalization_fixture(value_raw="1.50.0"))
    assert any(issue.code == "number_invalid" for issue in result.issues)
~~~

- [ ] Step 5: Run the three service tests and verify the generic-header and missing-marker cases fail while malformed numeric input remains explicitly invalid.

- [ ] Step 6: Gate parser calls in service.py.

Call normalize_period only when has_period_evidence is true. For value cells, return null without an issue for legitimate missing markers or nonnumeric text; call parse_number only for numeric candidates. Call resolve_unit only when cell, column or table context has unit evidence. Preserve all raw fields.

- [ ] Step 7: Run pytest tests/unit/normalization -v.

Expected: PASS; fixture table counts and table IDs are unchanged.

- [ ] Step 8: Commit with message fix: gate normalization on explicit evidence.

### Task 5: Scope metric issues and define downstream eligibility

**Files:**
- Modify: src/financial_report_qa/normalization/service.py
- Create: src/financial_report_qa/normalization/eligibility.py
- Modify: src/financial_report_qa/normalization/__init__.py
- Test: tests/unit/normalization/test_service.py
- Create: tests/unit/normalization/test_eligibility.py

**Interfaces:**
- Known metric aliases are normalized in every table, but metric_unknown is emitted only when the table has a non-null normalized statement_type.
- Produce CellEligibility(searchable: bool, comparable: bool, calculable: bool, blocking_reasons: tuple[str, ...]).
- Produce classify_cell_eligibility(cell: CellRecord, issue_codes: Collection[NormalizationIssueCode]) -> CellEligibility.

- [ ] Step 1: Write failing metric-scope tests.

~~~python
def test_unknown_row_in_unclassified_notes_table_emits_no_metric_issue(notes_fixture):
    result = normalize_extraction(*notes_fixture(row_label="Diễn giải bổ sung"))
    assert not any(issue.code == "metric_unknown" for issue in result.issues)


def test_unknown_row_in_financial_statement_keeps_metric_issue(statement_fixture):
    result = normalize_extraction(*statement_fixture(row_label="Chỉ tiêu chưa ánh xạ"))
    assert any(issue.code == "metric_unknown" for issue in result.issues)
~~~

- [ ] Step 2: Implement issue scoping without changing normalize_metric. Always retain row_label_raw and any recognized row_label_canonical.

- [ ] Step 3: Write failing eligibility tests.

~~~python
def test_cell_eligibility_levels(cell_factory):
    raw = classify_cell_eligibility(cell_factory(value_numeric=None), set())
    assert raw.searchable and not raw.comparable and not raw.calculable

    comparable = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit=None), set()
    )
    assert comparable.comparable and not comparable.calculable

    calculable = classify_cell_eligibility(
        cell_factory(value_numeric=Decimal("10"), period="2024", unit="VND"), set()
    )
    assert calculable.calculable
~~~

Also assert unit_conflict, number_ambiguous and period_ambiguous block calculable status.

- [ ] Step 4: Implement immutable CellEligibility and classify_cell_eligibility. searchable requires non-empty value_raw; comparable requires numeric value and period; calculable additionally requires unit and no blocking issue.

- [ ] Step 5: Run pytest tests/unit/normalization/test_service.py tests/unit/normalization/test_eligibility.py -v.

Expected: PASS.

- [ ] Step 6: Commit with message feat: define normalized cell eligibility.

### Task 6: Before/after comparison and quality gate

**Files:**
- Modify: src/financial_report_qa/evaluation/normalization_audit.py
- Modify: src/financial_report_qa/evaluation/normalization_audit_cli.py
- Modify: tests/unit/evaluation/test_normalization_audit.py
- Modify: tests/integration/evaluation/test_normalization_audit_cli.py

**Interfaces:**
- Produce compare_releases(before_path, after_path, sample, labels) -> AuditComparison.
- Produce enforce_quality_gate(comparison, remediated_codes) -> None.
- Add normalization-audit compare --before PATH --after PATH --sample PATH --labels PATH --output-dir PATH.
- Produce comparison.json and comparison.md; return non-zero when any gate fails.

- [ ] Step 1: Write failing tests for changed canonical table IDs, table count not equal to 146,011, unresolved sample context, coverage below 0.90 and false-positive rate above 0.05.

~~~python
with pytest.raises(QualityGateError, match="canonical table IDs changed"):
    compare_releases(before_release, changed_release, sample, labels)
~~~

- [ ] Step 2: Run pytest tests/unit/evaluation/test_normalization_audit.py -k "compare or quality_gate" -v.

Expected: FAIL because comparison interfaces are absent.

- [ ] Step 3: Implement exact table-ID set comparison and 146,011 row invariant. Replay every original sample against the after release; a disappeared issue remains in the denominator as a corrected outcome, while missing or changed source context fails the gate.

- [ ] Step 4: Implement coverage and false-positive thresholds only for explicitly remediated codes. Issue-count reduction is reported but never used as acceptance.

- [ ] Step 5: Write deterministic JSON/Markdown with fingerprints, before/after counts, coverage, false-positive rates, uncertain counts, cause distribution, rule summary, invariants and failed checks.

- [ ] Step 6: Run unit and CLI integration tests. Expected: PASS.

- [ ] Step 7: Commit with message feat: enforce normalization remediation quality gate.

### Task 7: Rebuild and final verification

**Files:**
- Generate: a new processed release and artifacts/normalization-audit reports.
- Verify: all source, tests, documentation and release invariants.

- [ ] Step 1: Run targeted tests.

~~~powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& .\.venv\Scripts\python.exe -m pytest tests/unit/normalization tests/unit/evaluation/test_normalization_audit.py tests/integration/evaluation/test_normalization_audit_cli.py -v
~~~

Expected: PASS.

- [ ] Step 2: Generate the deterministic sample from the immutable before release, complete labels, and run baseline. Preserve the release fingerprint in the report.

- [ ] Step 3: Build a fresh release using the documented dataset-builder command and a new output root. Never replace the before release.

Expected: exactly 146,011 canonical tables.

- [ ] Step 4: Run normalization-audit compare using the before release, after release, original sample and reviewed labels.

Expected: exit 0; coverage at least 90%, false-positive rate at most 5%, table count 146,011 and exact table-ID equality.

- [ ] Step 5: Run full verification.

~~~powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests
& .\.venv\Scripts\python.exe -m mypy src tests
~~~

Expected: all executable checks pass. Record Windows symlink skips as environmental when privilege is unavailable.

- [ ] Step 6: Commit source, tests, audit config and documentation only. Do not commit generated processed releases, labels or reports unless repository policy explicitly requires them.

Commit message: chore: verify normalization audit remediation.

## Plan Self-Review

- Tasks cover deterministic sampling, strict labeling, baseline reporting, conservative evidence gates, downstream eligibility, before/after gates and full release verification.
- Sampling, metric and comparison interfaces use the same names throughout.
- Acceptance is consistently at least 90% coverage, at most 5% false positives, exactly 146,011 rows and exact table-ID equality.
- There are no extraction changes, classifiers, count-driven suppression rules or deferred implementation markers.
