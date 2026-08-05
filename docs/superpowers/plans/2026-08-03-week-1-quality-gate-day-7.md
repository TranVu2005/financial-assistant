# Week 1 Quality Gate Day 7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, annotation-backed Week 1 gate that evaluates a 60-document ViFinQA pilot, verifies all accepted-cell provenance plus 30 manual traces, and publishes reproducible gate and Pareto reports.

**Architecture:** The evaluation package loads and validates one immutable canonical release, selects a stable 20-company × 3-document pilot, matches expert table annotations to observed tables, and evaluates usability and provenance with typed failure events. Three CLI phases—`prepare`, `sample-cells`, and `evaluate`—keep generated selection, human annotation, and final scoring distinct while binding every artifact to dataset and manifest fingerprints.

**Tech Stack:** Python 3.11, Pydantic 2, PyArrow/Parquet, NetworkX, orjson, pytest, Hypothesis, Ruff, mypy, uv.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-03-week-1-quality-gate-day-7-design.md` exactly.
- Treat raw TXT, source manifest, and canonical Parquet release as read-only.
- Use `SAMPLING_VERSION = "week1-pilot-v1"`; never use Python `hash()` or an unseeded PRNG.
- Default pilot size is exactly 20 companies × 3 documents = 60 documents.
- Require at least 30 annotations for each of `balance_sheet`, `income_statement`, and `cash_flow_statement`.
- Pass only at overall usability ≥85%, provenance validity exactly 100% with a positive denominator, 30/30 manual cell confirmations, and every eligible stratum ≥70%.
- Use integer comparisons for 85% and 70% pass/fail decisions; Wilson intervals are descriptive only.
- Emit no timestamp, hostname, absolute path, nondeterministic identifier, or locale-dependent value.
- Quality defects produce failure events and exit `1`; invalid inputs/workflow state produce typed errors and exit `2`.
- Commit annotations under `data/qa/week1_pilot/`; keep generated reports under ignored `data/interim/week1_gate/`.
- Preserve unrelated worktree changes, especially `src/financial_report_qa/data/dataset_builder.py`, `plan.md`, notebook files, `.agents/`, and the uncommitted Day 5-6 plan.
- Write a failing test before each implementation change and commit only the paths listed by that task.

---

## File Structure

### New application files

- `src/financial_report_qa/evaluation/week1_contracts.py`: Pydantic contracts, exact CSV columns, canonical CSV/JSON I/O, and stable annotation IDs.
- `src/financial_report_qa/evaluation/week1_dataset.py`: release/manifest loading, schema checks, identity validation, and indexed read-only rows.
- `src/financial_report_qa/evaluation/week1_sampling.py`: stable SHA-256 ranks, 20×3 document selection, and 30-cell stratified selection.
- `src/financial_report_qa/evaluation/week1_matching.py`: exact overlap scoring, one-to-one maximum-weight assignment, and table usability assessment.
- `src/financial_report_qa/evaluation/week1_provenance.py`: release-wide cell checks, source re-extraction comparison, and safe source excerpts.
- `src/financial_report_qa/evaluation/week1_pareto.py`: event collection, Pareto aggregation, and Wilson intervals.
- `src/financial_report_qa/evaluation/week1_gate.py`: prepare/sample/evaluate orchestration, report publication, and command parser.
- `scripts/week1_gate.py`: thin executable wrapper.

### Modified application files

- `src/financial_report_qa/evaluation/__init__.py`: export the public Week 1 gate API.
- `src/financial_report_qa/core/errors.py`: add gate-specific typed errors.
- `src/financial_report_qa/cli.py`: dispatch `week1-gate`.
- `README.md`: document the Day 7 workflow.
- `docs/development.md`: document annotation review and gate interpretation.

### New test files

- `tests/unit/evaluation/test_week1_contracts.py`
- `tests/unit/evaluation/test_week1_dataset.py`
- `tests/unit/evaluation/test_week1_sampling.py`
- `tests/unit/evaluation/test_week1_matching.py`
- `tests/unit/evaluation/test_week1_provenance.py`
- `tests/unit/evaluation/test_week1_pareto.py`
- `tests/unit/evaluation/test_week1_gate.py`
- `tests/integration/test_week1_gate.py`

### Test helper contracts

All underscore-prefixed helpers shown below are local test-data builders, defined above
their first test in the same file. They are not application interfaces:

- `test_week1_dataset.py`: `_write_release(tmp_path)` writes the exact five-artifact
  one-document release described in Task 2 and returns manifest path, release path,
  canonical document, table, and cell; `_mutate_release(release_path, mutation)` applies
  only the named corruption from the parameter table.
- `test_week1_sampling.py`: `_candidate_documents(company_count)` creates three ready,
  table-bearing canonical documents per company; `_cell_candidates(count)` creates
  candidates across at least 15 tables and six strata so the two-per-table cap can satisfy
  a 30-cell request.
- `test_week1_gate.py`: `_gate_dataset()`, `_pilot_documents()`,
  `_write_expected_rows()`, and `_complete_gate_case()` create internally consistent
  frozen models/files with the counts passed by each test.
- `test_week1_matching.py`: `_expected()` derives a valid annotation ID and returns a
  positive-shape `ExpectedTable`; `_table()` derives `table_id` with `stable_table_id()`;
  `_assessment_case()` starts from one valid usable pair and applies only the named
  mutation.
- `test_week1_provenance.py`: `_verified_source_case()` writes a real UTF-8 source,
  inventories it, extracts and normalizes it, then returns a matching in-memory
  `GateDataset`; no expected value is copied from the function under test.
- `test_week1_pareto.py`: `_event(code)` returns a `FailureEvent` with fixed canonical IDs.
- `test_week1_gate.py` and the integration test: `_report_bytes()` returns a filename-to-
  bytes mapping sorted by filename.
- `test_week1_gate.py`: `_complete_gate_case()` varies only table usability and manual
  verification counts while retaining all other valid inputs.
- `tests/integration/test_week1_gate.py`: `_build_60_document_case()`,
  `_fill_expected_tables_from_source_contract()`, and `_mark_all_cell_audits()` implement
  the exact synthetic workflow specified in Task 8.

---

### Task 1: Gate Contracts, Stable IDs, and Canonical CSV I/O

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_contracts.py`
- Modify: `src/financial_report_qa/core/errors.py`
- Test: `tests/unit/evaluation/test_week1_contracts.py`

**Interfaces:**
- Consumes: primitive CSV/JSON values and canonical IDs.
- Produces: `PilotMetadata`, `PilotDocument`, `ExpectedTable`, `CellAudit`, `FailureEvent`, `TableAssessment`, `GateCheck`, `GateResult`, `ParetoRow`, `stable_annotation_id()`, `read_csv_rows()`, `write_csv_rows()`, and `write_canonical_json()`.

- [ ] **Step 1: Write failing stable-ID and model-validation tests**

Create `tests/unit/evaluation/test_week1_contracts.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_report_qa.evaluation.week1_contracts import (
    EXPECTED_TABLE_COLUMNS,
    ExpectedTable,
    PilotDocument,
    read_csv_rows,
    stable_annotation_id,
    write_csv_rows,
)

DOC_ID = f"doc_{'a' * 64}"


def test_stable_annotation_id_uses_exact_canonical_payload() -> None:
    first = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    second = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    changed = stable_annotation_id(DOC_ID, 10, 21, "balance_sheet")
    assert first == second
    assert first.startswith("ann_")
    assert len(first) == 68
    assert changed != first


def test_expected_table_requires_derived_id_sorted_periods_and_positive_shape() -> None:
    annotation_id = stable_annotation_id(DOC_ID, 10, 20, "balance_sheet")
    expected = ExpectedTable(
        annotation_schema_version="1",
        annotation_id=annotation_id,
        doc_id=DOC_ID,
        relative_path="VCB/2024/Consolidated/report.txt",
        statement_type="balance_sheet",
        line_start=10,
        line_end=20,
        row_count=5,
        column_count=3,
        unit_normalized="VND_million",
        expected_periods=("2023", "2024"),
        notes="",
    )
    assert expected.expected_periods == ("2023", "2024")

    invalid_payload = expected.model_dump()
    invalid_payload["expected_periods"] = ("2024", "2023")
    with pytest.raises(ValidationError, match="sorted and duplicate-free"):
        ExpectedTable.model_validate(invalid_payload)


def test_pilot_document_rejects_unsafe_relative_path() -> None:
    with pytest.raises(ValidationError, match="safe POSIX"):
        PilotDocument(
            annotation_schema_version="1",
            dataset_fingerprint="b" * 64,
            source_manifest_sha256="c" * 64,
            doc_id=DOC_ID,
            relative_path="../report.txt",
            company_code="VCB",
            report_year=2024,
            statement_scope="consolidated",
        )
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_contracts.py
```

Expected: collection fails with `ModuleNotFoundError` for `week1_contracts`.

- [ ] **Step 3: Add the typed error hierarchy**

Append to `core/errors.py`:

```python
class Week1GateError(FinancialReportQAError):
    """Base class for expected Week 1 gate workflow failures."""


class Week1GateInputError(Week1GateError):
    """Gate inputs, annotations, or release identity are invalid."""


class Week1GateSourceError(Week1GateError):
    """A source document cannot be re-verified against the manifest."""


class Week1GatePublicationError(Week1GateError):
    """Gate artifacts cannot be safely verified or published."""
```

- [ ] **Step 4: Implement immutable contracts and exact vocabularies**

Use `ConfigDict(extra="forbid", frozen=True)` for every Pydantic model. Define:

```python
SAMPLING_VERSION = "week1-pilot-v1"
ANNOTATION_SCHEMA_VERSION = "1"
StatementType = Literal[
    "balance_sheet", "income_statement", "cash_flow_statement"
]
GateFailureCode = Literal[
    "missing_table",
    "span_mismatch",
    "shape_mismatch",
    "statement_mismatch",
    "unit_mismatch",
    "period_mismatch",
    "no_numeric_value",
    "invalid_provenance",
    "manual_provenance_failure",
    "unclosed_html_table",
    "nested_html_table",
    "unsupported_html_structure",
    "invalid_span_value",
    "span_collision",
    "expansion_limit_exceeded",
    "ragged_structured_rows",
    "insufficient_structural_evidence",
    "empty_extracted_table",
    "company_conflict",
    "period_incomplete",
    "period_ambiguous",
    "period_invalid",
    "statement_conflict",
    "metric_unknown",
    "number_missing",
    "number_ambiguous",
    "number_invalid",
    "unit_unknown",
    "unit_conflict",
]
```

Define the exact CSV constants from the spec. Model `expected_periods` as
`tuple[str, ...]`, but serialize it as `|`-joined text. `CellAudit.verified` is
`bool | None`; CSV accepts only empty, `true`, or `false`.

Define `FailureEvent(code, doc_id, annotation_id, table_id, cell_id)`;
`TableAssessment(annotation, table_id, overlap_numerator, overlap_denominator,
failures, usable)`; `GateCheck(name, passed, numerator, denominator, threshold_percent)`;
and `ParetoRow(rank, code, count, share, cumulative_share)`. `GateResult` contains input
hashes, counts, statement metrics, stratum metrics, checks, Pareto rows, and `passed`.

- [ ] **Step 5: Implement stable annotation IDs and canonical serializers**

```python
def stable_annotation_id(
    doc_id: str,
    line_start: int,
    line_end: int,
    statement_type: StatementType,
) -> str:
    payload = f"{doc_id}\n{line_start}\n{line_end}\n{statement_type}".encode()
    return f"ann_{hashlib.sha256(payload).hexdigest()}"
```

`read_csv_rows(path, expected_columns)` must decode UTF-8 strictly, require LF/final
newline, reject duplicate/missing/extra headers, use `csv.DictReader(newline="")`, and
return ordered dictionaries with string values. `write_csv_rows()` uses
`csv.DictWriter(stream, fieldnames=columns, lineterminator="\n",
extrasaction="raise")`, a same-parent temporary
file, flush plus `os.fsync`, and refuses to overwrite unless `allow_identical=True` and
the resulting bytes match exactly. `write_canonical_json()` uses
`orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE`.

- [ ] **Step 6: Add CSV round-trip and corruption tests**

```python
def test_expected_table_csv_is_byte_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "expected-tables.csv"
    row = {
        "annotation_schema_version": "1",
        "annotation_id": stable_annotation_id(DOC_ID, 10, 20, "balance_sheet"),
        "doc_id": DOC_ID,
        "relative_path": "VCB/2024/Consolidated/report.txt",
        "statement_type": "balance_sheet",
        "line_start": "10",
        "line_end": "20",
        "row_count": "5",
        "column_count": "3",
        "unit_normalized": "VND_million",
        "expected_periods": "2023|2024",
        "notes": "Unicode: kiểm toán",
    }
    write_csv_rows(path, EXPECTED_TABLE_COLUMNS, (row,))
    first = path.read_bytes()
    write_csv_rows(path, EXPECTED_TABLE_COLUMNS, (row,), allow_identical=True)
    assert path.read_bytes() == first
    assert read_csv_rows(path, EXPECTED_TABLE_COLUMNS) == (row,)


@pytest.mark.parametrize(
    "raw",
    [
        b"wrong,header\n",
        b"annotation_schema_version,annotation_schema_version\n",
        b"annotation_schema_version\r\n",
        b"annotation_schema_version",
    ],
)
def test_csv_reader_rejects_contract_drift(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(raw)
    with pytest.raises(Week1GateInputError):
        read_csv_rows(path, EXPECTED_TABLE_COLUMNS)
```

- [ ] **Step 7: Run Task 1 checks**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_contracts.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_contracts.py src/financial_report_qa/core/errors.py tests/unit/evaluation/test_week1_contracts.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_contracts.py src/financial_report_qa/core/errors.py tests/unit/evaluation/test_week1_contracts.py
```

- [ ] **Step 8: Commit Task 1**

```powershell
git add src/financial_report_qa/core/errors.py src/financial_report_qa/evaluation/week1_contracts.py tests/unit/evaluation/test_week1_contracts.py
git commit -m "feat: define week one gate contracts"
```

---

### Task 2: Immutable Release Loader and Identity Validation

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_dataset.py`
- Test: `tests/unit/evaluation/test_week1_dataset.py`

**Interfaces:**
- Consumes: `read_manifest()`, one release directory, current builder Arrow schemas, and `manifest.json`.
- Produces: frozen `GateDataset` and `load_gate_dataset(manifest_path, release_path) -> GateDataset`.

- [ ] **Step 1: Write a minimal valid release fixture and failing loader test**

In `test_week1_dataset.py`, build a one-document `InventoryResult`, write it with
`write_manifest()`, create rows through `DOCUMENT_SCHEMA`, `TABLE_SCHEMA`, `CELL_SCHEMA`,
and `ISSUE_SCHEMA`, and write the five exact builder artifacts. Use
`stable_document_id()`, `stable_table_id()`, and `stable_cell_id()`.

```python
def test_load_gate_dataset_indexes_verified_release(tmp_path: Path) -> None:
    manifest_path, release_path, document, table, cell = _write_release(tmp_path)

    dataset = load_gate_dataset(manifest_path, release_path)

    assert dataset.dataset_fingerprint == "f" * 64
    assert dataset.source_manifest_sha256 == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert dataset.documents_by_id == {document.doc_id: document}
    assert dataset.tables_by_id == {table.table_id: table}
    assert dataset.cells_by_table_id == {table.table_id: (cell,)}
```

The helper writes `manifest.json` with schema version, fingerprint, source hash, and exact
document/table/cell/issue counts; no fake path or noncanonical ID is permitted.

- [ ] **Step 2: Run the focused test and confirm missing-module failure**

Run `uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_dataset.py`.

- [ ] **Step 3: Implement `GateDataset` and Parquet row reconstruction**

```python
@dataclass(frozen=True)
class GateDataset:
    dataset_fingerprint: str
    source_manifest_sha256: str
    release_path: Path
    manifest: ManifestSnapshot
    documents_by_id: dict[str, DocumentRecord]
    tables_by_id: dict[str, TableRecord]
    cells_by_table_id: dict[str, tuple[CellRecord, ...]]
    issues: tuple[NormalizationIssue, ...]
```

Read each Parquet file with `pq.read_table()`, require exact equality with the current
`DOCUMENT_SCHEMA`, `TABLE_SCHEMA`, `CELL_SCHEMA`, and `ISSUE_SCHEMA`, convert via
`to_pylist()`, and validate rows with the canonical Pydantic models. The document Parquet
row omits `notes`; compare its released fields to the ready manifest `DocumentRecord`
rather than constructing a second partial document model.

- [ ] **Step 4: Validate release identity and relational integrity**

Require:

- safe existing release directory;
- exact files `documents.parquet`, `tables.parquet`, `cells.parquet`, `issues.parquet`, and
  `manifest.json`;
- lowercase 64-hex dataset/source fingerprints;
- source fingerprint equals exact manifest bytes;
- declared counts equal Parquet counts;
- one release document row per ready manifest document and no other row;
- unique `doc_id`, `table_id`, and `cell_id`;
- every table references a released document;
- every cell references a released table;
- every normalization issue references its declared document and optional table/cell.

Return maps and tuples in stable ID order. Errors use `Week1GateInputError` and mention
artifact name or stable ID, never the absolute path.

- [ ] **Step 5: Add parameterized corruption tests**

Cover wrong Arrow schema, missing file, malformed JSON, source-fingerprint mismatch,
wrong count, duplicate ID, dangling table/cell/issue reference, released non-ready
document, and a document row that disagrees with company/year/path/digest.

```python
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_cells", "cells.parquet"),
        ("source_hash", "source manifest fingerprint"),
        ("table_count", "table count"),
        ("dangling_cell", "unknown table_id"),
    ],
)
def test_load_gate_dataset_fails_closed_on_release_corruption(
    tmp_path: Path, mutation: str, message: str
) -> None:
    manifest_path, release_path, _, _, _ = _write_release(tmp_path)
    _mutate_release(release_path, mutation)
    with pytest.raises(Week1GateInputError, match=message):
        load_gate_dataset(manifest_path, release_path)
```

- [ ] **Step 6: Run Task 2 checks and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_dataset.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_dataset.py tests/unit/evaluation/test_week1_dataset.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_dataset.py tests/unit/evaluation/test_week1_dataset.py
git add src/financial_report_qa/evaluation/week1_dataset.py tests/unit/evaluation/test_week1_dataset.py
git commit -m "feat: load verified canonical releases for evaluation"
```

---

### Task 3: Stable 20×3 Pilot Selection and Prepare Phase

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_sampling.py`
- Create: `src/financial_report_qa/evaluation/week1_gate.py`
- Test: `tests/unit/evaluation/test_week1_sampling.py`
- Test: `tests/unit/evaluation/test_week1_gate.py`

**Interfaces:**
- Consumes: `GateDataset`.
- Produces: `stable_rank()`, `select_pilot_documents()`, and `prepare_pilot()` plus the first `pilot-metadata.json`, `pilot-documents.csv`, and header-only `expected-tables.csv`.

- [ ] **Step 1: Write deterministic rank and selection tests**

Create synthetic documents for 22 companies, each with three years and two scopes. Mark
one table per document in `table_doc_ids`.

```python
def test_stable_rank_is_namespaced_and_repeatable() -> None:
    assert stable_rank("company", "VCB") == stable_rank("company", "VCB")
    assert stable_rank("company", "VCB") != stable_rank("document", "VCB")
    assert len(stable_rank("company", "VCB")) == 64


def test_select_pilot_is_20_by_3_and_input_order_independent() -> None:
    documents, table_doc_ids = _candidate_documents(company_count=22)
    forward = select_pilot_documents(documents, table_doc_ids)
    reverse = select_pilot_documents(tuple(reversed(documents)), table_doc_ids)

    assert forward == reverse
    assert len(forward) == 60
    counts = Counter(item.company_code for item in forward)
    assert len(counts) == 20
    assert set(counts.values()) == {3}
    assert len({item.doc_id for item in forward}) == 60
```

Add tests showing distinct `(report_year, statement_scope)` buckets are preferred,
documents without tables are excluded, companies with fewer than three eligible documents
are excluded, and fewer than 20 eligible companies raises `Week1GateInputError`.

- [ ] **Step 2: Implement exact SHA-256 ranks and two-pass selection**

```python
def stable_rank(namespace: str, *parts: object) -> str:
    payload = "\n".join((SAMPLING_VERSION, namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

`select_pilot_documents()` accepts keyword-only `company_count=20` and
`documents_per_company=3` so small synthetic tests can use the same production algorithm.
Filter ready released documents with tables; rank eligible companies; choose the first
20; then execute the spec's stratum pass and fill pass. Convert selections to
`PilotDocument` and sort by the exact final tuple.

- [ ] **Step 3: Write the prepare-phase test**

```python
def test_prepare_pilot_writes_immutable_selection_and_empty_template(
    tmp_path: Path,
) -> None:
    dataset = _gate_dataset(company_count=20, documents_per_company=3)
    annotation_root = tmp_path / "annotations"

    metadata = prepare_pilot(dataset, annotation_root)

    documents = read_csv_rows(
        annotation_root / "pilot-documents.csv", PILOT_DOCUMENT_COLUMNS
    )
    assert len(documents) == 60
    assert read_csv_rows(
        annotation_root / "expected-tables.csv", EXPECTED_TABLE_COLUMNS
    ) == ()
    assert metadata.pilot_documents_sha256 == hashlib.sha256(
        (annotation_root / "pilot-documents.csv").read_bytes()
    ).hexdigest()
    assert json.loads((annotation_root / "pilot-metadata.json").read_text())[
        "document_count"
    ] == 60
```

Also call `prepare_pilot()` a second time and assert it raises without changing any byte.

- [ ] **Step 4: Implement prepare without overwriting review work**

`prepare_pilot(dataset, annotation_root, *, company_count=20,
documents_per_company=3) -> PilotMetadata` must require the target to be absent or empty,
create all output under a same-parent temporary directory, serialize selection and empty
template, hash `pilot-documents.csv`, write metadata, re-read all three files, then rename
the directory into place. Cleanup only a resolved sibling directory with prefix
`.week1-prepare-`.

- [ ] **Step 5: Run Task 3 checks and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
git add src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
git commit -m "feat: prepare deterministic week one pilot"
```

---

### Task 4: Annotation Validation, Exact Table Matching, and Usability

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_matching.py`
- Modify: `src/financial_report_qa/evaluation/week1_gate.py`
- Test: `tests/unit/evaluation/test_week1_matching.py`
- Modify: `tests/unit/evaluation/test_week1_gate.py`

**Interfaces:**
- Consumes: pilot rows, expected-table rows, `GateDataset`, and invalid-provenance cell IDs.
- Produces: `load_expected_tables()`, `match_tables()`, and `assess_tables()` returning stable `TableAssessment` values.

- [ ] **Step 1: Write strict annotation-set tests**

```python
def test_load_expected_tables_requires_pilot_membership_and_type_coverage(
    tmp_path: Path,
) -> None:
    pilot = _pilot_documents(company_count=2, documents_per_company=2)
    path = tmp_path / "expected-tables.csv"
    _write_expected_rows(path, pilot, per_type=2)

    rows = load_expected_tables(
        path,
        pilot,
        minimum_per_statement=2,
    )

    assert Counter(row.statement_type for row in rows) == {
        "balance_sheet": 2,
        "income_statement": 2,
        "cash_flow_statement": 2,
    }
```

Add failures for unknown document/path mismatch, derived annotation ID mismatch, duplicate
annotation ID, overlapping same-family spans in one document, out-of-file lines, unsorted
periods, and one statement family below its minimum. File line bounds come from verified
source line counts passed as `source_line_counts`.

- [ ] **Step 2: Write overlap and one-to-one matching tests**

```python
@pytest.mark.parametrize(
    ("expected_span", "observed_span", "numerator", "denominator"),
    [
        ((10, 20), (10, 20), 11, 11),
        ((10, 20), (12, 20), 9, 11),
        ((10, 20), (20, 30), 1, 11),
        ((10, 20), (21, 30), 0, 11),
    ],
)
def test_span_overlap_is_inclusive_and_exact(
    expected_span: tuple[int, int],
    observed_span: tuple[int, int],
    numerator: int,
    denominator: int,
) -> None:
    assert span_overlap(expected_span, observed_span) == (numerator, denominator)


def test_matching_never_reuses_one_observed_table() -> None:
    DOC_ID = f"doc_{'a' * 64}"
    annotations = (
        _expected(DOC_ID, 10, 20, "balance_sheet"),
        _expected(DOC_ID, 21, 30, "income_statement"),
    )
    observed = (_table(DOC_ID, 10, 30),)
    matches = match_tables(annotations, observed)
    assert sum(item.table_id is not None for item in matches) == 1


def test_matching_uses_lexicographic_table_id_for_an_exact_tie() -> None:
    DOC_ID = f"doc_{'a' * 64}"
    annotation = _expected(DOC_ID, 10, 20, "balance_sheet")
    observed = (_table(DOC_ID, 9, 20), _table(DOC_ID, 10, 21))
    expected_table_id = min(table.table_id for table in observed)
    assert match_tables((annotation,), observed)[0].table_id == expected_table_id
```

- [ ] **Step 3: Implement exact composite weights**

Within each document, sort annotations by `annotation_id` and tables by `table_id`. Add
an edge only for positive overlap. Compute the LCM of annotation span lengths so overlap
fractions become exact integers. Pair quality is `2 * lcm` for exact span equality and
`intersection * lcm // annotation_span` otherwise.

Encode the objective hierarchy without floats:

```python
base = len(tables) + 1
lex_bound = base ** len(annotations)
max_distance = max(
    abs(annotation.line_start - table.line_start)
    + abs(annotation.line_end - table.line_end)
    for annotation in annotations
    for table in tables
)
quality_scale = (len(annotations) * max_distance + 1) * lex_bound

weight = (
    pair_quality * quality_scale
    + (max_distance - boundary_distance) * lex_bound
    + (len(tables) - table_index)
    * base ** (len(annotations) - annotation_index - 1)
)
```

Insert nodes/edges in sorted order and call
`networkx.max_weight_matching(graph, maxcardinality=False, weight="weight")`. Convert the
set result back to annotation order. Unmatched annotations have `table_id=None` and zero
overlap. Test exact span preference, greater overlap, lower boundary distance, lexical
table ID, and independence from input order.

- [ ] **Step 4: Write usability predicate tests**

```python
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("no_match", "missing_table"),
        ("overlap_79_percent", "span_mismatch"),
        ("wrong_shape", "shape_mismatch"),
        ("wrong_statement", "statement_mismatch"),
        ("wrong_unit", "unit_mismatch"),
        ("missing_period", "period_mismatch"),
        ("no_numeric", "no_numeric_value"),
        ("bad_cell", "invalid_provenance"),
    ],
)
def test_assessment_retains_each_applicable_failure(
    mutation: str, expected_code: str
) -> None:
    annotation, match, table, cells, invalid_cells = _assessment_case(mutation)
    assessment = assess_table(annotation, match, table, cells, invalid_cells)
    assert expected_code in {failure.code for failure in assessment.failures}
    assert not assessment.usable
```

Add one case with shape, unit, and period wrong simultaneously and assert all three events
exist. Implement the 80% test as
`overlap_numerator * 100 >= overlap_denominator * 80`; no floating decision is allowed.

- [ ] **Step 5: Run Task 4 checks and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_matching.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_matching.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_matching.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_matching.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_matching.py tests/unit/evaluation/test_week1_gate.py
git add src/financial_report_qa/evaluation/week1_matching.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_matching.py tests/unit/evaluation/test_week1_gate.py
git commit -m "feat: match and assess annotated financial tables"
```

---

### Task 5: Automated Provenance and Stratified 30-Cell Audit

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_provenance.py`
- Modify: `src/financial_report_qa/evaluation/week1_sampling.py`
- Modify: `src/financial_report_qa/evaluation/week1_gate.py`
- Test: `tests/unit/evaluation/test_week1_provenance.py`
- Modify: `tests/unit/evaluation/test_week1_sampling.py`
- Modify: `tests/unit/evaluation/test_week1_gate.py`

**Interfaces:**
- Consumes: matched pilot tables, `GateDataset`, snapshot root, and source manifest records.
- Produces: `ProvenanceAudit`, `audit_provenance()`, `select_audit_cells()`, `source_excerpt()`, and `sample_audit_cells()` writing `cell-audit.csv`.

- [ ] **Step 1: Define provenance result contracts and write structural tests**

Add frozen models in `week1_contracts.py`:

```python
class ProvenanceFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    doc_id: str
    table_id: str
    cell_id: str
    reason: str


class RejectionFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    doc_id: str
    code: GateFailureCode
    line_start: int
    line_end: int


class ProvenanceAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accepted_cell_ids: tuple[str, ...]
    valid_cell_ids: tuple[str, ...]
    findings: tuple[ProvenanceFinding, ...]
    rejections: tuple[RejectionFinding, ...]
```

Write tests for malformed cell ID, duplicate `(table_id,row_idx,col_idx,cell_id)`, unknown
table, out-of-bounds coordinates, reversed/out-of-table line span, and safe path escape.

- [ ] **Step 2: Write source re-extraction equality tests**

Create one real UTF-8 temporary TXT, matching `DocumentRecord`, extraction, normalized
Parquet-style cell, and table. Assert `audit_provenance()` accepts it. Then independently
mutate `value_raw`, `value_numeric`, `period`, unit, and source line; each must produce one
`ProvenanceFinding` for canonical drift. Change the source byte and assert
`Week1GateSourceError` without an absolute path in the message.

```python
def test_audit_provenance_reextracts_and_compares_canonical_cell(tmp_path: Path) -> None:
    dataset, snapshot_root, table_id, cell_id = _verified_source_case(tmp_path)
    audit = audit_provenance(dataset, snapshot_root, (table_id,))
    assert audit.accepted_cell_ids == (cell_id,)
    assert audit.valid_cell_ids == (cell_id,)
    assert audit.findings == ()
```

- [ ] **Step 3: Implement one re-extraction per document**

Validate all cheap relational/span rules first. Group matched table IDs by document. For
each document call `extract_document(snapshot_root, document)` once, then
`normalize_extraction(document, extraction)` once. Index regenerated cells and compare
these exact fields with release cells:

```python
CELL_COMPARISON_FIELDS = (
    "cell_id",
    "table_id",
    "row_idx",
    "col_idx",
    "row_label_raw",
    "row_label_canonical",
    "column_label_raw",
    "column_label_canonical",
    "value_raw",
    "value_numeric",
    "period",
    "unit",
    "source_line_start",
    "source_line_end",
    "extraction_confidence",
)
```

Convert ingestion rejections from pilot documents to `RejectionFinding` values sorted by
`(doc_id, line_start, line_end, code)`. A source integrity/read error becomes
`Week1GateSourceError`; cell-level mismatch remains an `invalid_provenance` finding so
evaluation can complete. Emit one table failure event per distinct invalid cell, carrying
that `cell_id`, so the Pareto measures provenance defects rather than only affected tables.

- [ ] **Step 4: Write cell sampling properties**

```python
@given(order=st.permutations(tuple(range(40))))
def test_cell_sample_is_input_order_independent(order: tuple[int, ...]) -> None:
    candidates = _cell_candidates(40)
    shuffled = tuple(candidates[index] for index in order)
    assert select_audit_cells(shuffled, sample_size=30, max_per_table=2) == (
        select_audit_cells(candidates, sample_size=30, max_per_table=2)
    )


def test_cell_sample_is_stratified_unique_and_table_capped() -> None:
    selected = select_audit_cells(_cell_candidates(60), sample_size=30, max_per_table=2)
    assert len(selected) == 30
    assert len({item.cell.cell_id for item in selected}) == 30
    assert max(Counter(item.cell.table_id for item in selected).values()) <= 2
```

Add insufficient-eligible-cell failure. `AuditCellCandidate` contains the cell, company,
year, annotation statement type, relative path, and annotation ID.

- [ ] **Step 5: Implement round-robin selection and exact excerpts**

Bucket candidates by `(company_code, report_year, statement_type)`. Rank buckets with
`stable_rank("cell-stratum", *bucket)` and cells with
`stable_rank("cell", cell_id, table_id)`. Repeatedly visit buckets in ranked order,
selecting the next candidate whose table has fewer than two selections, until 30 are
selected or progress stops. Raise `Week1GateInputError` if progress stops early.

Use `read_document(snapshot_root, document)` for verified lines. Excerpt logic is:

```python
text = "\n".join(
    line.text
    for line in decoded.lines[cell.source_line_start - 1 : cell.source_line_end]
)
excerpt = text if len(text) <= 500 else f"{text[:497]}..."
```

- [ ] **Step 6: Implement `sample_audit_cells()` safely**

Validate metadata, pilot hash, completed expected tables, matches, and automated
provenance. Select only non-empty cells in tables whose current assessment is usable.
Write exactly 30 `CellAudit` rows with `verified=None` and empty notes. Refuse to overwrite
an existing `cell-audit.csv`, write by temporary file, re-read it, and leave
`pilot-metadata.json` unchanged.

- [ ] **Step 7: Run Task 5 checks and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_provenance.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_provenance.py src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_provenance.py src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation
git add src/financial_report_qa/evaluation/week1_contracts.py src/financial_report_qa/evaluation/week1_provenance.py src/financial_report_qa/evaluation/week1_sampling.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_provenance.py tests/unit/evaluation/test_week1_sampling.py tests/unit/evaluation/test_week1_gate.py
git commit -m "feat: audit canonical cell provenance"
```

---

### Task 6: Gate Metrics, Wilson Intervals, and Error Pareto

**Files:**
- Create: `src/financial_report_qa/evaluation/week1_pareto.py`
- Modify: `src/financial_report_qa/evaluation/week1_gate.py`
- Test: `tests/unit/evaluation/test_week1_pareto.py`
- Modify: `tests/unit/evaluation/test_week1_gate.py`

**Interfaces:**
- Consumes: annotations, assessments, provenance audit, completed cell audit, release issues, and re-extraction rejection codes.
- Produces: `wilson_interval()`, `build_failure_events()`, `build_pareto()`, and `calculate_gate_result() -> GateResult`.

- [ ] **Step 1: Write exact threshold-boundary tests**

```python
@pytest.mark.parametrize(
    ("usable", "annotated", "threshold", "passed"),
    [
        (85, 100, 85, True),
        (84, 100, 85, False),
        (17, 20, 85, True),
        (7, 10, 70, True),
        (6, 10, 70, False),
        (0, 0, 85, False),
    ],
)
def test_percentage_gate_uses_integer_arithmetic(
    usable: int, annotated: int, threshold: int, passed: bool
) -> None:
    assert percentage_passes(usable, annotated, threshold) is passed
```

Add `calculate_gate_result()` cases for exactly 60 documents, per-type minimum 30,
positive accepted-cell denominator, exact 100% provenance, 30/30 manual true, overall
85%, eligible stratum 70%, and exclusion of strata with only nine tables.

- [ ] **Step 2: Implement descriptive Wilson intervals**

Use `z = 1.959963984540054`, return `(0.0, 0.0)` for zero denominator, and round each
bound to six decimal places only at serialization:

```python
def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return center - margin, center + margin
```

- [ ] **Step 3: Write failure-event and Pareto tests**

```python
def test_pareto_sorts_count_desc_then_code_and_accumulates() -> None:
    events = (
        _event("shape_mismatch"),
        _event("unit_mismatch"),
        _event("shape_mismatch"),
        _event("period_mismatch"),
    )
    rows = build_pareto(events)
    assert [(row.rank, row.code, row.count) for row in rows] == [
        (1, "shape_mismatch", 2),
        (2, "period_mismatch", 1),
        (3, "unit_mismatch", 1),
    ]
    assert rows[-1].cumulative_share == Decimal("1.000000")


def test_empty_pareto_has_no_rows() -> None:
    assert build_pareto(()) == ()
```

Use `Decimal(count) / Decimal(total)` with `quantize(Decimal("0.000001"),
rounding=ROUND_HALF_EVEN)`. Set the last cumulative share exactly to `1.000000` to avoid
accumulated rounding drift.

- [ ] **Step 4: Implement complete failure event collection**

Collect every `TableAssessment.failure`; one `manual_provenance_failure` per false manual
row; one event per re-extraction rejection; and every normalization issue whose table ID
belongs to a matched pilot table. Preserve distinct events before aggregation. Sort events
by `(code, doc_id, annotation_id, table_id, cell_id)` with `None` before strings.

- [ ] **Step 5: Implement exact gate result calculation**

Create checks named:

```text
pilot_document_count
statement_type_coverage
overall_table_usability
accepted_cell_provenance
manual_cell_audit
eligible_strata_usability
```

The result passes only when `all(check.passed for check in checks)`. Record exact SHA-256
of `pilot-documents.csv`, `expected-tables.csv`, and completed `cell-audit.csv`, plus
dataset/source fingerprints. Sort statement metrics by fixed vocabulary order and strata
by `(company_code, report_year, statement_type)`.

- [ ] **Step 6: Run Task 6 checks and commit**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation/test_week1_pareto.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation/week1_pareto.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_pareto.py tests/unit/evaluation/test_week1_gate.py
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation/week1_pareto.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_pareto.py tests/unit/evaluation/test_week1_gate.py
git add src/financial_report_qa/evaluation/week1_pareto.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_pareto.py tests/unit/evaluation/test_week1_gate.py
git commit -m "feat: calculate week one gate and error pareto"
```

---

### Task 7: Deterministic Evaluate Phase and Report Publication

**Files:**
- Modify: `src/financial_report_qa/evaluation/week1_gate.py`
- Modify: `src/financial_report_qa/evaluation/__init__.py`
- Modify: `tests/unit/evaluation/test_week1_gate.py`

**Interfaces:**
- Consumes: all validated gate inputs and completed annotation files.
- Produces: `evaluate_gate() -> GateResult`, `gate-result.json`, `gate-report.md`, and `pareto-errors.csv` under `report_root / dataset_fingerprint`.

- [ ] **Step 1: Write pass/fail evaluate-phase tests**

```python
def test_evaluate_gate_publishes_deterministic_reports(tmp_path: Path) -> None:
    inputs = _complete_gate_case(tmp_path, usable=85, annotated=100, manual_true=30)
    first = evaluate_gate(**inputs)
    first_bytes = _report_bytes(first.report_path)

    second = evaluate_gate(**inputs)

    assert first.result.passed
    assert second.result == first.result
    assert _report_bytes(second.report_path) == first_bytes


def test_evaluate_gate_returns_valid_failed_result(tmp_path: Path) -> None:
    inputs = _complete_gate_case(tmp_path, usable=84, annotated=100, manual_true=30)
    outcome = evaluate_gate(**inputs)
    assert not outcome.result.passed
    assert "overall_table_usability" in {
        check.name for check in outcome.result.checks if not check.passed
    }
```

`GateEvaluation` is a frozen model with `result: GateResult` and `report_path: Path`.

- [ ] **Step 2: Write invalid workflow and no-mutation tests**

Test metadata/release fingerprint mismatch, changed pilot CSV hash, missing expected file,
incomplete cell audit, duplicate audit cell, audit row outside the deterministic sample,
and `verified` empty. Snapshot every annotation byte before each failure and assert exact
equality afterward.

- [ ] **Step 3: Implement final workflow validation**

`evaluate_gate()` must:

1. load gate dataset and immutable `PilotMetadata`;
2. verify `pilot-documents.csv` hash against metadata;
3. validate exactly 60 pilot rows and identity against release;
4. validate expected tables and minimum type coverage;
5. match and audit provenance;
6. assess tables using the provenance findings;
7. recompute the deterministic 30-cell sample from usable assessments;
8. require `cell-audit.csv` to contain exactly those cell IDs and immutable generated
   fields, with only `verified` and `review_notes` allowed to differ;
9. calculate checks, events, Pareto, and `GateResult`;
10. serialize and verify reports before publication.

- [ ] **Step 4: Implement deterministic JSON, Markdown, and Pareto CSV**

`gate-result.json` is `GateResult.model_dump(mode="json")` through canonical JSON.
`pareto-errors.csv` uses exact columns
`rank,code,count,share,cumulative_share`; an empty Pareto writes only the header.

`gate-report.md` has fixed sections in this order:

```markdown
# Week 1 Quality Gate

## Decision
## Input Identity
## Gate Checks
## Statement Coverage
## Eligible Strata
## Pareto Errors
```

Render numeric rates to two decimal percentage places and Wilson bounds to six decimals.
Do not include execution time or absolute paths.

- [ ] **Step 5: Implement safe idempotent publication**

Write all reports beneath a sibling temporary directory with prefix `.week1-report-`,
flush files, re-read and validate JSON/CSV, then rename to
`report_root / dataset_fingerprint`. If the destination exists, compare every filename and
byte: reuse only an identical directory; otherwise raise `Week1GatePublicationError`.
Cleanup only a resolved temporary child of `report_root` with the private prefix.

- [ ] **Step 6: Export the public API and run checks**

Export only these application entry points from `evaluation/__init__.py`:

```python
__all__ = (
    "GateEvaluation",
    "evaluate_gate",
    "prepare_pilot",
    "sample_audit_cells",
)
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation tests/unit/evaluation
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation tests/unit/evaluation
```

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/financial_report_qa/evaluation/__init__.py src/financial_report_qa/evaluation/week1_gate.py tests/unit/evaluation/test_week1_gate.py
git commit -m "feat: publish deterministic week one gate reports"
```

---

### Task 8: Product CLI, End-to-End Pilot Workflow, and Documentation

**Files:**
- Create: `scripts/week1_gate.py`
- Modify: `src/financial_report_qa/evaluation/week1_gate.py`
- Modify: `src/financial_report_qa/cli.py`
- Modify: `tests/unit/test_cli.py`
- Create: `tests/integration/test_week1_gate.py`
- Modify: `README.md`
- Modify: `docs/development.md`

**Interfaces:**
- Consumes: `financial-report-qa week1-gate prepare|sample-cells|evaluate` arguments.
- Produces: exit `0` on successful phase/pass, exit `1` only for a valid failed evaluation, and exit `2` for invalid arguments/input/source/publication.

- [ ] **Step 1: Write product dispatcher test**

Add dependency injection following existing CLI style:

```python
def test_week1_gate_forwards_arguments() -> None:
    received: list[str] = []

    def fake_gate_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 0

    exit_code = main(
        ["week1-gate", "prepare", "--release", "data/processed/release"],
        gate_main_fn=fake_gate_main,
    )

    assert exit_code == 0
    assert received == ["prepare", "--release", "data/processed/release"]
```

- [ ] **Step 2: Implement one parser with three explicit subcommands**

Required common arguments are `--manifest`, `--snapshot-root`, `--release`, and
`--annotation-root`; `evaluate` additionally accepts `--report-root`, defaulting to
`data/interim/week1_gate`. Do not expose production thresholds, sample counts, or sampling
version as CLI overrides.

The command catches `Week1GateError`, `OSError`, and Pydantic `ValidationError`, prints
`error: source manifest fingerprint mismatch` to stderr for that representative failure,
and returns `2`. `evaluate` prints the dataset
fingerprint, six named checks, top five Pareto rows, and report-relative path; it returns
`0 if result.passed else 1`. Other successful phases return `0` and print counts plus
fingerprints.

Create the wrapper:

```python
from financial_report_qa.evaluation.week1_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write a real 60-document integration fixture**

Generate 20 ASCII company codes `C00` through `C19`, each with 2022, 2023, and 2024
documents. Every UTF-8 TXT contains exactly three HTML tables titled in English:

```text
Balance Sheet
Income Statement
Cash Flow Statement
```

Each table has header `Metric | 2024` for the 2024 fixture (using the corresponding year
for the other fixtures), one controlled English metric row, and one numeric
value. Use the existing inventory writer and `build_dataset()` to produce the real release;
derive expected annotation spans/counts from the known fixture construction, not from the
release being evaluated.

- [ ] **Step 4: Exercise the full passing workflow**

```python
def test_week1_gate_full_workflow_passes_and_is_reproducible(tmp_path: Path) -> None:
    case = _build_60_document_case(tmp_path)
    assert gate_main(["prepare", *case.args]) == 0
    _fill_expected_tables_from_source_contract(case.annotation_root)
    assert gate_main(["sample-cells", *case.args]) == 0
    _mark_all_cell_audits(case.annotation_root, verified=True)
    assert gate_main(["evaluate", *case.args, "--report-root", str(case.report_root)]) == 0

    first = _report_bytes(case.report_root / case.dataset_fingerprint)
    assert gate_main(["evaluate", *case.args, "--report-root", str(case.report_root)]) == 0
    assert _report_bytes(case.report_root / case.dataset_fingerprint) == first
```

Assert 60 pilot documents, at least 30 annotations per statement type, 30 audit cells, 100%
provenance, overall usability 100%, and no raw/release byte changes.

- [ ] **Step 5: Add failing-quality and invalid-input integration cases**

Change one annotation shape enough to keep overall above 85% but create a table failure;
verify Pareto contains `shape_mismatch` while exit remains `0`. Then mark six of 30 manual
rows false; verify exit `1`, failed `manual_cell_audit`, and six
`manual_provenance_failure` events. Finally corrupt one source byte and verify exit `2`, no
new report, and the previous report remains byte-identical.

- [ ] **Step 6: Document the operator workflow**

Add exact commands:

```powershell
$releasePath = Read-Host 'Verified immutable release path from build-dataset output'
uv run --frozen --no-sync financial-report-qa week1-gate prepare --manifest data/manifests/documents.jsonl --snapshot-root data/raw/ocr_annual_financials/financial_statement --release $releasePath --annotation-root data/qa/week1_pilot
uv run --frozen --no-sync financial-report-qa week1-gate sample-cells --manifest data/manifests/documents.jsonl --snapshot-root data/raw/ocr_annual_financials/financial_statement --release $releasePath --annotation-root data/qa/week1_pilot
uv run --frozen --no-sync financial-report-qa week1-gate evaluate --manifest data/manifests/documents.jsonl --snapshot-root data/raw/ocr_annual_financials/financial_statement --release $releasePath --annotation-root data/qa/week1_pilot --report-root data/interim/week1_gate
```

Explain the two human edits: populate `expected-tables.csv` after `prepare`, then set
`verified` in all 30 `cell-audit.csv` rows after `sample-cells`. Document exit codes and
all thresholds.

- [ ] **Step 7: Run the complete Day 7 quality gate**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation tests/integration/test_week1_gate.py
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data tests/unit/ingestion tests/unit/normalization tests/golden/extraction tests/integration/test_pipeline_e2e.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation scripts/week1_gate.py tests/unit/evaluation tests/integration/test_week1_gate.py src/financial_report_qa/cli.py tests/unit/test_cli.py
uv run --frozen --no-sync ruff format --check src tests scripts
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation scripts/week1_gate.py tests/unit/evaluation tests/integration/test_week1_gate.py src/financial_report_qa/cli.py tests/unit/test_cli.py
```

Expected: all commands exit `0`, with no skipped Day 7 tests.

- [ ] **Step 8: Verify scope and commit Task 8**

```powershell
git status --short
git diff --check
git add scripts/week1_gate.py src/financial_report_qa/evaluation/week1_gate.py src/financial_report_qa/cli.py tests/unit/test_cli.py tests/integration/test_week1_gate.py README.md docs/development.md
git commit -m "feat: add week one quality gate workflow"
```

Confirm no raw TXT, Parquet release, generated report, notebook, `plan.md`,
`dataset_builder.py`, `.agents/`, or unrelated file is staged.

---

## Final Verification Checklist

- [ ] Map every design section to Tasks 1-8; record no uncovered requirement.
- [ ] Scan the plan for incomplete markers, vague actions, undefined interfaces, and inconsistent field names; record no findings.
- [ ] Confirm the exact interfaces remain consistent across tasks: `GateDataset`, `PilotMetadata`, `PilotDocument`, `ExpectedTable`, `CellAudit`, `TableAssessment`, `ProvenanceAudit`, `GateResult`, `prepare_pilot`, `sample_audit_cells`, and `evaluate_gate`.
- [ ] Confirm the annotation metadata seals only `pilot-documents.csv`; `gate-result.json` records completed expected-table and cell-audit hashes.
- [ ] Run `git diff --check` and inspect `git status --short` before any final commit.
- [ ] Run `uv run --frozen --no-sync pytest -q` when the full local runtime is available.
- [ ] Use `superpowers:verification-before-completion` before claiming Day 7 implementation complete.
