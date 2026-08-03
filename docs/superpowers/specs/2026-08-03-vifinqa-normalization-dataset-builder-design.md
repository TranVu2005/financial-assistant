# ViFinQA Normalization and Dataset Builder Design

**Date:** 2026-08-03

**Status:** Approved in conversation; pending review of this written specification

## Goal

Implement the roadmap's Day 5 normalization and Day 6 canonical dataset builder as one
deterministic, provenance-preserving delivery. The pipeline converts verified ViFinQA
extractions into canonical financial records, records every conservative non-decision,
and publishes reproducible Parquet artifacts without risking a previously valid dataset.

## Scope

Included:

- deterministic company, period, statement-type, metric, number, and unit normalization;
- immutable normalization result and issue contracts;
- preservation of `value_raw`, `row_label_raw`, `column_label_raw`, `unit_raw`, source
  spans, extraction confidence, table evidence, and rejected candidates;
- versioned exact alias rules with Unicode-aware comparison;
- a dataset builder that runs inventory, ingestion, and normalization through public APIs;
- immutable release publication of documents, tables, cells, normalization issues, a
  quality summary, and a dataset fingerprint, followed by an atomic current-release
  pointer update;
- property, unit, integration, and regression tests.

Excluded:

- fuzzy matching, embeddings, LLM classification, or probabilistic normalization;
- manual corpus corrections, learned aliases, and external rule services;
- retrieval indexes and QA planning or execution;
- OCR or PDF processing;
- weakening ingestion validation or changing raw snapshot files.

## Architecture

Normalization is a pure, typed stage after ingestion:

```text
DocumentRecord + ExtractionResult + versioned rules
  -> independent field normalizers
  -> conservative conflict resolution
  -> NormalizedDocument + ordered NormalizationIssue values

inventory manifest + snapshot root + build configuration
  -> inventory selection
  -> extract_document
  -> normalize_extraction
  -> stable sort and flatten
  -> temporary Parquet dataset + metadata
  -> immutable fingerprinted release
  -> atomic current-release pointer update
```

Each module has one responsibility:

- `normalization/companies.py` verifies the canonical document company and checks only
  explicit company evidence found in table metadata.
- `normalization/periods.py` parses controlled annual, quarterly, and as-of-date labels.
- `normalization/statements.py` classifies controlled financial-statement title aliases.
- `normalization/metrics.py` maps exact normalized row-label aliases to stable metric slugs.
- `normalization/numbers.py` parses missing markers, signs, parentheses, percentages, and
  unambiguous decimal/grouping separators into `Decimal`.
- `normalization/units.py` resolves cell, column, and table units and exposes exact scale
  multipliers.
- `normalization/service.py` coordinates the focused normalizers without filesystem I/O.
- `schemas/normalization.py` owns immutable cross-module result and issue contracts.
- `data/dataset_builder.py` owns deterministic orchestration, flattening, fingerprinting,
  Parquet serialization, quality metadata, and atomic publication.
- `scripts/build_dataset.py` is a thin command-line wrapper around the builder API.

The builder lives in the package rather than in `scripts/` so application code and tests
consume a public, typed interface. The script contains argument parsing and exit handling
only.

## Public Contracts

All new contracts use Pydantic 2 with `extra="forbid"` and `frozen=True`. Collections are
tuples and issue ordering is deterministic.

```python
class NormalizationIssue(BaseModel):
    code: NormalizationIssueCode
    doc_id: str
    table_id: str | None
    cell_id: str | None
    field: Literal[
        "company", "period", "statement_type", "metric", "number", "unit"
    ]
    raw_value: str | None


class NormalizedDocument(BaseModel):
    document: DocumentRecord
    extraction: ExtractionResult
    issues: tuple[NormalizationIssue, ...]
    ruleset_version: str
    normalization_fingerprint: str
```

`NormalizedDocument.extraction` contains new immutable `TableRecord` and `CellRecord`
instances with canonical fields populated. It retains blocks, placements, evidence, and
rejections from ingestion; normalization never mutates an input model.

The orchestration API is:

```python
def normalize_extraction(
    document: DocumentRecord,
    result: ExtractionResult,
) -> NormalizedDocument: ...
```

It requires `document.doc_id == result.doc_id`; a mismatch is a contract error. Supplying
the document explicitly resolves company identity and report-year context without
re-parsing paths or performing filesystem access.

The builder API is:

```python
def build_dataset(config: DatasetBuildConfig) -> DatasetBuildResult: ...
```

`DatasetBuildConfig` contains safe snapshot, manifest, and processed-root paths plus an
explicit schema version. `DatasetBuildResult` returns the immutable release directory,
dataset fingerprint, row counts, issue counts by code, and source manifest fingerprint.

## Conservative Decision Policy

A rule emits a canonical value only when one interpretation is supported by the approved
controlled rules. Unknown, conflicting, or genuinely ambiguous input produces `None` in
the owned canonical field and one stable issue code. It never removes a table or cell.

Comparison operates on a temporary key created with NFKC, case-folding, outer trimming,
and internal whitespace collapse. Metric and statement aliases may additionally use a
controlled punctuation-removal transform declared next to the alias. Raw strings remain
byte-for-byte equal to the strings emitted by ingestion.

There is no fuzzy matching. Alias maps are immutable module constants with a public
`RULESET_VERSION`. A duplicate normalized alias mapped to different canonical values is
rejected when the rules are loaded.

## Company Normalization

`DocumentRecord.company_code` is the canonical company source of truth because inventory
already validates it as an uppercase 2-10 character code. Normalization propagates this
value without deriving a company from filenames or free text.

An explicit ticker in a table title may be compared with the document code. A different
valid ticker emits `company_conflict`; an unrecognized title is not itself an issue.
Company evidence never overrides `DocumentRecord.company_code`.

## Period Normalization

Canonical period strings use exactly one of:

- `YYYY` for a financial year;
- `YYYY-Q1` through `YYYY-Q4` for a quarter;
- `YYYY-MM-DD` for an as-of date.

Controlled inputs include four-digit years, Vietnamese/English year labels, Arabic or
Roman quarter labels, and day-first dates. Two-digit years, month-only labels, invalid
calendar dates, and values with multiple valid interpretations produce `None`. If a label
contains a period without a year, `DocumentRecord.report_year` may be used only for an
explicit quarter token; otherwise the service emits `period_incomplete`.

## Statement-Type Normalization

The closed canonical vocabulary is:

- `balance_sheet`;
- `income_statement`;
- `cash_flow_statement`;
- `equity_changes`;
- `notes`.

Classification uses exact controlled aliases from `TableRecord.title_raw`. Multiple
matched statement families emit `statement_conflict`; no match leaves
`TableRecord.statement_type=None` without guessing.

## Metric Normalization

Metric aliases map exact normalized `CellRecord.row_label_raw` values to stable lowercase
ASCII slugs. The initial controlled vocabulary covers the high-value ViFinQA metrics
needed by later retrieval and evaluation, including `revenue`, `net_revenue`,
`profit_before_tax`, `profit_after_tax`, `total_assets`, `total_liabilities`, `equity`,
`cash_and_cash_equivalents`, and `operating_cash_flow`.

Unknown labels leave `row_label_canonical=None` and emit `metric_unknown`. A normalized
alias collision is a ruleset error, not a per-cell issue. Header cells and cells without a
raw row label are not metric candidates.

Column labels are period-normalized into `column_label_canonical` when possible. The same
canonical value is also stored in `CellRecord.period`. Non-period column labels remain raw
only and emit no issue unless they look period-like but are invalid or ambiguous.

## Number Normalization

`value_numeric` remains in the display scale represented by `CellRecord.unit`; it is not
expanded to base VND. For example, raw `1.500` under `triệu VND` becomes
`Decimal("1500")` and `VND_million`.

Parsing supports:

- leading `+` or `-`;
- one surrounding pair of accounting parentheses for a negative number;
- ASCII and non-breaking spaces used as grouping separators;
- unambiguous `.` or `,` decimal/grouping conventions;
- a trailing percent sign when unit resolution agrees with `percent`.

Empty strings, `-`, Unicode dash variants, `N/A`, `NA`, and controlled Vietnamese missing
markers produce `value_numeric=None` with `number_missing`. Malformed signs, unmatched
parentheses, mixed grouping widths, multiple decimal candidates, embedded letters, and
separator forms with more than one valid interpretation produce `None` with
`number_ambiguous` or `number_invalid`.

Separator interpretation is local to the raw token and never depends on machine locale.
When both `.` and `,` occur, the rightmost separator may be decimal only when the prefix
uses valid three-digit groups and the fractional suffix has one or two digits; otherwise
the token must match one unique grouping interpretation. A single separator followed by
exactly three digits is treated as grouping. Other single-separator forms are accepted as
decimal only when exactly one controlled interpretation exists.

## Unit and Scale Normalization

The closed unit vocabulary is:

- `VND` with multiplier `1`;
- `VND_thousand` with multiplier `1000`;
- `VND_million` with multiplier `1000000`;
- `VND_billion` with multiplier `1000000000`;
- `percent` with multiplier `0.01`;
- `ratio` with multiplier `1`.

Resolution precedence is explicit cell suffix, column label, then `TableRecord.unit_raw`.
Equivalent evidence agrees; different canonical units emit `unit_conflict` and produce
`None`. Unknown unit-like text emits `unit_unknown`. A missing unit with no unit-like
evidence remains `None` without inventing VND.

`TableRecord.unit_normalized` is populated only when a single table-wide unit is
unambiguous. Each numeric cell receives its resolved `CellRecord.unit`, including a more
specific valid column or cell override when present.

The economic value helper is:

```python
def economic_value(value: Decimal, unit: CanonicalUnit) -> Decimal: ...
```

It returns `value * multiplier(unit)` and is the invariant used by property tests. Scale
conversion helpers must preserve this value exactly in decimal arithmetic.

## Dataset Builder and Artifacts

The builder reads an existing immutable JSONL manifest through the Day 2 public API and
processes only `ready` documents in stable `relative_path`, then `doc_id`, order. It calls
`extract_document` and `normalize_extraction`; it does not duplicate their internals.

It writes these files:

- `documents.parquet` with one canonical document row per ready source;
- `tables.parquet` with one normalized `TableRecord` row per extracted table;
- `cells.parquet` with one normalized `CellRecord` row per source cell;
- `normalization_issues.parquet` with ordered audit issues;
- `quality-summary.json` with counts by inventory status, extraction rejection code,
  statement type, unit, and normalization issue code;
- `dataset-metadata.json` with schema version, ruleset version, source manifest
  fingerprint, artifact hashes, row counts, and final dataset fingerprint.

Nested values are serialized using explicit Arrow schemas. Decimal precision and scale
are fixed by the schema rather than inferred from a batch. Empty tables still produce
valid Parquet files with the declared columns.

Rows are sorted by canonical stable keys: documents by `(relative_path, doc_id)`, tables
by `(doc_id, line_start, table_id)`, cells by `(table_id, row_idx, col_idx, cell_id)`, and
issues by `(doc_id, table_id, cell_id, field, code, raw_value)` with `None` sorted before
strings. Output never contains absolute paths, timestamps, random IDs, or host metadata.

## Fingerprinting and Atomic Publication

The source manifest fingerprint is the SHA-256 of its exact verified bytes. The
normalization fingerprint hashes the document ID, canonical serialized tables/cells and
issues, and `RULESET_VERSION`. The dataset fingerprint hashes:

- source manifest fingerprint;
- explicit schema version;
- `RULESET_VERSION`;
- canonical build configuration excluding output and temporary paths;
- ordered payload artifact names and SHA-256 digests. Payload artifacts are the four
  Parquet files and `quality-summary.json`; `dataset-metadata.json` and `current.json` are
  excluded to avoid a self-referential hash.

The builder creates a uniquely named temporary directory below the configured processed
root, writes and closes all artifacts, re-opens them for schema/count/hash verification,
computes the dataset fingerprint, and writes `dataset-metadata.json`. It publishes the
verified directory as `releases/<dataset_fingerprint>` with one same-volume rename. A
pre-existing release is reused only after every declared artifact and digest verifies;
otherwise the build fails without overwriting it.

After the immutable release is available, the builder writes a temporary pointer file
containing the fingerprint and safe relative release path, flushes it, and replaces
`current.json` with `os.replace`. Consumers resolve a dataset only through this pointer,
so they observe either the complete previous release or the complete new release. Old
releases are retained; garbage collection is outside this task.

Temporary directories are cleaned only after their resolved paths are proven to be below
the configured processed root and carry the builder's private naming prefix. Unmanaged
paths and immutable releases are never deleted or overwritten.

## Errors and Audit Issues

Contract, source-integrity, manifest, Arrow schema, fingerprint, verification, and I/O
failures stop the build with typed domain errors. A failed build never updates
`current.json` to partial artifacts and never damages the previous valid release.

Field-level normalization failures are data observations, not workflow exceptions. They
produce stable issue codes such as:

- `company_conflict`;
- `period_incomplete`, `period_ambiguous`, `period_invalid`;
- `statement_conflict`;
- `metric_unknown`;
- `number_missing`, `number_ambiguous`, `number_invalid`;
- `unit_unknown`, `unit_conflict`.

Issue records contain no locale-dependent messages; user-facing explanations can map
codes later. Repeated normalization of equal inputs returns equal models.

## Testing Strategy

Tests are written before implementation.

Focused unit tests prove:

- company propagation and explicit conflict reporting;
- annual, quarterly, date, incomplete, invalid, and ambiguous period behavior;
- exact statement and metric aliases, unknown labels, and ruleset collision rejection;
- signs, accounting parentheses, spaces, decimal/grouping separators, percentages,
  missing markers, invalid forms, and raw preservation;
- unit precedence, agreement, conflict, unknown evidence, and exact multipliers;
- service-level preservation of all ingestion provenance and deterministic issue order.

Hypothesis property tests generate bounded `Decimal` coefficients and all canonical VND
scales. They prove that a value converted between compatible scales has the same
`economic_value`, that rendering and parsing controlled normalized numbers round-trips,
and that normalization never changes raw fields.

Integration tests build a small committed synthetic snapshot and manifest twice into
separate processed roots and assert equal schemas, rows, artifact hashes, summaries, and
dataset fingerprints. Failure injection during write, verification, release rename, and
pointer replacement proves that `current.json` never selects a partial release and the
previous release remains readable. Tests also reject unsafe processed roots and path
escape.

Regression gates rerun the complete schema, data, ingestion, and golden extraction suites
because the normalized records wrap those contracts.

## Quality Gate

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization src/financial_report_qa/data/dataset_builder.py src/financial_report_qa/schemas/normalization.py scripts/build_dataset.py tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync mypy src/financial_report_qa/normalization src/financial_report_qa/data/dataset_builder.py src/financial_report_qa/schemas/normalization.py scripts/build_dataset.py tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
```

## Completion Criteria

The combined delivery is complete when:

1. approved aliases normalize deterministically and unknown or conflicting evidence is
   never guessed;
2. every raw value, label, unit, and source span remains unchanged;
3. numeric scale property tests preserve exact economic value;
4. equal inputs and versions produce equal normalization and dataset fingerprints;
5. all declared Parquet and metadata artifacts pass explicit schema and count checks;
6. release publication plus pointer replacement is atomic from a consumer's perspective,
   and failure tests preserve the prior valid release;
7. all quality-gate commands pass;
8. raw snapshots and unrelated worktree files remain unchanged.
