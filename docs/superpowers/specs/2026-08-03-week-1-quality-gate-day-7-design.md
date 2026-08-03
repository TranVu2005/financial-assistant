# Week 1 Quality Gate Day 7 Design

**Date:** 2026-08-03

**Status:** Approved in conversation; pending review of this written specification

## Goal

Create a reproducible, annotation-backed Week 1 quality gate for the ViFinQA canonical
dataset. The gate measures whether at least 85% of manually identified main financial
tables are usable, proves that every accepted cell has valid source provenance, manually
traces 30 stratified cells to the source TXT, and produces a deterministic Pareto of
extraction and normalization failures.

## Scope

Included:

- deterministic selection of a 60-document pilot spanning 20 companies;
- versioned CSV contracts for pilot documents, expected main tables, and manual cell
  verification;
- automated matching of annotated tables to canonical Parquet tables;
- automated structural, normalization, and provenance checks;
- deterministic selection of 30 accepted cells for manual source tracing;
- gate metrics at corpus and sufficiently large stratum levels;
- deterministic JSON, Markdown, and CSV audit artifacts;
- CLI commands for preparation, cell sampling, and final evaluation;
- unit, property, integration, and regression tests.

Excluded:

- changing extraction, normalization, inventory, or dataset-builder behavior;
- automatically repairing failed tables or cells;
- fuzzy or model-based table matching;
- replacing expert annotation with extraction heuristics;
- notebook-only analysis;
- committing generated reports, raw reports, or processed Parquet releases.

## Architecture

The implementation is a read-only evaluation stage:

```text
manifest + immutable dataset release
  -> validate release identity and contracts
  -> deterministic 20-company x 3-document selection
  -> pilot-documents.csv + expected-tables.csv template

completed expected-tables.csv + canonical Parquet
  -> exact/overlap table matching
  -> automated usability and provenance checks
  -> deterministic 30-cell sample + source excerpts
  -> cell-audit.csv

completed cell-audit.csv + prior inputs
  -> aggregate gate metrics
  -> stable failure taxonomy and Pareto
  -> gate-result.json + gate-report.md + pareto-errors.csv
  -> exit 0 pass, 1 gate failure, or 2 invalid input/workflow failure
```

The code lives in `financial_report_qa.evaluation`:

- `evaluation/week1_contracts.py` owns immutable input/output contracts, CSV schemas,
  enums, and validators.
- `evaluation/week1_sampling.py` owns stable ranking, stratified document selection, and
  cell selection.
- `evaluation/week1_matching.py` owns annotated-to-observed table matching and usability
  checks.
- `evaluation/week1_provenance.py` owns release-wide structural provenance validation,
  source re-verification, and safe source excerpts.
- `evaluation/week1_pareto.py` owns failure-event aggregation and deterministic Pareto
  ordering.
- `evaluation/week1_gate.py` orchestrates `prepare`, `sample-cells`, and `evaluate`, and
  contains the command implementation.
- `scripts/week1_gate.py` is only an executable wrapper.

The product CLI adds `financial-report-qa week1-gate`. Business modules never import from
`scripts/`, `tests/`, notebooks, generated reports, or raw-data directories.

## Inputs and Release Identity

Every command receives explicit paths for:

- the immutable JSONL inventory manifest;
- the raw snapshot root;
- the immutable canonical dataset release directory;
- the versioned annotation directory;
- the generated report root.

The release must contain the exact current dataset-builder artifacts:
`documents.parquet`, `tables.parquet`, `cells.parquet`, `issues.parquet`, and
`manifest.json`. The gate does not silently accept missing or renamed equivalents. It
verifies that release and source manifest fingerprints agree before sampling or
evaluation.

The gate never follows an unverified mutable pointer during an audit. A caller may resolve
a current release before invocation, but all three commands persist and require the same
immutable `dataset_fingerprint` and `source_manifest_sha256`.

## Stable Sampling

`SAMPLING_VERSION` is `week1-pilot-v1`. Ranking uses SHA-256 over UTF-8 canonical strings;
it never uses Python's process-randomized `hash()` or an unseeded PRNG.

Eligible documents must:

- have `inventory_status == "ready"` in the source manifest;
- appear exactly once in `documents.parquet` with matching company, year, path, and digest;
- contain at least one extracted table in `tables.parquet`;
- belong to a company with at least three eligible documents.

Company rank is `SHA-256("week1-pilot-v1\ncompany\n<company_code>")`. The first 20 ranked
eligible companies are selected. Fewer than 20 eligible companies is an invalid pilot and
exits `2`.

For each selected company, documents are chosen in two passes:

1. Construct `(report_year, statement_scope)` buckets and rank each bucket with
   `SHA-256("week1-pilot-v1\nstratum\n<company>\n<year>\n<scope>")`. Take one document from
   each ranked bucket until three documents are selected or all buckets are exhausted.
2. If fewer than three were selected, fill from remaining company documents ranked by
   `SHA-256("week1-pilot-v1\ndocument\n<doc_id>\n<relative_path>")`.

Within a bucket, documents use the same document rank. The final pilot contains exactly
60 unique documents sorted by `(company_code, report_year, statement_scope,
relative_path, doc_id)`.

## Annotation Directory and Versioning

The committed annotation root is `data/qa/week1_pilot/`. It contains:

- `pilot-metadata.json`;
- `pilot-documents.csv`;
- `expected-tables.csv`;
- `cell-audit.csv` after cell sampling and manual review.

`pilot-metadata.json` contains only deterministic fields:

- `annotation_schema_version`;
- `sampling_version`;
- `dataset_fingerprint`;
- `source_manifest_sha256`;
- `document_count` fixed at `60`;
- `pilot_documents_sha256`, which seals the generated document selection.

It contains no timestamp, absolute path, hostname, or reviewer identity. Human review
identity and process evidence belong in version-control history rather than canonical
data. The metadata file is immutable after `prepare`. `gate-result.json` records the exact
SHA-256 digests of the completed `expected-tables.csv` and `cell-audit.csv` used for the
decision, avoiding mutation of annotations during evaluation.

`prepare` refuses to overwrite a non-empty annotation directory. Regeneration requires a
new annotation version or an explicitly empty target, preventing accidental loss of
manual work.

## Pilot Document Contract

`pilot-documents.csv` has this exact column order:

```text
annotation_schema_version,dataset_fingerprint,source_manifest_sha256,
doc_id,relative_path,company_code,report_year,statement_scope
```

The file contains exactly 60 data rows, uses UTF-8, RFC 4180 CSV quoting, LF line endings,
and a final newline. Rows follow the final stable pilot order. All metadata must match the
manifest and `documents.parquet` exactly.

## Expected Table Contract

`expected-tables.csv` is completed by expert review of the 60 source documents. It has
this exact column order:

```text
annotation_schema_version,annotation_id,doc_id,relative_path,statement_type,
line_start,line_end,row_count,column_count,unit_normalized,expected_periods,notes
```

Rules:

- one row represents one expected main financial table;
- `annotation_id` is `ann_` plus SHA-256 of
  `"<doc_id>\n<line_start>\n<line_end>\n<statement_type>"`;
- `statement_type` is exactly `balance_sheet`, `income_statement`, or
  `cash_flow_statement`;
- line spans are one-based, inclusive, ordered, inside the verified TXT file, and do not
  overlap another annotation of the same statement family in the same document;
- row and column counts are positive logical dimensions after span expansion and
  continuation merging;
- `unit_normalized` is empty when the source does not state one; otherwise it uses the
  normalization vocabulary;
- `expected_periods` is an empty string or a `|`-separated, sorted, duplicate-free list of
  canonical periods;
- `notes` may explain an unusual source but never changes scoring.

Every expected row must reference one of the 60 pilot documents. The completed file must
contain at least 30 rows for each of the three statement types. Failure to meet this
coverage makes the pilot invalid and exits `2`; the threshold is not relaxed and no
automatic top-up changes the approved sample.

## Table Matching

Matching is deterministic and performed independently within each `doc_id`.

An observed table is eligible for an annotation when their inclusive line spans overlap
by at least one line. The match score is the intersection length divided by the annotation
span length. Exact span equality ranks above partial overlap, then higher overlap ranks
first, then lower absolute boundary distance, then lexicographically smaller `table_id`.
An assigned pair below `0.80` remains matched for diagnosis but fails usability with
`span_mismatch`.

The matcher computes a one-to-one maximum-weight assignment over eligible pairs. It does
not greedily reuse one observed table for multiple annotations. Since each pilot document
contains few tables, an exact deterministic assignment algorithm is preferred over a
heuristic. Equal optimal assignments use the lexicographically smallest ordered tuple of
`table_id` values.

An annotation with no overlapping observed table produces `missing_table`. An observed
table not matched to a main-table annotation is not automatically a false positive
because the release may legitimately contain note or supporting tables.

## Table Usability

An annotated main table is usable only when all conditions hold:

1. a unique observed table is matched;
2. observed and annotated spans have an overlap score of at least `0.80`;
3. `row_count` and `column_count` equal the annotation;
4. `statement_type` equals the annotation;
5. at least one source cell in the matched table has non-null `value_numeric`;
6. if annotated `unit_normalized` is non-empty, the observed table unit equals it;
7. every annotated expected period occurs in at least one matched-table cell;
8. every cell in the matched table passes automated provenance validation.

Failures do not short-circuit; all applicable failure events are recorded so the Pareto
shows the true error mix. One expected table still contributes at most one denominator
unit and one usable numerator unit.

## Automated Provenance Validation

Every cell in every matched table is an accepted cell for the provenance gate. The audit
validates all accepted cells, not only the manual sample:

- `cell_id` and `table_id` match canonical ID syntax;
- `(table_id, row_idx, col_idx, cell_id)` is unique in `cells.parquet`;
- the referenced table exists and has the same document;
- row and column indices are inside table dimensions;
- `source_line_start >= table.line_start` and
  `source_line_end <= table.line_end`;
- source spans are one-based and ordered;
- the manifest path is safe and resolves below the snapshot root;
- current source size and SHA-256 equal the immutable document record;
- re-running `extract_document()` and `normalize_extraction()` returns an equal cell for
  the same `cell_id`, including raw value, canonical fields, and source span.

Any failure emits `invalid_provenance` and makes both the table and the 100% provenance
gate fail. The evaluator reports counts without exposing absolute machine paths.

## Manual 30-Cell Audit

`sample-cells` runs only after expected-table validation and automated matching. It samples
from cells in matched tables that currently satisfy the table-usability checks and have a
non-empty `value_raw`.

Cells are bucketed by `(company_code, report_year, statement_type)`. Bucket and cell rank
use SHA-256 with `SAMPLING_VERSION` and the stable IDs. Round-robin selection takes one
cell per ranked bucket before a second pass, never more than two cells from one table, and
continues until exactly 30 unique cells are selected. Fewer than 30 eligible cells is an
invalid audit and exits `2`.

`cell-audit.csv` has this exact column order:

```text
annotation_schema_version,dataset_fingerprint,cell_id,table_id,doc_id,
relative_path,company_code,report_year,statement_type,source_line_start,
source_line_end,value_raw,source_excerpt,verified,review_notes
```

`source_excerpt` is the exact joined TXT content for the inclusive cell span, preserving
line text but canonicalizing line separators to `\n` for CSV portability. It is capped at
500 Unicode code points; longer excerpts keep the first 497 followed by `...`. This cap
does not affect automated provenance checks.

`sample-cells` writes `verified` and `review_notes` empty. The reviewer sets `verified` to
lowercase `true` or `false`; any other value is invalid input. Final evaluation requires
30 rows and 30 `true` values to pass.

## Gate Metrics

The final gate passes only when all conditions hold:

- exactly 60 pilot documents are valid;
- each main statement type has at least 30 expected tables;
- `usable_main_tables / annotated_main_tables >= 0.85` using exact integer comparison
  `usable * 100 >= annotated * 85`;
- `provenance_valid_cells == accepted_cells`, including the zero-denominator guard that
  requires `accepted_cells > 0`;
- all 30 manual cell audits have `verified=true`;
- every `(company_code, report_year, statement_type)` stratum containing at least 10
  annotated tables has usability of at least 70%, checked as
  `usable * 100 >= annotated * 70`.

The report includes Wilson 95% confidence intervals for overall and statement-level
usability as descriptive statistics. Confidence intervals do not alter pass/fail.

## Failure Taxonomy and Pareto

Stable gate failure codes are:

- `missing_table`;
- `span_mismatch`;
- `shape_mismatch`;
- `statement_mismatch`;
- `unit_mismatch`;
- `period_mismatch`;
- `no_numeric_value`;
- `invalid_provenance`;
- `manual_provenance_failure`;
- every ingestion rejection code present in the pilot documents;
- every normalization issue code present in matched pilot tables.

The Pareto counts failure events, not failed tables. It sorts by descending count, then
code ascending. Each row contains `rank`, `code`, `count`, `share`, and
`cumulative_share`; shares use the total failure-event count and deterministic decimal
rounding to six places. An empty error set produces a header-only CSV and an empty JSON
list.

## Outputs

Generated artifacts live below
`data/interim/week1_gate/<dataset_fingerprint>/` and are reproducible:

- `gate-result.json`: typed inputs, counts, rates, thresholds, per-statement metrics,
  eligible strata, gate checks, and final `passed` boolean;
- `gate-report.md`: concise human-readable summary, failed checks, worst strata, and top
  Pareto contributors;
- `pareto-errors.csv`: complete deterministic error distribution.

Serialization uses UTF-8, LF, sorted JSON keys, final newlines, stable row ordering, and no
timestamps or absolute paths. The command writes into a same-parent temporary directory,
verifies all artifacts, and publishes with a rename. If the report directory already
exists, the command reuses it only when every generated byte is equal; a differing report
or embedded dataset fingerprint fails without overwrite.

## CLI

The product command is:

```text
financial-report-qa week1-gate prepare ...
financial-report-qa week1-gate sample-cells ...
financial-report-qa week1-gate evaluate ...
```

All subcommands accept explicit manifest, snapshot-root, release, annotation-root, and
report-root paths where relevant. Expected outcomes:

- exit `0`: command succeeds and, for `evaluate`, every gate passes;
- exit `1`: inputs are valid but one or more quality thresholds fail;
- exit `2`: arguments, annotation contracts, release identity, source integrity, or I/O
  are invalid.

Messages show relative source paths, stable IDs, counts, and fingerprints only. They do
not reveal absolute machine paths.

## Error Handling

Invalid workflow state raises typed subclasses of `FinancialReportQAError`:

- `Week1GateInputError` for malformed annotations, missing columns, duplicate IDs,
  insufficient pilot coverage, release mismatch, or unsafe paths;
- `Week1GateSourceError` for source size/hash mismatch or failed re-extraction;
- `Week1GatePublicationError` for report write, verification, or publication failures.

Quality defects remain data: they create failure events and exit `1`, not exceptions.
Unexpected exceptions are not converted into passing or partial reports. Existing
annotations, raw data, and canonical release artifacts are never mutated.

## Testing Strategy

Tests are written before implementation.

Contract tests prove exact CSV columns, strict enums, stable annotation IDs, duplicate
rejection, span validation, sorted period lists, 60-document identity, minimum per-type
coverage, and fingerprint consistency.

Sampling tests prove stable SHA-256 ranking, exactly 20 companies and three documents per
company, preference for distinct year/scope buckets, independence from input ordering,
failure on insufficient eligibility, 30 unique cells, at most two cells per table, and
stratum round-robin behavior.

Matching tests prove the 0.80 overlap boundary, exact-span preference, deterministic ties,
one-to-one maximum-weight assignment, missing tables, and no false-positive assumption for
unmatched observed tables.

Metric tests prove every usability predicate, integer threshold boundaries at 85% and
70%, zero-denominator failure, statement-type minimums, Wilson interval calculation, and
that all applicable failures are retained.

Provenance tests prove valid re-extraction, invalid IDs, duplicate coordinates, missing
tables, out-of-bounds coordinates, reversed/out-of-table source spans, unsafe paths,
source hash mismatch, canonical field drift, and safe 500-code-point excerpts.

Pareto tests prove deterministic counts, descending ordering with code tie-breaks, exact
six-place shares, cumulative shares, and header-only empty output.

An integration fixture creates a small synthetic snapshot, inventory, extraction, and
canonical release with parameterized sampling counts. It exercises the full
prepare-to-sample-to-evaluate workflow for one passing and one failing audit without
requiring the real corpus. CLI tests verify exit codes `0`, `1`, and `2`.

Regression gates rerun schema, data, ingestion, normalization, and canonical dataset
builder tests because Day 7 consumes all those contracts.

## Quality Gate for the Implementation

```powershell
uv run --frozen --no-sync pytest -q tests/unit/evaluation tests/integration/test_week1_gate.py
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data tests/unit/ingestion tests/unit/normalization tests/golden/extraction tests/integration/test_build_dataset.py
uv run --frozen --no-sync ruff check src/financial_report_qa/evaluation scripts/week1_gate.py tests/unit/evaluation tests/integration/test_week1_gate.py
uv run --frozen --no-sync ruff format --check src tests scripts
uv run --frozen --no-sync mypy src/financial_report_qa/evaluation scripts/week1_gate.py tests/unit/evaluation tests/integration/test_week1_gate.py
```

## Completion Criteria

Day 7 is complete when:

1. the same verified release and sampling version always produce the same 60 documents;
2. invalid or under-covered annotations fail closed before scoring;
3. table matching is one-to-one, deterministic, and auditable;
4. usability and stratum thresholds use exact integer comparisons;
5. every accepted cell receives automated provenance validation;
6. exactly 30 stratified cells receive completed manual verification;
7. gate JSON, Markdown, and Pareto CSV are byte-deterministic for equal inputs;
8. CLI exit codes distinguish pass, quality failure, and invalid workflow state;
9. all implementation quality-gate commands pass;
10. raw snapshots, canonical releases, and unrelated worktree files remain unchanged.
