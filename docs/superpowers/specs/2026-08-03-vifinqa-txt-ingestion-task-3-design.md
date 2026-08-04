# ViFinQA TXT Ingestion Task 3 Design

**Date:** 2026-08-03

**Status:** Approved

## Goal

Build the complete ViFinQA TXT ingestion pipeline for Task 3: lossless text reading,
one-based source provenance, deterministic block segmentation, HTML-first table
detection, conservative structured-text fallback, rectangular table extraction, and
auditable rejection. The same immutable snapshot must produce equal extraction models
on repeated runs.

## Scope

This design combines the roadmap's Day 3 Task 3a and Day 4 Task 3b. It supports only
the current ViFinQA TXT snapshot and consumes canonical `DocumentRecord` values produced
by the Day 2 inventory.

Included:

- lossless reading of `utf-8` and `utf-8-sig` files;
- verification of file size and SHA-256 before parsing;
- one-based inclusive line provenance;
- segmentation into paragraphs, page markers, table regions, and note regions;
- HTML table parsing for ViFinQA's `table`, `tr`, `td`, and `th` markup;
- a conservative fallback for structured text without usable HTML markers;
- multi-line header composition;
- compatible table-continuation merging across page boundaries;
- `rowspan` and `colspan` expansion into a rectangular logical grid;
- immutable extraction results, stable source-cell IDs, confidence, evidence, and
  deterministic rejection codes.

Excluded:

- PDF parsing, OCR, or network access;
- support for arbitrary dataset layouts or encodings not approved by inventory;
- numeric, metric, company, period, or unit normalization;
- CSV, Parquet, index, or normalized-dataset publication;
- probabilistic or model-based table detection;
- forcing explanatory prose into tables.

## Corpus Evidence

The local ViFinQA snapshot contains 1,973 TXT files. Of these, 1,965 contain both
`<table>` and `</table>` markers. The eight files without HTML tables are PRT explanatory
letters dominated by prose and bullet-like financial explanations. Therefore HTML is
the primary extraction signal, while structured-text fallback must prefer zero output
over false positives when column evidence is weak.

## Architecture

The implementation is a typed, deterministic pipeline:

```text
DocumentRecord + snapshot root
  -> verified lossless TXT reader
  -> source lines and semantic blocks
  -> HTML-first candidate detection
  -> conservative non-HTML fallback
  -> candidate extraction and span expansion
  -> continuation/header reconciliation
  -> immutable ExtractionResult
```

Each unit has one responsibility:

- `ingestion/txt_reader.py` owns safe path resolution, byte verification, decoding,
  line-ending preservation, and block segmentation.
- `ingestion/provenance.py` owns immutable ingestion contracts, reason codes, stable
  source-cell IDs, and span validation. It performs no filesystem I/O.
- `ingestion/table_detector.py` owns HTML-region discovery and structured-text fallback.
- `ingestion/table_extractor.py` owns HTML/text parsing, rectangular placement,
  header composition, continuation merging, and canonical table/cell construction.
- `ingestion/__init__.py` exports the approved public API.

The implementation uses Python's standard-library `html.parser.HTMLParser`. No parser
dependency is added for the limited, regular ViFinQA markup.

## Public API

The package-level orchestration interface is:

```python
def extract_document(
    root: Path,
    document: DocumentRecord,
) -> ExtractionResult: ...
```

`root` is the ViFinQA financial-statements snapshot root used to create the inventory.
The implementation resolves `document.relative_path` below this root and rejects path
escape. It accepts only `inventory_status == "ready"`; empty, duplicate, and quarantined
records are not extraction inputs.

Focused modules also expose:

```python
def read_document(root: Path, document: DocumentRecord) -> DecodedDocument: ...


def detect_table_candidates(document: DecodedDocument) -> DetectionResult: ...


def extract_candidates(
    document: DecodedDocument,
    detection: DetectionResult,
) -> ExtractionResult: ...


def stable_cell_id(table_id: str, origin_row: int, origin_col: int) -> str: ...
```

## Immutable Ingestion Contracts

All ingestion models use Pydantic 2 with `extra="forbid"` and `frozen=True`. Collection
fields are tuples. Source spans are one-based and inclusive.

### Reader contracts

`SourceLine` contains:

- `number: int`, starting at 1;
- `text: str`, excluding the line terminator;
- `line_ending: Literal["\\n", "\\r\\n", "\\r", ""]`.

`DecodedDocument` contains the input `DocumentRecord`, the exact decoded `text`, the
ordered `tuple[SourceLine, ...]`, and the ordered `tuple[TextBlock, ...]`. Concatenating
every `line.text + line.line_ending` must reconstruct `text` exactly. An empty ready
document cannot exist because inventory marks zero-byte files as `empty`.

`TextBlock` contains `kind`, `line_start`, `line_end`, and exact `text`. Supported kinds
are `paragraph`, `table`, `notes`, and `page_marker`.

### Detection contracts

`TableCandidate` contains:

- stable candidate order within the document;
- `kind: Literal["html", "structured_text"]`;
- exact raw source and inclusive line span;
- `confidence: float` in `[0, 1]`;
- a non-empty tuple of deterministic evidence codes.

`RejectedCandidate` contains candidate kind, exact raw source, inclusive span, and one
stable rejection code. Expected codes are:

- `unclosed_html_table`;
- `nested_html_table`;
- `unsupported_html_structure`;
- `invalid_span_value`;
- `span_collision`;
- `expansion_limit_exceeded`;
- `ragged_structured_rows`;
- `insufficient_structural_evidence`;
- `empty_extracted_table`.

`DetectionResult` contains ordered candidates, ordered rejected candidates, and blocks.

### Extraction contracts

`CellPlacement` contains a logical `row_idx`, `col_idx`, and the `cell_id` of its source
cell. Multiple placements may reference one source cell after `rowspan`/`colspan`
expansion.

`ExtractedTable` contains:

- one canonical `TableRecord`;
- one `CellRecord` per source `td`/`th` or structured-text cell;
- rectangular logical-grid `CellPlacement` values;
- candidate evidence and any continuation-merge evidence.

`ExtractionResult` contains the document ID, blocks, extracted tables, and all rejected
candidates. Ordering follows source line and then source cell order.

## Reader and Integrity Rules

The reader opens the source in binary mode and streams SHA-256 calculation. Before
decoding it verifies both `file_size_bytes` and `sha256` against `DocumentRecord`. A
mismatch raises `SourceSnapshotMismatchError`; extraction never proceeds on changed
content.

The reader uses exactly the inventory-approved encoding:

- `utf-8` decodes without BOM handling;
- `utf-8-sig` requires and consumes the UTF-8 BOM;
- any other or missing encoding raises `UnsupportedSourceEncodingError`.

Decode errors and filesystem failures are workflow failures, not rejected table
candidates. Exceptions contain the safe relative path but never absolute machine paths.

Line splitting preserves `LF`, `CRLF`, `CR`, missing final newline, blank lines, Unicode,
HTML entities, and all other decoded characters. No whitespace is changed in stored
source text.

## Block Segmentation

Page markers match the complete line `===== PAGE <positive integer> =====` after only
outer-whitespace removal. A marker is its own `page_marker` block.

HTML table regions begin at `<table` and end at the matching `</table>` in source order.
Markers may occur on one line or across multiple lines. These regions become `table`
blocks and are not included in fallback detection. An opening marker without a close
marker reserves the remainder of the document as a rejected HTML region, so fallback
cannot reinterpret its contents.

Non-empty non-marker lines are grouped into paragraphs until a blank line, page marker,
or table boundary. Stored text and spans remain exact.

A notes region begins at a standalone heading matching the controlled aliases
`THUYẾT MINH`, `THUYÊT MINH`, `THUYẾT MINH BÁO CÁO TÀI CHÍNH`, or
`THUYÊT MINH BÁO CÁO TÀI CHÍNH`. Matching may use NFKC, case-folding, and collapsed
whitespace on a temporary copy; stored text is never normalized. Later non-table prose
blocks are labeled `notes`; table and page-marker blocks retain their own kinds.

## Candidate Detection

### HTML-first detection

Every closed, non-nested HTML table region becomes an HTML candidate. Valid HTML
candidates have confidence `1.0` and evidence `html_table_marker`. A missing closing tag,
nested table, or unexpected table structure is rejected with its specific reason code.
Fallback does not reinterpret the bytes inside a rejected HTML region.

### Conservative structured-text fallback

Fallback examines only non-HTML paragraph or notes lines. A candidate must satisfy all
of these rules:

1. at least three consecutive non-blank source lines;
2. every row splits using the same delimiter class: tab, or a run of at least two ASCII
   spaces;
3. every row has the same number of non-empty columns, from 2 through 20;
4. at least two rows contain a numeric-looking token outside the first column;
5. either the first row contains a controlled header signal (`mã số`, `chỉ tiêu`,
   `thuyết minh`, `năm`, `kỳ`, `đơn vị`, `đvt`) or at least half of populated non-first
   cells are numeric-looking;
6. no row is a page marker, prose sentence longer than 200 characters, or a pure
   bullet/list item.

Numeric-looking checks are detection-only and never create numeric values. They accept
digits with punctuation, parentheses, percent signs, or a lone dash.

Fallback confidence starts at `0.75`, adds `0.05` for a header signal, `0.05` when at
least half of non-first cells are numeric-looking, and `0.05` for five or more rows,
capped at `0.90`. Candidates below `0.75` are rejected. Evidence codes record each
satisfied signal so results are explainable and deterministic.

## HTML Extraction and Rectangular Placement

The HTML parser recognizes `tr`, `td`, `th`, `rowspan`, `colspan`, and `br`. HTML entities
are decoded into visible cell text. A `br` becomes `\n`. Outer whitespace introduced by
markup boundaries is trimmed, but internal Unicode and whitespace are otherwise kept.
The exact candidate HTML remains available separately.

`rowspan` and `colspan` must be positive decimal integers. Each source cell gets one
stable ID from `SHA-256("<table_id>\\n<origin_row>\\n<origin_col>")`, prefixed with
`cell_`. Expansion places references to this ID at all covered logical coordinates.
Overlaps are rejected as `span_collision`.

Each candidate may expand to at most 100,000 placements. Exceeding the cap rejects the
whole candidate as `expansion_limit_exceeded`; no partial `TableRecord` is emitted.
Rows are right-padded with absent placements to the maximum width, making the logical
grid rectangular without inventing source cells or values.

For each source cell, `CellRecord` stores:

- origin grid coordinates;
- decoded visible text as `value_raw`;
- exact one-based source-line span of its opening through closing tag, or its structured
  source line;
- raw row and column labels when deterministically available;
- candidate confidence as `extraction_confidence`;
- `None` for canonical labels, numeric value, period, and unit.

No numeric punctuation, signs, dash values, units, labels, or dates are normalized.

## Headers, Titles, Units, and Continuations

The header band is the maximal prefix of at most three logical rows. An HTML row belongs
to the band when all populated placements refer to `th` source cells (including a header
cell continued by `rowspan`) or fewer than half of populated non-first cells are
numeric-looking. The first structured-text row also belongs when it contains a controlled
header signal. Non-empty distinct header texts are composed top-to-bottom per column with
`\n`; this composed raw string becomes `column_label_raw` for data cells. Header source
cells remain in the extraction result.

`row_label_raw` for a data cell is the first populated source-cell text in that logical
row. Canonical label fields remain `None`.

`title_raw` is the nearest preceding non-table, non-marker line within three source lines
when its trimmed length is 1 through 200 and it is not predominantly numeric. Otherwise
it is `None`.

An explicit nearby line or header cell containing `đơn vị`, `đơn vị tính`, or `đvt` may
populate `unit_raw` with the original matched text. `unit_normalized` and
`statement_type` remain `None`. `csv_path` remains `None`.

Adjacent tables are merged as continuations only when:

- they are separated solely by blank lines, one page marker, and optionally a repeated
  title;
- the second table begins within 20 source lines of the first table's end;
- expanded column counts match;
- composed header fingerprints match after temporary NFKC, case-folding, and whitespace
  collapse.

The merged `TableRecord` spans the first through last source line and uses a deterministic
ID from that span. Repeated continuation header cells are omitted from the merged
`CellRecord` collection and active placements; their exact text and provenance remain in
the immutable source lines, blocks, and candidates. Remaining data rows are re-indexed in
source order before stable cell IDs are calculated. Tables that fail any merge condition
remain independent.

## Canonical Schema Mapping

Each successful table creates the existing immutable `TableRecord`:

- `table_id`: `stable_table_id(doc_id, line_start, line_end)`;
- `doc_id`: source document ID;
- `title_raw` and `unit_raw`: conservative raw metadata described above;
- `statement_type=None`, `unit_normalized=None`, `csv_path=None`;
- `row_count` and `column_count`: active rectangular grid dimensions;
- `quality_score`: candidate confidence, or the minimum confidence of merged candidates.

Each source cell creates the existing immutable `CellRecord`. This task adds only
`stable_cell_id()` to the ingestion provenance module; it does not weaken or restructure
the Day 1 schema.

## Error Handling and Determinism

Document-level contract violations stop extraction with a typed domain error:

- unsafe or missing source path;
- non-ready inventory status;
- unsupported or inconsistent encoding;
- byte-size or SHA-256 mismatch;
- filesystem read failure.

Candidate-local defects produce ordered `RejectedCandidate` records and allow later
candidates in the same verified document to proceed. A candidate is atomic: either its
complete table and provenance are emitted or none of them are.

No result contains timestamps, absolute paths, random IDs, locale-dependent parsing, or
unordered collections. Stable sorting uses source order and explicit integer coordinates.

## Testing Strategy

Tests are written before implementation.

`tests/unit/ingestion/test_txt_reader.py` proves:

- UTF-8 and UTF-8-SIG Unicode round trips;
- LF, CRLF, CR, blank-line, and missing-final-newline preservation;
- exact one-based line numbering and block spans;
- page-marker and notes-region classification without source mutation;
- path escape, non-ready status, unsupported encoding, size mismatch, digest mismatch,
  decode error, and safe I/O failure behavior.

`tests/unit/ingestion/test_table_detector.py` proves:

- single-line and multi-line HTML region detection;
- exclusion of HTML regions from fallback;
- exact conservative-fallback thresholds and confidence;
- rejection of prose, bullets, signatures, ragged rows, nested tables, and unclosed HTML;
- deterministic candidate/rejection ordering.

`tests/unit/ingestion/test_table_extractor.py` proves:

- Unicode/entity decoding and raw candidate preservation;
- one `CellRecord` with multiple placements for spans;
- rectangular padding, stable cell IDs, and exact cell line provenance;
- invalid spans, collisions, expansion caps, and candidate atomicity;
- multi-line header composition and raw label mapping;
- compatible continuation merging and rejection of incompatible merges;
- all normalization-owned `CellRecord` fields remain `None`.

`tests/golden/extraction/test_txt_extraction.py` uses small synthetic ViFinQA-shaped
fixtures and committed JSON golden outputs. Fixtures cover Vietnamese Unicode, a
multi-line HTML table, span expansion, a page continuation, a valid structured-text
fallback, and an explanatory letter that produces zero tables. Golden serialization uses
Unicode-preserving deterministic JSON.

A local corpus smoke test runs against the available ViFinQA snapshot without committing
raw reports or corpus-derived output. It verifies all ready documents complete without
uncaught candidate errors, the eight HTML-free explanatory letters do not become false
tables under the current evidence, and a repeated sample run returns equal models.

## Quality Gate

```powershell
uv run --frozen --no-sync pytest -q tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data
uv run --frozen --no-sync ruff check src/financial_report_qa/ingestion tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync mypy src/financial_report_qa/ingestion tests/unit/ingestion tests/golden/extraction
```

The local ViFinQA smoke command is documented separately because raw data is not a test
suite prerequisite.

## Completion Criteria

Task 3 is complete when:

1. approved UTF-8 ViFinQA files round-trip without character or line-ending loss;
2. all emitted table and cell records have exact one-based source provenance;
3. HTML tables, span expansion, header composition, and compatible continuations match
   golden outputs;
4. conservative fallback accepts its structured fixture and rejects prose fixtures;
5. invalid candidates are auditable and never produce partial tables;
6. repeated extraction of unchanged input produces equal immutable models;
7. all quality-gate commands pass;
8. source reports remain unchanged and no unrelated project files are modified.
