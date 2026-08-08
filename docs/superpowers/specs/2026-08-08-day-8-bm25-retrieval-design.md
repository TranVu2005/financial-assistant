# Day 8 BM25 Retrieval Baseline Design

## Scope

Implement the Day 8 lexical retrieval baseline against the immutable
`dataset-pilot-v1` release lock. The milestone creates and validates 30 gold retrieval
questions, builds one deterministic BM25 document per canonical table, applies explicit
metadata filters before lexical ranking, records auditable score data, and reports Recall@10
and F2@10.

Dense retrieval, automatic entity parsing, rank fusion, graph expansion, QA planning, and
answer execution remain out of scope.

## Source boundary

All real-corpus commands consume the release lock rather than an arbitrary processed
directory. The loader must verify:

- the lock alias is `dataset-pilot-v1`;
- the referenced Week 1 gate result has `passed: true`;
- lock, gate result, and release manifest contain the same dataset fingerprint;
- the release contains `documents.parquet`, `tables.parquet`, and `cells.parquet`;
- the current release fingerprint is
  `37a61be7aebae1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.

Source TXT and canonical Parquet files are immutable. Retrieval artifacts are rebuildable and
live below `data/indexes/` or `artifacts/evaluations/`.

## Gold retrieval contract

Create `data/qa/retrieval-gold-v1.jsonl` with exactly 30 UTF-8 JSON Lines records. Each record
has this logical shape:

```python
class RetrievalFilters(BaseModel):
    company_codes: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    statement_types: tuple[str, ...] = ()


class GoldRetrievalQuestion(BaseModel):
    question_id: str                 # retq_<64 lowercase hex>
    question: str                    # non-empty Vietnamese question
    intent: Literal["lookup", "compare", "growth"]
    filters: RetrievalFilters
    gold_table_ids: tuple[str, ...]  # sorted, unique, non-empty
```

`question_id` is derived from normalized question text, filters, gold table IDs, contract
version, and dataset fingerprint. Gold table IDs must exist in the locked release and must be
compatible with the declared filters. Filters are explicit expert annotations for Day 8;
extracting filters from free text belongs to Day 10.

The 30 questions cover all three intents, multiple companies and periods, and both single-table
and multi-table retrieval. Gold labels must be reviewed against the release and source
provenance; generated candidates cannot silently become ground truth.

## Retrieval document contract

One `TableDocument` is derived for every canonical table:

```python
class TableMetadata(BaseModel):
    company_code: str
    periods: tuple[str, ...]
    statement_type: str | None
    unit: str | None


class TableDocument(BaseModel):
    table_id: str
    doc_id: str
    text: str
    metadata: TableMetadata
```

The deterministic text serialization contains labeled fields in this order:

1. raw table title;
2. normalized statement type;
3. sorted canonical metric names and their curated aliases observed from row labels;
4. company ticker and canonical company name when registered;
5. sorted cell periods plus document report year;
6. normalized table/cell units.

Empty fields are omitted. Values are Unicode-normalized and whitespace-collapsed without
dropping Vietnamese accents. Lists are deduplicated and sorted. Raw numeric cell values are not
indexed because they add lexical noise and do not serve Day 8 table discovery.

## Index artifacts

Build artifacts under:

```text
data/indexes/bm25/<dataset-fingerprint>/
  manifest.json
  documents.jsonl
  bm25/
```

The manifest records schema version, builder version, `bm25s` version, tokenizer settings,
dataset fingerprint, release-lock hash, table count, document hash, and artifact hashes. Build
through a temporary sibling directory and publish atomically. Rebuilding the same input must
produce identical `documents.jsonl` and manifest content; an existing non-identical target is an
error.

Use `bm25s` with a fixed Vietnamese-compatible tokenization policy: Unicode NFKC, casefold,
whitespace/punctuation token boundaries, no stemming, and no language stop-word removal.

## Query and ranking flow

```text
Question + explicit filters
  -> validate release/index identity
  -> intersect company/period/statement postings
  -> BM25 over only eligible table rows
  -> stable sort by score descending, table_id ascending
  -> top-k candidates with rank and score trace
```

Filters use AND across fields and OR within each field. An empty field is unconstrained. A table
with no known value does not satisfy a non-empty filter. An empty eligible set returns an empty
result with an auditable reason rather than silently falling back to the full corpus.

`bm25_score` is the Day 8 total score. The trace also records matched query tokens and the
filter decisions. No undocumented boosts or penalties are allowed. Non-finite scores are an
error; exact score ties are resolved by `table_id`.

## Evaluation

For each question at `k=10`:

```text
true_positive = |top_10_table_ids intersect gold_table_ids|
precision@10  = true_positive / 10
recall@10     = true_positive / |gold_table_ids|
F2@10         = 5 * precision * recall / (4 * precision + recall)
```

F2 is zero when both precision and recall are zero. Primary reported metrics are the macro means
across exactly 30 questions. The report also includes metrics by intent, query-level predictions,
gold IDs, scores, matched tokens, filter counts, missing-gold details, and failure categories.

Write deterministic artifacts to:

```text
artifacts/evaluations/retrieval-day8-<dataset-prefix>.json
artifacts/evaluations/retrieval-day8-<dataset-prefix>.md
```

Day 8 records the observed baseline without pretending to satisfy the Week 2 gate. The Week 2
acceptance targets remain F2 >= 0.80 and Recall@10 >= 0.90 for Day 14.

## Error handling

Fail closed for an invalid/missing release lock, fingerprint drift, malformed gold JSONL,
nonexistent or filter-incompatible gold IDs, corrupt index manifest, duplicate table/question
IDs, and non-finite BM25 scores. CLI commands return `0` for success and `2` for invalid inputs or
artifact integrity failures.

Do not report real-corpus metrics when the 30 reviewed questions are absent. Synthetic fixtures
may verify code behavior but cannot satisfy the Day 8 output.

## Delivery split

### Hard tasks: coding and data contracts

1. Release-lock resolver and retrieval schemas.
2. Gold-question contract, validator, stable IDs, and reviewed 30-question artifact.
3. Deterministic table-document builder and metadata postings.
4. Atomic BM25 index builder and integrity manifest.
5. Filter-first retriever with stable ranking and score traces.
6. Evaluator, deterministic JSON/Markdown reports, and CLI integration.

Every production behavior follows test-driven development with a failing unit or integration test
before implementation.

### Easier tasks: verification and evidence

1. Run focused unit/integration suites after each hard task.
2. Validate the release lock and all 30 gold records.
3. Build the real index twice and compare deterministic artifact hashes.
4. Run all 30 questions; inspect empty results, missing gold IDs, and score ties.
5. Generate and compare evaluation JSON/Markdown on replay.
6. Run full pytest, Ruff, mypy, and `git diff --check`; record exact outputs and observed metrics.

## Test strategy

- Schema tests: canonical IDs, tuple ordering, duplicates, unknown IDs, incompatible filters.
- Document tests: deterministic serialization, alias inclusion, empty fields, Unicode, no numeric
  noise, release joins.
- Filter tests: AND/OR semantics, unknown metadata, empty result, stable row mapping.
- Ranking tests: eligible-only ranking, tie-breaks, finite scores, top-k bounds, trace content.
- Evaluation tests: known precision/recall/F2 values, multi-gold questions, zero hits, macro/by-intent
  aggregation, deterministic serialization.
- Integration tests: release lock -> build index -> retrieve -> evaluate using a small Parquet
  fixture; CLI exit codes and corruption detection.
- Real-corpus checks: 146,011 table documents, exact locked fingerprint, 30 validated questions,
  replay-identical reports.

## Skill deliverable

Create `.agents/skills/vifinqa-bm25-retrieval/` with a concise `SKILL.md` and
`agents/openai.yaml`. The skill triggers for Day 8 BM25 indexing, filter-first table retrieval,
gold retrieval validation, score tracing, evaluation, and retrieval regression work. It mandates
release-lock consumption, TDD, deterministic artifacts, and truthful real-corpus verification.
