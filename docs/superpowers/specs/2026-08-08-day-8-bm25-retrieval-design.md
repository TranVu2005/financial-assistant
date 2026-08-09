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
  `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.

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


class GoldTableEvidence(BaseModel):
    table_id: str
    relative_path: str
    line_start: int
    line_end: int
    verified: Literal[True]


class GoldRetrievalQuestion(BaseModel):
    question_id: str                 # retq_<64 lowercase hex>
    question: str                    # non-empty Vietnamese question
    intent: Literal["lookup", "compare", "growth"]
    filters: RetrievalFilters
    gold_table_ids: tuple[str, ...]  # sorted, unique, non-empty
    reviewed_by: str
    reviewed_at: datetime
    gold_evidence: tuple[GoldTableEvidence, ...]
    dataset_fingerprint: str
```

`question_id` is derived from normalized question text, filters, gold table IDs, contract
version, and dataset fingerprint. Gold table IDs must exist in the locked release and must be
compatible with the declared filters. Evidence must cover every gold ID exactly once and match
the released table's document path and source span. `reviewed_by` and `reviewed_at` must be
non-empty/valid, and every evidence row must have `verified=true`. Filters are explicit expert
annotations for Day 8;
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
4. authoritative company ticker;
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

## BM25 v2 corpus-bound metric expansion

The BM25 v2 document contract persists structured metric observations alongside the serialized
text:

```text
MetricLabelObservation(canonical: str, raw: str | None)
```

Observations are retained whenever the canonical identity exists, including `raw=None`. Raw
labels without a canonical identity are excluded. The persisted manifest uses
`schema_version="bm25-index-v2"`, `builder_version="v2"`, and
`query_expansion_version="v1"`; v1 indexes require a rebuild and are rejected by the loader.

The query lexicon is derived only from raw/canonical pairs observed in the locked corpus. An
alias mapping to more than one canonical metric is excluded. Expansion matches complete token
sequences, chooses the longest non-overlapping match, and merges canonical tokens with stable
deduplication. `RetrievalTrace.metric_expansions` records the alias, canonical metric, and only
the added tokens that are present in the index vocabulary. Retrieval has no runtime dependency
on `financial_report_qa.normalization`.

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

If no query token exists in the index vocabulary, return no candidates with
`empty_reason="no_index_tokens"`; never rank arbitrary zero-score tables.

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

## 2026-08-09 remediation evidence

The clean committed snapshot at `4ee2fa6` imported `RetrievalService` and passed 45 focused
retrieval tests, Ruff, and mypy. Two independent real-data BM25 v2 builds used lock
`data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`, fingerprint
`37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`, and 146,011 documents.
Their seven artifact files had zero hash differences; the manifest hash was
`B6007B13301E62E259C86BF23FE8ACC1014EB51E44D7E7E9B86A724DDB2E8484` and the document hash was
`b1206d17e7a870da727fd4ec70bb06bc707ede14de6641814ce1dfba418b7dd6`.

The reviewed gold file `data/qa/retrieval-gold-v1.jsonl` validated 30 questions (10 lookup,
10 compare, 10 growth; 10 companies; 18 multi-table) and retained SHA-256
`13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5`. Reports from
`artifacts/evaluations/remediation-v2-a/` and `remediation-v2-b/` were byte-identical:
JSON SHA-256 `70280CC6A277128F1F9C7CC05A5C6C96AEEDA3C62D4FF40C1BE150285F6E2AE9` and Markdown
SHA-256 `349F9927D9D5654F07E75B91A8ED58F0911EBCC07303D266B9B6F95136E28374`.

Observed clean-source macro metrics were Precision@10 `0.1366667`, Recall@10 `0.8333333`,
F2@10 `0.4034392`, and 41 true positives. By intent: lookup Recall/F2 `0.9000/0.3214`,
compare `0.8000/0.4444`, growth `0.8000/0.4444`. There were five `zero_gold_hits`, zero
partial hits, and no `no_eligible_documents` or `no_index_tokens` failures. The misses are
HDB/NVL ranking-fragmentation cases with matching metadata and metric expansions; no gold or
query-ID rule was changed. Therefore the remediation evidence is reproducible but does not meet
the provisional Recall/F2 floors `0.8833333/0.4179894`.

The fixed metric contract remains `precision=TP/10`, `recall=TP/|gold|`, and
`F2=5PR/(4P+R)`. For the current gold cardinalities the theoretical macro-F2 ceiling is
`0.476190476190476`; changing that formula or the Day-14 target requires a separate versioned
design decision.

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
