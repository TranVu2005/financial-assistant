# Day 9 Dense Retrieval Baseline Design

## Objective

Build two independently measurable dense table-retrieval baselines over the immutable
`dataset-pilot-v1` corpus:

- `BAAI/bge-m3` at revision `5617a9f61b028005a4858fdac845db406aefb181`;
- `intfloat/multilingual-e5-small` at revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`.

Both baselines reuse the Day 8 table-document and explicit-filter contracts, use exact FAISS
CPU search, cache query embeddings, and produce an auditable comparison with BM25 v3 on the
same 30 reviewed gold questions. Day 9 is an honest baseline milestone: completion does not
require either dense encoder to outperform BM25.

## Source and evaluation boundary

All real-corpus work consumes the same release lock as Day 8:

```text
data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json
```

The resolver must continue to enforce alias `dataset-pilot-v1`, a passing Week 1 gate, and the
dataset fingerprint:

```text
37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f
```

Dense evaluation consumes the existing 30-question reviewed gold file without changing its
questions, filters, IDs, evidence, or labels:

```text
data/qa/retrieval-gold-v1.jsonl
SHA-256: 13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5
```

The BM25 v3 reference remains Precision@10 `0.1466667`, Recall@10 `0.8833333`, F2@10
`0.4312169`, 44 true positives, three zero-hit questions, and one partial-hit question. Dense
reports compare against these locked observations; they do not silently rebuild or relabel the
BM25 reference.

## Scope

Day 9 includes:

- a shared dense corpus derived from every Day 8 `TableDocument`;
- two pinned sentence-transformer encoder specifications;
- deterministic batched float32 document encoding on CPU;
- one exact `faiss.IndexFlatIP` index per encoder;
- explicit metadata filtering before dense ranking;
- an integrity-checked query embedding cache;
- dense evaluation and BM25-versus-dense comparison artifacts;
- replay evidence from independent builds.

Day 9 excludes automatic entity parsing, fusion, graph expansion, reranking, fine-tuning,
quantization, GPU-only evidence, approximate nearest-neighbor indexes, and changes to the gold
set. These belong to later milestones or separate versioned decisions.

## Encoder contract

Each encoder is represented by a frozen `DenseEncoderSpec` containing:

```python
class DenseEncoderSpec(BaseModel):
    model_id: str
    revision: str
    dimension: int
    max_sequence_length: int
    query_prefix: str
    document_prefix: str
    pooling: Literal["sentence_transformers"]
    normalize_embeddings: Literal[True]
    dtype: Literal["float32"]
    device: Literal["cpu"]
    batch_size: int
```

The two approved specifications are:

| Field | BGE-M3 | multilingual-e5-small |
|---|---|---|
| `model_id` | `BAAI/bge-m3` | `intfloat/multilingual-e5-small` |
| `revision` | `5617a9f61b028005a4858fdac845db406aefb181` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `dimension` | `1024` | `384` |
| `max_sequence_length` | `512` | `512` |
| `query_prefix` | empty | `query: ` |
| `document_prefix` | empty | `passage: ` |
| `batch_size` | `8` | `32` |

BGE-M3 supports longer input, but Day 9 deliberately pins both challengers to 512 tokens. The
retrieval documents contain compact table metadata rather than raw table cells, so this reduces
CPU build time and keeps the input budget comparable without adding a long-document experiment.
BGE-M3 receives no query instruction. Multilingual E5 always receives its documented asymmetric
`query: ` and `passage: ` prefixes, including for Vietnamese text.

The implementation loads only the pinned revision, disables remote custom code, encodes on CPU
in inference mode, requests NumPy float32 output, and L2-normalizes every vector. A produced
dimension that differs from the specification is an error. The manifest records the exact
resolved model revision and installed versions of Sentence Transformers, Transformers, Torch,
NumPy, and FAISS.

## Shared dense corpus

The corpus builder reuses `resolve_retrieval_release()` and `build_table_documents()` rather
than defining a second text serialization. The resulting documents must have the same
`table_id`, text, metadata, and sorted ordering as the Day 8 corpus contract. Metric observations
remain part of the persisted document representation for identity consistency, but dense
retrieval performs no Day 8 lexical query expansion.

The shared corpus is written once below the dataset fingerprint:

```text
data/indexes/dense/<dataset-fingerprint>/
  corpus/
    manifest.json
    documents.jsonl
```

`documents.jsonl` is sorted by `table_id`; its zero-based row position is the permanent dense
row ID. The corpus manifest records `schema_version="dense-corpus-v1"`, builder version, dataset
fingerprint, release-lock hash, document count, document hash, and artifact hashes. The builder
writes to a temporary sibling and publishes atomically. An existing target with different
content fails closed.

## Dense index artifacts

Each encoder writes an independent artifact directory:

```text
data/indexes/dense/<dataset-fingerprint>/
  encoders/
    bge-m3-<encoder-spec-prefix>/
      manifest.json
      index.faiss
    multilingual-e5-small-<encoder-spec-prefix>/
      manifest.json
      index.faiss
```

The full encoder spec is serialized canonically and hashed with SHA-256. The first 12 lowercase
hex characters form `<encoder-spec-prefix>`; the full hash remains in the manifest. The dense
manifest uses `schema_version="dense-index-v1"`, `builder_version="v1"`,
`index_type="IndexFlatIP"`, `metric="inner_product"`, `dtype="float32"`, and
`normalized=true`. It also records the corpus document hash, encoder contract, dataset identity,
document count, vector dimension, library versions, index byte size, and hashes of the final
files. Wall-clock build duration belongs to the evaluation evidence rather than the integrity
manifest so identical builds can retain identical manifests.

Document vectors are added to `IndexFlatIP` in corpus-row order. No separate embedding matrix is
persisted because a flat FAISS index already stores the vectors and can reconstruct them. This
avoids duplicating roughly 598 MB for BGE-M3 and 224 MB for multilingual E5 at 146,011 rows.
Artifacts are created through a temporary sibling and atomically renamed. Load verifies every
manifest field, file hash, row count, dimension, finite reconstructed vectors, and corpus
identity before returning an index.

## Query cache

Query text is normalized with Unicode NFKC and whitespace collapse; case and Vietnamese accents
are preserved. The encoder-specific query input is then produced by prepending the spec's query
prefix. The cache key is SHA-256 over canonical JSON containing:

- `encoder_spec_sha256`;
- normalized query text;
- normalization version `v1`.

Entries live at:

```text
data/indexes/dense/<dataset-fingerprint>/query-cache/
  <encoder-spec-prefix>/<query-sha256>.npy
```

Each entry is a NumPy `.npy` file containing exactly one normalized float32 vector. Writes use a
temporary sibling and atomic replacement. Reads verify the key-derived path, dtype, shape,
dimension, finite values, and unit norm within tolerance `1e-5`. An invalid cache entry fails
closed rather than being silently reused or overwritten. A separate explicit rebuild command
may replace a named corrupt entry after the operator removes or quarantines it.

## Filter-first retrieval and stable ranking

Dense retrieval keeps the exact Day 8 filter semantics:

- OR within company, period, and statement fields;
- AND across non-empty fields;
- unknown metadata never satisfies a requested value;
- empty filters leave the corpus unconstrained;
- an empty intersection returns `no_eligible_documents` without fallback.

The service resolves eligible corpus row IDs before search and passes those IDs through
`faiss.IDSelectorBatch` in `SearchParameters`. This is exact restricted search, not global
oversampling followed by post-filtering. It requests all `eligible_count` exact scores, rejects
negative FAISS sentinel IDs and non-finite scores, maps each row back to its immutable
`table_id`, applies the stable order `(-score, table_id)`, and only then slices the requested
top-k. Retrieving all eligible scores is deliberate: requesting only k would make an exact-score
tie at the cutoff depend on FAISS's internal tie order rather than `table_id`.

The dense trace records encoder spec hash, normalized query hash, cache hit/miss, eligible count,
filter decisions, candidate row IDs, table IDs, cosine scores, ranks, and empty reason. Because
all stored and query vectors are L2-normalized, the inner-product score is cosine similarity.

## Evaluation and comparison

Each dense encoder is evaluated independently at fixed `k=10` using the existing formulas:

```text
true_positive = |top_10_table_ids intersect gold_table_ids|
precision@10  = true_positive / 10
recall@10     = true_positive / |gold_table_ids|
F2@10         = 5 * precision * recall / (4 * precision + recall)
```

The evaluator reports macro Precision@10, Recall@10, and F2@10 over exactly 30 questions, the
same metrics by intent, total true positives, and counts of zero-hit, partial-hit, and full-hit
questions. Every query record includes gold IDs, predicted IDs, ranks, scores, filter trace,
cache state, and failure category.

One deterministic comparison artifact combines the locked BM25 v3 observations with both dense
runs:

```text
artifacts/evaluations/
  retrieval-day9-dense-<dataset-prefix>.json
  retrieval-day9-dense-<dataset-prefix>.md
```

The comparison records absolute metrics and dense-minus-BM25 deltas overall and by intent. It
also records encoder build duration, index byte size, and cold-cache and warm-cache query latency
with p50/p95 summaries. Latency is operational evidence and is excluded from byte-for-byte
report equality because wall-clock measurements vary; deterministic report comparison omits
timings from its canonical hash projection. Peak memory is not an acceptance metric because the
current Windows stack has no reliable cross-library peak-RSS measurement without adding another
runtime dependency.

Day 9 has no metric floor. A dense model may be selected for Day 10 fusion only after its quality,
latency, and storage trade-offs are visible in this report.

## CLI surface

Extend the existing retrieval CLI with:

```text
financial-report-qa retrieval build-dense-corpus
financial-report-qa retrieval build-dense-index --encoder bge-m3
financial-report-qa retrieval build-dense-index --encoder multilingual-e5-small
financial-report-qa retrieval evaluate-dense --encoder bge-m3
financial-report-qa retrieval evaluate-dense --encoder multilingual-e5-small
financial-report-qa retrieval compare-day9
```

All commands require the release lock explicitly. Index and evaluation commands also require
explicit artifact paths; they do not search for the newest directory. CLI success returns `0`.
Invalid identity, unavailable pinned model revision, corrupt artifacts/cache, invalid gold, and
non-finite vectors or scores return `2` with an actionable dense-retrieval error.

## Failure handling

Fail closed for:

- missing, invalid, or fingerprint-incompatible release locks;
- model ID or resolved revision drift;
- remote-code requirements;
- corpus count/order/hash drift;
- manifest schema, spec hash, dimension, or library-contract mismatch;
- duplicate table IDs or broken FAISS row mapping;
- non-finite or non-unit document/query vectors;
- corrupt index, cache, or report inputs;
- gold questions incompatible with the immutable release;
- non-finite FAISS scores or unexpected result IDs.

Model download failures are reported as unavailable pinned inputs. Unit and integration tests
never access the network; only explicitly invoked real-corpus build commands may populate the
local Hugging Face cache.

## Test strategy

Unit tests use a deterministic fake encoder and cover:

- frozen encoder specs and canonical spec hashes;
- query/document prefixes and 512-token configuration;
- corpus ordering and identity hashes;
- normalized finite float32 vectors and dimension rejection;
- atomic build/load and artifact corruption detection;
- query cache hit, miss, corruption, and encoder isolation;
- AND/OR filter semantics and `IDSelectorBatch` restricted search;
- exact cosine ranking, stable table-ID ties, and empty results;
- metric arithmetic and BM25 delta calculation;
- deterministic JSON/Markdown serialization excluding timing fields.

A fixture-only integration test exercises release resolution, dense corpus publication, fake
batch encoding, FAISS build/load, filtered retrieval, cold/warm cache behavior, evaluation, CLI
exit codes, and corruption rejection. It does not download either production model.

Real-corpus verification must:

1. validate the immutable release and all 30 reviewed gold questions;
2. build the shared 146,011-document dense corpus twice into independent roots;
3. build each pinned encoder twice on CPU into independent roots;
4. compare corpus and index hashes for each replay;
5. evaluate both encoder builds and compare deterministic report projections;
6. compare both dense systems with the locked BM25 v3 baseline;
7. inspect every zero-hit, partial-hit, non-finite score, tie, and empty result;
8. run focused retrieval tests, the full test suite, Ruff, mypy, and `git diff --check`.

If full encoder replay is too slow to finish within the Day 9 execution window, the milestone is
reported as incomplete with the completed build/evaluation evidence preserved. Cached document
embeddings or a single build cannot be relabeled as independent replay evidence.

## Definition of Done

Day 9 is complete only when:

- the shared corpus contains exactly 146,011 sorted, unique table documents bound to the locked
  fingerprint;
- both pinned encoder indexes pass integrity checks and preserve exact row-to-table identity;
- explicit filters are applied before exact FAISS ranking;
- query-cache cold and warm results are identical;
- both encoders evaluate all 30 reviewed questions at `k=10`;
- the JSON and Markdown comparison artifacts report overall and per-intent results against BM25
  v3, including latency and index size;
- two independent real-corpus builds and deterministic evaluation projections agree for each
  encoder;
- focused tests, full pytest, Ruff, mypy, and `git diff --check` have recorded outcomes;
- any unavailable model, replay, or quality result is reported truthfully rather than replaced
  with fixture evidence.

Dense retrieval is allowed to underperform BM25. Fusion, production-model selection, and entity
parsing remain Day 10 decisions driven by the Day 9 evidence.

## Primary references

- BGE-M3 model card: <https://huggingface.co/BAAI/bge-m3>
- Multilingual E5 small model card: <https://huggingface.co/intfloat/multilingual-e5-small>
- Day 8 design: `docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md`
- BM25 v3 remediation design:
  `docs/superpowers/specs/2026-08-09-bm25-length-normalization-design.md`
