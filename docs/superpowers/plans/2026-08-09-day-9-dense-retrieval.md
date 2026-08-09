# Day 9 Dense Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible BGE-M3 and multilingual-e5-small dense table-retrieval baselines, evaluate both against the immutable 30-question gold set, and compare them honestly with BM25 v3.

**Architecture:** Reuse the locked Day 8 `TableDocument` corpus and metadata filters. Publish one shared dense corpus plus one exact normalized `faiss.IndexFlatIP` artifact per pinned encoder, restrict every search with eligible row IDs before ranking, cache query vectors by encoder/query hash, and produce deterministic evaluation projections with separate operational timings.

**Tech Stack:** Python 3.11, Pydantic 2, NumPy float32, Sentence Transformers, Torch CPU, FAISS CPU, pytest, Ruff, mypy.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-08-09-day-9-dense-retrieval-design.md`.
- Consume only `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`; require fingerprint `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Preserve `data/qa/retrieval-gold-v1.jsonl` byte-for-byte; expected SHA-256 is `13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5`.
- Pin BGE-M3 to `5617a9f61b028005a4858fdac845db406aefb181` and multilingual-e5-small to `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Official evidence uses CPU, float32, normalized embeddings, exact `IndexFlatIP`, and maximum sequence length 512.
- E5 inputs require `query: ` and `passage: `; BGE-M3 inputs have no prefix.
- Apply the existing explicit metadata filters before dense ranking; automatic parsing remains Day 10.
- Rank all eligible rows by `(-cosine_score, table_id)` before slicing top-k; do not use oversampling or post-filter fallback.
- Evaluation remains fixed at `k=10`, `precision=TP/10`, `recall=TP/|gold|`, and `F2=5PR/(4P+R)`.
- Day 9 has no quality floor; dense retrieval may underperform BM25.
- Unit and fixture integration tests must not download models or access the network.
- Preserve unrelated worktree changes. In particular, do not stage, restore, or delete the currently user-owned `data/qa/week1_pilot_37a61be7aebd/*` changes.
- If the release lock or either pinned model is unavailable, report Day 9 evidence as incomplete; do not substitute fixtures or another revision.

---

## File responsibility map

| File | Responsibility |
|---|---|
| `src/financial_report_qa/retrieval/dense_contracts.py` | Frozen encoder, corpus, index, candidate, trace, latency, and comparison contracts |
| `src/financial_report_qa/retrieval/dense_artifacts.py` | Canonical JSON, SHA-256, atomic text/NumPy/FAISS publication helpers |
| `src/financial_report_qa/retrieval/dense_corpus.py` | Build, save, and load the shared sorted `TableDocument` corpus |
| `src/financial_report_qa/retrieval/dense_encoder.py` | Approved encoder specs, canonical spec hash, fakeable encoder protocol, real Sentence Transformers adapter |
| `src/financial_report_qa/retrieval/dense_index.py` | Batched vector validation, exact FAISS build/save/load, row identity checks |
| `src/financial_report_qa/retrieval/filtering.py` | Shared Day 8/Day 9 metadata eligibility logic |
| `src/financial_report_qa/retrieval/dense_cache.py` | NFKC query normalization, cache keys, atomic `.npy` read/write and validation |
| `src/financial_report_qa/retrieval/dense_service.py` | Filter-first exact dense search and stable trace construction |
| `src/financial_report_qa/retrieval/dense_evaluation.py` | Dense scoring, cold/warm runs, deterministic projection, BM25 comparison, reports |
| `src/financial_report_qa/retrieval/cli.py` | Day 9 subcommands while preserving Day 8 behavior |
| `src/financial_report_qa/core/errors.py` | Dense input/artifact/model error types under `RetrievalError` |
| `tests/unit/retrieval/test_dense_*.py` | Focused contracts, corpus, encoder, index, cache, service, and evaluation tests |
| `tests/integration/retrieval/test_day9_dense_cli.py` | Network-free fixture lifecycle for every Day 9 CLI command |
| `README.md` | Reproducible Day 9 commands and observed evidence after real runs |
| `plan.md` | Mark Day 9 complete only after both real encoder replays and comparison evidence exist |

---

### Task 1: Add dense contracts and artifact primitives

**Files:**
- Create: `src/financial_report_qa/retrieval/dense_contracts.py`
- Create: `src/financial_report_qa/retrieval/dense_artifacts.py`
- Modify: `src/financial_report_qa/core/errors.py`
- Create: `tests/unit/retrieval/test_dense_contracts.py`

**Interfaces:**
- Produces: `DenseEncoderSpec`, `DenseCorpusManifest`, `DenseIndexManifest`, `DenseRetrievalCandidate`, `DenseRetrievalTrace`, `LatencySummary`, `DenseRunObservation`.
- Produces: `canonical_json_bytes(value) -> bytes`, `sha256_bytes(value) -> str`, `file_sha256(path) -> str`, `write_text_atomic(path, content) -> None`, `write_numpy_atomic(path, vector) -> None`.
- Produces: `DenseInputError`, `DenseArtifactError`, and `DenseModelError`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_encoder_spec_rejects_unpinned_or_invalid_contracts() -> None:
    valid = {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "revision": "5" * 40,
        "dimension": 1024,
        "max_sequence_length": 512,
        "query_prefix": "",
        "document_prefix": "",
        "pooling": "sentence_transformers",
        "normalize_embeddings": True,
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 8,
    }
    assert DenseEncoderSpec.model_validate(valid).dimension == 1024
    with pytest.raises(ValidationError):
        DenseEncoderSpec.model_validate({**valid, "revision": "main"})
    with pytest.raises(ValidationError):
        DenseEncoderSpec.model_validate({**valid, "normalize_embeddings": False})


def test_dense_candidate_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValidationError, match="finite"):
        DenseRetrievalCandidate(
            row_id=0,
            table_id="tbl_" + "a" * 64,
            score=float("nan"),
            rank=1,
            metadata=_metadata("a"),
            snippet="title: revenue",
        )


def test_canonical_json_hash_is_key_order_independent() -> None:
    assert sha256_bytes(canonical_json_bytes({"b": 2, "a": 1})) == sha256_bytes(
        canonical_json_bytes({"a": 1, "b": 2})
    )
```

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_contracts.py
```

Expected: FAIL during collection because `financial_report_qa.retrieval.dense_contracts` does not exist.

- [ ] **Step 3: Implement the frozen public contracts**

Define exact literals and fields:

```python
EncoderName = Literal["bge-m3", "multilingual-e5-small"]
DenseEmptyReason = Literal["no_eligible_documents"]


class DenseEncoderSpec(_FrozenModel):
    name: EncoderName
    model_id: NonEmptyString
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    dimension: int = Field(gt=0)
    max_sequence_length: Literal[512] = 512
    query_prefix: str
    document_prefix: str
    pooling: Literal["sentence_transformers"] = "sentence_transformers"
    normalize_embeddings: Literal[True] = True
    dtype: Literal["float32"] = "float32"
    device: Literal["cpu"] = "cpu"
    batch_size: int = Field(gt=0)


class DenseCorpusManifest(_FrozenModel):
    schema_version: Literal["dense-corpus-v1"] = "dense-corpus-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_count: int = Field(ge=0)
    document_sha256: Fingerprint
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)


class DenseIndexManifest(_FrozenModel):
    schema_version: Literal["dense-index-v1"] = "dense-index-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_sha256: Fingerprint
    encoder: DenseEncoderSpec
    encoder_spec_sha256: Fingerprint
    document_count: int = Field(ge=0)
    dimension: int = Field(gt=0)
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    metric: Literal["inner_product"] = "inner_product"
    dtype: Literal["float32"] = "float32"
    normalized: Literal[True] = True
    index_byte_size: int = Field(ge=0)
    library_versions: dict[str, NonEmptyString]
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)
```

Add `DenseRetrievalCandidate(row_id, table_id, score, rank, metadata, snippet)` with the same finite-score validator as `RetrievalCandidate`. Add `DenseRetrievalTrace(question_id, query, normalized_query_sha256, encoder_spec_sha256, cache_hit, eligible_count, filter_decisions, results, empty_reason)` and frozen latency/run-observation contracts with non-negative values.

- [ ] **Step 4: Implement deterministic artifact helpers and error types**

Use compact UTF-8 canonical JSON:

```python
def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

`write_text_atomic` and `write_numpy_atomic` must create a sibling temporary file, flush and `os.fsync`, then `Path.replace`; on exceptions they unlink only that resolved temporary file. Add the three dense errors under `RetrievalError`, with model download/revision failures classified as `DenseModelError` rather than unexpected programming errors.

- [ ] **Step 5: Run focused quality gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_contracts.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/dense_contracts.py src/financial_report_qa/retrieval/dense_artifacts.py src/financial_report_qa/core/errors.py tests/unit/retrieval/test_dense_contracts.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/dense_contracts.py src/financial_report_qa/retrieval/dense_artifacts.py src/financial_report_qa/core/errors.py tests/unit/retrieval/test_dense_contracts.py
git diff --check
```

Expected: all commands PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/financial_report_qa/retrieval/dense_contracts.py src/financial_report_qa/retrieval/dense_artifacts.py src/financial_report_qa/core/errors.py tests/unit/retrieval/test_dense_contracts.py
git commit -m "feat(retrieval): add dense artifact contracts"
```

---

### Task 2: Publish the shared dense corpus

**Files:**
- Create: `src/financial_report_qa/retrieval/dense_corpus.py`
- Create: `tests/unit/retrieval/test_dense_corpus.py`

**Interfaces:**
- Consumes: `TableDocument`, `DenseCorpusManifest`, canonical artifact helpers.
- Produces: `DenseCorpus(documents: tuple[TableDocument, ...], manifest: DenseCorpusManifest)`.
- Produces: `build_dense_corpus(documents, *, dataset_fingerprint, release_lock_sha256) -> DenseCorpus`.
- Produces: `save_dense_corpus(corpus, output_dir) -> Path` and `load_dense_corpus(corpus_dir, *, release_lock_sha256) -> DenseCorpus`.

- [ ] **Step 1: Write failing corpus identity tests**

```python
def test_dense_corpus_sorts_documents_and_persists_stable_rows(tmp_path: Path) -> None:
    corpus = build_dense_corpus(
        tuple(reversed(_documents())),
        dataset_fingerprint="f" * 64,
        release_lock_sha256="e" * 64,
    )
    target = tmp_path / "corpus"
    save_dense_corpus(corpus, target)
    loaded = load_dense_corpus(target, release_lock_sha256="e" * 64)
    assert [item.table_id for item in loaded.documents] == [
        "tbl_" + "a" * 64,
        "tbl_" + "b" * 64,
    ]
    assert loaded.manifest.artifact_sha256["documents.jsonl"] == hashlib.sha256(
        (target / "documents.jsonl").read_bytes()
    ).hexdigest()


def test_dense_corpus_rejects_duplicate_table_ids() -> None:
    with pytest.raises(ValueError, match="unique table IDs"):
        build_dense_corpus(
            (_documents()[0], _documents()[0]),
            dataset_fingerprint="f" * 64,
            release_lock_sha256="e" * 64,
        )


def test_dense_corpus_loader_rejects_document_corruption(tmp_path: Path) -> None:
    target = _saved_corpus(tmp_path)
    (target / "documents.jsonl").write_bytes(
        (target / "documents.jsonl").read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="artifact hash"):
        load_dense_corpus(target, release_lock_sha256="e" * 64)
```

- [ ] **Step 2: Run the tests and verify failure**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_corpus.py
```

Expected: FAIL because `dense_corpus.py` is absent.

- [ ] **Step 3: Implement the in-memory corpus and deterministic document bytes**

```python
@dataclass(frozen=True)
class DenseCorpus:
    documents: tuple[TableDocument, ...]
    manifest: DenseCorpusManifest


def document_line(document: TableDocument) -> bytes:
    return canonical_json_bytes(document.model_dump(mode="json"))


def documents_sha256(documents: tuple[TableDocument, ...]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document_line(document))
    return digest.hexdigest()
```

`build_dense_corpus` sorts by `table_id`, rejects duplicates, computes the hash from exact emitted lines, and constructs `dense-corpus-v1` with the explicit lock hash.

- [ ] **Step 4: Implement atomic save/load and existing-target behavior**

`save_dense_corpus` writes `documents.jsonl`, hashes it, writes sorted/indented `manifest.json`, and atomically renames a temporary sibling directory. If the target exists, call `load_dense_corpus`; return only when manifest identity and document bytes match, otherwise raise `DenseArtifactError`.

`load_dense_corpus` must validate the manifest is a JSON object before Pydantic parsing, require `dense-corpus-v1`, verify the exact artifact set `{"documents.jsonl"}`, compare hashes, parse every nonblank JSONL row as `TableDocument`, and recheck sorted unique IDs, count, document hash, and release-lock hash.

- [ ] **Step 5: Run corpus and Day 8 regression gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_corpus.py tests/unit/retrieval/test_table_documents.py tests/unit/retrieval/test_index_service.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/dense_corpus.py tests/unit/retrieval/test_dense_corpus.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/dense_corpus.py tests/unit/retrieval/test_dense_corpus.py
git diff --check
```

Expected: all commands PASS and BM25 v3 document behavior remains unchanged.

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/financial_report_qa/retrieval/dense_corpus.py tests/unit/retrieval/test_dense_corpus.py
git commit -m "feat(retrieval): publish shared dense corpus"
```

---

### Task 3: Build and verify both exact FAISS indexes

**Files:**
- Create: `src/financial_report_qa/retrieval/dense_encoder.py`
- Create: `src/financial_report_qa/retrieval/dense_index.py`
- Create: `tests/unit/retrieval/test_dense_encoder.py`
- Create: `tests/unit/retrieval/test_dense_index.py`

**Interfaces:**
- Consumes: `DenseCorpus`, `DenseEncoderSpec`, FAISS and Sentence Transformers.
- Produces: `approved_encoder_spec(name: EncoderName) -> DenseEncoderSpec`.
- Produces: `encoder_spec_sha256(spec: DenseEncoderSpec) -> str`.
- Produces: `DenseEncoder` protocol with `spec`, `encode_documents(texts)`, and `encode_query(text)`.
- Produces: `SentenceTransformerDenseEncoder(spec, *, local_files_only: bool)`.
- Produces: `DenseIndex(corpus, faiss_index, manifest)` plus build/save/load functions.

- [ ] **Step 1: Write failing encoder-spec and fake-backend tests**

```python
def test_approved_encoder_specs_are_fully_pinned() -> None:
    bge = approved_encoder_spec("bge-m3")
    e5 = approved_encoder_spec("multilingual-e5-small")
    assert (bge.revision, bge.dimension, bge.query_prefix, bge.batch_size) == (
        "5617a9f61b028005a4858fdac845db406aefb181", 1024, "", 8
    )
    assert (e5.revision, e5.dimension, e5.query_prefix, e5.document_prefix) == (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3", 384, "query: ", "passage: "
    )


def test_encoder_spec_hash_changes_with_prefix_policy() -> None:
    spec = approved_encoder_spec("multilingual-e5-small")
    changed = spec.model_copy(update={"query_prefix": ""})
    assert encoder_spec_sha256(spec) != encoder_spec_sha256(changed)
```

The real adapter test monkeypatches `sentence_transformers.SentenceTransformer` and asserts the constructor receives the full revision, `device="cpu"`, `trust_remote_code=False`, and `local_files_only=True`; it also asserts documents receive `passage: ` and queries receive `query: `.

- [ ] **Step 2: Write failing index behavior tests**

```python
def test_dense_index_batches_normalized_float32_vectors(tmp_path: Path) -> None:
    corpus = _corpus(three_documents=True)
    encoder = FakeEncoder(
        approved_encoder_spec("multilingual-e5-small").model_copy(
            update={"dimension": 2, "batch_size": 2}
        )
    )
    built = build_dense_index(corpus, encoder)
    assert built.faiss_index.ntotal == 3
    assert encoder.document_batches == [2, 1]
    target = tmp_path / "e5-index"
    save_dense_index(built, target)
    loaded = load_dense_index(
        target,
        corpus,
        expected_encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
        release_lock_sha256="e" * 64,
    )
    assert loaded.faiss_index.d == 2


@pytest.mark.parametrize(
    "bad_vector",
    (
        np.asarray([[np.nan, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
    ),
)
def test_dense_index_rejects_invalid_encoder_output(bad_vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_dense_index(_one_document_corpus(), FixedOutputEncoder(bad_vector))
```

- [ ] **Step 3: Run the focused tests and verify failure**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_encoder.py tests/unit/retrieval/test_dense_index.py
```

Expected: FAIL because the encoder and index modules do not exist.

- [ ] **Step 4: Implement approved specs and the real encoder adapter**

The approved constants must be literal code, not environment-derived aliases. The adapter uses a lazy import inside `__init__` so importing retrieval does not load Torch:

```python
self._model = SentenceTransformer(
    spec.model_id,
    revision=spec.revision,
    device=spec.device,
    trust_remote_code=False,
    local_files_only=local_files_only,
)
self._model.max_seq_length = spec.max_sequence_length
```

`encode_documents` prepends `document_prefix`; `encode_query` prepends `query_prefix`. Both call `SentenceTransformer.encode` with `convert_to_numpy=True`, `normalize_embeddings=True`, and `show_progress_bar=False`, cast with `np.asarray(values, dtype=np.float32)`, and return 2-D document batches or one 1-D query vector. Convert model/revision/download failures to `DenseModelError` while allowing shape bugs in project code to surface during validation.

- [ ] **Step 5: Implement batched build and integrity-checked persistence**

```python
@dataclass(frozen=True)
class DenseIndex:
    corpus: DenseCorpus
    faiss_index: faiss.IndexFlatIP
    manifest: DenseIndexManifest


def validate_vector_batch(values: np.ndarray, *, rows: int, dimension: int) -> np.ndarray:
    if values.dtype != np.float32 or values.shape != (rows, dimension):
        raise ValueError("dense encoder returned an invalid float32 shape")
    if not np.isfinite(values).all():
        raise ValueError("dense encoder returned non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("dense encoder returned non-unit vectors")
    return np.ascontiguousarray(values)
```

`build_dense_index` slices documents in `spec.batch_size` chunks, encodes only that chunk, validates it, and calls `IndexFlatIP.add`. It records exact library versions and corpus identity. `save_dense_index` writes `index.faiss` into a temporary sibling, records file bytes/hash in the final manifest, writes the manifest, and atomically publishes. `load_dense_index` verifies schema, identity, artifact set/hash, exact `IndexFlatIP` type, dimension, count, and reconstructs vectors in chunks of 4,096 to validate finiteness and unit norms without allocating a second full matrix.

- [ ] **Step 6: Run focused and import-isolation gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_encoder.py tests/unit/retrieval/test_dense_index.py
uv run --frozen --no-sync python -c "import sys; import financial_report_qa.retrieval.dense_index; assert 'sentence_transformers' not in sys.modules; assert 'torch' not in sys.modules"
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/dense_encoder.py src/financial_report_qa/retrieval/dense_index.py tests/unit/retrieval/test_dense_encoder.py tests/unit/retrieval/test_dense_index.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/dense_encoder.py src/financial_report_qa/retrieval/dense_index.py tests/unit/retrieval/test_dense_encoder.py tests/unit/retrieval/test_dense_index.py
git diff --check
```

Expected: all commands PASS without downloading a model.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/financial_report_qa/retrieval/dense_encoder.py src/financial_report_qa/retrieval/dense_index.py tests/unit/retrieval/test_dense_encoder.py tests/unit/retrieval/test_dense_index.py
git commit -m "feat(retrieval): build exact dense indexes"
```

---

### Task 4: Add shared filtering, query cache, and exact dense retrieval

**Files:**
- Create: `src/financial_report_qa/retrieval/filtering.py`
- Modify: `src/financial_report_qa/retrieval/service.py`
- Create: `src/financial_report_qa/retrieval/dense_cache.py`
- Create: `src/financial_report_qa/retrieval/dense_service.py`
- Create: `tests/unit/retrieval/test_dense_cache.py`
- Create: `tests/unit/retrieval/test_dense_service.py`
- Modify: `tests/unit/retrieval/test_index_service.py`

**Interfaces:**
- Produces: `eligible_positions(documents, filters) -> tuple[tuple[int, ...], tuple[FilterDecision, ...]]`.
- Produces: `normalize_dense_query(query) -> str` and `dense_query_sha256(query, spec_hash) -> str`.
- Produces: `QueryEmbeddingCache.get_or_encode(query, encoder) -> CachedQueryEmbedding`.
- Produces: `DenseRetrievalService.retrieve(query, *, filters, k=10, question_id=None) -> DenseRetrievalTrace`.

- [ ] **Step 1: Write failing cache tests**

```python
def test_query_cache_is_encoder_scoped_and_reuses_valid_vector(tmp_path: Path) -> None:
    encoder = CountingEncoder(_two_dimensional_spec())
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)
    first = cache.get_or_encode("  Lợi  nhuận  ", encoder)
    second = cache.get_or_encode("Lợi nhuận", encoder)
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.query_sha256 == second.query_sha256
    assert encoder.query_calls == 1
    np.testing.assert_array_equal(first.vector, second.vector)


def test_query_cache_fails_closed_on_wrong_dtype(tmp_path: Path) -> None:
    encoder = CountingEncoder(_two_dimensional_spec())
    cache = QueryEmbeddingCache(tmp_path, encoder.spec)
    cached = cache.get_or_encode("doanh thu", encoder)
    np.save(cached.path, np.asarray([1.0, 0.0], dtype=np.float64))
    with pytest.raises(DenseArtifactError, match="dtype"):
        cache.get_or_encode("doanh thu", encoder)
```

- [ ] **Step 2: Write failing filter-first and stable-tie tests**

```python
def test_dense_service_filters_inside_faiss_and_stabilizes_ties(tmp_path: Path) -> None:
    index, encoder = _equal_score_index_with_ineligible_best_row()
    service = DenseRetrievalService(index, encoder, QueryEmbeddingCache(tmp_path, encoder.spec))
    trace = service.retrieve(
        "doanh thu",
        filters=RetrievalFilters(company_codes=("VCB",)),
        k=2,
        question_id="retq_" + "1" * 64,
    )
    assert [item.table_id for item in trace.results] == [
        "tbl_" + "a" * 64,
        "tbl_" + "b" * 64,
    ]
    assert all(item.metadata.company_code == "VCB" for item in trace.results)
    assert trace.eligible_count == 2


def test_dense_service_returns_audited_empty_filter_result(tmp_path: Path) -> None:
    index, encoder = _fixture_index()
    service = DenseRetrievalService(index, encoder, QueryEmbeddingCache(tmp_path, encoder.spec))
    trace = service.retrieve(
        "doanh thu", filters=RetrievalFilters(company_codes=("MISSING",)), k=10
    )
    assert trace.results == ()
    assert trace.empty_reason == "no_eligible_documents"
    assert encoder.query_calls == 0
```

- [ ] **Step 3: Run tests and confirm missing behavior**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_cache.py tests/unit/retrieval/test_dense_service.py
```

Expected: FAIL because cache/service modules do not exist.

- [ ] **Step 4: Extract the exact Day 8 filter function**

Move `_eligible_positions` without semantic changes into:

```python
def eligible_positions(
    documents: Sequence[TableDocument], filters: RetrievalFilters
) -> tuple[tuple[int, ...], tuple[FilterDecision, ...]]:
```

Modify `RetrievalService.retrieve` to call this function. Keep OR within fields, AND across fields, sorted row IDs, and identical `FilterDecision` counts. Add a BM25 regression assertion comparing the decisions before and after extraction.

- [ ] **Step 5: Implement normalized atomic query caching**

Normalize with `" ".join(unicodedata.normalize("NFKC", query).split())` while preserving case and accents. Hash canonical JSON containing `encoder_spec_sha256`, normalized query, and `normalization_version="v1"`. Cache below `<root>/<spec-prefix>/<query-hash>.npy`.

Validate one-dimensional float32 shape `(spec.dimension,)`, finite values, and unit norm tolerance `1e-5` on both writes and reads. `get_or_encode` returns a frozen record containing vector, `cache_hit`, query hash, normalized query, and path. Never auto-repair a corrupt existing file.

- [ ] **Step 6: Implement exact restricted FAISS search**

`DenseRetrievalService.__init__` must require the encoder spec hash to equal both the loaded
index manifest and the cache spec; reject any mismatch before accepting a query.

```python
eligible_array = np.asarray(eligible, dtype=np.int64)
selector = faiss.IDSelectorBatch(eligible_array)
parameters = faiss.SearchParameters()
parameters.sel = selector
scores, rows = self._index.faiss_index.search(
    cached.vector.reshape(1, -1), len(eligible), params=parameters
)
```

Reject any negative row, duplicate row, ineligible row, missing row, or non-finite score. Build candidates for every eligible row, sort by `(-score, table_id)`, then slice `[:k]`. This all-row sort is required for deterministic ties at the top-k cutoff. Do not encode the query when the eligible set is empty.

- [ ] **Step 7: Run cache/service plus BM25 regression gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_cache.py tests/unit/retrieval/test_dense_service.py tests/unit/retrieval/test_index_service.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/filtering.py src/financial_report_qa/retrieval/service.py src/financial_report_qa/retrieval/dense_cache.py src/financial_report_qa/retrieval/dense_service.py tests/unit/retrieval/test_dense_cache.py tests/unit/retrieval/test_dense_service.py tests/unit/retrieval/test_index_service.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/filtering.py src/financial_report_qa/retrieval/service.py src/financial_report_qa/retrieval/dense_cache.py src/financial_report_qa/retrieval/dense_service.py tests/unit/retrieval/test_dense_cache.py tests/unit/retrieval/test_dense_service.py tests/unit/retrieval/test_index_service.py
git diff --check
```

Expected: all commands PASS.

- [ ] **Step 8: Commit Task 4**

```powershell
git add src/financial_report_qa/retrieval/filtering.py src/financial_report_qa/retrieval/service.py src/financial_report_qa/retrieval/dense_cache.py src/financial_report_qa/retrieval/dense_service.py tests/unit/retrieval/test_dense_cache.py tests/unit/retrieval/test_dense_service.py tests/unit/retrieval/test_index_service.py
git commit -m "feat(retrieval): add filter-first dense search"
```

---

### Task 5: Evaluate dense runs and compare with BM25 v3

**Files:**
- Create: `src/financial_report_qa/retrieval/dense_evaluation.py`
- Create: `tests/unit/retrieval/test_dense_evaluation.py`

**Interfaces:**
- Consumes: 30 `GoldRetrievalQuestion` records, `DenseRetrievalService`, existing `score_at_10` and `RetrievalMetrics`.
- Produces: `evaluate_dense_retrieval(retriever, questions, *, k=10) -> DenseEvaluationReport`.
- Produces: `evaluate_cold_and_warm(service, questions) -> DenseEvaluationRun`.
- Produces: `build_day9_comparison(bm25_report, bge_run, e5_run) -> Day9ComparisonReport`.
- Produces: `deterministic_projection(report) -> dict[str, object]` and report writers.

- [ ] **Step 1: Write failing dense evaluation tests**

```python
def test_dense_evaluation_reuses_fixed_day8_metric_math() -> None:
    report = evaluate_dense_retrieval(_DenseRetrieverFixture(), (_question(),))
    assert report.macro.precision == pytest.approx(0.1)
    assert report.macro.recall == pytest.approx(0.5)
    assert report.macro.f2 == pytest.approx(5 * 0.1 * 0.5 / 0.9)
    assert report.failure_counts == {
        "full_gold_hits": 0,
        "partial_gold_hits": 1,
        "zero_gold_hits": 0,
        "no_eligible_documents": 0,
    }


def test_cold_and_warm_predictions_match_but_cache_states_change() -> None:
    run = evaluate_cold_and_warm(_CachingDenseFixture(), (_question(),))
    assert run.cold_report.per_question[0].trace.cache_hit is False
    assert run.warm_report.per_question[0].trace.cache_hit is True
    assert run.cold_report.per_question[0].predicted_table_ids == (
        run.warm_report.per_question[0].predicted_table_ids
    )
```

- [ ] **Step 2: Write failing comparison and deterministic-projection tests**

```python
def test_day9_comparison_reports_dense_minus_bm25_delta() -> None:
    comparison = build_day9_comparison(_bm25_reference(), _bge_run(), _e5_run())
    assert comparison.systems["bm25-v3"].macro.recall == pytest.approx(0.8833333333333333)
    assert comparison.systems["bge-m3"].delta_vs_bm25.recall == pytest.approx(
        _bge_run().cold_report.macro.recall - 0.8833333333333333
    )


def test_deterministic_projection_excludes_wall_clock_and_cache_state() -> None:
    first = _comparison(cold_p95=0.1, cache_hit=False)
    second = _comparison(cold_p95=9.9, cache_hit=True)
    assert deterministic_projection(first) == deterministic_projection(second)
```

- [ ] **Step 3: Run tests and verify missing-module failure**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_evaluation.py
```

Expected: FAIL because `dense_evaluation.py` does not exist.

- [ ] **Step 4: Implement dense question scoring and failure taxonomy**

Sort questions by `question_id`, require `k == 10`, call the dense service with the gold filters, and reuse `score_at_10`. Persist predicted/gold/missing IDs and the full dense trace. Failure names are exactly `no_eligible_documents`, `zero_gold_hits`, `partial_gold_hits`, and `full_gold_hits`. Macro and per-intent means must include all questions, including empty results.

- [ ] **Step 5: Implement cold/warm timing and equivalence checks**

Time each question with `time.perf_counter`. The cold pass requires every non-empty-filter result to have `cache_hit=False`; the warm pass requires the corresponding trace to have `cache_hit=True`. Reject any prediction, score, filter decision, or metric difference between passes. Compute p50 with `statistics.median` and p95 by the nearest-rank index `ceil(0.95 * n) - 1` over sorted samples.

- [ ] **Step 6: Implement BM25 validation, comparison, and report rendering**

Load the existing Day 8 JSON as `RetrievalEvaluationReport`; require the locked fingerprint, 30 questions, and macro values within absolute tolerance `5e-8` of Precision `0.14666666666666667`, Recall `0.8833333333333333`, and F2 `0.4312169312169312`. Build system summaries for `bm25-v3`, `bge-m3`, and `multilingual-e5-small`, with overall/per-intent absolute values and dense-minus-BM25 deltas.

Write:

```text
retrieval-day9-dense-<fingerprint-prefix>.json
retrieval-day9-dense-<fingerprint-prefix>.md
```

Both files end with a newline and use stable key/system/question ordering. The full report retains build/index/latency observations. `deterministic_projection` recursively removes latency, build duration, and `cache_hit` before replay hashing; it retains encoder specs, scores, predictions, metrics, failures, filters, and identities.

- [ ] **Step 7: Run evaluation regression gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_dense_evaluation.py tests/unit/retrieval/test_evaluation.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/dense_evaluation.py tests/unit/retrieval/test_dense_evaluation.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/dense_evaluation.py tests/unit/retrieval/test_dense_evaluation.py
git diff --check
```

Expected: all commands PASS; Day 8 filenames and byte-stable reports remain unchanged.

- [ ] **Step 8: Commit Task 5**

```powershell
git add src/financial_report_qa/retrieval/dense_evaluation.py tests/unit/retrieval/test_dense_evaluation.py
git commit -m "feat(retrieval): evaluate day 9 dense baselines"
```

---

### Task 6: Add the network-free Day 9 CLI lifecycle

**Files:**
- Modify: `src/financial_report_qa/retrieval/cli.py`
- Modify: `tests/unit/test_cli.py`
- Create: `tests/integration/retrieval/test_day9_dense_cli.py`

**Interfaces:**
- Adds: `build-dense-corpus`, `build-dense-index`, `evaluate-dense`, and `compare-day9`.
- Preserves: `build-index`, `validate-gold`, and `evaluate` argument and exit behavior.
- Produces: `_load_dense_encoder(name, *, local_files_only)` as the single monkeypatch seam for fixture tests.

- [ ] **Step 1: Write a failing fixture CLI lifecycle**

Use the existing Day 8 Parquet release fixture and 30-record gold helper. Patch `resolve_retrieval_release` and `_load_dense_encoder` only. The fake encoders return deterministic normalized 2-D vectors but expose distinct encoder spec hashes.

```python
def test_day9_dense_cli_fixture_lifecycle_is_network_free_and_replayable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _patch_release_resolver(monkeypatch, _fixture_release(tmp_path))
    _patch_fake_encoder_loader(monkeypatch)
    assert main(_build_corpus_args(tmp_path)) == 0
    assert main(_build_index_args(tmp_path, "bge-m3", "bge-a")) == 0
    assert main(_build_index_args(tmp_path, "multilingual-e5-small", "e5-a")) == 0
    assert main(_evaluate_args(tmp_path, "bge-m3", "bge-a", "bge-report.json")) == 0
    assert main(_evaluate_args(tmp_path, "multilingual-e5-small", "e5-a", "e5-report.json")) == 0
    assert main(_compare_args(tmp_path)) == 0
    comparison = json.loads(_comparison_path(tmp_path).read_text(encoding="utf-8"))
    assert list(comparison["systems"]) == ["bge-m3", "bm25-v3", "multilingual-e5-small"]
```

Add corruption coverage by appending bytes to `index.faiss`; `evaluate-dense` must return `2` and print `retrieval error:`. Add a monkeypatch sentinel proving no `SentenceTransformer` constructor is called by the fixture lifecycle.

- [ ] **Step 2: Run the integration test and verify parser failure**

```powershell
uv run --frozen --no-sync pytest -q tests/integration/retrieval/test_day9_dense_cli.py tests/unit/test_cli.py
```

Expected: FAIL because the Day 9 subcommands are unknown.

- [ ] **Step 3: Add exact CLI arguments**

```text
build-dense-corpus:
  --release-lock PATH --output-root PATH

build-dense-index:
  --release-lock PATH --corpus-dir PATH --encoder {bge-m3,multilingual-e5-small}
  --output-root PATH --observation-path PATH [--local-files-only]

evaluate-dense:
  --release-lock PATH --corpus-dir PATH --index-dir PATH
  --encoder {bge-m3,multilingual-e5-small} --gold-path PATH
  --cache-dir PATH --observation-path PATH --output-path PATH

compare-day9:
  --release-lock PATH --bm25-report PATH --bge-report PATH --e5-report PATH
  --output-dir PATH
```

`build-dense-index` measures wall time around document encoding/index construction and writes atomic operational JSON containing encoder name/spec hash, build seconds, index bytes, and dataset fingerprint. `evaluate-dense` performs the cold/warm run and requires the matching build observation. All artifact paths are explicit; no command selects the newest directory.

Path construction is fixed: `build-dense-corpus` publishes and prints
`<output-root>/<fingerprint>/corpus`; `build-dense-index` publishes and prints
`<output-root>/<encoder-name>-<encoder-spec-prefix>`. `evaluate-dense` never searches for an
observation file.

- [ ] **Step 4: Implement command dispatch with narrow expected-error handling**

Resolve the release first for every command. Build the shared corpus with `build_table_documents`; verify corpus/index/gold fingerprint and lock hash at every boundary. Catch existing retrieval errors, dense expected errors, Pydantic validation, JSON decode, FAISS read/write `RuntimeError`, and `OSError`, returning `2`. Do not catch `ValueError` globally; unexpected programming errors must continue to reach the caller as enforced by `tests/unit/test_cli.py`.

- [ ] **Step 5: Run the complete fixture lifecycle and Day 8 CLI regression**

```powershell
uv run --frozen --no-sync pytest -q tests/integration/retrieval/test_day9_dense_cli.py tests/integration/retrieval/test_day8_cli.py tests/unit/test_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/cli.py tests/integration/retrieval/test_day9_dense_cli.py tests/unit/test_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/cli.py tests/integration/retrieval/test_day9_dense_cli.py tests/unit/test_cli.py
git diff --check
```

Expected: all commands PASS and the test records zero network/model loads.

- [ ] **Step 6: Run all focused retrieval gates**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval tests/integration/retrieval
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval tests/unit/retrieval tests/integration/retrieval
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval tests/unit/retrieval tests/integration/retrieval
git diff --check
```

Expected: all commands PASS.

- [ ] **Step 7: Commit Task 6**

```powershell
git add src/financial_report_qa/retrieval/cli.py tests/unit/test_cli.py tests/integration/retrieval/test_day9_dense_cli.py
git commit -m "feat(retrieval): add day 9 dense cli"
```

---

### Task 7: Build real indexes twice, compare evidence, and document Day 9

**Files:**
- Read only: `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`
- Read only: `data/qa/retrieval-gold-v1.jsonl`
- Read only: `artifacts/evaluations/retrieval-day8-37a61be7aebd.json`
- Generate: `data/indexes/dense/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/` and independent replay roots
- Generate: `artifacts/evaluations/day9/` and `artifacts/evaluations/day9-replay/`
- Modify after complete evidence only: `README.md`
- Modify after complete evidence only: `plan.md`

**Interfaces:**
- Consumes: completed Tasks 1-6 and both exact pinned model snapshots.
- Produces: two independent corpus/index builds per encoder, two dense evaluations per encoder, one BM25/dense comparison, exact hashes, and recorded verification output.

- [ ] **Step 1: Preflight immutable inputs and preserve the dirty worktree**

```powershell
git status --short
Test-Path 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
Test-Path 'data/qa/retrieval-gold-v1.jsonl'
Get-FileHash -Algorithm SHA256 'data/qa/retrieval-gold-v1.jsonl'
uv run --frozen --no-sync financial-report-qa retrieval validate-gold --release-lock 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json' --gold-path 'data/qa/retrieval-gold-v1.jsonl'
```

Expected: lock exists; gold hash is exactly `13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5`; validation reports 30 questions. If the lock is absent or inaccessible, stop real-data work and report incomplete evidence without restoring the deleted paths.

- [ ] **Step 2: Rebuild and verify the locked BM25 v3 reference report**

The root Day 8 report currently present in some workspaces may be the stale pre-remediation baseline. Build v3 from the immutable release instead of selecting it by filename:

```powershell
$lockPath = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
$goldPath = 'data/qa/retrieval-gold-v1.jsonl'
$bm25Root = 'data/indexes/bm25-day9-reference'
$bm25Output = 'artifacts/evaluations/day9/bm25-v3'
uv run --frozen --no-sync financial-report-qa retrieval build-index --release-lock $lockPath --output-root $bm25Root
$bm25Index = "$bm25Root/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
uv run --frozen --no-sync financial-report-qa retrieval evaluate --release-lock $lockPath --index-dir $bm25Index --gold-path $goldPath --output-dir $bm25Output
```

Expected: the index manifest is `bm25-index-v3` with `b=0.25`; the report at `artifacts/evaluations/day9/bm25-v3/retrieval-day8-37a61be7aebd.json` has TP `44`, Precision@10 `0.14666666666666667`, Recall@10 `0.8833333333333333`, and F2@10 `0.4312169312169312`. Stop if these locked values do not match; never fall back to `artifacts/evaluations/retrieval-day8-37a61be7aebd.json` when it contains the old 23-TP run.

- [ ] **Step 3: Populate and verify the two pinned model snapshots**

Run only with approved network access:

```powershell
uv run --frozen --no-sync python -c "from huggingface_hub import snapshot_download; print(snapshot_download('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181'))"
uv run --frozen --no-sync python -c "from huggingface_hub import snapshot_download; print(snapshot_download('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3'))"
```

Expected: each command prints a local snapshot path whose final component is the requested full SHA. Do not continue with `main`, a short SHA resolved later, or a different cached revision.

- [ ] **Step 4: Build and compare two independent dense corpora**

```powershell
$denseA = 'data/indexes/dense-day9-a'
$denseB = 'data/indexes/dense-day9-b'
uv run --frozen --no-sync financial-report-qa retrieval build-dense-corpus --release-lock $lockPath --output-root $denseA
uv run --frozen --no-sync financial-report-qa retrieval build-dense-corpus --release-lock $lockPath --output-root $denseB
Get-FileHash -Algorithm SHA256 "$denseA/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/corpus/documents.jsonl"
Get-FileHash -Algorithm SHA256 "$denseB/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/corpus/documents.jsonl"
```

Expected: both manifests report 146,011 documents and both document hashes match exactly.

- [ ] **Step 5: Build BGE-M3 twice on CPU**

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$corpusA = "$denseA/$fingerprint/corpus"
$corpusB = "$denseB/$fingerprint/corpus"
$bgeIndexA = (& uv run --frozen --no-sync financial-report-qa retrieval build-dense-index --release-lock $lockPath --corpus-dir $corpusA --encoder bge-m3 --output-root "$denseA/$fingerprint/encoders" --observation-path 'artifacts/evaluations/day9/bge-build.json' --local-files-only | Select-Object -Last 1).Trim()
$bgeIndexB = (& uv run --frozen --no-sync financial-report-qa retrieval build-dense-index --release-lock $lockPath --corpus-dir $corpusB --encoder bge-m3 --output-root "$denseB/$fingerprint/encoders" --observation-path 'artifacts/evaluations/day9-replay/bge-build.json' --local-files-only | Select-Object -Last 1).Trim()
```

Expected: each manifest reports revision `5617a9f61b028005a4858fdac845db406aefb181`, dimension 1024, count 146,011, CPU/float32/normalized, and `IndexFlatIP`.

- [ ] **Step 6: Build multilingual-e5-small twice on CPU**

```powershell
$e5IndexA = (& uv run --frozen --no-sync financial-report-qa retrieval build-dense-index --release-lock $lockPath --corpus-dir $corpusA --encoder multilingual-e5-small --output-root "$denseA/$fingerprint/encoders" --observation-path 'artifacts/evaluations/day9/e5-build.json' --local-files-only | Select-Object -Last 1).Trim()
$e5IndexB = (& uv run --frozen --no-sync financial-report-qa retrieval build-dense-index --release-lock $lockPath --corpus-dir $corpusB --encoder multilingual-e5-small --output-root "$denseB/$fingerprint/encoders" --observation-path 'artifacts/evaluations/day9-replay/e5-build.json' --local-files-only | Select-Object -Last 1).Trim()
```

Expected: each manifest reports revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, dimension 384, count 146,011, CPU/float32/normalized, and `IndexFlatIP`.

- [ ] **Step 7: Compare exact index and deterministic manifest hashes**

The previous build commands capture each emitted encoder directory, so run:

```powershell
Get-FileHash -Algorithm SHA256 "$bgeIndexA/index.faiss"
Get-FileHash -Algorithm SHA256 "$bgeIndexB/index.faiss"
Get-FileHash -Algorithm SHA256 "$e5IndexA/index.faiss"
Get-FileHash -Algorithm SHA256 "$e5IndexB/index.faiss"
Get-FileHash -Algorithm SHA256 "$bgeIndexA/manifest.json"
Get-FileHash -Algorithm SHA256 "$bgeIndexB/manifest.json"
Get-FileHash -Algorithm SHA256 "$e5IndexA/manifest.json"
Get-FileHash -Algorithm SHA256 "$e5IndexB/manifest.json"
```

Expected: A/B hashes match for both `index.faiss` and `manifest.json`. The build-observation hashes may differ because wall-clock duration is operational evidence.

- [ ] **Step 8: Evaluate cold/warm cache behavior for all four builds**

First require fresh cache roots; do not delete or silently reuse an existing cache:

```powershell
$cacheRoots = @(
  'data/indexes/dense-query-cache/day9-a-bge',
  'data/indexes/dense-query-cache/day9-b-bge',
  'data/indexes/dense-query-cache/day9-a-e5',
  'data/indexes/dense-query-cache/day9-b-e5'
)
if ($cacheRoots | Where-Object { Test-Path -LiteralPath $_ }) { throw 'Choose new explicit cache roots for a cold-cache run' }
```

```powershell
uv run --frozen --no-sync financial-report-qa retrieval evaluate-dense --release-lock $lockPath --corpus-dir $corpusA --index-dir $bgeIndexA --encoder bge-m3 --gold-path $goldPath --cache-dir 'data/indexes/dense-query-cache/day9-a-bge' --observation-path 'artifacts/evaluations/day9/bge-build.json' --output-path 'artifacts/evaluations/day9/bge-report.json'
uv run --frozen --no-sync financial-report-qa retrieval evaluate-dense --release-lock $lockPath --corpus-dir $corpusB --index-dir $bgeIndexB --encoder bge-m3 --gold-path $goldPath --cache-dir 'data/indexes/dense-query-cache/day9-b-bge' --observation-path 'artifacts/evaluations/day9-replay/bge-build.json' --output-path 'artifacts/evaluations/day9-replay/bge-report.json'
uv run --frozen --no-sync financial-report-qa retrieval evaluate-dense --release-lock $lockPath --corpus-dir $corpusA --index-dir $e5IndexA --encoder multilingual-e5-small --gold-path $goldPath --cache-dir 'data/indexes/dense-query-cache/day9-a-e5' --observation-path 'artifacts/evaluations/day9/e5-build.json' --output-path 'artifacts/evaluations/day9/e5-report.json'
uv run --frozen --no-sync financial-report-qa retrieval evaluate-dense --release-lock $lockPath --corpus-dir $corpusB --index-dir $e5IndexB --encoder multilingual-e5-small --gold-path $goldPath --cache-dir 'data/indexes/dense-query-cache/day9-b-e5' --observation-path 'artifacts/evaluations/day9-replay/e5-build.json' --output-path 'artifacts/evaluations/day9-replay/e5-report.json'
```

Expected: every report contains 30 cold misses followed by 30 warm hits; cold/warm predictions and metrics match exactly; no query has a non-finite score.

- [ ] **Step 9: Compare deterministic evaluation projections**

```powershell
uv run --frozen --no-sync python -c "import json; from pathlib import Path; from financial_report_qa.retrieval.dense_evaluation import DenseEvaluationRun, deterministic_projection; a=DenseEvaluationRun.model_validate_json(Path('artifacts/evaluations/day9/bge-report.json').read_text(encoding='utf-8')); b=DenseEvaluationRun.model_validate_json(Path('artifacts/evaluations/day9-replay/bge-report.json').read_text(encoding='utf-8')); assert deterministic_projection(a)==deterministic_projection(b); print('bge deterministic projection: identical')"
uv run --frozen --no-sync python -c "import json; from pathlib import Path; from financial_report_qa.retrieval.dense_evaluation import DenseEvaluationRun, deterministic_projection; a=DenseEvaluationRun.model_validate_json(Path('artifacts/evaluations/day9/e5-report.json').read_text(encoding='utf-8')); b=DenseEvaluationRun.model_validate_json(Path('artifacts/evaluations/day9-replay/e5-report.json').read_text(encoding='utf-8')); assert deterministic_projection(a)==deterministic_projection(b); print('e5 deterministic projection: identical')"
```

Expected: both commands print `identical`.

- [ ] **Step 10: Generate the Day 9 BM25-versus-dense comparison twice**

```powershell
uv run --frozen --no-sync financial-report-qa retrieval compare-day9 --release-lock $lockPath --bm25-report 'artifacts/evaluations/day9/bm25-v3/retrieval-day8-37a61be7aebd.json' --bge-report 'artifacts/evaluations/day9/bge-report.json' --e5-report 'artifacts/evaluations/day9/e5-report.json' --output-dir 'artifacts/evaluations/day9'
uv run --frozen --no-sync financial-report-qa retrieval compare-day9 --release-lock $lockPath --bm25-report 'artifacts/evaluations/day9/bm25-v3/retrieval-day8-37a61be7aebd.json' --bge-report 'artifacts/evaluations/day9-replay/bge-report.json' --e5-report 'artifacts/evaluations/day9-replay/e5-report.json' --output-dir 'artifacts/evaluations/day9-replay'
```

Expected: both comparison reports name all three systems, report overall and per-intent Precision@10/Recall@10/F2@10, failure counts, dense-minus-BM25 deltas, build/index size, and cold/warm p50/p95. Compare deterministic projections rather than full bytes because timings are intentionally retained.

- [ ] **Step 11: Inspect failures and record evidence without tuning gold**

For each encoder, list query IDs in `zero_gold_hits`, `partial_gold_hits`, and `no_eligible_documents`; inspect top-10 IDs/scores and filter decisions. Record whether dense recovers any of the three BM25 v3 zero-hit questions. Do not alter questions, explicit filters, model prefixes, gold IDs, or table documents in response to these 30 results.

- [ ] **Step 12: Run full verification**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval tests/integration/retrieval
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

Expected: record exact pass/fail counts. Fix only regressions introduced by Day 9. If repository-wide pre-existing failures remain, identify their files and prove focused dense/retrieval gates are green; do not claim the full gate passed.

- [ ] **Step 13: Update README and roadmap only after complete evidence**

In `README.md`, add the exact corpus/build/evaluate/compare commands, both model revisions, artifact hashes, index sizes, build times, cold/warm p50/p95, overall/per-intent metrics, failure counts, and deterministic-projection evidence. In `plan.md`, check the three Day 9 bullets and add a dated completion block only if both full encoder replays and both 30-question evaluations completed. State plainly when either dense model underperforms BM25.

- [ ] **Step 14: Commit only documentation and tracked evidence intended for source control**

```powershell
git add README.md plan.md
git diff --cached --check
git commit -m "docs(retrieval): record day 9 dense evidence"
```

Do not stage model caches, FAISS indexes, query caches, ignored evaluation artifacts, or unrelated `data/qa` changes.

---

## Final self-review checklist

- Every production behavior began with a failing focused test.
- Day 8 BM25 tests and CLI remain green.
- No unit/integration test downloads a model.
- Both model IDs and full revisions match the approved design.
- Corpus rows are sorted unique table IDs and both replays contain 146,011 rows.
- Every query applies metadata eligibility before exact FAISS scoring.
- Stable ranking sorts all eligible results by `(-score, table_id)` before top-k slicing.
- E5 prefixes are present; BGE prefixes are empty; both max lengths equal 512.
- Cache keys include the full encoder spec hash and normalized query.
- Corrupt cache/index/corpus artifacts fail closed.
- Evaluation uses exactly 30 reviewed questions and fixed-denominator Precision@10.
- Full reports retain timings; deterministic replay projections exclude only operational fields.
- No gold, query, filter, or document text was changed after observing dense results.
- Missing lock/model/replay evidence is reported as incomplete, never replaced by fixtures.
