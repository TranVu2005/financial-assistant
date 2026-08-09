# Day 8 BM25 Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a deterministic, filter-first BM25 table retriever and truthful Recall@10/F2@10 baseline on exactly 30 reviewed gold questions bound to `dataset-pilot-v1`.

**Architecture:** Resolve every real run through the immutable Week-1 release lock, derive one auditable text document per canonical table, build a content-addressed `bm25s` index, restrict candidates with explicit expert metadata filters, and rank with stable BM25 scores. Evaluate deterministic query traces and macro metrics; never substitute synthetic fixtures for the missing reviewed gold artifact.

**Tech Stack:** Python 3.11, Pydantic 2, PyArrow/Parquet, DuckDB, NumPy, bm25s 0.3.x, orjson, pytest, Ruff, mypy, PowerShell.

## Global Constraints

- Design authority: `docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md`.
- Preserve unrelated dirty changes. Stage only files listed by the active task.
- Real retrieval must consume `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`; do not select an arbitrary `release_v2_*` directory.
- Expected dataset fingerprint: `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Expected release size: 1,971 documents, 146,011 canonical tables, and 6,199,661 cells.
- Source TXT, release Parquet, Week-1 gate artifacts, and release lock are immutable.
- Retrieval indexes and evaluation reports are rebuildable artifacts.
- Day 8 uses explicit expert filters. Automatic company/year/metric extraction remains Day 10 scope.
- Day 8 reports observed metrics; F2 >= 0.80 and Recall@10 >= 0.90 remain the Day-14 gate.
- Production behavior follows RED -> GREEN -> REFACTOR. Record the expected failing reason before implementation.
- If the release lock or 30 reviewed gold questions are unavailable, stop real-corpus verification and report the missing input. Synthetic results are not Day-8 metrics.

## File Map

| Path | Responsibility |
|---|---|
| `src/financial_report_qa/retrieval/contracts.py` | Immutable query, filter, document, candidate, trace, metric, and manifest contracts |
| `src/financial_report_qa/retrieval/release.py` | Fail-closed release-lock resolution and Parquet identity checks |
| `src/financial_report_qa/retrieval/gold.py` | Stable question IDs and strict JSONL loading/validation |
| `src/financial_report_qa/retrieval/documents.py` | Deterministic Parquet joins and one text document per table |
| `src/financial_report_qa/retrieval/index.py` | Tokenization, BM25 build/load, hashes, and atomic publication |
| `src/financial_report_qa/retrieval/service.py` | Metadata candidate restriction, stable ranking, and score traces |
| `src/financial_report_qa/retrieval/evaluation.py` | Per-query and macro Recall@10/F2@10 plus deterministic reports |
| `src/financial_report_qa/retrieval/cli.py` | `build-index`, `validate-gold`, and `evaluate` commands |
| `src/financial_report_qa/core/errors.py` | Retrieval input/artifact exception hierarchy |
| `src/financial_report_qa/cli.py` | Top-level `retrieval` command dispatch |
| `tests/unit/retrieval/` | Focused schema, gold, document, index, service, and metric tests |
| `tests/integration/retrieval/test_day8_cli.py` | Small Parquet fixture exercising the full CLI lifecycle |
| `data/qa/retrieval-gold-v1.jsonl` | Exactly 30 reviewed questions bound to the locked dataset |
| `data/indexes/bm25/<fingerprint>/` | Rebuildable BM25 documents, sparse index, and integrity manifest |
| `artifacts/evaluations/` | Deterministic Day-8 JSON and Markdown evidence |
| `.agents/skills/vifinqa-bm25-retrieval/SKILL.md` | Reusable agent guardrails for implementing and verifying this plan |
| `.agents/skills/vifinqa-bm25-retrieval/agents/openai.yaml` | Skill discovery metadata and default invocation prompt |

---

# Part I — Hard Tasks: Coding and Gold Contracts

### Task 1: Add retrieval contracts and fail-closed release resolution

**Files:**

- Create: `src/financial_report_qa/retrieval/contracts.py`
- Create: `src/financial_report_qa/retrieval/release.py`
- Modify: `src/financial_report_qa/core/errors.py`
- Modify: `src/financial_report_qa/retrieval/__init__.py`
- Create: `tests/unit/retrieval/test_contracts.py`
- Create: `tests/unit/retrieval/test_release.py`

**Interfaces:**

- Produces: `RetrievalFilters`, `GoldTableEvidence`, `GoldRetrievalQuestion`, `TableMetadata`, `TableDocument`, `RetrievalCandidate`, `FilterDecision`, `RetrievalTrace`, and `Bm25IndexManifest`.
- Produces: `ResolvedRetrievalRelease(lock, release_path, manifest, lock_sha256)`.
- Produces: `resolve_retrieval_release(lock_path: Path, *, repo_root: Path) -> ResolvedRetrievalRelease`.
- Consumers: every later task receives the resolved release rather than accepting a raw release directory.

- [ ] **Step 1: Write failing immutable-contract tests**

Add tests proving filters are sorted/unique, table/question IDs are canonical, unknown fields are rejected, and tuples are frozen:

```python
def test_retrieval_filters_reject_noncanonical_values() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilters(company_codes=("VCB", "ACB"))


def test_gold_question_requires_sorted_unique_gold_ids() -> None:
    with pytest.raises(ValidationError):
        GoldRetrievalQuestion(
            question_id="retq_" + "0" * 64,
            question="Doanh thu thuần của VCB năm 2023?",
            intent="lookup",
            filters=RetrievalFilters(company_codes=("VCB",), periods=("2023",)),
            gold_table_ids=("tbl_" + "f" * 64, "tbl_" + "0" * 64),
            dataset_fingerprint="a" * 64,
        )
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_contracts.py
```

Expected: FAIL because the retrieval contracts do not exist.

- [ ] **Step 2: Implement the minimal Pydantic contracts**

Use `ConfigDict(extra="forbid", frozen=True)` on every persisted or public model. Canonical tuple validators must require the supplied tuple to equal `tuple(sorted(set(value)))`; do not silently reorder user input.

Define:

```python
RetrievalIntent = Literal["lookup", "compare", "growth"]


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    company_codes: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    statement_types: tuple[str, ...] = ()


class GoldTableEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    table_id: str = Field(pattern=r"^tbl_[0-9a-f]{64}$")
    relative_path: Annotated[str, StringConstraints(min_length=1)]
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    verified: Literal[True]


class GoldRetrievalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    question_id: str = Field(pattern=r"^retq_[0-9a-f]{64}$")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    intent: RetrievalIntent
    filters: RetrievalFilters
    gold_table_ids: tuple[str, ...]
    reviewed_by: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    reviewed_at: datetime
    gold_evidence: tuple[GoldTableEvidence, ...]
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
```

Include `table_id`, `doc_id`, `text`, structured metadata, rank, finite float score, matched tokens, eligible count, and optional empty-result reason in the remaining contracts.

`FilterDecision` has `field: Literal["company_codes", "periods", "statement_types"]`, canonical `requested_values`, `matched_count_before_intersection`, and `eligible_count_after_intersection`. `RetrievalTrace.filter_decisions` contains one decision for every non-empty filter field in company/period/statement order.

- [ ] **Step 3: Write failing release identity tests**

Build a small temporary release containing manifest JSON plus three Parquet files. Test:

- missing lock;
- alias other than `dataset-pilot-v1`;
- missing/failed Week-1 gate;
- lock/gate/release fingerprint disagreement;
- unsafe `..` paths;
- missing required Parquet;
- manifest table count differing from `tables.parquet`;
- valid relative lock resolution.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_release.py
```

Expected: FAIL because `resolve_retrieval_release` does not exist.

- [ ] **Step 4: Implement release resolution**

Add:

```python
class RetrievalError(Exception):
    """Base error for lexical retrieval."""


class RetrievalInputError(RetrievalError):
    """Invalid release, gold question, or CLI input."""


class RetrievalArtifactError(RetrievalError):
    """Corrupt or identity-mismatched generated artifact."""
```

Resolve `repo_root` and `lock_path`, require the lock to stay inside `repo_root`, and resolve every relative field against `repo_root` (never against the nested lock directory or current file parent). Reject traversal, read the existing `ReleaseLock`, require a passing gate, compare all fingerprints, and use `pyarrow.parquet.read_metadata` to verify table/document/cell row counts without loading full data. Product CLI calls pass `repo_root=Path.cwd()`; temporary tests create a complete repository-shaped root and pass it explicitly.

- [ ] **Step 5: Verify and commit Task 1**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_release.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval src/financial_report_qa/core/errors.py tests/unit/retrieval
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval src/financial_report_qa/core/errors.py tests/unit/retrieval
git diff --check
git add src/financial_report_qa/retrieval src/financial_report_qa/core/errors.py tests/unit/retrieval
git commit -m "feat(retrieval): add day 8 contracts and release boundary"
```

Expected: every verification command exits 0; only Task-1 paths are staged.

---

### Task 2: Define and validate the 30-question gold retrieval set

**Files:**

- Create: `src/financial_report_qa/retrieval/gold.py`
- Create: `tests/unit/retrieval/test_gold.py`
- Create: `data/qa/retrieval-gold-v1.jsonl`

**Interfaces:**

- Produces: `stable_question_id(question, filters, gold_table_ids, dataset_fingerprint) -> str`.
- Produces: `load_gold_questions(path: Path, release: ResolvedRetrievalRelease, *, require_count: int = 30) -> tuple[GoldRetrievalQuestion, ...]`.
- Consumers: validation CLI and evaluator.

- [ ] **Step 1: Write failing stable-ID and JSONL tests**

```python
def test_stable_question_id_is_order_independent_after_canonical_validation() -> None:
    filters = RetrievalFilters(company_codes=("VCB",), periods=("2023",))
    value = stable_question_id(
        "Doanh thu thuần của VCB năm 2023?",
        filters,
        ("tbl_" + "1" * 64,),
        "a" * 64,
    )
    assert re.fullmatch(r"retq_[0-9a-f]{64}", value)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_count",
        "duplicate_question_id",
        "duplicate_question_text",
        "wrong_fingerprint",
        "unknown_table_id",
        "filter_mismatch",
        "missing_reviewer",
        "invalid_reviewed_at",
        "unverified_evidence",
        "missing_gold_evidence",
        "evidence_span_mismatch",
        "invalid_json",
        "blank_line",
    ],
)
def test_load_gold_questions_rejects_invalid_artifact(
    retrieval_release: ResolvedRetrievalRelease,
    gold_file: Path,
    mutation: str,
) -> None:
    mutate_gold_file(gold_file, mutation)
    with pytest.raises(RetrievalInputError):
        load_gold_questions(gold_file, retrieval_release)
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_gold.py
```

Expected: FAIL because the gold loader does not exist.

- [ ] **Step 2: Implement canonical IDs and strict streaming validation**

Normalize question identity with Unicode NFKC and collapsed whitespace, serialize this exact payload with sorted keys, then SHA-256 it:

```python
payload = {
    "contract_version": "retrieval-gold-v1",
    "dataset_fingerprint": dataset_fingerprint,
    "filters": filters.model_dump(mode="json"),
    "gold_table_ids": list(gold_table_ids),
    "question": normalized_question,
}
```

The loader must read strict UTF-8 one line at a time, reject blank lines, validate exactly 30 records, recompute every ID, verify all gold table IDs exist, join gold tables to document/cell metadata, and enforce AND-across/OR-within filter compatibility.

Every record also requires `reviewed_by` as a non-empty reviewer identifier, timezone-aware `reviewed_at`, and `gold_evidence`. Evidence must cover every `gold_table_id` exactly once, set `verified=true`, and equal the released table's `relative_path`, `line_start`, and `line_end`. Structural validity without this persisted review evidence is not a reviewed gold artifact.

- [ ] **Step 3: Select and review the real 30 questions**

Use the locked release to identify tables; do not copy BM25 predictions into `gold_table_ids`. Create exactly:

| Intent | Count | Gold shape |
|---|---:|---|
| `lookup` | 10 | one source table |
| `compare` | 10 | at least two tables or one table containing both reviewed periods |
| `growth` | 10 | at least two reviewed periods with every required source table labeled |

Coverage rules:

- at least 10 distinct company codes;
- all three main statement types appear;
- at least 10 multi-table questions;
- every question has a non-empty company filter and period filter;
- every gold table is inspected in canonical Parquet and traced to its immutable source span;
- every row records the human reviewer, review timestamp, and one verified evidence object per gold table;
- JSONL rows are sorted by `question_id` and LF-terminated.

Resolve the release first, then use DuckDB only to locate review candidates:

```powershell
$releasePath = uv run --frozen --no-sync python -c "from pathlib import Path; from financial_report_qa.retrieval.release import resolve_retrieval_release; print(resolve_retrieval_release(Path('data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'), repo_root=Path.cwd()).release_path)"
uv run --frozen --no-sync python -c "import duckdb,sys; p=sys.argv[1].replace('\\','/'); duckdb.sql(f\"SELECT d.company_code,d.report_year,t.statement_type,t.table_id,t.title_raw FROM read_parquet('{p}/documents.parquet') d JOIN read_parquet('{p}/tables.parquet') t USING(doc_id) WHERE t.statement_type IS NOT NULL ORDER BY d.company_code,d.report_year,t.statement_type,t.table_id LIMIT 200\").show()" $releasePath
```

Human review determines labels; the candidate query never writes the gold artifact.

- [ ] **Step 4: Verify and commit Task 2**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_gold.py
uv run --frozen --no-sync python -c "from pathlib import Path; from financial_report_qa.retrieval.gold import load_gold_questions; from financial_report_qa.retrieval.release import resolve_retrieval_release; r=resolve_retrieval_release(Path('data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'), repo_root=Path.cwd()); print(len(load_gold_questions(Path('data/qa/retrieval-gold-v1.jsonl'), r)))"
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/gold.py tests/unit/retrieval/test_gold.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/gold.py tests/unit/retrieval/test_gold.py
git diff --check
git add src/financial_report_qa/retrieval/gold.py tests/unit/retrieval/test_gold.py data/qa/retrieval-gold-v1.jsonl
git commit -m "test(retrieval): lock 30 gold table questions"
```

Expected: the loader prints `30` and exits 0. If real review is incomplete, do not commit a partial or generated substitute.

---

### Task 3: Build deterministic table documents from canonical Parquet

**Files:**

- Create: `src/financial_report_qa/retrieval/documents.py`
- Create: `tests/unit/retrieval/test_documents.py`

**Interfaces:**

- Produces: `iter_table_documents(release: ResolvedRetrievalRelease) -> Iterator[TableDocument]`.
- Produces: `write_table_documents(release, output_path: Path) -> DocumentBuildResult`.
- Consumers: BM25 index builder; JSONL order is ascending `table_id`.

- [ ] **Step 1: Write failing serialization and join tests**

Create a fixture with two documents, three tables, repeated cell labels, null metadata, Vietnamese diacritics, and numeric values. Assert:

```python
assert documents[0].text == (
    "title: Báo cáo kết quả hoạt động kinh doanh\n"
    "statement: income statement\n"
    "metrics: net revenue | profit after tax\n"
    "metric aliases: Doanh thu thuần | Lợi nhuận sau thuế\n"
    "company: VCB\n"
    "periods: 2022 | 2023\n"
    "units: VND million"
)
assert "1.234.567" not in documents[0].text
assert [item.table_id for item in documents] == sorted(item.table_id for item in documents)
```

Also prove list order and duplicate cells do not change output bytes.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_documents.py
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 2: Implement a bounded DuckDB aggregation**

Query `documents.parquet`, `tables.parquet`, and `cells.parquet` into one row per table. Aggregate only:

- raw title and normalized statement;
- canonical row labels plus their observed raw labels;
- company code and report year;
- cell periods;
- table and cell units.

Sort/deduplicate every list in Python after Unicode NFKC plus whitespace collapse. Convert canonical snake_case values to display tokens by replacing `_` with a space. Index the inventory `company_code`; do not depend on an uncommitted company-registry API.

- [ ] **Step 3: Write deterministic UTF-8 JSONL**

Serialize `TableDocument.model_dump(mode="json")` using:

```python
orjson.dumps(record, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)
```

Write to a temporary sibling file, flush, close, hash, and replace the target. Return table count and SHA-256. Reject duplicate/missing table IDs and a count unequal to the release manifest.

- [ ] **Step 4: Verify and commit Task 3**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_documents.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_documents.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_documents.py
git diff --check
git add src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_documents.py
git commit -m "feat(retrieval): derive deterministic table documents"
```

---

### Task 4: Build and load the content-addressed BM25 index

**Files:**

- Create: `src/financial_report_qa/retrieval/index.py`
- Create: `tests/unit/retrieval/test_index.py`

**Interfaces:**

- Produces: `tokenize_text(text: str) -> tuple[str, ...]`.
- Produces: `build_bm25_index(release, output_root: Path) -> Path`.
- Produces: `load_bm25_index(index_dir: Path, release) -> LoadedBm25Index`.
- Artifact target: `<output_root>/<full-dataset-fingerprint>/`.

- [ ] **Step 1: Write failing tokenizer/index integrity tests**

Cover:

```python
def test_tokenize_text_preserves_vietnamese_and_ticker() -> None:
    assert tokenize_text("  LỢI NHUẬN—VCB, năm 2023  ") == (
        "lợi",
        "nhuận",
        "vcb",
        "năm",
        "2023",
    )
```

Also test NFKC equivalence, no stop-word removal, empty text, index manifest identity, corrupted NumPy/vocabulary file, existing non-identical target, and identical rebuild acceptance.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index.py
```

Expected: FAIL because the index module does not exist.

- [ ] **Step 2: Implement fixed tokenization**

Use:

```python
TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def tokenize_text(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(TOKEN_RE.findall(normalized))
```

Do not call `bm25s.tokenize` with its default English stop words. Pin builder settings in the manifest: `method="lucene"`, `k1=1.5`, `b=0.75`, `dtype="float32"`, tokenizer version `v1`.

- [ ] **Step 3: Build and save BM25 atomically**

```python
retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene", dtype="float32")
retriever.index(token_lists, show_progress=False)
retriever.save(temp_dir / "bm25", corpus=table_ids, show_progress=False)
```

Copy the deterministic `documents.jsonl` into the temporary index directory. Hash every generated file by relative POSIX path, write sorted `manifest.json` last, then atomically rename. Loader verification recomputes hashes before `bm25s.BM25.load(..., load_corpus=True, mmap=True)`.

- [ ] **Step 4: Verify and commit Task 4**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index.py
git diff --check
git add src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index.py
git commit -m "feat(retrieval): build content-addressed bm25 index"
```

---

### Task 5: Implement metadata-first stable BM25 retrieval

**Files:**

- Create: `src/financial_report_qa/retrieval/service.py`
- Create: `tests/unit/retrieval/test_service.py`

**Interfaces:**

- Produces: `Bm25TableRetriever.search(question: str, filters: RetrievalFilters, *, top_k: int = 10) -> RetrievalTrace`.
- Consumes: validated `LoadedBm25Index` and `TableDocument` sequence aligned by BM25 row.

- [ ] **Step 1: Write failing filter/ranking tests**

Use a small real BM25 index. Prove:

- OR within `company_codes`, `periods`, and `statement_types`;
- AND across the three fields;
- empty filters allow all rows;
- unknown metadata never satisfies a non-empty filter;
- empty eligible set returns no candidates and `empty_reason="no_eligible_documents"`;
- zero in-vocabulary query tokens return no candidates and `empty_reason="no_index_tokens"`;
- ties sort by `table_id`;
- `top_k < 1` is rejected;
- candidate rank starts at 1;
- traces contain query tokens, matched tokens, BM25 scores, and ordered per-field `FilterDecision` counts.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_service.py
```

Expected: FAIL because the retrieval service does not exist.

- [ ] **Step 2: Build deterministic metadata postings**

At load time build `dict[str, frozenset[int]]` postings for company, period, and statement. Candidate selection starts with all row indices, unions requested values within each field, then intersects field results. For every non-empty field, append a `FilterDecision` containing requested values, union size before intersection, and eligible size after intersection.

- [ ] **Step 3: Rank only eligible candidates**

Tokenize and retain only vocabulary tokens. If none remain, return an empty trace with `empty_reason="no_index_tokens"`; do not score or rank arbitrary tables. Otherwise call `bm25.get_scores(list(tokens))`. Reject non-finite scores among eligible rows.

Do not use `bm25s.retrieve(weight_mask=...)`: bm25s 0.3.10 masks by multiplication and can return zero-scored excluded rows. Instead rank:

```python
ranked_indices = sorted(
    eligible_indices,
    key=lambda index: (-float(scores[index]), documents[index].table_id),
)[:top_k]
```

Matched tokens are the sorted intersection of query tokens and the selected document's token set.

- [ ] **Step 4: Verify and commit Task 5**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_service.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_service.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_service.py
git diff --check
git add src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_service.py
git commit -m "feat(retrieval): add filter-first bm25 ranking"
```

---

### Task 6: Add evaluator, deterministic reports, and product CLI

**Files:**

- Create: `src/financial_report_qa/retrieval/evaluation.py`
- Create: `src/financial_report_qa/retrieval/cli.py`
- Modify: `src/financial_report_qa/cli.py`
- Modify: `tests/unit/test_cli.py`
- Create: `tests/unit/retrieval/test_evaluation.py`
- Create: `tests/integration/retrieval/test_day8_cli.py`

**Interfaces:**

- Produces: `score_at_10(predicted, gold) -> RetrievalMetrics`.
- Produces: `evaluate_retrieval(retriever, questions, k=10) -> RetrievalEvaluationReport`.
- CLI commands: `retrieval build-index`, `retrieval validate-gold`, and `retrieval evaluate`.

- [ ] **Step 1: Write failing metric tests with exact expected values**

```python
def test_score_at_10_for_one_hit_out_of_two_gold_tables() -> None:
    metrics = score_at_10(
        predicted=("tbl_" + "a" * 64, "tbl_" + "f" * 64),
        gold=("tbl_" + "a" * 64, "tbl_" + "b" * 64),
    )
    assert metrics.true_positive == 1
    assert metrics.precision == pytest.approx(0.1)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f2 == pytest.approx(5 * 0.1 * 0.5 / (4 * 0.1 + 0.5))
```

Cover zero hits, all hits, duplicate predictions rejected, macro averaging, three intent groups, and deterministic report bytes.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_evaluation.py
```

Expected: FAIL because `score_at_10` and report models do not exist.

- [ ] **Step 2: Implement metrics and failure taxonomy**

Always use denominator `k=10` for precision. Use `5PR / (4P + R)`, returning zero only when the denominator is zero. Macro-average question metrics and group by intent.

Per-query failures are exactly:

- `no_eligible_documents`;
- `no_index_tokens`;
- `zero_gold_hits`;
- `partial_gold_hits`;
- `none`.

- [ ] **Step 3: Write failing end-to-end CLI integration test**

Build a small temporary release lock, three Parquet tables, and exactly 30 fixture questions; index it, validate gold, and evaluate. Assert:

- success exits 0;
- invalid input/corruption exits 2;
- output names contain the fingerprint prefix;
- JSON and Markdown contain the same macro metrics;
- replay output is byte-identical.

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/integration/retrieval/test_day8_cli.py tests/unit/test_cli.py
```

Expected: FAIL because the command is not registered.

- [ ] **Step 4: Implement CLI and atomic reports**

Required invocations:

```powershell
financial-report-qa retrieval build-index --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --output-root data/indexes/bm25
financial-report-qa retrieval validate-gold --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --gold-path data/qa/retrieval-gold-v1.jsonl
financial-report-qa retrieval evaluate --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --index-dir data/indexes/bm25/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f --gold-path data/qa/retrieval-gold-v1.jsonl --output-dir artifacts/evaluations
```

Catch only `RetrievalInputError`, `RetrievalArtifactError`, validation, JSON, and I/O errors at the CLI boundary; print a concise error to stderr and return 2. Unexpected programming errors must propagate in tests.

Write sorted, indented UTF-8 JSON with a terminal newline and render Markdown from the validated report model. Use temporary files plus `Path.replace`.

- [ ] **Step 5: Verify and commit Task 6**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py tests/unit/test_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval src/financial_report_qa/cli.py tests/unit/retrieval tests/integration/retrieval tests/unit/test_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval src/financial_report_qa/cli.py tests/unit/retrieval tests/integration/retrieval tests/unit/test_cli.py
git diff --check
git add src/financial_report_qa/retrieval src/financial_report_qa/cli.py tests/unit/retrieval tests/integration/retrieval tests/unit/test_cli.py
git commit -m "feat(retrieval): evaluate day 8 bm25 baseline"
```

---

### Task 7: Package and validate the reusable BM25 agent skill

**REQUIRED SUB-SKILL:** Use `superpowers:writing-skills`.

**Files:**

- Create or update: `.agents/skills/vifinqa-bm25-retrieval/SKILL.md`
- Create or update: `.agents/skills/vifinqa-bm25-retrieval/agents/openai.yaml`

**Interfaces:**

- Produces: discoverable `$vifinqa-bm25-retrieval` guidance for Tasks 1–11.
- Enforces: release-lock identity, reviewed gold evidence, explicit Day-8 filters, fixed metric arithmetic, replay determinism, and truthful blockers.

- [ ] **Step 1: Run a no-skill baseline scenario**

Ask a fresh agent to prepare Day-8 execution from `plan.md` and code only. Record whether it invents an entity parser, assumes gold exists, uses `TP/retrieved` precision, requires all periods in one table, or substitutes fixtures for real metrics.

- [ ] **Step 2: Write or update the minimal skill**

The YAML description starts with `Use when...` and contains retrieval triggers. The body links this design and plan, defines hard Tasks 1–7 versus verification Tasks 8–11, includes the non-negotiable contracts, full verification commands, common mistakes, and a handoff prompt. Keep `agents/openai.yaml` strings quoted and make `default_prompt` explicitly mention `$vifinqa-bm25-retrieval`.

- [ ] **Step 3: Validate structure**

```powershell
python C:/Users/Admin/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/vifinqa-bm25-retrieval
```

Expected: `Skill is valid!` and exit 0.

- [ ] **Step 4: Run a fresh application scenario with the skill**

Ask a new agent to use the skill and return scope, preconditions, filter/ranking semantics, formulas, task split, and missing-input behavior. It passes only when all RED failures from Step 1 are corrected and the agent stops real evaluation when lock/gold evidence is missing.

- [ ] **Step 5: Commit Task 7**

```powershell
git add .agents/skills/vifinqa-bm25-retrieval/SKILL.md .agents/skills/vifinqa-bm25-retrieval/agents/openai.yaml
git commit -m "docs(agent): add vifinqa bm25 retrieval skill"
```

---

# Part II — Easier Tasks: Real-Data Verification and Evidence

### Task 8: Preflight the immutable release and reviewed gold set

**Files:**

- Read only: `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`
- Read only: `data/qa/retrieval-gold-v1.jsonl`
- Read only: locked release and Week-1 gate artifacts

**Interfaces:**

- Consumes the exact outputs of Tasks 1-2.
- Produces terminal evidence only; no production files change.

- [x] **Step 1: Confirm inputs exist without restoring or editing user-deleted files**

```powershell
$lockPath = "data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json"
$goldPath = "data/qa/retrieval-gold-v1.jsonl"
Test-Path -LiteralPath $lockPath
Test-Path -LiteralPath $goldPath
```

Expected: both are `True`. If either is `False`, report the missing path and stop Tasks 8–10.

- [x] **Step 2: Validate the release and gold questions**

```powershell
uv run --frozen --no-sync financial-report-qa retrieval validate-gold --release-lock $lockPath --gold-path $goldPath
```

Expected: exit 0; dataset fingerprint `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`, 146,011 tables, and exactly 30 valid questions.

- [x] **Step 3: Inspect coverage counts**

Run a read-only Python command using `load_gold_questions` and print counts by intent, company, statement, and gold-table cardinality.

Expected:

- 10 lookup, 10 compare, 10 growth;
- at least 10 distinct companies;
- all three main statements;
- at least 10 questions with more than one gold table.

---

### Task 9: Build the real index twice and prove deterministic identity

**Files:**

- Generate: `data/indexes/bm25/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/`
- Generate then discard manually after comparison: `data/indexes/bm25-replay/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f/`

**Interfaces:**

- Consumes the Task-6 CLI and Task-4 index implementation.
- Produces two independently built artifact trees for hash comparison.

- [x] **Step 1: Build the canonical index**

```powershell
$fingerprint = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
$indexRoot = "data/indexes/bm25"
uv run --frozen --no-sync financial-report-qa retrieval build-index --release-lock $lockPath --output-root $indexRoot
```

Expected: exit 0, target path ends with the full fingerprint, manifest reports 146,011 table documents, and no source/release file changes.

- [x] **Step 2: Build an independent replay**

```powershell
$replayRoot = "data/indexes/bm25-replay"
uv run --frozen --no-sync financial-report-qa retrieval build-index --release-lock $lockPath --output-root $replayRoot
```

- [x] **Step 3: Compare every relative artifact hash**

```powershell
$canonical = Join-Path $indexRoot $fingerprint
$replay = Join-Path $replayRoot $fingerprint
$left = Get-ChildItem -LiteralPath $canonical -Recurse -File | ForEach-Object {
  [PSCustomObject]@{
    Path = $_.FullName.Substring($canonical.Length).TrimStart('\')
    Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
  }
}
$right = Get-ChildItem -LiteralPath $replay -Recurse -File | ForEach-Object {
  [PSCustomObject]@{
    Path = $_.FullName.Substring($replay.Length).TrimStart('\')
    Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
  }
}
Compare-Object ($left | Sort-Object Path) ($right | Sort-Object Path) -Property Path,Hash
```

Expected: no output. Any difference is a determinism defect; add a failing test before fixing.

---

### Task 10: Run the 30-question baseline and replay the reports

**Files:**

- Generate: `artifacts/evaluations/retrieval-day8-37a61be7aebd.json`
- Generate: `artifacts/evaluations/retrieval-day8-37a61be7aebd.md`
- Generate: `artifacts/evaluations/replay/retrieval-day8-37a61be7aebd.json`
- Generate: `artifacts/evaluations/replay/retrieval-day8-37a61be7aebd.md`

- [x] **Step 1: Evaluate the canonical index**

```powershell
$indexPath = Join-Path $indexRoot $fingerprint
uv run --frozen --no-sync financial-report-qa retrieval evaluate --release-lock $lockPath --index-dir $indexPath --gold-path $goldPath --output-dir artifacts/evaluations
```

Expected: exit 0 and both JSON/Markdown reports describe exactly 30 questions.

- [x] **Step 2: Evaluate again into a fresh directory**

```powershell
uv run --frozen --no-sync financial-report-qa retrieval evaluate --release-lock $lockPath --index-dir $indexPath --gold-path $goldPath --output-dir artifacts/evaluations/replay
$canonicalHashes = Get-ChildItem artifacts/evaluations -File -Filter 'retrieval-day8-37a61be7aebd.*' | ForEach-Object { [PSCustomObject]@{Name=$_.Name; Hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} }
$replayHashes = Get-ChildItem artifacts/evaluations/replay -File -Filter 'retrieval-day8-37a61be7aebd.*' | ForEach-Object { [PSCustomObject]@{Name=$_.Name; Hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} }
Compare-Object ($canonicalHashes | Sort-Object Name) ($replayHashes | Sort-Object Name) -Property Name,Hash
```

Compare by file name and hash. Expected: no content-hash differences.

- [x] **Step 3: Inspect retrieval failures**

From JSON, report:

- macro Recall@10 and F2@10;
- metrics for lookup, compare, and growth;
- count of `no_eligible_documents`, `no_index_tokens`, `zero_gold_hits`, and `partial_gold_hits`;
- ten lowest-F2 questions with predicted IDs, gold IDs, BM25 scores, matched tokens, and eligible counts.

Do not tune aliases or labels during this verification task. Open a new TDD remediation task for any source-backed defect.

---

### Task 11: Run the repository gate and close Day 8 documentation

**Files:**

- Modify after evidence exists: `README.md`
- Modify after evidence exists: `plan.md`

- [x] **Step 1: Run focused and full verification**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

Expected: every command exits 0 with no warnings attributed to changed code.

- [x] **Step 2: Record exact operator commands and observed metrics**

In `README.md`, document the three retrieval commands and immutable release-lock rule. In `plan.md`, mark Day-8 bullets complete only when:

- 30 reviewed gold questions validate;
- the real index contains 146,011 documents;
- metadata filtering precedes ranking;
- top-k traces include BM25 scores and matched tokens;
- Recall@10 and F2@10 reports exist and replay byte-identically.

Record measured values; do not write `passed` merely because code tests pass.

- [x] **Step 3: Review scope and commit**

```powershell
git status --short
git diff -- README.md plan.md
git add README.md plan.md
git commit -m "docs: record day 8 bm25 retrieval baseline"
```

Do not stage indexes, replay directories, unrelated dirty changes, source data, or unreviewed gold data.

## BM25 v2 remediation evidence (2026-08-09)

Tasks 1–5 were implemented on `codex/modular-foundation` through commits `42a0b5d`, `fe73684`,
`a7233f8`, `5bd985f`, and `4ee2fa6`. A clean archive of `4ee2fa6` imported
`RetrievalService`, passed 45 focused tests, Ruff, and mypy, and contained no retrieval import
from `financial_report_qa.normalization`.

The lock-bound gold (`data/qa/retrieval-gold-v1.jsonl`, SHA-256
`13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5`) validated 30 questions:
10 lookup, 10 compare, 10 growth; 10 companies; 18 multi-table questions. Two independently
built v2 indexes at `data/indexes/bm25-remediation-v2-a/` and `...-v2-b/` contained 146,011
documents and had zero relative artifact hash differences. Manifest SHA-256 was
`B6007B13301E62E259C86BF23FE8ACC1014EB51E44D7E7E9B86A724DDB2E8484`; document SHA-256 was
`b1206d17e7a870da727fd4ec70bb06bc707ede14de6641814ce1dfba418b7dd6`.

Reports under `artifacts/evaluations/remediation-v2-a/` and `...-v2-b/` were byte-identical
(JSON `70280CC6A277128F1F9C7CC05A5C6C96AEEDA3C62D4FF40C1BE150285F6E2AE9`, Markdown
`349F9927D9D5654F07E75B91A8ED58F0911EBCC07303D266B9B6F95136E28374`). Clean-source metrics:
Precision@10 `0.1366667`, Recall@10 `0.8333333`, F2@10 `0.4034392`, TP `41`; by intent
lookup `0.9000/0.3214`, compare `0.8000/0.4444`, growth `0.8000/0.4444` (Recall/F2).
Failure counts were five `zero_gold_hits`, zero partial hits, zero `no_eligible_documents`,
and zero `no_index_tokens`. The five misses are HDB/NVL ranking-fragmentation cases; gold,
filters, and query-ID-specific rules were not changed. The provisional floors
Recall `0.8833333` and F2 `0.4179894` were not met.

Full working-tree gates were recorded truthfully: `pytest -q` passed 556 with 1 skipped;
`ruff check .` reported 84 pre-existing errors outside retrieval; `mypy` reported 33
pre-existing errors in normalization/evaluation; `git diff --check` passed. The fixed F2
formula remains unchanged, with a current-gold theoretical ceiling of `0.476190476190476`.

## Final Evidence Contract

The Day-8 handoff must contain:

- release-lock path, full dataset fingerprint, and table count;
- gold path, SHA-256, question count, intent distribution, and multi-table count;
- canonical index path, manifest hash, document hash, and replay comparison;
- macro Recall@10 and F2@10 plus metrics by intent;
- failure-category counts and ten lowest-F2 queries;
- canonical/replay report paths and matching SHA-256 values;
- focused/full pytest, Ruff, mypy, and `git diff --check` outputs;
- explicit blockers if real release/gold access was unavailable.

## Self-Review

- Spec coverage: release lock, 30-question gold set, document fields, filter-first BM25, score traces, Recall@10/F2@10, deterministic artifacts, and truthful failure behavior all map to tasks.
- Task split: Tasks 1–6 contain coding and expert gold-contract work; Task 7 packages the reusable skill; Tasks 8–11 contain repeatable verification/evidence work.
- Interface consistency: all tasks use `ResolvedRetrievalRelease`, `RetrievalFilters`, `TableDocument`, and the full fingerprint path.
- Placeholder scan: no TBD/TODO/“implement later” instruction remains; runtime metrics are explicitly read from generated reports.
- Scope: dense retrieval, entity parsing, fusion, graph, planning, and execution remain excluded.
