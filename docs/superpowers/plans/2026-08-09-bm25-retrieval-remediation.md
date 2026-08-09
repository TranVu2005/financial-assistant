# BM25 Retrieval Corpus-Bound Alias Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dirty normalization dependency in commit `8e5baa0` with a deterministic, corpus-bound metric-alias lexicon while preserving or improving the verified Day-8 BM25 metrics.

**Architecture:** Persist sorted canonical/raw metric-label observations in each `TableDocument`, derive an unambiguous alias lexicon from those observations, and expand query tokens with longest whole-token matches. The retriever remains BM25-only and filter-first; the release lock, reviewed gold, score formula, and stable ranking contracts do not change.

**Tech Stack:** Python 3.11, Pydantic 2, DuckDB, PyArrow/Parquet, bm25s 0.3.x, orjson, pytest, Ruff, mypy, PowerShell, Git.

## Global Constraints

- Design authority: `docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md`.
- Baseline implementation plan: `docs/superpowers/plans/2026-08-08-day-8-bm25-retrieval.md`.
- Preserve unrelated dirty changes; stage only paths listed by the active task.
- Implement on `codex/modular-foundation`; do not merge or cherry-pick detached commit `8e5baa0` unchanged.
- Real retrieval consumes `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`.
- Expected fingerprint: `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`.
- Expected release size: 146,011 canonical table documents.
- Gold remains `data/qa/retrieval-gold-v1.jsonl`, exactly 30 reviewed questions, SHA-256 `13888830e7dde393bf3ed0e4561c02340912a6f36ab2b32503ef2fb2cfac63f5`.
- Do not change gold labels, reviewers, filters, evidence, or question IDs during remediation.
- Do not use BM25 predictions or the 30 gold questions to construct the alias lexicon.
- Day 8 continues to consume explicit expert company/period/statement filters. Do not parse company or year from free text.
- Ranking remains BM25-only over eligible rows, ordered by `(-score, table_id)`.
- Precision remains `TP/10`; Recall remains `TP/|gold|`; F2 remains `5PR/(4P+R)`.
- Query expansion must not introduce undocumented duplicate-token weighting.
- Production behavior follows RED -> GREEN -> REFACTOR, with the expected failing reason recorded before implementation.
- Every task ends with focused pytest, Ruff, mypy, `git diff --check`, and an independent review before proceeding.

## Verified Starting Evidence

| Measure | Original baseline | Dirty working-state remediation |
|---|---:|---:|
| Precision@10 | 0.0766667 | 0.1400000 |
| Recall@10 | 0.5166667 | 0.8833333 |
| F2@10 | 0.2341270 | 0.4179894 |
| True positives | 23 | 42 |
| Zero-gold-hit questions | 13 | 3 |
| Partial-gold-hit questions | 3 | 1 |

The dirty-state report is reproducible only while uncommitted normalization files are present. A clean archive of `8e5baa0` fails importing `RetrievalService` because the committed normalization package does not contain all symbols imported by `normalization/service.py`. This plan must produce the same or better retrieval result from committed retrieval code alone.

With the fixed `P@10=TP/10` definition and the current gold cardinalities, the theoretical macro-F2 ceiling is `0.476190476190476`. The Day-14 `F2 >= 0.80` target is a specification inconsistency; this remediation records it but does not silently change the metric.

## File Map

| Path | Responsibility |
|---|---|
| `src/financial_report_qa/retrieval/contracts.py` | Metric-label observations, expansion trace, and BM25 v2 manifest contract |
| `src/financial_report_qa/retrieval/documents.py` | Canonical/raw label pairing and deterministic document construction |
| `src/financial_report_qa/retrieval/metric_aliases.py` | Corpus-bound lexicon construction and boundary-safe query expansion |
| `src/financial_report_qa/retrieval/index.py` | BM25 v2 persistence and loading |
| `src/financial_report_qa/retrieval/service.py` | Filter-first retrieval using stable unique expanded tokens |
| `src/financial_report_qa/retrieval/evaluation.py` | Existing metric/report behavior; consumes expansion traces without formula changes |
| `tests/unit/retrieval/test_contracts.py` | New immutable contract validation |
| `tests/unit/retrieval/test_table_documents.py` | Canonical/raw-null and deterministic document tests |
| `tests/unit/retrieval/test_metric_aliases.py` | Lexicon ambiguity, boundaries, overlap, and token-deduplication tests |
| `tests/unit/retrieval/test_index_service.py` | Multi-document ranking and manifest migration tests |
| `tests/unit/retrieval/test_evaluation.py` | Expansion-trace serialization and unchanged metric arithmetic |
| `tests/integration/retrieval/test_day8_cli.py` | BM25 v2 build/load/evaluate replay lifecycle |

---

# Part I — Hard Tasks

### Task 1: Add immutable metric-observation and BM25 v2 contracts

**Files:**

- Modify: `src/financial_report_qa/retrieval/contracts.py:131-180`
- Modify: `tests/unit/retrieval/test_contracts.py`
- Modify: `tests/unit/retrieval/test_index_service.py`

**Interfaces:**

- Produces: `MetricLabelObservation(canonical: str, raw: str | None)`.
- Produces: `MetricExpansion(alias_tokens, canonical_metric, added_tokens)`.
- Extends: `TableDocument.metric_labels: tuple[MetricLabelObservation, ...] = ()`.
- Extends: `RetrievalTrace.metric_expansions: tuple[MetricExpansion, ...] = ()`.
- Changes: `BM25IndexManifest.schema_version` to `bm25-index-v2`.
- Changes: `BM25IndexManifest.builder_version` to `v2`.
- Adds: `BM25IndexManifest.query_expansion_version = "v1"`.

- [ ] **Step 1: Write failing contract tests**

Add tests with these exact behaviors:

```python
def test_metric_label_observation_requires_canonical_text() -> None:
    with pytest.raises(ValidationError):
        MetricLabelObservation(canonical="", raw="Doanh thu thuần")


def test_table_document_requires_sorted_unique_metric_labels() -> None:
    labels = (
        MetricLabelObservation(canonical="total_assets", raw="Tổng tài sản"),
        MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),
    )
    with pytest.raises(ValidationError):
        make_table_document(metric_labels=labels)


def test_metric_expansion_records_only_tokens_added_to_query() -> None:
    expansion = MetricExpansion(
        alias_tokens=("doanh", "thu", "thuần"),
        canonical_metric="net_revenue",
        added_tokens=("net", "revenue"),
    )
    assert expansion.added_tokens == ("net", "revenue")
```

Update the manifest test to require:

```python
assert manifest["schema_version"] == "bm25-index-v2"
assert manifest["builder_version"] == "v2"
assert manifest["tokenizer_version"] == "v1"
assert manifest["query_expansion_version"] == "v1"
```

- [ ] **Step 2: Run the tests and record RED evidence**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_index_service.py
```

Expected: FAIL because the new models, fields, and v2 manifest values do not exist.

- [ ] **Step 3: Implement minimal immutable contracts**

Use frozen Pydantic models and canonical tuple validation. `MetricLabelObservation` ordering is ascending by `(canonical, raw or "")`; duplicates are invalid rather than silently removed.

```python
class MetricLabelObservation(_FrozenModel):
    canonical: NonEmptyString
    raw: NonEmptyString | None = None


class MetricExpansion(_FrozenModel):
    alias_tokens: tuple[str, ...]
    canonical_metric: NonEmptyString
    added_tokens: tuple[str, ...]
```

`TableDocument.metric_labels` and `RetrievalTrace.metric_expansions` default to empty tuples so fixture construction remains explicit and old report parsing remains understandable. Old persisted indexes are not accepted as v2 indexes.

- [ ] **Step 4: Verify Task 1**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_index_service.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/contracts.py tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_index_service.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/contracts.py tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_index_service.py
git diff --check
```

Expected: all commands exit `0`.

- [ ] **Step 5: Review and commit Task 1**

```powershell
git add src/financial_report_qa/retrieval/contracts.py tests/unit/retrieval/test_contracts.py tests/unit/retrieval/test_index_service.py
git commit -m "feat(retrieval): add corpus alias contracts"
```

---

### Task 2: Preserve canonical metrics and persist canonical/raw observations

**Files:**

- Modify: `src/financial_report_qa/retrieval/documents.py:50-125`
- Modify: `tests/unit/retrieval/test_table_documents.py`

**Interfaces:**

- Consumes: `MetricLabelObservation` from Task 1.
- Produces: one sorted `metric_labels` tuple per `TableDocument`.
- Preserves: text field order `title`, `statement`, `metrics`, `metric aliases`, `company`, `periods`, `units`.

- [ ] **Step 1: Write failing canonical/raw-null tests**

Add three fixture rows:

```python
{
    "row_label_canonical": "net_revenue",
    "row_label_raw": "Doanh thu thuần",
}
{
    "row_label_canonical": "total_assets",
    "row_label_raw": None,
}
{
    "row_label_canonical": None,
    "row_label_raw": "Dòng trình bày không phải metric",
}
```

Assert:

```python
assert document.metric_labels == (
    MetricLabelObservation(canonical="net_revenue", raw="Doanh thu thuần"),
    MetricLabelObservation(canonical="total_assets", raw=None),
)
assert "metrics: net revenue | total assets" in document.text
assert "metric aliases: Doanh thu thuần" in document.text
assert "Dòng trình bày không phải metric" not in document.text
```

Add a second test proving shuffled and duplicated cell rows produce byte-identical documents.

- [ ] **Step 2: Run the tests and record RED evidence**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_table_documents.py
```

Expected: FAIL because canonical labels with `raw=None` are dropped and observations are not persisted.

- [ ] **Step 3: Correct the DuckDB aggregation**

Aggregate structured pairs only when canonical identity exists:

```sql
list(DISTINCT struct_pack(
    canonical := c.row_label_canonical,
    raw := c.row_label_raw
)) FILTER (WHERE c.row_label_canonical IS NOT NULL)
```

In Python:

- Normalize canonical values without translating them through another registry.
- Normalize raw values only when non-null and non-empty.
- Sort/deduplicate observations by `(canonical, raw or "")`.
- Derive `metrics:` from every canonical observation.
- Derive `metric aliases:` only from observations with non-null raw text.
- Continue excluding `value_raw` from the SQL selection and output text.

- [ ] **Step 4: Verify Task 2**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_table_documents.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_table_documents.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_table_documents.py
git diff --check
```

- [ ] **Step 5: Review and commit Task 2**

```powershell
git add src/financial_report_qa/retrieval/documents.py tests/unit/retrieval/test_table_documents.py
git commit -m "fix(retrieval): preserve canonical metric observations"
```

---

### Task 3: Build an unambiguous corpus-bound alias lexicon

**Files:**

- Create: `src/financial_report_qa/retrieval/metric_aliases.py`
- Create: `tests/unit/retrieval/test_metric_aliases.py`

**Interfaces:**

- Produces: `MetricAliasRule(alias_tokens, canonical_metric, canonical_tokens)`.
- Produces: `build_metric_alias_lexicon(documents: tuple[TableDocument, ...]) -> tuple[MetricAliasRule, ...]`.
- Produces: `expand_metric_query(query_tokens: tuple[str, ...], lexicon: tuple[MetricAliasRule, ...]) -> tuple[tuple[str, ...], tuple[MetricExpansion, ...]]`.
- Consumes: `tokenize_text` from `retrieval.index`; imports nothing from `financial_report_qa.normalization`.

- [ ] **Step 1: Write failing lexicon-construction tests**

Cover deterministic construction and ambiguity rejection:

```python
def test_build_metric_alias_lexicon_excludes_ambiguous_aliases() -> None:
    documents = (
        make_document("a", canonical="profit_after_tax", raw="Lợi nhuận"),
        make_document("b", canonical="operating_profit", raw="Lợi nhuận"),
        make_document("c", canonical="net_revenue", raw="Doanh thu thuần"),
    )

    lexicon = build_metric_alias_lexicon(documents)

    assert all(rule.alias_tokens != ("lợi", "nhuận") for rule in lexicon)
    assert any(rule.canonical_metric == "net_revenue" for rule in lexicon)
```

Also prove duplicate observations and document ordering do not change lexicon equality.

- [ ] **Step 2: Write failing query-expansion tests**

Required cases:

```python
def test_expansion_uses_whole_token_boundaries() -> None:
    tokens, expansions = expand_metric_query(
        ("profit", "after", "taxation"),
        (rule(("profit", "after", "tax"), "profit_after_tax"),),
    )
    assert tokens == ("profit", "after", "taxation")
    assert expansions == ()


def test_expansion_prefers_longest_non_overlapping_alias() -> None:
    lexicon = (
        rule(("lợi", "nhuận", "sau", "thuế", "chưa", "phân", "phối"), "retained_earnings"),
        rule(("lợi", "nhuận", "sau", "thuế"), "profit_after_tax"),
    )
    tokens, expansions = expand_metric_query(
        ("lợi", "nhuận", "sau", "thuế", "chưa", "phân", "phối"),
        lexicon,
    )
    assert "retained" in tokens
    assert "earnings" in tokens
    assert "profit" not in expansions[0].added_tokens


def test_expansion_does_not_duplicate_existing_tokens() -> None:
    tokens, expansions = expand_metric_query(
        ("net", "revenue"),
        (rule(("net", "revenue"), "net_revenue"),),
    )
    assert tokens == ("net", "revenue")
    assert expansions == ()
```

- [ ] **Step 3: Run the tests and record RED evidence**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_metric_aliases.py
```

Expected: FAIL because `metric_aliases.py` does not exist.

- [ ] **Step 4: Implement deterministic lexicon construction**

For each observation with non-null raw text:

```python
alias_tokens = tokenize_text(observation.raw)
canonical_tokens = tokenize_text(observation.canonical.replace("_", " "))
```

Build `alias_tokens -> set[canonical_metric]`. Emit a rule only when:

- alias tokens and canonical tokens are non-empty;
- exactly one canonical metric is observed for the alias;
- canonical expansion would add at least one token not already present in the alias.

Sort rules by `(-len(alias_tokens), alias_tokens, canonical_metric)`.

- [ ] **Step 5: Implement boundary-safe longest-match expansion**

Scan the query from left to right. At each token offset, select the first sorted rule whose complete alias-token sequence matches. Record at most one rule for that span, advance by the matched span, and continue. Merge tokens with stable uniqueness:

```python
expanded_tokens = tuple(dict.fromkeys((*query_tokens, *candidate_tokens)))
```

Record `MetricExpansion` only when at least one canonical token is newly added.

- [ ] **Step 6: Verify Task 3**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_metric_aliases.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/metric_aliases.py tests/unit/retrieval/test_metric_aliases.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/metric_aliases.py tests/unit/retrieval/test_metric_aliases.py
git diff --check
```

- [ ] **Step 7: Review and commit Task 3**

```powershell
git add src/financial_report_qa/retrieval/metric_aliases.py tests/unit/retrieval/test_metric_aliases.py
git commit -m "feat(retrieval): add boundary-safe metric expansion"
```

---

### Task 4: Integrate expansion without changing filter or ranking contracts

**Files:**

- Modify: `src/financial_report_qa/retrieval/service.py:15-80`
- Modify: `tests/unit/retrieval/test_index_service.py`
- Modify: `tests/unit/retrieval/test_evaluation.py`

**Interfaces:**

- `RetrievalService.__init__` builds one immutable lexicon from `index.documents`.
- `RetrievalService.retrieve` returns stable unique vocabulary tokens plus expansion traces.
- `_eligible_positions` and `(-score, table_id)` sorting remain byte-for-byte behaviorally unchanged.

- [ ] **Step 1: Write a failing multi-document ranking test**

Create three documents:

- target document: canonical `net_revenue`, observed alias `Doanh thu thuần`;
- distractor document: unrelated canonical metric but repeated `doanh`, `thu`, and `thuần` words;
- second distractor: `profit_after_tax`.

Assert:

```python
trace = service.retrieve(
    "Doanh thu thuần",
    filters=RetrievalFilters(company_codes=("VGT",)),
    k=10,
)
assert trace.results[0].table_id == target_table_id
assert trace.query_tokens.count("net") == 1
assert trace.query_tokens.count("revenue") == 1
assert trace.metric_expansions[0].canonical_metric == "net_revenue"
```

Add negative cases proving a non-boundary substring produces no expansion and an OOV query still returns `empty_reason="no_index_tokens"`.

- [ ] **Step 2: Run the tests and record RED evidence**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_evaluation.py
```

Expected: FAIL because service integration and serialized expansion traces are absent.

- [ ] **Step 3: Integrate the corpus lexicon**

Implementation shape:

```python
class RetrievalService:
    def __init__(self, index: BM25Index) -> None:
        self._index = index
        self._metric_alias_lexicon = build_metric_alias_lexicon(index.documents)

    def retrieve(...):
        base_tokens = tokenize_text(query)
        query_tokens, metric_expansions = expand_metric_query(
            base_tokens,
            self._metric_alias_lexicon,
        )
```

Then keep only vocabulary tokens for scoring. If all expanded/base tokens are OOV, return no candidates. Persist only expansions whose added tokens survive vocabulary filtering, so traces describe effective scoring behavior.

- [ ] **Step 4: Prove unchanged filter/ranking semantics**

Run all existing service tests for:

- OR within fields and AND across fields;
- no fallback from empty eligible sets;
- no zero-score padding for OOV queries;
- finite scores;
- stable `table_id` tie-breaking;
- rank starting at one.

- [ ] **Step 5: Verify Task 4**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_evaluation.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_evaluation.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_evaluation.py
git diff --check
```

- [ ] **Step 6: Review and commit Task 4**

```powershell
git add src/financial_report_qa/retrieval/service.py tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_evaluation.py
git commit -m "fix(retrieval): integrate auditable alias expansion"
```

---

### Task 5: Migrate persistence and fail closed on BM25 v1 indexes

**Files:**

- Modify: `src/financial_report_qa/retrieval/index.py:40-190`
- Modify: `tests/unit/retrieval/test_index_service.py`
- Modify: `tests/integration/retrieval/test_day8_cli.py`

**Interfaces:**

- Persists `TableDocument.metric_labels` in `documents.jsonl`.
- Writes `bm25-index-v2`, builder `v2`, tokenizer `v1`, query expansion `v1`.
- Rejects v1 indexes with `RetrievalArtifactError` at the CLI boundary.

- [ ] **Step 1: Write failing persistence/migration tests**

Add tests proving:

```python
saved = load_bm25_index(output_dir)
assert saved.documents[0].metric_labels == index.documents[0].metric_labels
assert saved.manifest.schema_version == "bm25-index-v2"
assert saved.manifest.query_expansion_version == "v1"
```

Mutate a fixture manifest to `bm25-index-v1` and assert loading fails before BM25 artifacts are used. Update the CLI fixture lifecycle to build/evaluate v2 and replay byte-identically.

- [ ] **Step 2: Run the tests and record RED evidence**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index_service.py tests/integration/retrieval/test_day8_cli.py
```

Expected: FAIL because the persisted schema is still v1 and no explicit migration error exists.

- [ ] **Step 3: Implement v2 persistence and explicit rejection**

Before validating `BM25IndexManifest`, read the JSON object and require:

```python
if payload.get("schema_version") != "bm25-index-v2":
    raise ValueError("unsupported BM25 index schema; rebuild the index")
```

Keep artifact hashing, atomic publication, document ordering, and existing-target identity checks unchanged.

- [ ] **Step 4: Verify Task 5**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval/test_index_service.py tests/integration/retrieval/test_day8_cli.py
uv run --frozen --no-sync ruff check src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index_service.py tests/integration/retrieval/test_day8_cli.py
uv run --frozen --no-sync mypy src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index_service.py tests/integration/retrieval/test_day8_cli.py
git diff --check
```

- [ ] **Step 5: Review and commit Task 5**

```powershell
git add src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index_service.py tests/integration/retrieval/test_day8_cli.py
git commit -m "feat(retrieval): publish bm25 index v2"
```

---

# Part II — Easier Verification and Evidence Tasks

### Task 6: Prove retrieval works from committed source without dirty normalization

**Files:**

- Read only: committed retrieval source and tests.
- Generate outside the repository: a clean Git archive under the OS temporary directory.

**Interfaces:**

- Consumes the completed Task-1 through Task-5 commits.
- Produces terminal evidence that retrieval imports and tests without uncommitted files.

- [ ] **Step 1: Confirm target files and dependency boundary**

```powershell
git status --short
rg -n "financial_report_qa\.normalization" src/financial_report_qa/retrieval tests/unit/retrieval
```

Expected: no normalization import in retrieval production code; unrelated dirty files remain unstaged.

- [ ] **Step 2: Create a clean source snapshot**

```powershell
$shortSha = git rev-parse --short HEAD
$verifyRoot = Join-Path $env:TEMP "financial-assistant-retrieval-$shortSha-clean"
$zipPath = "$verifyRoot.zip"
if (Test-Path -LiteralPath $verifyRoot) { throw "clean verification path already exists: $verifyRoot" }
if (Test-Path -LiteralPath $zipPath) { throw "clean verification archive already exists: $zipPath" }
git archive --format=zip --output=$zipPath HEAD
Expand-Archive -LiteralPath $zipPath -DestinationPath $verifyRoot
```

- [ ] **Step 3: Run the clean retrieval slice**

From `$verifyRoot`:

```powershell
$env:PYTHONPATH = (Join-Path $verifyRoot "src")
& "D:\GitHub\financial-assistant\.venv\Scripts\python.exe" -c "from financial_report_qa.retrieval.service import RetrievalService; print(RetrievalService.__name__)"
& "D:\GitHub\financial-assistant\.venv\Scripts\python.exe" -m pytest -q tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py
& "D:\GitHub\financial-assistant\.venv\Scripts\ruff.exe" check src/financial_report_qa/retrieval tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py
& "D:\GitHub\financial-assistant\.venv\Scripts\mypy.exe" --no-incremental src/financial_report_qa/retrieval tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py
```

Expected: import succeeds and every focused command exits `0`.

- [ ] **Step 4: Run working-state repository gates**

```powershell
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

Report exact pre-existing failures separately. Do not attribute unrelated notebook or normalization failures to retrieval, and do not claim the full repository is clean unless all four commands exit `0`.

---

### Task 7: Build and evaluate BM25 v2 twice from clean source

**Files:**

- Generate: `data/indexes/bm25-remediation-v2-a/<fingerprint>/`
- Generate: `data/indexes/bm25-remediation-v2-b/<fingerprint>/`
- Generate: `artifacts/evaluations/remediation-v2-a/retrieval-day8-37a61be7aebd.{json,md}`
- Generate: `artifacts/evaluations/remediation-v2-b/retrieval-day8-37a61be7aebd.{json,md}`

**Interfaces:**

- Uses source from the clean archive created in Task 6 through `PYTHONPATH`.
- Uses the immutable release lock and reviewed gold from the project checkout.

- [ ] **Step 1: Validate immutable inputs**

```powershell
$lockPath = "data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json"
$goldPath = "data/qa/retrieval-gold-v1.jsonl"
$fingerprint = "37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f"
Test-Path -LiteralPath $lockPath
Test-Path -LiteralPath $goldPath
Get-FileHash -LiteralPath $goldPath -Algorithm SHA256
```

Expected: both paths exist and gold hash equals the value in Global Constraints.

- [ ] **Step 2: Build two independent indexes**

From the project checkout, retain `$env:PYTHONPATH = Join-Path $verifyRoot "src"`:

```powershell
$python = "D:\GitHub\financial-assistant\.venv\Scripts\python.exe"
& $python -m financial_report_qa.cli retrieval build-index --release-lock $lockPath --output-root data/indexes/bm25-remediation-v2-a
& $python -m financial_report_qa.cli retrieval build-index --release-lock $lockPath --output-root data/indexes/bm25-remediation-v2-b
```

- [ ] **Step 3: Compare every relative artifact hash**

```powershell
$leftRoot = Join-Path "data/indexes/bm25-remediation-v2-a" $fingerprint
$rightRoot = Join-Path "data/indexes/bm25-remediation-v2-b" $fingerprint
$left = Get-ChildItem -LiteralPath $leftRoot -Recurse -File | ForEach-Object {
    [PSCustomObject]@{
        Path = $_.FullName.Substring($leftRoot.Length).TrimStart("\")
        Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
}
$right = Get-ChildItem -LiteralPath $rightRoot -Recurse -File | ForEach-Object {
    [PSCustomObject]@{
        Path = $_.FullName.Substring($rightRoot.Length).TrimStart("\")
        Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    }
}
Compare-Object ($left | Sort-Object Path) ($right | Sort-Object Path) -Property Path,Hash
```

Expected: no output.

- [ ] **Step 4: Evaluate both indexes**

```powershell
& $python -m financial_report_qa.cli retrieval evaluate --release-lock $lockPath --index-dir $leftRoot --gold-path $goldPath --output-dir artifacts/evaluations/remediation-v2-a
& $python -m financial_report_qa.cli retrieval evaluate --release-lock $lockPath --index-dir $rightRoot --gold-path $goldPath --output-dir artifacts/evaluations/remediation-v2-b
```

- [ ] **Step 5: Compare report hashes and metrics**

Require:

- JSON A/B hashes equal.
- Markdown A/B hashes equal.
- question count `30` and fingerprint exact.
- Recall@10 `>= 0.8833333333333333`.
- F2@10 `>= 0.41798941798941797`.
- zero-gold-hit count `<= 3`.
- partial-gold-hit count `<= 1`.
- no `no_eligible_documents` or `no_index_tokens` failures.

The stretch criterion is Recall@10 `>= 0.90`. Failure to reach the stretch criterion does not authorize gold edits or query-ID-specific rules.

- [ ] **Step 6: Inspect the remaining failures without tuning**

Print question ID, intent, company, gold IDs, predictions, scores, matched tokens, metric expansions, and missing gold IDs for every non-`none` failure. Classify each as annotation, document, lexicon, filter, or ranking evidence.

If a new generalizable defect is found, stop closure and write a separate failing regression test before changing production code. If no generalizable defect is found, preserve the measured baseline and report the unresolved examples.

---

### Task 8: Record evidence, metric ceiling, and integration status

**Files:**

- Modify: `docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md`
- Modify: `docs/superpowers/plans/2026-08-08-day-8-bm25-retrieval.md`
- Modify only after successful evidence: `README.md`
- Modify only after successful evidence: `plan.md`

**Interfaces:**

- Records BM25 v2 schema, corpus-bound alias expansion, hashes, metrics, and blockers.
- Does not change evaluation arithmetic.

- [ ] **Step 1: Update design documentation**

Document:

- structured canonical/raw observations in `TableDocument`;
- exclusion of ambiguous aliases;
- whole-token longest non-overlapping matching;
- stable token deduplication;
- expansion trace fields;
- mandatory rebuild from v1 to v2;
- no dependency on normalization runtime modules.

- [ ] **Step 2: Record exact evidence**

Include release path, fingerprint, 146,011 document count, gold hash, index manifest/document hashes, report hashes, macro and intent metrics, failure counts, and exact test/lint/type-check outputs.

- [ ] **Step 3: Record the F2 target inconsistency**

State that the current metric definition yields a theoretical macro-F2 ceiling of `0.476190476190476` for the current gold cardinalities. Preserve the `TP/10` formula. A future change to either the target or formula requires a separate versioned design decision.

- [ ] **Step 4: Review documentation diff**

```powershell
git diff -- docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md docs/superpowers/plans/2026-08-08-day-8-bm25-retrieval.md README.md plan.md
git diff --check
```

- [ ] **Step 5: Commit documentation only after evidence exists**

```powershell
git add docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md docs/superpowers/plans/2026-08-08-day-8-bm25-retrieval.md README.md plan.md
git commit -m "docs: record reproducible bm25 v2 remediation"
```

- [ ] **Step 6: Confirm branch integration**

```powershell
git branch --show-current
git log --oneline --decorate -8
git status --short
```

Expected: branch is `codex/modular-foundation`; all remediation commits are ancestors of `HEAD`; unrelated dirty changes remain unstaged and preserved.

## Final Acceptance Contract

The remediation is complete only when all conditions below are evidenced:

- A clean committed snapshot imports `RetrievalService` without any uncommitted normalization file.
- Retrieval production code has no import from `financial_report_qa.normalization`.
- Canonical metrics survive when raw aliases are absent.
- Raw labels without canonical identity are excluded.
- Alias matching uses complete token sequences, longest non-overlapping selection, and stable deduplication.
- Ambiguous corpus aliases are excluded deterministically.
- Query traces record effective metric expansion.
- Filter-first behavior, stable ranking, OOV behavior, and metric formulas are unchanged.
- BM25 v1 indexes fail closed with a rebuild instruction.
- Two independently built BM25 v2 indexes have identical relative artifact hashes.
- Two independent evaluations produce byte-identical JSON and Markdown.
- Gold content and SHA-256 are unchanged.
- Recall@10 and F2@10 do not regress below `0.8833333333333333` and `0.41798941798941797`.
- Focused tests, Ruff, mypy, and `git diff --check` pass from clean committed source.
- Full repository gate results are reported truthfully, including unrelated blockers.
- Remediation commits are on `codex/modular-foundation`, not only on a detached worktree.

## Self-Review

- Spec coverage: release identity, immutable reviewed gold, deterministic documents, filter-first BM25, score traces, fixed metrics, and replay evidence map to Tasks 1-8.
- Defect coverage: hidden dependency, substring matching, duplicate weighting, canonical/raw-null loss, weak tests, and detached integration each have an explicit task and acceptance check.
- Type consistency: `MetricLabelObservation`, `MetricAliasRule`, `MetricExpansion`, `build_metric_alias_lexicon`, and `expand_metric_query` use the same names across producer and consumer tasks.
- Scope: dense retrieval, entity parsing, gold mutation, prediction-derived labels, rank fusion, and metric-formula changes remain excluded.
- Execution order: contracts -> documents -> lexicon -> service -> persistence -> clean verification -> real replay -> documentation.

## Execution Handoff

Two supported execution modes:

1. **Subagent-Driven (recommended):** use `superpowers:subagent-driven-development`, one implementer per task, followed by spec and code review.
2. **Inline Execution:** use `superpowers:executing-plans`, execute Tasks 1-5 in small batches, then Tasks 6-8 as verification checkpoints.

