# BM25 Length-Normalization Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the deterministic BM25 baseline with `b=0.25` so primary statement tables are not over-penalized for legitimate document length.

**Architecture:** Keep the v2 retrieval flow unchanged except for the pinned BM25 `b` parameter and persisted index version. Make this a v3 artifact boundary so no v2 index can be evaluated accidentally.

**Tech Stack:** Python 3.11, bm25s, pytest, Ruff, mypy, PowerShell.

## Global Constraints

- Authority: `docs/superpowers/specs/2026-08-09-bm25-length-normalization-design.md`.
- Preserve reviewed gold, filters, query IDs, tokenization, alias expansion, and ranking order.
- Pin `k1=1.5`, `b=0.25`, `delta=0.5`, and `method="lucene"`.
- Persist `schema_version="bm25-index-v3"`, `builder_version="v3"`, and `query_expansion_version="v1"`.
- Build and evaluate twice against the immutable `dataset-pilot-v1` lock; do not stage generated indexes or reports.

---

### Task 1: Pin the v3 BM25 contract

**Files:**

- Modify: `src/financial_report_qa/retrieval/contracts.py`
- Modify: `src/financial_report_qa/retrieval/index.py`
- Modify: `tests/unit/retrieval/test_index_service.py`

**Interfaces:**

- Produces: v3 `BM25IndexManifest` with `b=0.25`.
- Rejects: persisted v2 manifests before executable artifacts are loaded.

- [x] **Step 1: Write the failing regression test**

Add a `TableDocument` pair with the same company and metric tokens: a short note and a longer
primary statement containing the additional relevant `consolidated` title token. Assert that
`RetrievalService.retrieve("total assets consolidated", ...)` ranks the primary statement first.
Also change persistence assertions to require `bm25-index-v3`/`v3` and make the rejection
fixture use `bm25-index-v2`.

- [x] **Step 2: Run the focused test and observe RED**

Run:

```powershell
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m pytest -q tests/unit/retrieval/test_index_service.py
```

Expected: the new ranking regression fails with `b=0.75`, and v3 manifest assertions fail
because production code still publishes v2.

- [x] **Step 3: Write the minimal v3 change**

In `index.py`, set `B = 0.25`. In `contracts.py`, change only manifest literals to
`bm25-index-v3` and `v3`. In `load_bm25_index`, reject every schema other than v3 with the
existing rebuild-required error. Do not add score boosts, query rules, or new metadata fields.

- [x] **Step 4: Run focused quality gates**

```powershell
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m pytest -q tests/unit/retrieval/test_index_service.py tests/unit/retrieval/test_contracts.py
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m ruff check src/financial_report_qa/retrieval tests/unit/retrieval
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m mypy src/financial_report_qa/retrieval tests/unit/retrieval
git diff --check
```

- [x] **Step 5: Commit the code change**

```powershell
git add src/financial_report_qa/retrieval/contracts.py src/financial_report_qa/retrieval/index.py tests/unit/retrieval/test_index_service.py
git commit -m "fix(retrieval): reduce bm25 length normalization"
```

### Task 2: Prove v3 real-corpus evidence

**Files:**

- Generate only: `data/indexes/bm25-length-v3-a/<fingerprint>/`
- Generate only: `data/indexes/bm25-length-v3-b/<fingerprint>/`
- Generate only: `artifacts/evaluations/bm25-length-v3-a/`
- Generate only: `artifacts/evaluations/bm25-length-v3-b/`
- Modify: `docs/superpowers/specs/2026-08-09-bm25-length-normalization-design.md`
- Modify: `docs/superpowers/plans/2026-08-09-bm25-length-normalization.md`

**Interfaces:**

- Consumes: exact release lock and reviewed v1 gold.
- Produces: replay-identical v3 index/report evidence and recorded metrics.

- [x] **Step 1: Validate immutable inputs**

```powershell
$lock = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
$gold = 'data/qa/retrieval-gold-v1.jsonl'
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m financial_report_qa.cli retrieval validate-gold --release-lock $lock --gold-path $gold
```

- [x] **Step 2: Build two v3 indexes**

```powershell
$lock = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m financial_report_qa.cli retrieval build-index --release-lock $lock --output-root data/indexes/bm25-length-v3-a
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m financial_report_qa.cli retrieval build-index --release-lock $lock --output-root data/indexes/bm25-length-v3-b
```

- [x] **Step 3: Compare deterministic index artifacts**

Compare all relative file SHA-256 values of the two fingerprint directories. Expected: zero differences.

- [x] **Step 4: Evaluate both indexes and compare reports**

```powershell
$fingerprint = '37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f'
$lock = 'data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json'
$gold = 'data/qa/retrieval-gold-v1.jsonl'
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m financial_report_qa.cli retrieval evaluate --release-lock $lock --index-dir "data/indexes/bm25-length-v3-a/$fingerprint" --gold-path $gold --output-dir artifacts/evaluations/bm25-length-v3-a
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m financial_report_qa.cli retrieval evaluate --release-lock $lock --index-dir "data/indexes/bm25-length-v3-b/$fingerprint" --gold-path $gold --output-dir artifacts/evaluations/bm25-length-v3-b
```

Compare JSON and Markdown SHA-256 values. Expected: exact identity, Recall >= `0.8833333333333333`,
F2 >= `0.41798941798941797`, and zero-hit <= `3`.

- [x] **Step 5: Record evidence and commit docs**

Update the design with measured hashes/metrics and mark the plan evidence steps complete. Stage
only the two docs and commit with `docs(retrieval): record bm25 v3 length evidence`.

### Task 3: Verify, review, and integrate

**Files:**

- No additional production files.

- [ ] **Step 1: Run repository tests**

```powershell
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m pytest -q
```

- [ ] **Step 2: Run retrieval static gates and diff check**

```powershell
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m ruff check src/financial_report_qa/retrieval tests/unit/retrieval
D:\GitHub\financial-assistant\.venv\Scripts\python.exe -m mypy src/financial_report_qa/retrieval tests/unit/retrieval
git diff --check
```

- [ ] **Step 3: Request code review**

Review the committed diff against this spec: v3 rejection, no gold/filter/query changes,
determinism, and evidence values. Resolve every Critical or Important finding.

- [ ] **Step 4: Merge the reviewed branch into `main`**

Merge only after fresh verification and review. Keep generated index/report artifacts ignored.
