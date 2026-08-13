# Day 13 final review fix report

Date: 2026-08-14

Branch: `codex/day13-retrieval-evaluation`

Review range: `b99be08..7a4209f`

## Scope and outcome

This wave closes every final-review finding without rebuilding a dense model or rerunning GPU
work. Existing locked reports are the ranking inputs for deterministic V2 transformations; the
locked BM25 index is used only for the product-CLI diagnostic replay.

### 1. Versioned BM25 reference identity

- Added immutable `gold30` and `gold70` descriptors. Each binds the locked dataset fingerprint,
  gold JSONL SHA-256, ordered question-ID digest, question count, expected legacy metrics, and
  exact BM25 report SHA-256.
- `gold70` is the current default dataset contract. `gold30` remains replayable either from its
  historical JSONL or as the byte-identical locked subset selected from the current gold70 file.
- Reference validation now rejects incomplete, duplicate, reordered, incorrectly scored, or
  incorrectly aggregated reports. Exact artifact loading additionally checks the report SHA-256.
- Legacy report models were not changed, so old Day 8--12 JSON remains parseable.

### 2. Product CLI and deterministic writers

- Added `retrieval evaluate-v2` for a full V2 BM25 report with optional diagnostic depth.
- Added `retrieval derive-v2` to rescore persisted legacy/dense/fusion/expansion rankings into a
  source-SHA-bound full V2 system report, without rerunning GPU models.
- Added `retrieval export-failures --annotations` and a strict deterministic JSONL annotation
  loader.
- Added deterministic JSON/Markdown writers for diagnostic V2, per-system V2, and failure
  reports.
- Added the tracked `data/qa/retrieval-failure-annotations-v1.jsonl`. Its 11 manual labels and
  notes are source-backed and kept separate from ranked output; ranking never creates gold labels.
- README now documents clean-clone commands using existing locked release/index inputs.

### 3. Full V2 evidence artifacts

- Replaced the insufficient macro-only `metrics-v2-422df141c935.json` with six complete,
  per-system JSON/Markdown report pairs under `artifacts/evaluations/day13/v2/`:
  `bm25-v3`, `dense-bge-m3`, `dense-e5-small`, `fusion-bge`, `fusion-e5`, and
  `graph-expansion`.
- Every JSON report stores 70/70 question records, source path/SHA, selection identity, macro
  metrics, and all planned breakdowns: intent, gold cardinality, period cardinality, statement
  filter, and report era.
- Added a complete BM25 diagnostic V2 pair under `day13/v2/bm25-diagnostic/` and regenerated the
  failure pair through the new CLI.
- Artifact regression tests require the exact six claimed systems, the complete breakdown label
  sets, 70/70 records, source SHA agreement, expected F2@R claims, and safe diagnostic ranks.

JSON artifact SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `retrieval-v2-bm25-v3-422df141c935.json` | `9db8ff98687445074ee2ba5d743200c7e2cac781f2fa247a468002d4f0ab4539` |
| `retrieval-v2-dense-bge-m3-422df141c935.json` | `95e10fc0d17dd0804b6117acddc72fad957090f9a5031ac2bbece661ece0cb17` |
| `retrieval-v2-dense-e5-small-422df141c935.json` | `ad3f86ed97cadfe187205671306291e1bbe154d44623527c87207a216633d4a9` |
| `retrieval-v2-fusion-bge-422df141c935.json` | `05b0504e9d8d7d157d98ff8eb95e6c0a1e4ed200096f79ec1b899e966310e43a` |
| `retrieval-v2-fusion-e5-422df141c935.json` | `f3c08cf51b0642faecc63d879e2c77af2d7719c14e1b19955a5f73b0a6688928` |
| `retrieval-v2-graph-expansion-422df141c935.json` | `2f9a4557c41ffa827801a2a9986ce075f9ec509fa642d758885767e98e053273` |
| `bm25-diagnostic/retrieval-v2-422df141c935.json` | `cdf4cf515725d29a7f95303e2f0385f91b91adef2cb6b53a2c00ac5e57dea815` |
| `failures-422df141c935.json` | `0530751d4275074385ea4a63a2da0fe5cc249c1b4d3b243ccabc750103296cac` |
| `retrieval-failure-annotations-v1.jsonl` | `2de345784fc673cc07ae61b3d03e616129fffb6d62898fd1972da470841c7913` |

### 4. Diagnostic rank safety

- V2 still scores an independent fixed top-10 trace.
- A deeper diagnostic retrieval is accepted only if its first ten table IDs exactly preserve the
  metric trace. A k-dependent retriever that changes that prefix now fails closed with an explicit
  artifact error; ranks are never silently rewritten.
- README clarifies that only the BM25 diagnostic artifact stores rank-100 evidence. The persisted
  dense/fusion/expansion reports make no diagnostic-cutoff claim.

### 5. Documentation corrections

- `plan.md` now names `artifacts/evaluations/day13/**/*.json` and `.md` as the Day 13 output.
- README points claims to the full V2 artifacts, documents all three new CLI paths and both
  reference versions, and corrects diagnostic-rank wording.

## TDD evidence

Production behavior was introduced only after focused RED failures:

1. Reference tests first failed because the versioned reference module did not exist; the
   truncated current report was accepted by the old global metric-only lock. GREEN adds the two
   descriptors, exact artifact loader, deep content validation, and historical subset resolver.
2. The k-dependent retriever regression first failed because V2 accepted divergent `k=10` and
   `k=100` rankings. GREEN adds the top-10 prefix invariant.
3. Writer tests first failed because no V2 writer existed. GREEN adds deterministic full
   JSON/Markdown output and completeness validation.
4. Offline CLI tests first failed because `evaluate-v2`, `derive-v2`, and `export-failures` were
   not commands. GREEN exercises all three on local fixtures twice and checks byte identity.
5. Artifact tests first failed because only the macro-only summary existed. GREEN validates six
   full system reports and the diagnostic artifact.

## Verification evidence

- Focused retrieval/CLI suite: `192 passed, 3 skipped`.
- Full suite: `767 passed, 4 skipped`.
- Ruff on every changed Python/test file: `All checks passed!`.
- Mypy on every changed source file: `Success: no issues found in 7 source files`.
- Full Ruff: exactly the existing `102 errors`; no new Day 13 error.
- Full mypy: exactly the existing `33 errors in 5 files (checked 162 source files)`; no new error.
- `git diff --check`: clean.
- Real product-CLI replay against the locked 146,011-table BM25 index compared 16 generated
  JSON/Markdown files with the tracked outputs: `16 compared, 0 mismatches`.

## Self-review

The final staged diff was audited requirement-by-requirement and by production boundary. No open
Critical or Important finding remains. In particular:

- reference selection is content-locked before release provenance validation and exact report
  files are SHA-locked at every product-CLI boundary;
- diagnostic metrics retain their independent top-10 trace and reject a divergent deep prefix;
- every README metric row has one source-bound full V2 artifact with all planned breakdowns;
- all ignored generated artifact paths and this report are explicitly staged;
- the documented clean-clone commands use `uv run --frozen --no-sync` and no network/GPU step;
- the staged scope contains no model/index rebuild, gold relabeling, or unrelated source change.

Known repository-wide Ruff/mypy debt is unchanged from the supplied baseline and is the only
remaining concern outside this fix scope.
