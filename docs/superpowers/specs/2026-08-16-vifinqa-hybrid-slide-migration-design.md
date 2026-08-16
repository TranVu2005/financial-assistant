# ViFinQA Hybrid Slide-Aligned Migration Design

**Date:** 2026-08-16  
**Status:** Proposed for user review  
**Objective:** Replace the low-ceiling Day 25 planning/reasoning path with a slide-aligned, evidence-first hybrid pipeline while preserving the project's proven ingestion, provenance, execution safety, and submission validation components.

## 1. Context and Measured Problem

The Day 25 final export covers all 1,012 required IDs, but only 113 items are produced by a reasoning path; 899 are contract-valid backstops that select an arbitrary real numeric cell. The 113 reasoned items are distributed as follows:

| Tier | Questions | Reasoned | Reasoned coverage |
|---|---:|---:|---:|
| Easy | 361 | 70 | 19.4% |
| Medium | 235 | 5 | 2.1% |
| Intermediate | 200 | 22 | 11.0% |
| Hard | 216 | 16 | 7.4% |

The dominant failure codes are `metric_not_found` (362), invalid LLM plans or JSON (229), `period_unresolved` (133), `cell_ambiguous` (72), `unit_missing` (43), `no_candidate_tables` (42), and scope-empty candidates (18). This shows that the main constraint is evidence representation and plan expressiveness, not arithmetic or ZIP validity.

The official slide deck describes a different successful shape: preserve table structure and metadata, retrieve top-50 candidates, rerank to top-10 evidence tables, apply adaptive CoT/PoT reasoning, and independently audit the selected evidence. This migration adopts that shape without abandoning deterministic execution or submission replay.

## 2. Scope

### 2.1 In scope

- A new hybrid submission path behind an explicit configuration flag.
- Context-preserving evidence views built from the locked release.
- Query decomposition and four-tier difficulty routing.
- Hybrid lexical/dense retrieval with reranking and soft metadata constraints.
- A typed computation DAG that can express Easy, Medium, Intermediate, and Hard questions.
- Frontier-model planning over real table/cell identifiers.
- Deterministic DAG compilation, sandbox execution, semantic auditing, and replayable submission export.
- A semantic fallback for every question before the existing contract-only backstop.
- Tier-stratified evaluation and experiment provenance.

### 2.2 Out of scope

- Rebuilding the Streamlit UI.
- Graph expansion or graph neural retrieval.
- Fine-tuning a local model.
- Expanding the canonical metric dictionary by hand.
- Re-ingesting the entire corpus unless a measured blocker cannot be solved from existing Parquet and source TXT artifacts.
- Replacing the stable submission contract, archive validator, or provenance identifiers.

## 3. Preserve, Replace, and Retire

### 3.1 Preserve

- Release inventory and document/table/cell provenance.
- Existing normalized numeric values and source line references.
- Submission contracts, deterministic ZIP packaging, archive security checks, and full-ID validation.
- Sandbox boundaries and the requirement that every submitted `pandas_query` replay to the declared answer.
- Existing rule-based answers when they pass the new semantic auditor.

### 3.2 Replace on the primary submission path

- BM25-only top-10 retrieval.
- Hard company/period/scope filters that can empty the candidate set.
- The nine-operation `FinancialQueryPlan` as the universal reasoning representation.
- Qwen2.5-7B generation of a complete plan.
- Row-label-only LLM grounding.
- Arbitrary-cell backstop as the first response to reasoning failure.

### 3.3 Retire from score optimization

- `answered_count` as a proxy for accuracy.
- Gold-filter retrieval results as a proxy for raw-query retrieval quality.
- UI, latency, and graph work until the score gates in Section 11 are reached.

## 4. Target Architecture

```text
Raw question
  -> Query analyzer and difficulty router
  -> Atomic evidence requests
  -> Hybrid candidate generation (BM25 + dense, top 50)
  -> Cross-encoder reranker (top 10 tables)
  -> Context-preserving evidence pack
  -> Frontier planner selects CellRefs and a ComputationDAG
  -> Deterministic compiler generates Pandas
  -> Sandbox executes
  -> Deterministic checks + independent semantic auditor
  -> One bounded repair attempt when rejected
  -> Replayable SubmissionItem
  -> Semantic fallback if the primary path fails
  -> Contract-only backstop only after every semantic attempt fails
```

The current pipeline remains available as a control and rollback path. The hybrid path does not silently overwrite it.

## 5. Evidence Representation

The current execution frame is intentionally lossy. The new `EvidenceCell` and `EvidenceTable` views preserve the context required to select the correct row and column.

### 5.1 `EvidenceCell`

Each numeric cell exposes:

- `cell_id` and `table_id`.
- `company_code`, report year, statement scope, and statement type when available.
- Raw row label, canonical row label when available, and reconstructed row hierarchy.
- Raw column label, flattened header path, resolved period, and whether period was inferred.
- Raw value, normalized numeric value, explicit unit, inherited table/page unit, and scale.
- Source document, source line, row index, and column index.

### 5.2 `EvidenceTable`

Each candidate table exposes:

- Table title and surrounding section text.
- Document metadata and statement scope.
- Flattened header paths.
- All numeric `EvidenceCell` records in stable row/column order.
- A compact rendering for LLM prompts and a complete tidy DataFrame for execution/export.

These views are derived at query time or cached from existing `cells.parquet`, `tables.parquet`, `documents.parquet`, and source TXT files. A corpus-wide re-ingestion is not the default.

## 6. Retrieval Design

### 6.1 Query decomposition

The analyzer emits one or more `EvidenceRequest` objects containing:

- Company candidates.
- Period or period range.
- Scope if explicitly stated.
- Metric phrase copied from the question.
- Requested role such as operand, filter, ranking key, or final output.

The analyzer must preserve the original metric text. Canonical aliases may add search terms but may not veto a raw phrase.

### 6.2 Candidate generation

- Retrieve lexical top-50 and dense top-50 for every atomic evidence request.
- Merge by weighted reciprocal rank fusion.
- Treat company, period, and scope as score features. They are hard constraints only when explicitly stated and at least one matching candidate exists.
- If a hard constraint would return zero candidates, retry once with that constraint converted to a demotion feature and record the relaxation.
- Union candidates across atomic requests before reranking.

### 6.3 Reranking

A cross-encoder reranks the union using the question, table title, section context, row labels, flattened headers, scope, and year. The planner receives at most ten tables, but retrieval traces retain the top 50 for semantic fallback and auditing.

## 7. Difficulty Routing and Reasoning

Difficulty is inferred from question structure, never from question ID.

### 7.1 Easy

One company, one period, one requested value, and no derived operation. The planner selects exactly one `CellRef`; conversion and rounding are deterministic.

### 7.2 Medium

Exactly two operands and one arithmetic operation. The planner selects two `CellRef` values plus an enum operator. It does not generate Pandas or free-form formulas.

### 7.3 Intermediate

More than two values, repeated formulas, aggregation, ranking, or filtering without dependent retrieval. The planner emits a typed computation DAG.

### 7.4 Hard

An intermediate result determines the company, year, or subsequent evidence request. The planner emits stages. A stage may execute and then trigger one additional retrieval round using its bounded intermediate result.

## 8. Computation DAG

The new DAG replaces the nine-operation universal plan while remaining deterministic and auditable.

Node families:

- `CellValue(cell_id)` and `Constant(value)`.
- `Add`, `Subtract`, `Multiply`, `Divide`, and `AbsoluteDifference`.
- `Sum`, `Mean`, `Median`, `Min`, `Max`, and `Count`.
- `Compare` predicates and `Filter`.
- `ArgMin` and `ArgMax` returning the associated company or period.
- `PeriodShift` with an integer year offset.
- `Round` applied only to the final result.

Every `CellValue` must reference a cell included in the evidence pack. Division by zero, unresolved units, incompatible scales, empty filters, and non-scalar final results are deterministic failures.

The compiler translates the DAG into a safe Pandas expression over one or more evidence DataFrames. The sandbox is extended to accept a mapping of `df1..dfN`, but it continues to deny imports, I/O, network access, arbitrary calls, and unsupported AST nodes.

## 9. Model Collaboration

### 9.1 Planner

A configurable frontier model receives the question, difficulty, atomic evidence requests, and compact top-10 evidence tables. Its output is constrained JSON containing only real table IDs, cell IDs, and DAG nodes.

### 9.2 Coder

The deterministic DAG compiler is the only production coder. Model-generated Pandas is retained solely as an offline benchmark and is never packaged in the final submission. This keeps implementation scope bounded and preserves the current safety contract.

### 9.3 Auditor

The auditor receives the question, chosen cells, table context, formula, result, unit conversions, and scope decisions. It returns `accept` or a typed rejection code. It cannot directly replace the numeric answer. One repair attempt may ask the planner to choose alternative evidence or correct the DAG.

The same provider may implement planner and auditor with independent prompts, but their calls and outputs are logged separately. Local models below 10B are limited to classification or candidate selection and are not used as universal planners.

## 10. Fallback and Error Handling

1. Existing verified rule answer, if accepted by the new auditor.
2. Hybrid primary plan over reranked top-10 evidence.
3. One planner repair using the auditor rejection.
4. Semantic fallback over the retained top-50 evidence, using the strongest available planner and a less restrictive evidence pack.
5. Retrieval relaxation when no candidate exists.
6. Existing contract-only backstop solely to preserve all 1,012 IDs.

Every stage records whether the result is `rule_verified`, `hybrid_primary`, `hybrid_repaired`, `semantic_fallback`, or `contract_backstop`. The final report must never combine these categories into a single `answered_count`.

## 11. Evaluation and Release Gates

### 11.1 Development data

- Build an 80-case development set with known evidence, formulas, and answers: 20 cases per tier, derived from the same corpus but distinct from the locked audit cases.
- Lock a 40-question audit sample from the official set with 14 Easy, 9 Medium, 8 Intermediate, and 9 Hard questions. Each audit answer and evidence trace is verified by two independent calculations before the sample is locked; audit cases are used for release selection, not prompt-specific editing.
- Do not select prompts or thresholds from Dashboard scores alone.

### 11.2 Metrics

- Evidence-table Recall@10 and required-cell recall.
- Cell selection accuracy.
- DAG/formula accuracy.
- Execution accuracy.
- Exact/tolerance answer accuracy over all questions in the evaluated split.
- Contract-backstop rate.
- Invalid structured output rate.
- Per-tier latency and cost.

At most three configurable frontier planner candidates are benchmarked with identical evidence, prompt version, token budget, and audit sample. Selection order is weighted audit answer accuracy, execution accuracy, required-cell accuracy, then cost and latency; no provider name is hard-coded into the architecture.

### 11.3 Minimum gates for the final migration

- 100% submission contract validity and query replay.
- 100% of questions receive a semantic attempt before contract backstop.
- Contract-backstop rate below 20%, with a stretch goal below 5%.
- Required evidence present in top-10 for at least 75% of the locked audit set.
- Audit answer accuracy: Easy at least 65%, Medium at least 45%, Intermediate at least 30%, Hard at least 15%.
- Weighted audit accuracy at least 40% before replacing the Day 25 submission path.
- The new path must beat the frozen Day 25 control on the same audit set and dataset fingerprint.

## 12. Five-Day Delivery Sequence

### Day 26: evaluation lock and evidence contracts

- Freeze the Day 25 artifact, config, model identifier, and dataset fingerprint as control.
- Add tier labels, evidence contracts, and a locked audit set.
- Implement context-preserving evidence rendering from the existing release.

### Day 27: Easy and Medium score path

- Implement query decomposition, difficulty routing, single-cell selection, two-cell arithmetic, unit conversion, and exact cell-ID replay.
- Replace arbitrary fallback for Easy and Medium with semantic fallback.

### Day 28: hybrid retrieval

- Add dense candidates, fusion, reranking, and soft-filter relaxation.
- Measure table and cell recall by tier before changing reasoning prompts.

### Day 29: Intermediate and Hard reasoning

- Implement the computation DAG, deterministic compiler, one-hop dependent retrieval, and auditor/repair loop.
- Run the full 1,012-question export with complete traces.

### Day 30: ablation, selection, and freeze

- Compare the Day 25 control, hybrid without reranker, hybrid with reranker, and hybrid with auditor.
- Select the artifact by weighted audit accuracy, then execution accuracy, then contract-backstop rate.
- Validate the final ZIP against all 1,012 IDs, replay every query, record SHA-256/config/model hashes, and freeze the repository state.

## 13. Testing Strategy

- Contract tests for every DAG node and invalid node combination.
- Golden tests for one Easy, Medium, Intermediate, and Hard example per core formula family.
- Retrieval tests proving soft filters cannot permanently empty a valid candidate set.
- Property tests for unit-scale invariance, row-order invariance, and deterministic compilation.
- Security tests for malicious model output and unsupported Pandas AST.
- Integration tests from raw question through multi-DataFrame submission replay.
- Regression tests for every Day 25 failure converted into a valid semantic answer.

## 14. Operational Safety

- Work starts from a new feature branch or isolated worktree; existing uncommitted Day 23-25 changes are preserved.
- Every experiment records Git commit, dataset fingerprint, retrieval index fingerprint, planner model, auditor model, prompts, seeds, latency, cost, and artifact SHA-256.
- No Dashboard upload is used as the first verification step.
- The current submission exporter remains selectable until the hybrid gates pass.

## 15. Decision

Adopt the hybrid migration. Preserve data/provenance/export safety, replace retrieval and reasoning on the score path, and retain the Day 25 pipeline only as a frozen control and rollback option.
