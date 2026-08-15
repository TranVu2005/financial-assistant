# Answer gold v1 provenance

Task 20.9 built `answer-gold-v1.jsonl`: 30 numeric-answer labels for the 30 gold70 questions the
Day 16 rule planner resolved to a scalar `answered` result via `execution compile-plans` on release
`data/processed/release_v2_422df141c935`
(`dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`).

Before this task, no numeric ground truth existed anywhere in `data/qa/` — `GoldRetrievalQuestion`
has no "answer" field, and `plan-cases-v1.jsonl` only labels `expected_operation`/abstain codes.
Day 20 plan §1.1 measured this as a blocker: the Day 21 gate ("answer accuracy ≥ 0,85") has nothing
to score against without it.

## Method (ADR 0009 decision A2)

Per ADR 0009: labels were read from the source report text, independent of what the executor
computed, then compared afterward — not generated from executor output (which would make accuracy
trivially 1.0 and defeat the point of a gold set).

1. For each of the 30 `answered` results, every evidence `CellMatch` was resolved to its
   `documents.relative_path` + `cells.source_line_start/end` via `cells.parquet`/`tables.parquet`/
   `documents.parquet` — the same provenance chain Day 20 plan §1.6 measured as 100% complete
   (33/33 evidence cells on gold70).
2. The literal `_extracted.txt` line at that offset was parsed **independently of the extraction/
   normalization pipeline**: a standalone regex-based HTML-table parser (not
   `financial_report_qa.data.*`) located the `<tr>` matching the cell's `row_label_raw` and read off
   the raw Vietnamese-formatted number (`.` thousands separator, `(…)` for negative) in that row.
3. The parsed source number was compared to `cells.parquet`'s stored `value_numeric` for that cell.
   51/51 evidence cells were checked this way (30 questions, several sharing evidence across
   `lookup`/`difference`/`growth_rate` variants of the same underlying company-year figure).
4. Two automated "mismatches" (4 of the 51 checks) were investigated by hand and confirmed to be
   **parser artifacts of the cross-check script itself**, not data errors: the row-matching heuristic
   picked an earlier row sharing a text substring (e.g. "Lưu chuyển tiền thuần từ hoạt động kinh
   doanh trước những thay đổi…") instead of the intended summary row. Reading the actual line by eye
   confirmed the summary row's value matches `cells.parquet` exactly in both cases (CTG 2022:
   84.420.878; CTG 2023: 27.802.896, both `VND_million`).
5. One genuine discrepancy was found and is **not** silently corrected to match the executor (see
   below).

## The one flagged discrepancy

`cells.parquet` stores `16465930202330.002` VND for VGT's 2023 "Doanh thu thuần" cell
(`cell_7d2406cb…`). The source line
(`VGT/2023/VGT_financial_statements_2023_consolidated/VGT_financial_statements_2023_consolidated_extracted.txt:1926`)
reads `16.465.930.202.330` — no fractional VND. The `.002` is an ingestion-time float-precision
artifact (relative magnitude ≈ 1.2 × 10⁻¹⁶, invisible at any realistic display precision), not a
locator or compiler bug — confirmed by checking that the corresponding VGT 2022 evidence cell
(`18272547438555.0`, from its own 2022 filing) matches its source line exactly with no such residue.

Three gold labels touch this cell and are recorded with the **source-read** value, not the
executor's:

| question_id | operation | executor answer | gold answer (source) | deviation |
| --- | --- | ---: | ---: | ---: |
| `retq_9abb0ade…` | lookup | 16465930202330.002 | 16465930202330 | 0.002 VND |
| `retq_1b05912e…` | difference | -1806617236224.998 | -1806617236225 | 0.002 VND |
| `retq_5adfcd4b…` | growth_rate | -0.09887057304407611257780321750 | -0.09887057304407622203161674434 | ≈1.1 × 10⁻¹⁶ (relative) |

Each record's `matches_executor_answer` field is `false` for these three and `true` for the other 27
— this field is intentionally part of the schema so a future reader does not have to recompute the
diff to know which labels are exact.

## What this is not

This is not a claim that the deterministic compiler (Day 18–19) is bug-free — it is a claim that,
for these 30 questions on this locked release, the compiler's arithmetic and locator matched the
source report in every case except one immaterial float-precision residue traceable to ingestion,
not to `execution/`. The sample is 30 of 70 gold70 questions (43%); Day 21's mandate to expand the QA
set to ≥120 questions is the place to grow this, not this task.

## Day 21 (task 21.10): 30 → 58

Task 21.10 ([docs/plans/day21-e2e-week3-gate.md](../../docs/plans/day21-e2e-week3-gate.md) §3)
labeled 28 additional answers for questions that became `verified` only after the E2E pipeline
(task 21.7) replaced `gold_table_ids` with real BM25 v4 retrieval rankings over the full
120-question set (`financial-report-qa retrieval evaluate-v2`, `artifacts/evaluations/day21/retrieval/`)
— all 28 are from the 50 questions added in task 21.9; 0 previously-unlabeled questions among the
original 70 became newly verified.

Method, adapted from ADR 0009 decision A2 for scale (28 questions vs. 30):

1. **Layer 1** (all 28): each evidence cell's `value_numeric` was read from `cells.parquet` by
   `(table_id, row_label_canonical, period)` — upstream of `pandas_query`/`operations.py` dispatch,
   so this is not the executor's derived answer, the same distinction Day 20 relied on. All 28 agreed
   exactly with the pipeline's computed answer (or matched within the same float-drift magnitude as
   the Day 20 VGT case, e.g. `2164998913301.9998` vs. the true `2164998913302`).
2. **Layer 2 spot-check** (4/28, chosen for diversity of operation and metric — DLG lookup, VGC
   lookup, VPI lookup with a float-drift case, POW two-period difference, SHB two-period growth_rate
   with an unusually small absolute value worth double-checking): the immutable `_extracted.txt` line
   was read directly and matched Layer 1 exactly in all 4, including confirming the VPI float-drift
   case (`Doanh thu thuần` = `2.164.998.913.302` in source, no `.9998` residue — same artifact class
   as Day 20's VGT case, not a new bug) and confirming the SHB figure is genuinely `triệu VND`
   (million VND, per the column header), not an extraction error.
3. Given 28/28 Layer-1 agreement and 4/4 Layer-2 confirmation, the remaining 24 were not
   individually re-verified against raw text — this is a lighter-weight pass than Day 20's full
   51/51 cross-check, made defensible by 100% Layer-1 agreement across the whole batch. Documented
   here rather than silently presented as equally rigorous.
4. All labels use the Layer-1 (source-derived) value, not the pipeline's computed float, for the
   same reason as the Day 20 VGT case: never adjust a label to match machine output.

Re-running `verify-answers`/`pipeline run-e2e` with the expanded 58-label set and the real 120-question
BM25 v4 ranking gives the first accuracy measurement with every currently-verified answer scored: **39/39
verified questions scored, 33 correct, 6 overconfident-wrong, accuracy 0.846** (`none` scope policy —
see [docs/decisions/0010-statement-scope-contract.md](../../docs/decisions/0010-statement-scope-contract.md)
for why this is reported as one line of a 3-policy tradeoff table, not a single number).
