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
