# Retrieval gold v1 provenance

Task 13.2 expanded `retrieval-gold-v1.jsonl` from 30 to 70 reviewed questions for locked dataset fingerprint `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.

The original 30 JSONL record strings are unchanged. Their source blob SHA-256 is `1b4646e5b2adac433522bfaa6d3de87951f0aef9a0600140336cd1ac65034404`; all 30 exact byte strings occur in the sorted 70-record artifact.

The 40 additions were authored leakage-safely:

1. Documents were selected from `documents.parquet` by company, year, and reporting scope to satisfy the published quotas.
2. Tables were selected from `tables.parquet` by statement metadata, source span, and required cardinality before any question text was written.
3. Questions and labels were written from canonical `tables.parquet`/`cells.parquet` content and immutable `_extracted.txt` provenance. No BM25, dense, fusion, or graph ranked list was opened or used to choose a gold table.
4. Every table was joined back to its source document and persisted with its exact `relative_path`, `line_start`, `line_end`, and `verified=true` evidence.
5. IDs were derived with `stable_question_id`, and all 70 records were sorted by `question_id`.

The additions contain 14 lookup, 13 compare, and 13 growth questions; cover 30 previously unseen companies with at most two additions per company; include 26 early-period questions and eight 2024–2025 questions; use all four statement filters; contain 12 notes questions; and have cardinalities 24×1, 4×2, 8×3, and 4×4. Three questions require both separate and consolidated reports for the same company-year, and 11 intentionally use Vietnamese abbreviations or aliases not copied from table titles.

## Day 21 expansion: 70 → 120

Task 21.9 ([docs/plans/day21-e2e-week3-gate.md](../../docs/plans/day21-e2e-week3-gate.md) §3, ADR [0010](../../docs/decisions/0010-statement-scope-contract.md) decision F1) added 50 questions to satisfy the plan.md Week-3 gate ("nâng bộ QA lên ít nhất 120 câu") and to correct a scope-wording skew measured in the Day 21 diagnosis (§1.6): gold70 stated `riêng`/`hợp nhất` in only 22.9% of questions and leaned `consolidated` (8 vs. 5), while the official 1,012-question ViFinQA set states a scope in 37.7% of questions and leans `separate` 28× over `consolidated` (36.4% vs. 1.3%).

The original 70 JSONL record strings are unchanged, byte-for-byte, including line endings — every original line was carried forward verbatim, never re-serialized. Their whole-file source blob SHA-256 (`5ed12e6abfe03009a4792d45c2e437bbe615257fc2eeb20d8feb32ac9dbd8b9e`, still `CURRENT_BM25_REFERENCE.gold_sha256` in `retrieval/reference.py`) is reconstructible byte-exactly as a subset of the new 120-record file — `resolve_gold_reference(path, version="gold70")` was generalized to extract it by locked `question_ids`, the same mechanism `gold30` already used, since it is no longer the whole file. The new full-file SHA-256 is `74048fb845e8d50c750d0c3bfcf98155c44177c8e86b468fc682a2e7b45bbb3d`.

Every additional table selected also requires its containing document's own `report_year` to equal the target period exactly (not just an agreeing `TRY_CAST(LEFT(period,4))` value from any table) — an early draft without this constraint twice produced a gold table whose containing report's fiscal year did not match its own question's `periods` filter, which `load_reviewed_gold`'s existing period-filter invariant correctly rejected. Real end-to-end retrieval evaluation on the full 120-question set (`financial-report-qa retrieval evaluate-v2`, BM25 v4 index, `artifacts/evaluations/day21/retrieval/`) confirms the fixed set loads and scores cleanly: Recall@10 0.9458, F2@R 0.5085 — both above the 70-question baseline (0.9143, 0.4836).

The 50 additions follow the same five rules as the original 40, unchanged:

1. Documents/tables were selected from `documents.parquet`/`tables.parquet` by metadata (company, statement_scope, canonical metric, period) before any question text was written — specifically, `(company, statement_scope, canonical_metric, period)` groups with **exactly one distinct value** (the same "safe lookup group" invariant ADR 0010 decision C1 relies on), so every generated question is unambiguously answerable without needing to guess.
2. Questions and evidence were written from canonical `cells.parquet` content — `source_line_start`/`source_line_end` for every contributing cell — never from a BM25/dense/fusion/graph ranked list.
3. IDs were derived with `stable_question_id`; all 120 records are sorted by `question_id`.
4. At most 2 additions per company (43 distinct companies drawn from 50 additions).
5. A scope phrase (`riêng` for a `separate`-scope table, `hợp nhất` for `consolidated`) was appended to the question text **only when it matches the table's actual scope** — never contradicting the gold table, matching the Day 21 diagnosis finding that 0/70 gold70 questions had a scope-wording contradiction.

Composition of the 50 additions: 25 lookup, 14 growth, 11 compare (two-period, same metric); year range 2014–2025; 25 questions state a scope (25 `riêng`, 2 `hợp nhất` — the low `hợp nhất` count is deliberate, matching the official corpus's 1.3% consolidated-stated rate). Combined with gold70's original 5 separate / 8 consolidated / 3 both-worded, the full 120-question set states a scope in 40/120 (33.3%) — a real, measured improvement toward the 37.7% official rate, not full parity: the original 70 records could not be reworded (byte-identical is a hard invariant carried from Day 13), and their own skew (8 consolidated vs. 5 separate) sets a floor the new 50 cannot fully correct without exceeding the 2-per-company quota. This is documented honestly rather than silently accepted as "done."

Generation script: [scripts/day21_expand_gold.py](../../scripts/day21_expand_gold.py), run once against the locked release.
