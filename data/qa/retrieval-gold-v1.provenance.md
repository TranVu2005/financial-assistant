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
