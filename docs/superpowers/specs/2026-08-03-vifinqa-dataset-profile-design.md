# ViFinQA Dataset Profile Notebook Design

## Goal

Replace the legacy OCR-corpus analysis in `notebooks/01_dataset_profile.ipynb` with a read-only profile of the ViFinQA release stored at `data/raw`. The notebook should summarize the reports, question collection, and stock-code mapping; expose coverage and data-quality risks; and remain fast enough to run locally.

## Inputs and boundaries

The notebook resolves the repository root and reads these inputs:

- `data/raw/code_stock.csv`
- `data/raw/questions/questions.jsonl`
- `data/raw/financial_statements/**/*.txt`

It never modifies files under `data/raw`. Metadata is collected for every report, while report contents are read only through a deterministic bounded sample. The notebook replaces the previous `ocr_result` assumptions instead of supporting both layouts.

## Notebook structure

1. Configuration and dataset-path validation.
2. Reusable parsers and profiling helpers.
3. Synthetic self-checks for report paths, questions, and bounded text inspection.
4. Dataset overview covering question, report, company, year, and byte counts.
5. Report coverage by ticker, year, and statement type.
6. Question analysis covering length, duplicate text, IDs, mentioned years, mentioned tickers, and numerical-language indicators.
7. Cross-file integrity checks between report paths, question mentions, and `code_stock.csv`.
8. A deterministic report-content sample for UTF-8, OCR, and table-marker indicators.
9. A concise readiness summary translating findings into follow-up actions.

## Components and data flow

`load_company_map` validates the stock-code CSV and normalizes ticker strings. `load_questions` parses JSON Lines one record at a time, preserves line numbers, and reports malformed rows or invalid fields. `build_report_inventory` walks the statement tree and derives ticker, year, document name, statement type, size, and structural status from each path. Text-inspection helpers read at most the configured byte limit from a stable random sample.

The three inventories remain separate pandas data frames and are joined only for explicit cross-checks. This keeps input-specific errors visible and avoids silently dropping unmatched records.

## Outputs

The notebook displays:

- KPI tables for the whole release.
- Report counts by year, ticker, and statement type.
- A ticker-year coverage heatmap.
- Question-length and mentioned-year distributions.
- Duplicate, missing-ID, malformed-path, empty-file, and unmatched-ticker tables.
- Sample-based OCR/encoding/table-marker metrics.
- A final prioritized readiness table.

Charts use matplotlib and pandas only, matching the existing project dependencies.

## Error handling

Missing required paths or files stop execution with a message naming the expected ViFinQA layout. Individual malformed JSONL rows, stat failures, invalid report paths, and decoding failures are retained as quality records rather than aborting the complete profile. Empty collections produce explicit messages and avoid invalid divisions or plots.

## Verification

Tests load the tagged helper cell from the notebook and exercise it against temporary synthetic ViFinQA files. They cover valid and malformed report paths, JSONL validation, company-map normalization, deterministic sampling, truncated reads, invalid UTF-8, duplicate question detection inputs, and cross-file ticker checks.

Before completion, the notebook JSON and every Python code cell are syntax-checked, the focused tests are run, and the notebook is executed from top to bottom against the downloaded dataset. The saved notebook must finish without exceptions and show counts consistent with the release card: 1,973 reports, 1,012 questions, and 100 mapped companies.
