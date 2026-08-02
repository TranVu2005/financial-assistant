# Dataset Profile Notebook Design

## Goal

Create one exploratory notebook at `notebooks/01_dataset_profile.ipynb` that profiles the downloaded `ocr_result` tree without modifying raw files. The notebook must make the corpus structure and data-quality risks visible enough to guide inventory, ingestion, and normalization work.

## Scope

The notebook reads TXT files below `data/raw/ocr_annual_financials/ocr_result` by default. A configuration cell allows another root path, a deterministic content-sample size, and a random seed. The notebook remains useful when the corpus is stored elsewhere under `data/`.

It does not extract canonical financial tables, create retrieval indexes, infer missing financial values, or modify source files.

## Analysis Flow

1. Validate the configured root and discover TXT files recursively.
2. Parse ticker, year, report name, and relative path from the expected hierarchy.
3. Build an in-memory Pandas inventory containing path metadata, file size, parsed report attributes, and structural warnings.
4. Show corpus-level KPIs and distributions by year, ticker, report type, and file size.
5. Show a ticker-year coverage matrix and ranked anomaly tables.
6. Read a deterministic sample of file contents to measure encoding/read failures, line counts, table markers, replacement characters, and OCR-noise indicators.
7. Produce a concise readiness summary that maps observed issues to the next ingestion tasks.

## Visual Design

The notebook uses Pandas tables and Matplotlib charts so it works locally in Jupyter without a separate web service. Visuals include year coverage, top tickers, report-type distribution, file-size histogram, and a bounded ticker-year heatmap. Large tables are displayed as sortable DataFrame outputs and limited to useful top-N views.

## Performance and Safety

Metadata is collected for every TXT file. Content inspection is sampled by default to avoid reading the entire corpus. Sampling is deterministic. Raw files are opened read-only, Unicode decoding failures are recorded rather than aborting the run, and all derived values stay in memory unless the user explicitly adds an export step later.

## Error Handling

The notebook stops early with a clear message when the configured root does not exist or contains no TXT files. Individual malformed paths and unreadable files are retained in anomaly tables with an error field. A single bad file never stops the remaining scan.

## Verification

Notebook helper functions are exercised against a temporary synthetic `ocr_result` tree before the real-data analysis cells run. Verification covers valid path parsing, malformed paths, deterministic sampling, readable UTF-8 content, and unreadable/invalid content handling. The final notebook is also parsed as JSON and all code cells are syntax-checked.

## Success Criteria

- A user can set one path and run all cells from top to bottom.
- The notebook reports corpus size, company/year coverage, report categories, file-size distribution, and actionable anomalies.
- Default execution never writes to or changes `ocr_result`.
- Content analysis is bounded and reproducible.
- The notebook explains which findings should influence the next inventory and ingestion implementation.
