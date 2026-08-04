# Normalization Issue Audit Workflow

This guide details the procedure for generating deterministic samples of normalization issues, labeling them, and producing reproducible baseline evaluation reports.

## Important Policy Notice
> [!NOTE]
> Generated audit samples, human labeling CSV files, and evaluation reports contain localized quality inspection data and **must not be committed by default**. They should remain local or be published to external QA archival repositories.

## 1. Sample Generation
To extract a deterministic stratified sample of normalization issues from a dataset release:

```bash
normalization-audit sample \
    --release /path/to/normalized_dataset_release \
    --output data/qa/normalization_issue_sample.parquet \
    --config configs/normalization_audit.yaml
```
- The sample generation algorithm uses a stable SHA-256 hash rank to guarantee order independence.
- Overwriting existing sample Parquet files is restricted if the embedded dataset fingerprint differs from the source release.

## 2. CSV Labeling
Inspect the generated sample table (`data/qa/normalization_issue_sample.parquet`) and create a corresponding human-reviewed labels file at `data/qa/normalization_issue_labels.csv`.

Required columns in `normalization_issue_labels.csv`:
- `sample_id`: 64-character SHA-256 sample identifier.
- `label`: Exactly one of `true_issue`, `false_positive`, or `uncertain`.
- `cause_code`: Valid taxonomy cause code (e.g., `ocr_corruption`, `year_header_as_unit`).
- `reviewer_note`: Optional explanation or rationale.

## 3. Baseline Validation & Reporting
To validate human labels and generate deterministic JSON and Markdown evaluation reports:

```bash
normalization-audit baseline \
    --sample data/qa/normalization_issue_sample.parquet \
    --labels data/qa/normalization_issue_labels.csv \
    --output-dir artifacts/normalization-audit/
```
This produces:
- `artifacts/normalization-audit/baseline.json` (deterministic sorted keys)
- `artifacts/normalization-audit/baseline.md` (issue-code-ordered summary tables)

## 4. Source Context Inspection
To investigate root causes or verify extraction boundaries during labeling, inspect the source context fields included directly within each sampled row of `normalization_issue_sample.parquet`:
- `doc_id`, `table_id`, `cell_id`
- `table_title_raw`, `table_unit_raw`
- `row_label_raw`, `column_label_raw`, `value_raw`
- `source_line_start`, `source_line_end`
