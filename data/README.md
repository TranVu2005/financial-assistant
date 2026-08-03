# Data layout

Only small, redistributable fixtures and manifests may be committed. Do not
commit source PDFs, generated parquet files, OCR output, indexes, or model files.

`data/raw/` is append-only. After downloading a revision-pinned ViFinQA snapshot,
create its deterministic manifest without modifying source files:

```bash
financial-report-qa inventory-data \
  --root data/raw/ViFinQA/financial_statements \
  --repo-id tinixai/ViFinQA \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --manifest data/manifests/documents.jsonl
```

Only commit small, redistributable manifests. Investigate every `record_type="issue"`
entry before ingestion.

See [`docs/data-download.md`](../docs/data-download.md) for the WSL/Linux
commands that dry-run, download, and resume the public OCR corpus.
