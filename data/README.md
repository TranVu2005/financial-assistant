# Data layout

Only small, redistributable fixtures and manifests may be committed. Do not
commit source PDFs, generated parquet files, OCR output, indexes, or model files.

`data/raw/` is append-only. Record downloaded documents and checksums in
`data/manifests/documents.csv` before processing them.

See [`docs/data-download.md`](../docs/data-download.md) for the WSL/Linux
commands that dry-run, download, and resume the public OCR corpus.
