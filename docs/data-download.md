# Downloading the OCR financial-report corpus

The downloader fetches a revision-pinned snapshot of
`tinixai/ocr_annual_financials` into
`data/raw/ocr_annual_financials/`. It preserves the repository's Unicode
paths and uses Hugging Face's local-directory metadata to resume interrupted
downloads without creating a second full cache.

The source corpus contains OCR text with embedded HTML tables. Downloading it
does not include the unreleased ViFinQA questions or hidden competition
answers, and it does not create CSV or Parquet files. Table extraction is a
separate processing stage.

## WSL setup

Open the repository from WSL and install the locked environment:

```bash
cd /mnt/d/GitHub/financial-assistant
uv sync --frozen --extra dev
```

If the dataset is gated or private, authenticate without placing a token in
the command history:

```bash
uv run --frozen --no-sync hf auth login
```

## Inspect the download first

The safe default is a dry run. It resolves the current `main` branch to an
immutable commit and reports the exact pending size without transferring the
dataset:

```bash
bash scripts/download_dataset.sh
```

You can also invoke the Python entry point directly:

```bash
uv run --frozen --no-sync financial-report-qa download-data
```

## Download the full snapshot

After checking the dry-run output and free space, start or resume the full
download:

```bash
uv run --frozen --no-sync financial-report-qa download-data --download
```

The command leaves at least 20 GiB free by default. Increase the reserve when
the same disk will also hold normalized tables and indexes:

```bash
uv run --frozen --no-sync financial-report-qa download-data --reserve-gb 100 --download
```

To use a different destination under `data/`:

```bash
bash scripts/download_dataset.sh \
  --target data/raw/tinix-full \
  --manifest data/raw/tinix-full/download_manifest.json \
  --download
```

Re-running the same command performs another dry run, recognizes completed
files, and downloads only missing or changed content. Keep the generated
`.cache/huggingface/` directory inside the target if you want efficient
resume and update checks.

## Useful options

```text
--repo-id OWNER/DATASET     Hugging Face dataset repository
--revision REVISION         Branch, tag, or full commit hash (default: main)
--target PATH               Destination (default: data/raw/ocr_annual_financials)
--include GLOB              Download matching paths; repeat as needed
--exclude GLOB              Skip matching paths; repeat as needed
--workers N                 Concurrent file downloads (default: 8)
--reserve-gb N              Free space to preserve after download (default: 20)
--download                  Perform transfer; omit for dry run
```

Example for one ticker while validating the pipeline:

```bash
bash scripts/download_dataset.sh --include 'FPT/**' --download
```

Do not commit the contents of `data/raw/`. The directory is intentionally
ignored by Git.
