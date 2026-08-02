# Dataset Profile Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one Jupyter notebook that safely profiles the downloaded `ocr_result` corpus and turns its structure, coverage, and OCR-quality risks into actionable next steps.

**Architecture:** Keep all analysis and visualization code in `notebooks/01_dataset_profile.ipynb`. Tag the reusable helper cell so an automated test can load and execute the exact notebook code against a synthetic corpus; real-data cells call the same helpers. Scan metadata for every TXT file, but inspect file contents only through deterministic bounded sampling.

**Tech Stack:** Python 3.11, pathlib, Pandas, Matplotlib, JupyterLab, nbformat, pytest, uv.

## Global Constraints

- Default input is `data/raw/ocr_annual_financials/ocr_result` resolved from the repository root.
- Never modify, rename, or write files below `ocr_result`.
- Scan metadata for every TXT file; sample content deterministically with a configurable seed.
- One unreadable or malformed file must be recorded and must not stop the remaining scan.
- Keep all derived profile data in memory; no export is added in this task.
- The notebook must run on WSL with Python 3.11 and the locked `dev` environment.

---

## File Structure

- Create `notebooks/01_dataset_profile.ipynb`: configuration, helper functions, self-checks, inventory, charts, anomaly tables, content sample, and readiness recommendations.
- Create `tests/notebooks/test_dataset_profile_notebook.py`: loads the tagged helper cell from the notebook and verifies its behavior on a temporary `ocr_result` tree.
- Modify `pyproject.toml`: add explicit notebook, notebook-parsing, and plotting dependencies to the `dev` extra.
- Modify `uv.lock`: lock the added development dependencies.

### Task 1: Lock the notebook contract with failing tests

**Files:**
- Create: `tests/notebooks/test_dataset_profile_notebook.py`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**
- Consumes: notebook JSON at `notebooks/01_dataset_profile.ipynb`.
- Produces: a test loader for the code cell tagged `profile-helpers` and behavioral contracts for `parse_report_path`, `build_inventory`, `sample_paths`, and `inspect_text_file`.

- [ ] **Step 1: Write a test loader and path-parsing test**

```python
NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "01_dataset_profile.ipynb"


def load_helpers() -> dict[str, object]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = next(
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and "profile-helpers" in cell.metadata.get("tags", [])
    )
    namespace: dict[str, object] = {}
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_parse_report_path_extracts_hierarchy(tmp_path: Path) -> None:
    root = tmp_path / "ocr_result"
    path = (
        root / "HTG" / "2022" / "HTG_Baocaotaichinh_2022_Kiemtoan_Hopnhat" / "report_extracted.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")
    parsed = load_helpers()["parse_report_path"](path, root)
    assert parsed["ticker"] == "HTG"
    assert parsed["year"] == 2022
    assert parsed["scope"] == "consolidated"
    assert parsed["assurance"] == "audited"
    assert parsed["structure_status"] == "valid"
```

- [ ] **Step 2: Add inventory, deterministic sampling, and content-profile tests**

```python
def test_inventory_keeps_malformed_paths_as_anomalies(tmp_path: Path) -> None:
    root = tmp_path / "ocr_result"
    valid = root / "FPT" / "2023" / "FPT_Baocaotaichinh_2023" / "valid.txt"
    malformed = root / "loose.txt"
    valid.parent.mkdir(parents=True)
    valid.write_text("<table>100</table>", encoding="utf-8")
    malformed.write_text("noise", encoding="utf-8")
    frame = load_helpers()["build_inventory"](root)
    assert len(frame) == 2
    assert set(frame["structure_status"]) == {"valid", "malformed"}


def test_sample_paths_is_bounded_and_deterministic(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.txt" for index in range(10)]
    sample_paths = load_helpers()["sample_paths"]
    assert sample_paths(paths, sample_size=4, seed=42) == sample_paths(
        paths, sample_size=4, seed=42
    )
    assert len(sample_paths(paths, sample_size=40, seed=42)) == 10


def test_inspect_text_file_reports_utf8_and_table_markers(tmp_path: Path) -> None:
    path = tmp_path / "report.txt"
    path.write_text("Bảng tài chính\n<table><tr><td>100</td></tr></table>", encoding="utf-8")
    result = load_helpers()["inspect_text_file"](path, max_bytes=1_000_000)
    assert result["utf8_valid"] is True
    assert result["has_html_table"] is True
    assert result["line_count"] == 2
```

- [ ] **Step 3: Run the tests and verify RED**

Run: `uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py`

Expected: FAIL because `notebooks/01_dataset_profile.ipynb` does not exist.

### Task 2: Implement the notebook helpers and self-check

**Files:**
- Create: `notebooks/01_dataset_profile.ipynb`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**
- Consumes: a `Path` pointing to an `ocr_result` directory.
- Produces:
  - `parse_report_path(path: Path, root: Path) -> dict[str, object]`
  - `build_inventory(root: Path) -> pd.DataFrame`
  - `sample_paths(paths: Sequence[Path], sample_size: int, seed: int) -> list[Path]`
  - `inspect_text_file(path: Path, max_bytes: int) -> dict[str, object]`
  - `build_readiness_summary(inventory: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Add explicit development dependencies**

Add to `[project.optional-dependencies].dev`:

```toml
"jupyterlab>=4.4,<5",
"matplotlib>=3.10,<4",
"nbformat>=5.10,<6",
```

Run: `uv lock`

- [ ] **Step 2: Create notebook configuration and tagged helper cell**

The configuration cell defines:

```python
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ocr_annual_financials" / "ocr_result"
CONTENT_SAMPLE_SIZE = 500
RANDOM_SEED = 42
MAX_CONTENT_BYTES = 2_000_000
EXPECTED_YEAR_RANGE = range(2015, 2026)
```

The `profile-helpers` cell implements the interfaces above. `build_inventory` uses `root.rglob("*.txt")`, stable path sorting, `Path.stat`, and per-file exception capture. `inspect_text_file` reads at most `max_bytes` in binary mode, tries strict UTF-8, then decodes with replacement only for metrics.

- [ ] **Step 3: Add an in-notebook synthetic self-check cell**

Use `tempfile.TemporaryDirectory` to create one valid and one malformed path, assert both inventory statuses, and display `"Self-check passed"`. The temporary directory is outside the raw corpus and is automatically removed.

- [ ] **Step 4: Run the targeted tests and verify GREEN**

Run: `uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py`

Expected: all notebook contract tests PASS.

### Task 3: Add analysis views and readiness recommendations

**Files:**
- Modify: `notebooks/01_dataset_profile.ipynb`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**
- Consumes: `inventory` from `build_inventory(DATA_ROOT)` and sampled records from `inspect_text_file`.
- Produces: notebook displays only; no raw-data mutation and no file export.

- [ ] **Step 1: Add guarded real-data execution**

Validate `DATA_ROOT.exists()` and raise a `FileNotFoundError` that prints the configured path and downloader command. If no TXT files are found, raise a `RuntimeError` explaining the expected hierarchy.

- [ ] **Step 2: Add KPI and coverage views**

Display total TXT files, valid tickers, year range, total GiB, malformed paths, empty files, and median file size. Add Matplotlib charts for reports per year, top 20 tickers, report scope/assurance/period, and log-scaled file-size distribution.

- [ ] **Step 3: Add bounded ticker-year heatmap and anomaly tables**

Build a pivot table from the 40 tickers with the most reports and years 2015–2025. Display malformed paths, empty/tiny files below 1 KiB, unusually large files above the 99.5th percentile, and duplicated `(ticker, year, report_name)` groups.

- [ ] **Step 4: Add sampled content-quality analysis**

Sample at most `CONTENT_SAMPLE_SIZE` valid paths with `RANDOM_SEED`, profile each file with `MAX_CONTENT_BYTES`, and show UTF-8 validity, read failures, truncation, line-count distribution, HTML-table marker coverage, replacement-character ratio, and numeric-character ratio.

- [ ] **Step 5: Add actionable readiness table**

`build_readiness_summary` returns columns `priority`, `finding`, `evidence`, and `next_action`. It must cover path parsing, quarantine candidates, encoding handling, table detection strategy, and coverage-aware retrieval.

- [ ] **Step 6: Extend the structural test**

Assert the notebook contains headings for `Corpus overview`, `Coverage`, `Anomalies`, `Content quality sample`, and `Recommended next steps`, and assert no code cell writes beneath `DATA_ROOT`.

- [ ] **Step 7: Run all verification commands**

Run:

```bash
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check src tests
uv run --frozen --no-sync mypy src tests
uv run --frozen --no-sync python -c "import ast,json,pathlib; p=pathlib.Path('notebooks/01_dataset_profile.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); [ast.parse(''.join(c['source'])) for c in n['cells'] if c['cell_type']=='code' and not ''.join(c['source']).lstrip().startswith('%')]; print('notebook syntax: OK')"
```

Expected: all commands exit 0.

- [ ] **Step 8: Commit**

```bash
git add notebooks/01_dataset_profile.ipynb tests/notebooks/test_dataset_profile_notebook.py pyproject.toml uv.lock
git commit -m "feat: add OCR dataset profiling notebook"
```
