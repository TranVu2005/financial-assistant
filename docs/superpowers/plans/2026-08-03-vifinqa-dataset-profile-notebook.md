# ViFinQA Dataset Profile Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy OCR profiler with an executable notebook that profiles the downloaded ViFinQA reports, questions, and company map.

**Architecture:** Keep configuration, reusable helpers, synthetic checks, and presentation cells in `notebooks/01_dataset_profile.ipynb`. Tests execute the tagged helper cell directly against temporary ViFinQA fixtures; real-data cells call the same helpers and keep the three source inventories separate until explicit integrity checks.

**Tech Stack:** Python 3.11, Jupyter/nbformat, pandas, matplotlib, pytest, standard-library `json`, `pathlib`, `re`, and `codecs`.

## Global Constraints

- Read only `data/raw/code_stock.csv`, `data/raw/questions/questions.jsonl`, and `data/raw/financial_statements/**/*.txt`.
- Never create, modify, rename, or delete files below `data/raw`.
- Inspect metadata for all reports but content for only a deterministic bounded sample.
- Replace legacy `ocr_result` support; do not retain dual-layout branches.
- Preserve malformed rows and paths as quality records instead of silently discarding them.
- The executed notebook must report 1,973 reports, 1,012 questions, and 100 mapped companies.

---

## File structure

- `notebooks/01_dataset_profile.ipynb`: complete ViFinQA profiling workflow and saved outputs.
- `tests/notebooks/test_dataset_profile_notebook.py`: executable contract for the tagged notebook helpers and required analysis sections.
- `docs/superpowers/specs/2026-08-03-vifinqa-dataset-profile-design.md`: approved scope; read-only reference.

### Task 1: Define the ViFinQA helper contract with failing tests

**Files:**

- Modify: `tests/notebooks/test_dataset_profile_notebook.py`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**

- Consumes: the code cell tagged `profile-helpers` in `notebooks/01_dataset_profile.ipynb`.
- Produces: required functions `load_company_map(path: Path) -> pd.DataFrame`, `load_questions(path: Path) -> pd.DataFrame`, `parse_report_path(path: Path, root: Path) -> dict[str, object]`, `build_report_inventory(root: Path) -> pd.DataFrame`, `sample_paths(paths: Sequence[Path], sample_size: int, seed: int) -> list[Path]`, `inspect_text_file(path: Path, max_bytes: int) -> dict[str, object]`, and `extract_mentioned_tickers(question: str, known_tickers: Sequence[str]) -> tuple[str, ...]`.

- [ ] **Step 1: Replace legacy hierarchy tests with ViFinQA report-path tests**

```python
def test_parse_report_path_extracts_vifinqa_hierarchy(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = root / "AAA" / "2015" / "AAA_financial_statements_2015_consolidated" / "report.txt"
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")

    parse_report_path = cast(Callable[[Path, Path], dict[str, object]], _load_helpers()["parse_report_path"])
    parsed = parse_report_path(path, root)

    assert parsed["ticker"] == "AAA"
    assert parsed["year"] == 2015
    assert parsed["statement_type"] == "consolidated"
    assert parsed["structure_status"] == "valid"


def test_parse_report_path_keeps_unexpected_depth_as_anomaly(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    path = root / "AAA" / "2015" / "document" / "extra" / "report.txt"
    path.parent.mkdir(parents=True)
    path.write_text("sample", encoding="utf-8")

    parsed = _load_helpers()["parse_report_path"](path, root)
    assert parsed["structure_status"] == "malformed"
    assert "expected exactly" in cast(str, parsed["structure_issue"])
```

- [ ] **Step 2: Add company-map and JSONL validation tests**

```python
def test_load_company_map_normalizes_and_flags_rows(tmp_path: Path) -> None:
    source = tmp_path / "code_stock.csv"
    source.write_text("Mã CK,Tên công ty\naaa,Công ty AAA\n,Thiếu mã\n", encoding="utf-8")
    frame = _load_helpers()["load_company_map"](source)
    assert frame.loc[0, "ticker"] == "AAA"
    assert frame["is_valid"].tolist() == [True, False]


def test_load_questions_preserves_malformed_and_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        '{"id": 1, "question": "Doanh thu HPG năm 2022?"}\n'
        '{bad json}\n'
        '{"id": 2, "question": ""}\n',
        encoding="utf-8",
    )
    frame = _load_helpers()["load_questions"](source)
    assert frame["line_number"].tolist() == [1, 2, 3]
    assert frame["is_valid"].tolist() == [True, False, False]
    assert "JSON" in cast(str, frame.loc[1, "validation_issue"])
```

- [ ] **Step 3: Add inventory, ticker-extraction, bounded-read, and notebook-section tests**

```python
def test_extract_mentioned_tickers_uses_token_boundaries() -> None:
    extract = _load_helpers()["extract_mentioned_tickers"]
    assert extract("So sánh HPG và VCB năm 2022", ["HPG", "VCB", "HP"]) == ("HPG", "VCB")


def test_build_report_inventory_is_sorted_and_keeps_malformed_paths(tmp_path: Path) -> None:
    root = tmp_path / "financial_statements"
    valid = root / "VCB" / "2022" / "VCB_financial_statements_2022_separate" / "report.txt"
    malformed = root / "loose.txt"
    valid.parent.mkdir(parents=True)
    valid.write_text("<table>100</table>", encoding="utf-8")
    malformed.write_text("noise", encoding="utf-8")
    frame = _load_helpers()["build_report_inventory"](root)
    assert set(frame["structure_status"]) == {"valid", "malformed"}
    assert frame["relative_path"].tolist() == sorted(frame["relative_path"].tolist())
```

Retain the existing deterministic sampling and UTF-8 boundary tests, update the section assertion to require `Dataset overview`, `Report coverage`, `Question analysis`, `Integrity checks`, `Report content sample`, and `Readiness summary`, and retain the read-only and unique-cell-ID assertions.

- [ ] **Step 4: Run the focused tests and confirm the contract fails against the legacy notebook**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py
```

Expected: FAIL because `load_company_map`, `load_questions`, `extract_mentioned_tickers`, ViFinQA statement types, and the new section headings are not implemented.

- [ ] **Step 5: Commit the failing contract**

```powershell
git add tests/notebooks/test_dataset_profile_notebook.py
git commit -m "test: define ViFinQA notebook profile contract"
```

### Task 2: Replace notebook configuration and reusable helpers

**Files:**

- Modify: `notebooks/01_dataset_profile.ipynb`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**

- Consumes: paths below `PROJECT_ROOT / "data" / "raw"` and the Task 1 test contract.
- Produces: `COMPANY_MAP_PATH`, `QUESTIONS_PATH`, `STATEMENTS_ROOT`, `CONTENT_SAMPLE_SIZE`, `RANDOM_SEED`, `MAX_CONTENT_BYTES`, plus every helper signature named in Task 1.

- [ ] **Step 1: Replace title and configuration cells**

Use this configuration in the notebook:

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

candidate_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
PROJECT_ROOT = next(
    (path for path in candidate_roots if (path / "pyproject.toml").exists()),
    Path.cwd().resolve(),
)
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
COMPANY_MAP_PATH = DATA_ROOT / "code_stock.csv"
QUESTIONS_PATH = DATA_ROOT / "questions" / "questions.jsonl"
STATEMENTS_ROOT = DATA_ROOT / "financial_statements"
CONTENT_SAMPLE_SIZE = 250
RANDOM_SEED = 42
MAX_CONTENT_BYTES = 2_000_000
EXPECTED_YEAR_RANGE = range(2015, 2026)
```

The markdown must state that all three ViFinQA inputs are read-only and that report-content metrics are sample based.

- [ ] **Step 2: Implement CSV and JSONL loaders in the tagged helper cell**

```python
def load_company_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    required = {"Mã CK", "Tên công ty"}
    if not required <= set(frame.columns):
        raise ValueError(f"Company map must contain columns: {sorted(required)}")
    result = pd.DataFrame({
        "row_number": range(2, len(frame) + 2),
        "ticker": frame["Mã CK"].str.strip().str.upper(),
        "company_name": frame["Tên công ty"].str.strip(),
    })
    valid_ticker = result["ticker"].str.fullmatch(r"[A-Z0-9]{2,10}")
    result["is_valid"] = valid_ticker & result["company_name"].ne("")
    result["validation_issue"] = None
    result.loc[~valid_ticker, "validation_issue"] = "invalid or missing ticker"
    result.loc[valid_ticker & result["company_name"].eq(""), "validation_issue"] = "missing company name"
    return result


def load_questions(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            record = {"line_number": line_number, "id": None, "question": None,
                      "is_valid": False, "validation_issue": None}
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                record["validation_issue"] = f"invalid JSON: {error.msg}"
            else:
                record["id"] = payload.get("id") if isinstance(payload, dict) else None
                record["question"] = payload.get("question") if isinstance(payload, dict) else None
                issues = []
                if not isinstance(record["id"], int):
                    issues.append("id must be an integer")
                if not isinstance(record["question"], str) or not record["question"].strip():
                    issues.append("question must be a non-empty string")
                record["is_valid"] = not issues
                record["validation_issue"] = "; ".join(issues) or None
            records.append(record)
    return pd.DataFrame.from_records(records)
```

- [ ] **Step 3: Implement report inventory, question enrichment, and bounded inspection**

Implement the inventory and mention helpers with these rules:

```python
def parse_report_path(path: Path, root: Path) -> dict[str, object]:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = relative.parts
    ticker_raw = parts[0] if len(parts) >= 1 else ""
    year_raw = parts[1] if len(parts) >= 2 else ""
    document_name = parts[2] if len(parts) >= 3 else ""
    ticker = ticker_raw.upper() if re.fullmatch(r"[A-Za-z0-9]{2,10}", ticker_raw) else None
    year = int(year_raw) if re.fullmatch(r"\d{4}", year_raw) else None
    normalized_name = document_name.casefold()
    if "consolidated" in normalized_name:
        statement_type = "consolidated"
    elif "separate" in normalized_name:
        statement_type = "separate"
    elif "aggregated" in normalized_name:
        statement_type = "aggregated"
    else:
        statement_type = "other"
    issues = []
    if len(parts) != 4:
        issues.append("expected exactly ticker/year/document/file hierarchy")
    if ticker is None:
        issues.append("invalid ticker directory")
    if year is None or not 1900 <= year <= 2100:
        issues.append("invalid year directory")
    return {
        "path": path,
        "relative_path": relative.as_posix(),
        "ticker": ticker,
        "year": year,
        "year_in_expected_range": year in range(2015, 2026) if year else False,
        "document_name": document_name or None,
        "statement_type": statement_type,
        "structure_status": "valid" if not issues else "malformed",
        "structure_issue": "; ".join(issues) or None,
    }


def build_report_inventory(root: Path) -> pd.DataFrame:
    if not root.exists():
        raise FileNotFoundError(f"Statements root does not exist: {root}")
    records = []
    paths = sorted(root.rglob("*.txt"), key=lambda item: item.as_posix().casefold())
    for path in paths:
        record = parse_report_path(path, root)
        try:
            stat = path.stat()
            record.update(size_bytes=stat.st_size, is_empty=stat.st_size == 0,
                          modified_at=pd.Timestamp(stat.st_mtime, unit="s", tz="UTC"),
                          stat_error=None)
        except OSError as error:
            record.update(size_bytes=None, is_empty=False, modified_at=pd.NaT,
                          stat_error=str(error))
        records.append(record)
    return pd.DataFrame.from_records(records)


def extract_mentioned_tickers(question: str, known_tickers: Sequence[str]) -> tuple[str, ...]:
    matches = []
    for ticker in sorted({value.upper() for value in known_tickers}):
        if re.search(rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])", question.upper()):
            matches.append(ticker)
    return tuple(matches)
```

Retain `sample_paths` and the tested incremental UTF-8 decoder in `inspect_text_file`; the latter reads `max_bytes + 1`, truncates to the cap, calls the strict incremental decoder with `final=not truncated`, falls back to `errors="replace"`, and records `bytes_read`, `truncated`, `utf8_valid`, `read_error`, line/character/replacement/control counts, numeric ratio, and HTML/pipe/tabular markers.

- [ ] **Step 4: Add and run a synthetic self-check cell**

Create temporary valid and malformed report paths, a two-row company CSV, and valid/invalid question JSONL lines. Assert the expected types, statuses, deterministic sample, and HTML-table detection, then print `Synthetic self-check passed`.

- [ ] **Step 5: Run the helper tests**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py
```

Expected: helper behavior tests PASS; section tests may still FAIL until Task 3.

- [ ] **Step 6: Commit helper implementation**

```powershell
git add notebooks/01_dataset_profile.ipynb tests/notebooks/test_dataset_profile_notebook.py
git commit -m "feat: add ViFinQA profiling helpers"
```

### Task 3: Build ViFinQA analysis, visualizations, and integrity reporting

**Files:**

- Modify: `notebooks/01_dataset_profile.ipynb`
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**

- Consumes: Task 2 loaders and inventory helpers.
- Produces: notebook data frames `company_map`, `questions`, `report_inventory`, `valid_companies`, `valid_questions`, `valid_reports`, `question_profile`, `content_profile`, and `readiness`.

- [ ] **Step 1: Add required-input validation and build all inventories**

```python
required_paths = [COMPANY_MAP_PATH, QUESTIONS_PATH, STATEMENTS_ROOT]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    formatted = "\n".join(f"- {path}" for path in missing_paths)
    raise FileNotFoundError(f"Missing ViFinQA inputs:\n{formatted}")

company_map = load_company_map(COMPANY_MAP_PATH)
questions = load_questions(QUESTIONS_PATH)
report_inventory = build_report_inventory(STATEMENTS_ROOT)
valid_companies = company_map[company_map["is_valid"]].copy()
valid_questions = questions[questions["is_valid"]].copy()
valid_reports = report_inventory[report_inventory["structure_status"].eq("valid")].copy()
```

Raise a clear `RuntimeError` if any of the three valid collections is empty.

- [ ] **Step 2: Add `Dataset overview` KPIs and report visualizations**

Build the KPI frame and chart inputs as follows, then render four matplotlib axes and the heatmap:

```python
valid_years = valid_reports["year"].dropna().astype(int)
kpis = pd.DataFrame([
    ("Question rows", f"{len(questions):,}"),
    ("Valid questions", f"{len(valid_questions):,}"),
    ("Report files", f"{len(report_inventory):,}"),
    ("Mapped companies", f"{len(valid_companies):,}"),
    ("Observed report tickers", f"{valid_reports['ticker'].nunique():,}"),
    ("Observed years", f"{valid_years.min()}–{valid_years.max()}"),
    ("Report text size", f"{report_inventory['size_bytes'].fillna(0).sum() / 1024**2:,.1f} MiB"),
    ("Malformed report paths", f"{report_inventory['structure_status'].eq('malformed').sum():,}"),
    ("Empty reports", f"{report_inventory['is_empty'].fillna(False).sum():,}"),
], columns=["Metric", "Value"])

reports_by_year = valid_reports["year"].value_counts().sort_index()
top_tickers = valid_reports["ticker"].value_counts().head(20).sort_values()
statement_types = valid_reports["statement_type"].value_counts().sort_values()
positive_sizes_mib = report_inventory.loc[report_inventory["size_bytes"].gt(0), "size_bytes"] / 1024**2

coverage_tickers = valid_reports["ticker"].value_counts().head(40).index
coverage = valid_reports[valid_reports["ticker"].isin(coverage_tickers)].pivot_table(
    index="ticker", columns="year", values="relative_path", aggfunc="size", fill_value=0
).reindex(index=coverage_tickers, columns=list(EXPECTED_YEAR_RANGE), fill_value=0)
```

- [ ] **Step 3: Add `Question analysis` metrics and charts**

Derive the profile and anomaly frames with this code, then plot `word_count`, `mentioned_year_counts`, and `mentioned_ticker_counts`:

```python
known_tickers = valid_companies["ticker"].tolist()
question_profile = valid_questions.copy()
question_profile["character_count"] = question_profile["question"].str.len()
question_profile["word_count"] = question_profile["question"].str.split().str.len()
question_profile["mentioned_years"] = question_profile["question"].map(
    lambda value: tuple(sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", value))))
)
question_profile["mentioned_tickers"] = question_profile["question"].map(
    lambda value: extract_mentioned_tickers(value, known_tickers)
)
numeric_terms = ("bao nhiêu", "tỷ lệ", "chênh lệch", "tăng", "giảm", "trung bình", "tổng", "phần trăm")
question_profile["has_numeric_language"] = question_profile["question"].map(
    lambda value: bool(re.search(r"\d", value)) or any(term in value.casefold() for term in numeric_terms)
)

duplicate_ids = question_profile[question_profile.duplicated("id", keep=False)].sort_values("id")
duplicate_text = question_profile[
    question_profile.duplicated("question", keep=False)
].sort_values("question")
observed_ids = set(question_profile["id"].astype(int))
missing_ids = sorted(set(range(1, max(observed_ids) + 1)) - observed_ids) if observed_ids else []
mentioned_year_counts = question_profile["mentioned_years"].explode().dropna().value_counts().sort_index()
mentioned_ticker_counts = question_profile["mentioned_tickers"].explode().dropna().value_counts().head(20)
```

- [ ] **Step 4: Add `Integrity checks` and `Report content sample`**

Compute integrity sets and sample metrics exactly as follows:

```python
company_tickers = set(valid_companies["ticker"])
report_tickers = set(valid_reports["ticker"].dropna())
question_tickers = set(question_profile["mentioned_tickers"].explode().dropna())
report_tickers_not_mapped = sorted(report_tickers - company_tickers)
mapped_tickers_without_reports = sorted(company_tickers - report_tickers)
mentioned_tickers_without_reports = sorted(question_tickers - report_tickers)
questions_without_ticker = question_profile[question_profile["mentioned_tickers"].str.len().eq(0)]
invalid_questions = questions[~questions["is_valid"]]
malformed_reports = report_inventory[report_inventory["structure_status"].eq("malformed")]
duplicate_company_tickers = valid_companies[
    valid_companies.duplicated("ticker", keep=False)
].sort_values("ticker")

integrity_summary = pd.DataFrame([
    ("Report tickers absent from map", len(report_tickers_not_mapped)),
    ("Mapped tickers without reports", len(mapped_tickers_without_reports)),
    ("Mentioned tickers without reports", len(mentioned_tickers_without_reports)),
    ("Questions without recognized ticker", len(questions_without_ticker)),
    ("Invalid question rows", len(invalid_questions)),
    ("Malformed report paths", len(malformed_reports)),
    ("Duplicate company-map rows", len(duplicate_company_tickers)),
], columns=["Check", "Count"])

selected_paths = sample_paths(valid_reports["path"].tolist(), CONTENT_SAMPLE_SIZE, RANDOM_SEED)
content_profile = pd.DataFrame([inspect_text_file(path, MAX_CONTENT_BYTES) for path in selected_paths])
readable = content_profile[content_profile["read_error"].isna()]
content_kpis = pd.DataFrame([
    ("Sampled reports", len(content_profile)),
    ("Read errors", int(content_profile["read_error"].notna().sum())),
    ("Strict UTF-8", float(readable["utf8_valid"].eq(True).mean()) if len(readable) else None),
    ("Truncated at byte cap", float(content_profile["truncated"].mean()) if len(content_profile) else None),
    ("HTML table markers", float(content_profile["has_html_table"].mean()) if len(content_profile) else None),
    ("Any table marker", float(content_profile["has_tabular_markers"].mean()) if len(content_profile) else None),
], columns=["Metric", "Value"])
```

- [ ] **Step 5: Add `Readiness summary`**

Create the summary from live evidence:

```python
readiness = pd.DataFrame([
    ("P0", "question validation", f"{len(invalid_questions):,} invalid rows",
     "Quarantine invalid JSONL rows while retaining line-number provenance."),
    ("P0", "question IDs", f"{len(duplicate_ids):,} duplicate rows; {len(missing_ids):,} missing IDs",
     "Use validated question IDs as stable external identifiers."),
    ("P0", "report paths", f"{len(malformed_reports):,} malformed paths",
     "Keep tolerant parsing and record structure issues in the inventory."),
    ("P1", "ticker alignment",
     f"{len(report_tickers_not_mapped):,} unmapped report tickers; {len(mapped_tickers_without_reports):,} mapped without reports",
     "Resolve ticker mismatches before retrieval indexing."),
    ("P1", "encoding", f"{int(content_profile['utf8_valid'].eq(False).sum()):,} sampled non-UTF-8 reports",
     "Preserve raw bytes and record decoder fallbacks."),
    ("P1", "table handling", f"{content_profile['has_tabular_markers'].mean():.1%} sampled with table markers",
     "Retain inline tables and add structure-aware chunking."),
], columns=["priority", "finding", "evidence", "next_action"])
```

- [ ] **Step 6: Run the complete focused test file**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py
```

Expected: PASS.

- [ ] **Step 7: Commit the analysis cells**

```powershell
git add notebooks/01_dataset_profile.ipynb tests/notebooks/test_dataset_profile_notebook.py
git commit -m "feat: profile ViFinQA dataset"
```

### Task 4: Execute and verify the notebook on the downloaded release

**Files:**

- Modify: `notebooks/01_dataset_profile.ipynb` (saved execution outputs only)
- Test: `tests/notebooks/test_dataset_profile_notebook.py`

**Interfaces:**

- Consumes: the complete Task 3 notebook and local `data/raw` release.
- Produces: an executed notebook with no error outputs and saved release statistics.

- [ ] **Step 1: Validate notebook JSON and Python syntax before execution**

Run:

```powershell
uv run --frozen --no-sync python -c "import ast,json,pathlib; p=pathlib.Path('notebooks/01_dataset_profile.ipynb'); n=json.loads(p.read_text(encoding='utf-8')); [ast.parse(''.join(c['source'])) for c in n['cells'] if c['cell_type']=='code']; print('notebook syntax: OK')"
```

Expected: `notebook syntax: OK`.

- [ ] **Step 2: Execute the notebook in place from the repository root**

Run:

```powershell
uv run --frozen --no-sync jupyter nbconvert --to notebook --execute notebooks/01_dataset_profile.ipynb --inplace --ExecutePreprocessor.timeout=600
```

Expected: exit code 0 and no traceback or error output in the notebook.

- [ ] **Step 3: Verify saved release counts and absence of execution errors**

Run:

```powershell
uv run --frozen --no-sync python -c "import json,pathlib; n=json.loads(pathlib.Path('notebooks/01_dataset_profile.ipynb').read_text(encoding='utf-8')); outputs=[o for c in n['cells'] for o in c.get('outputs',[])]; assert not [o for o in outputs if o.get('output_type')=='error']; rendered=json.dumps(outputs,ensure_ascii=False); assert all(x in rendered for x in ['1,973','1,012','100']); print('executed counts: OK')"
```

Expected: `executed counts: OK`.

- [ ] **Step 4: Run all focused quality checks**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/notebooks/test_dataset_profile_notebook.py
uv run --frozen --no-sync ruff check tests/notebooks/test_dataset_profile_notebook.py
git diff --check
```

Expected: tests PASS, Ruff reports no violations, and `git diff --check` exits 0.

- [ ] **Step 5: Commit verified notebook outputs**

```powershell
git add notebooks/01_dataset_profile.ipynb
git commit -m "docs: save ViFinQA profile results"
```
