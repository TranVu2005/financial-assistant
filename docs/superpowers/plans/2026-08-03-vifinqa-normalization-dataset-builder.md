# ViFinQA Normalization and Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize ViFinQA financial tables conservatively and publish deterministic, provenance-preserving Parquet releases selected through an atomic current-release pointer.

**Architecture:** Pure field normalizers return canonical values plus stable issue codes; `normalization/service.py` applies them to immutable ingestion models and produces a fingerprinted `NormalizedDocument`. `data/dataset_builder.py` consumes the existing manifest and ingestion APIs, writes explicit Arrow schemas into an immutable fingerprinted release, verifies it, and atomically updates `current.json`.

**Tech Stack:** Python 3.11, Pydantic 2, `Decimal`, orjson, PyArrow/Parquet, pytest, Hypothesis, Ruff, mypy, uv.

## Global Constraints

- Follow the approved design in `docs/superpowers/specs/2026-08-03-vifinqa-normalization-dataset-builder-design.md`.
- Preserve every raw field and all source provenance exactly; normalization creates new frozen models and never mutates inputs.
- Unknown, conflicting, or ambiguous evidence must not be guessed; use canonical `None` plus a deterministic issue when the design requires one.
- Keep numeric values in display scale: raw `1.500` under `triệu VND` becomes `Decimal("1500")` plus `VND_million`.
- Do not use fuzzy matching, an LLM, machine locale, timestamps, absolute paths, or random identifiers in canonical output.
- The only versioned normalization source is immutable Python rule data exposed through `RULESET_VERSION`.
- Dataset consumers select releases only through atomically replaced `current.json`.
- Do not include `scripts/build_dataset.py` in business-module imports.
- Write failing tests before implementation and commit only the files named by each task.
- Preserve unrelated worktree changes in `plan.md`, the profiling notebook, notebook tests, and `.agents/`.

---

## File Structure

### New files

- `src/financial_report_qa/schemas/normalization.py`: immutable normalized-document and issue contracts.
- `src/financial_report_qa/normalization/_shared.py`: comparison keys, generic decisions, ruleset validation, and issue sorting.
- `src/financial_report_qa/normalization/companies.py`: company propagation and explicit ticker conflict detection.
- `src/financial_report_qa/normalization/periods.py`: annual, quarterly, and day-first date normalization.
- `src/financial_report_qa/normalization/statements.py`: controlled statement-title classification.
- `src/financial_report_qa/normalization/metrics.py`: controlled metric aliases and canonical row slugs.
- `src/financial_report_qa/normalization/numbers.py`: locale-independent Decimal parsing and numeric issue classification.
- `src/financial_report_qa/normalization/units.py`: unit evidence resolution, scale multipliers, and exact scale conversion.
- `src/financial_report_qa/normalization/service.py`: immutable orchestration and normalization fingerprinting.
- `src/financial_report_qa/data/dataset_builder.py`: manifest orchestration, Arrow flattening, release verification, publication, and command implementation.
- `scripts/build_dataset.py`: thin executable wrapper.
- `tests/unit/schemas/test_normalization.py`: normalization contract validation.
- `tests/unit/normalization/test_shared.py`: normalized keys and rule collision tests.
- `tests/unit/normalization/test_companies.py`: company tests.
- `tests/unit/normalization/test_periods.py`: period tests.
- `tests/unit/normalization/test_statements.py`: statement tests.
- `tests/unit/normalization/test_metrics.py`: metric tests.
- `tests/unit/normalization/test_numbers.py`: numeric parser examples and properties.
- `tests/unit/normalization/test_units.py`: unit resolution and scale properties.
- `tests/unit/normalization/test_service.py`: end-to-end immutable normalization tests.
- `tests/unit/data/test_dataset_builder.py`: flattening, schema, verification, and failure-injection tests.
- `tests/integration/test_build_dataset.py`: reproducible release and pointer integration test.
- `tests/integration/fixtures/normalization/VCB/2024/Consolidated/report.txt`: small UTF-8 source snapshot.

### Modified files

- `src/financial_report_qa/schemas/__init__.py`: export normalized contracts.
- `src/financial_report_qa/normalization/__init__.py`: export only the approved public normalization API.
- `src/financial_report_qa/data/manifests.py`: add strict deterministic manifest reading and source-byte fingerprinting.
- `src/financial_report_qa/data/__init__.py`: export the builder API.
- `src/financial_report_qa/core/errors.py`: add normalization and dataset-publication domain errors.
- `src/financial_report_qa/cli.py`: dispatch `build-dataset`.
- `tests/unit/data/test_manifests.py`: cover strict manifest reads.
- `tests/unit/test_cli.py`: cover build command dispatch.
- `README.md`: document the build command and immutable release layout.
- `docs/data-download.md`: document processed release selection through `current.json`.

---

### Task 1: Immutable Normalization Contracts and Shared Rule Primitives

**Files:**
- Create: `src/financial_report_qa/schemas/normalization.py`
- Create: `src/financial_report_qa/normalization/_shared.py`
- Modify: `src/financial_report_qa/schemas/__init__.py`
- Modify: `src/financial_report_qa/core/errors.py`
- Test: `tests/unit/schemas/test_normalization.py`
- Test: `tests/unit/normalization/test_shared.py`

**Interfaces:**
- Consumes: `DocumentRecord`, `ExtractionResult`, and strings from ingestion.
- Produces: `NormalizationIssueCode`, `NormalizationIssue`, `NormalizedDocument`, `RULESET_VERSION`, `Decision[T]`, `normalized_key()`, `validate_aliases()`, and `issue_sort_key()`.

- [ ] **Step 1: Write failing schema tests**

Create `tests/unit/schemas/test_normalization.py` with a minimal empty extraction and these assertions:

```python
from financial_report_qa.ingestion.provenance import ExtractionResult
from financial_report_qa.schemas.documents import DocumentRecord, stable_document_id
from financial_report_qa.schemas.normalization import NormalizationIssue, NormalizedDocument


def _document() -> DocumentRecord:
    digest = "a" * 64
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )


def test_normalized_document_is_frozen_and_requires_matching_document() -> None:
    document = _document()
    extraction = ExtractionResult(doc_id=document.doc_id, blocks=(), tables=(), rejected=())
    normalized = NormalizedDocument(
        document=document,
        extraction=extraction,
        issues=(),
        ruleset_version="2026.08.1",
        normalization_fingerprint="b" * 64,
    )

    assert normalized.document.doc_id == normalized.extraction.doc_id
    with pytest.raises(ValidationError, match="frozen"):
        normalized.ruleset_version = "changed"  # type: ignore[misc]


def test_normalized_document_rejects_mismatched_doc_id() -> None:
    document = _document()
    extraction = ExtractionResult(doc_id=f"doc_{'c' * 64}", blocks=(), tables=(), rejected=())

    with pytest.raises(ValidationError, match="document and extraction IDs must match"):
        NormalizedDocument(
            document=document,
            extraction=extraction,
            issues=(),
            ruleset_version="2026.08.1",
            normalization_fingerprint="b" * 64,
        )


def test_issue_rejects_unknown_fields_and_noncanonical_ids() -> None:
    with pytest.raises(ValidationError):
        NormalizationIssue(
            code="metric_unknown",
            doc_id="bad",
            table_id=None,
            cell_id=None,
            field="metric",
            raw_value="Doanh thu",
        )
```

- [ ] **Step 2: Run the schema tests and confirm the missing module failure**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_normalization.py
```

Expected: collection fails with `ModuleNotFoundError: financial_report_qa.schemas.normalization`.

- [ ] **Step 3: Implement the frozen contracts and domain errors**

Add to `src/financial_report_qa/core/errors.py`:

```python
class NormalizationError(FinancialReportQAError):
    """A normalization contract or ruleset is invalid."""


class DatasetBuildError(FinancialReportQAError):
    """A canonical dataset could not be built or verified."""


class DatasetPublicationError(DatasetBuildError):
    """A verified dataset release could not be safely published."""
```

Implement `schemas/normalization.py` with:

```python
NormalizationIssueCode = Literal[
    "company_conflict",
    "period_incomplete",
    "period_ambiguous",
    "period_invalid",
    "statement_conflict",
    "metric_unknown",
    "number_missing",
    "number_ambiguous",
    "number_invalid",
    "unit_unknown",
    "unit_conflict",
]
NormalizationField = Literal[
    "company", "period", "statement_type", "metric", "number", "unit"
]
ISSUE_FIELD_BY_CODE: dict[NormalizationIssueCode, NormalizationField] = {
    "company_conflict": "company",
    "period_incomplete": "period",
    "period_ambiguous": "period",
    "period_invalid": "period",
    "statement_conflict": "statement_type",
    "metric_unknown": "metric",
    "number_missing": "number",
    "number_ambiguous": "number",
    "number_invalid": "number",
    "unit_unknown": "unit",
    "unit_conflict": "unit",
}


class NormalizationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: NormalizationIssueCode
    doc_id: str = Field(pattern=r"^doc_[0-9a-f]{64}$")
    table_id: str | None = Field(default=None, pattern=r"^tbl_[0-9a-f]{64}$")
    cell_id: str | None = Field(default=None, pattern=r"^cell_[0-9a-f]{64}$")
    field: NormalizationField
    raw_value: str | None

    @model_validator(mode="after")
    def validate_code_field_pair(self) -> Self:
        expected = ISSUE_FIELD_BY_CODE[self.code]
        if self.field != expected:
            raise ValueError(f"issue code {self.code} requires field {expected}")
        return self


class NormalizedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: DocumentRecord
    extraction: ExtractionResult
    issues: tuple[NormalizationIssue, ...]
    ruleset_version: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    normalization_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_document_identity(self) -> Self:
        if self.document.doc_id != self.extraction.doc_id:
            raise ValueError("document and extraction IDs must match")
        return self
```

Export `NormalizationIssue` and `NormalizedDocument` from `schemas/__init__.py` without exporting internal helpers.

- [ ] **Step 4: Write failing shared-primitive tests**

Create `tests/unit/normalization/test_shared.py`:

```python
def test_normalized_key_is_unicode_aware_without_changing_source() -> None:
    raw = "  BÁO   CÁO\u00a0TÀI CHÍNH  "
    assert normalized_key(raw) == "báo cáo tài chính"
    assert raw == "  BÁO   CÁO\u00a0TÀI CHÍNH  "


def test_validate_aliases_rejects_conflicting_normalized_keys() -> None:
    with pytest.raises(NormalizationError, match="conflicting alias"):
        validate_aliases({"Báo cáo": "first", "  BÁO  CÁO ": "second"})


def test_issue_sort_key_orders_none_before_identifiers() -> None:
    def issue(table_id: str | None) -> NormalizationIssue:
        return NormalizationIssue(
            code="metric_unknown",
            doc_id=f"doc_{'a' * 64}",
            table_id=table_id,
            cell_id=None,
            field="metric",
            raw_value="unknown",
        )

    table_issue = issue(table_id=f"tbl_{'a' * 64}")
    document_issue = issue(table_id=None)
    assert sorted((table_issue, document_issue), key=issue_sort_key) == [
        document_issue,
        table_issue,
    ]
```

- [ ] **Step 5: Implement shared primitives**

Implement `_shared.py` with Python 3.11-compatible generics:

```python
T = TypeVar("T")
RULESET_VERSION = "2026.08.1"


@dataclass(frozen=True)
class Decision(Generic[T]):
    value: T | None
    issue_code: NormalizationIssueCode | None = None


def normalized_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def validate_aliases(aliases: Mapping[str, T]) -> dict[str, T]:
    validated: dict[str, T] = {}
    for raw, canonical in aliases.items():
        key = normalized_key(raw)
        if key in validated and validated[key] != canonical:
            raise NormalizationError(f"conflicting alias: {raw!r}")
        validated[key] = canonical
    return validated


def _none_first(value: str | None) -> tuple[bool, str]:
    return value is not None, value or ""


def issue_sort_key(issue: NormalizationIssue) -> tuple[object, ...]:
    return (
        issue.doc_id,
        _none_first(issue.table_id),
        _none_first(issue.cell_id),
        issue.field,
        issue.code,
        _none_first(issue.raw_value),
    )
```

- [ ] **Step 6: Run focused tests and static checks**

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/schemas/test_normalization.py tests/unit/normalization/test_shared.py
uv run --frozen --no-sync ruff check src/financial_report_qa/schemas/normalization.py src/financial_report_qa/normalization/_shared.py tests/unit/schemas/test_normalization.py tests/unit/normalization/test_shared.py
uv run --frozen --no-sync mypy src/financial_report_qa/schemas/normalization.py src/financial_report_qa/normalization/_shared.py tests/unit/schemas/test_normalization.py tests/unit/normalization/test_shared.py
```

Expected: all commands pass.

- [ ] **Step 7: Commit Task 1**

```powershell
git add src/financial_report_qa/core/errors.py src/financial_report_qa/schemas/__init__.py src/financial_report_qa/schemas/normalization.py src/financial_report_qa/normalization/_shared.py tests/unit/schemas/test_normalization.py tests/unit/normalization/test_shared.py
git commit -m "feat: add immutable normalization contracts"
```

---

### Task 2: Company, Period, and Statement Normalizers

**Files:**
- Create: `src/financial_report_qa/normalization/companies.py`
- Create: `src/financial_report_qa/normalization/periods.py`
- Create: `src/financial_report_qa/normalization/statements.py`
- Test: `tests/unit/normalization/test_companies.py`
- Test: `tests/unit/normalization/test_periods.py`
- Test: `tests/unit/normalization/test_statements.py`

**Interfaces:**
- Consumes: `DocumentRecord`, table title, raw column label, and report year.
- Produces: `normalize_company(document, title_raw) -> Decision[str]`, `normalize_period(raw, report_year) -> Decision[str]`, and `normalize_statement_type(title_raw) -> Decision[str]`.

- [ ] **Step 1: Write company examples first**

```python
def _document(company_code: str) -> DocumentRecord:
    digest = "a" * 64
    return DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path=f"{company_code}/2024/Consolidated/report.txt",
        company_code=company_code,
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )


@pytest.mark.parametrize(
    ("title", "expected_code"),
    [(None, "VCB"), ("BẢNG CÂN ĐỐI KẾ TOÁN", "VCB"), ("Mã CK: VCB", "VCB")],
)
def test_company_uses_inventory_code(title: str | None, expected_code: str) -> None:
    decision = normalize_company(_document("VCB"), title)
    assert decision == Decision(value=expected_code)


def test_company_reports_explicit_conflicting_ticker_without_overriding_document() -> None:
    assert normalize_company(_document("VCB"), "Mã chứng khoán: ACB") == Decision(
        value="VCB", issue_code="company_conflict"
    )
```

Run `uv run --frozen --no-sync pytest -q tests/unit/normalization/test_companies.py`; expect missing-module failure. Implement a controlled regex for labels `mã ck`, `mã chứng khoán`, `stock code`, and `ticker`, requiring an ASCII alphanumeric 2-10 character token. Never scan unlabeled uppercase words.

- [ ] **Step 2: Write exhaustive period examples**

```python
@pytest.mark.parametrize(
    ("raw", "report_year", "expected"),
    [
        ("2024", 2024, Decision(value="2024")),
        ("Năm 2023", 2024, Decision(value="2023")),
        ("Quý IV/2024", 2024, Decision(value="2024-Q4")),
        ("Q1 2022", 2024, Decision(value="2022-Q1")),
        ("Quý 2", 2024, Decision(value="2024-Q2")),
        ("31/12/2024", 2024, Decision(value="2024-12-31")),
        ("31-02-2024", 2024, Decision(value=None, issue_code="period_invalid")),
        ("12/11/24", 2024, Decision(value=None, issue_code="period_ambiguous")),
        ("Tháng 12", 2024, Decision(value=None, issue_code="period_incomplete")),
        ("Chỉ tiêu", 2024, Decision(value=None)),
    ],
)
def test_normalize_period(raw: str, report_year: int, expected: Decision[str]) -> None:
    assert normalize_period(raw, report_year) == expected
```

Implement anchored regular expressions, an explicit Roman-quarter map `{I: 1, II: 2, III: 3, IV: 4}`, and `datetime.date(year, month, day)` validation. Run the period file; expect all cases to pass.

- [ ] **Step 3: Write statement vocabulary tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bảng cân đối kế toán", "balance_sheet"),
        ("Báo cáo kết quả hoạt động kinh doanh", "income_statement"),
        ("Báo cáo lưu chuyển tiền tệ", "cash_flow_statement"),
        ("Báo cáo thay đổi vốn chủ sở hữu", "equity_changes"),
        ("Thuyết minh báo cáo tài chính", "notes"),
    ],
)
def test_statement_aliases(raw: str, expected: str) -> None:
    assert normalize_statement_type(raw) == Decision(value=expected)


def test_statement_conflict_is_not_guessed() -> None:
    assert normalize_statement_type(
        "Bảng cân đối kế toán / Báo cáo lưu chuyển tiền tệ"
    ) == Decision(value=None, issue_code="statement_conflict")
```

Use controlled normalized phrase containment with longest aliases checked first. Return `Decision(None)` for no family and `statement_conflict` when aliases from multiple families occur.

- [ ] **Step 4: Run Task 2 quality checks**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization/test_companies.py tests/unit/normalization/test_periods.py tests/unit/normalization/test_statements.py
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization/companies.py src/financial_report_qa/normalization/periods.py src/financial_report_qa/normalization/statements.py tests/unit/normalization
uv run --frozen --no-sync mypy src/financial_report_qa/normalization/companies.py src/financial_report_qa/normalization/periods.py src/financial_report_qa/normalization/statements.py tests/unit/normalization/test_companies.py tests/unit/normalization/test_periods.py tests/unit/normalization/test_statements.py
```

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/financial_report_qa/normalization/companies.py src/financial_report_qa/normalization/periods.py src/financial_report_qa/normalization/statements.py tests/unit/normalization/test_companies.py tests/unit/normalization/test_periods.py tests/unit/normalization/test_statements.py
git commit -m "feat: normalize company period and statement evidence"
```

---

### Task 3: Versioned Metric Alias Rules

**Files:**
- Create: `src/financial_report_qa/normalization/metrics.py`
- Test: `tests/unit/normalization/test_metrics.py`

**Interfaces:**
- Consumes: one `row_label_raw` string and the shared `RULESET_VERSION`.
- Produces: `METRIC_ALIASES` and `normalize_metric(raw) -> Decision[str]`.

- [ ] **Step 1: Write the alias and unknown-label tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Doanh thu bán hàng và cung cấp dịch vụ", "revenue"),
        ("Doanh thu thuần về bán hàng và cung cấp dịch vụ", "net_revenue"),
        ("Lợi nhuận kế toán trước thuế", "profit_before_tax"),
        ("Lợi nhuận sau thuế thu nhập doanh nghiệp", "profit_after_tax"),
        ("Tổng cộng tài sản", "total_assets"),
        ("Nợ phải trả", "total_liabilities"),
        ("Vốn chủ sở hữu", "equity"),
        ("Tiền và các khoản tương đương tiền", "cash_and_cash_equivalents"),
        ("Lưu chuyển tiền thuần từ hoạt động kinh doanh", "operating_cash_flow"),
    ],
)
def test_metric_aliases(raw: str, expected: str) -> None:
    assert normalize_metric(raw) == Decision(value=expected)


def test_metric_matching_collapses_unicode_and_whitespace_only() -> None:
    assert normalize_metric("  TỔNG   CỘNG TÀI SẢN ") == Decision(value="total_assets")


def test_unknown_metric_is_auditable() -> None:
    assert normalize_metric("Chỉ tiêu chưa ánh xạ") == Decision(
        value=None, issue_code="metric_unknown"
    )
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run `uv run --frozen --no-sync pytest -q tests/unit/normalization/test_metrics.py`; expect missing-module failure.

- [ ] **Step 3: Implement the exact versioned map**

Import `RULESET_VERSION` from `_shared.py` and build `METRIC_ALIASES` through `validate_aliases()` at import time. Include the nine approved canonical metrics and only explicit Vietnamese/English synonyms written in the constant. `normalize_metric()` performs one normalized-key lookup; it does not use substring or edit-distance matching.

```python
METRIC_ALIASES = validate_aliases(
    {
        "Doanh thu bán hàng và cung cấp dịch vụ": "revenue",
        "Revenue": "revenue",
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ": "net_revenue",
        "Net revenue": "net_revenue",
        "Lợi nhuận kế toán trước thuế": "profit_before_tax",
        "Profit before tax": "profit_before_tax",
        "Lợi nhuận sau thuế thu nhập doanh nghiệp": "profit_after_tax",
        "Profit after tax": "profit_after_tax",
        "Tổng cộng tài sản": "total_assets",
        "Total assets": "total_assets",
        "Nợ phải trả": "total_liabilities",
        "Total liabilities": "total_liabilities",
        "Vốn chủ sở hữu": "equity",
        "Owners' equity": "equity",
        "Tiền và các khoản tương đương tiền": "cash_and_cash_equivalents",
        "Cash and cash equivalents": "cash_and_cash_equivalents",
        "Lưu chuyển tiền thuần từ hoạt động kinh doanh": "operating_cash_flow",
        "Net cash flows from operating activities": "operating_cash_flow",
    }
)
```

- [ ] **Step 4: Run tests, lint, and type checking**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization/test_metrics.py tests/unit/normalization/test_shared.py
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization/metrics.py tests/unit/normalization/test_metrics.py
uv run --frozen --no-sync mypy src/financial_report_qa/normalization/metrics.py tests/unit/normalization/test_metrics.py
```

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/financial_report_qa/normalization/metrics.py tests/unit/normalization/test_metrics.py
git commit -m "feat: add versioned financial metric aliases"
```

---

### Task 4: Locale-Independent Financial Number Parsing

**Files:**
- Create: `src/financial_report_qa/normalization/numbers.py`
- Test: `tests/unit/normalization/test_numbers.py`

**Interfaces:**
- Consumes: one exact `value_raw` string.
- Produces: frozen `NumberDecision(value: Decimal | None, unit_hint: Literal["percent"] | None, issue_code: NormalizationIssueCode | None)` and `parse_number(raw) -> NumberDecision`.

- [ ] **Step 1: Write table-driven number examples**

```python
@pytest.mark.parametrize(
    ("raw", "value", "unit_hint", "issue"),
    [
        ("1.500", Decimal("1500"), None, None),
        ("1,500", Decimal("1500"), None, None),
        ("1 500 000", Decimal("1500000"), None, None),
        ("1.500,25", Decimal("1500.25"), None, None),
        ("1,500.25", Decimal("1500.25"), None, None),
        ("(1.500)", Decimal("-1500"), None, None),
        ("+12,5", Decimal("12.5"), None, None),
        ("12,5%", Decimal("12.5"), "percent", None),
        ("-", None, None, "number_missing"),
        ("N/A", None, None, "number_missing"),
        ("1.50.0", None, None, "number_invalid"),
        ("(100", None, None, "number_invalid"),
        ("1,23,456", None, None, "number_ambiguous"),
    ],
)
def test_parse_number_examples(
    raw: str,
    value: Decimal | None,
    unit_hint: str | None,
    issue: str | None,
) -> None:
    assert parse_number(raw) == NumberDecision(
        value=value, unit_hint=unit_hint, issue_code=issue
    )
```

- [ ] **Step 2: Add raw-preservation and bounded round-trip properties**

```python
@given(
    value=st.decimals(
        min_value=Decimal("-1000000000000000000"),
        max_value=Decimal("1000000000000000000"),
        allow_nan=False,
        allow_infinity=False,
        places=2,
    ),
)
def test_controlled_decimal_rendering_round_trips(value: Decimal) -> None:
    raw = format(value, "f")
    assert parse_number(raw).value == value


@given(raw=st.text(max_size=40))
def test_parse_number_never_mutates_input(raw: str) -> None:
    before = raw
    parse_number(raw)
    assert raw == before
```

Use a separate renderer helper in the test if the signed fractional expression obscures intent; assert exact `Decimal` equality, never float proximity.

- [ ] **Step 3: Implement parsing as explicit validation stages**

Implement in this order:

1. preserve `raw`, normalize only Unicode space variants in a local working string;
2. classify controlled missing markers after NFKC/case-folding;
3. peel one trailing `%` into `unit_hint`;
4. validate one optional sign or one complete accounting-parentheses pair;
5. reject all characters except ASCII digits, `.`, `,`, and approved spaces;
6. validate/remove space grouping;
7. resolve punctuation using pure `_resolve_separators()`;
8. construct `Decimal` from an ASCII canonical token and apply the sign.

Use these separator rules exactly:

```python
def _resolve_single_separator(integer: str, separator: str) -> str | None:
    groups = integer.split(separator)
    if len(groups) > 2:
        return "".join(groups) if _valid_grouping(groups) else None
    left, right = groups
    if len(right) == 3 and left.isdigit() and 1 <= len(left) <= 3:
        return left + right
    if len(right) in {1, 2} and left.isdigit() and right.isdigit():
        return f"{left}.{right}"
    return None
```

For mixed `.` and `,`, treat the rightmost mark as decimal only with a one- or two-digit suffix and require valid three-digit grouping by the other mark in the prefix. Classify structurally plausible but non-unique grouping as `number_ambiguous`; classify illegal characters, signs, or parentheses as `number_invalid`.

- [ ] **Step 4: Run focused and property tests**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization/test_numbers.py
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization/numbers.py tests/unit/normalization/test_numbers.py
uv run --frozen --no-sync mypy src/financial_report_qa/normalization/numbers.py tests/unit/normalization/test_numbers.py
```

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/financial_report_qa/normalization/numbers.py tests/unit/normalization/test_numbers.py
git commit -m "feat: parse financial numbers conservatively"
```

---

### Task 5: Unit Resolution and Exact Scale Conversion

**Files:**
- Create: `src/financial_report_qa/normalization/units.py`
- Test: `tests/unit/normalization/test_units.py`

**Interfaces:**
- Consumes: optional cell unit hint, raw column label, and raw table unit.
- Produces: `CanonicalUnit`, `normalize_unit()`, `resolve_unit()`, `unit_multiplier()`, `economic_value()`, and `convert_scale()`.

- [ ] **Step 1: Write unit aliases, precedence, and conflict tests**

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Đơn vị tính: VND", "VND"),
        ("ĐVT: nghìn đồng", "VND_thousand"),
        ("Triệu VND", "VND_million"),
        ("tỷ đồng", "VND_billion"),
        ("%", "percent"),
        ("lần", "ratio"),
    ],
)
def test_normalize_unit_aliases(raw: str, expected: CanonicalUnit) -> None:
    assert normalize_unit(raw) == Decision(value=expected)


def test_resolve_unit_prefers_more_specific_agreeing_evidence() -> None:
    assert resolve_unit(
        cell_hint="percent", column_raw="Tỷ lệ (%)", table_raw=None
    ) == Decision(value="percent")


def test_resolve_unit_rejects_conflicting_evidence() -> None:
    assert resolve_unit(
        cell_hint=None, column_raw="ĐVT: triệu đồng", table_raw="ĐVT: tỷ đồng"
    ) == Decision(value=None, issue_code="unit_conflict")
```

- [ ] **Step 2: Write the economic-value property**

```python
VND_UNITS = st.sampled_from(["VND", "VND_thousand", "VND_million", "VND_billion"])


@given(
    coefficient=st.integers(min_value=-(10**18), max_value=10**18),
    source=VND_UNITS,
    target=VND_UNITS,
)
def test_scale_conversion_preserves_economic_value(
    coefficient: int, source: CanonicalUnit, target: CanonicalUnit
) -> None:
    value = Decimal(coefficient)
    converted = convert_scale(value, source=source, target=target)
    assert economic_value(converted, target) == economic_value(value, source)
```

Add examples proving `economic_value(Decimal("1500"), "VND_million") == Decimal("1500000000")` and rejecting conversion between monetary units and `percent`/`ratio`.

- [ ] **Step 3: Implement controlled aliases and exact Decimal multipliers**

```python
CanonicalUnit = Literal[
    "VND", "VND_thousand", "VND_million", "VND_billion", "percent", "ratio"
]
_MULTIPLIERS: dict[CanonicalUnit, Decimal] = {
    "VND": Decimal(1),
    "VND_thousand": Decimal(1000),
    "VND_million": Decimal(1000000),
    "VND_billion": Decimal(1000000000),
    "percent": Decimal("0.01"),
    "ratio": Decimal(1),
}
```

Strip only controlled prefixes (`đơn vị tính:`, `đơn vị:`, `đvt:`) from the comparison key. Detect unit aliases as complete controlled phrases or explicit parenthesized suffixes; arbitrary substrings do not count. `resolve_unit()` gathers non-`None` evidence in cell, column, table order, returns the one unique canonical unit, and returns `unit_conflict` when the set has more than one member.

- [ ] **Step 4: Run Task 5 checks**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization/test_units.py
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization/units.py tests/unit/normalization/test_units.py
uv run --frozen --no-sync mypy src/financial_report_qa/normalization/units.py tests/unit/normalization/test_units.py
```

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/financial_report_qa/normalization/units.py tests/unit/normalization/test_units.py
git commit -m "feat: normalize units and preserve economic scale"
```

---

### Task 6: Normalization Service and Deterministic Fingerprint

**Files:**
- Create: `src/financial_report_qa/normalization/service.py`
- Modify: `src/financial_report_qa/normalization/__init__.py`
- Test: `tests/unit/normalization/test_service.py`

**Interfaces:**
- Consumes: `normalize_extraction(document: DocumentRecord, result: ExtractionResult)` and all Task 2-5 decisions.
- Produces: a new `NormalizedDocument`; public package exports `RULESET_VERSION`, `normalize_extraction`, `economic_value`, and `convert_scale`.

- [ ] **Step 1: Build a representative extracted-table fixture**

In `test_service.py`, create one immutable `ExtractedTable` with:

- title `Báo cáo kết quả hoạt động kinh doanh`;
- table unit `Đơn vị tính: triệu đồng`;
- period columns `2023` and `2024`;
- one `Doanh thu thuần về bán hàng và cung cấp dịch vụ` row;
- values `(1.500)` and `2.000`;
- exact source lines, cell IDs, placements, and extraction evidence.

Use existing `stable_table_id()` and `stable_cell_id()` rather than literal fake IDs.

Implement the fixture with the exact construction below; keep the helper local to the
test module:

```python
def _extraction_fixture() -> tuple[DocumentRecord, ExtractionResult]:
    digest = "a" * 64
    document = DocumentRecord(
        doc_id=stable_document_id(digest),
        repo_id="org/vifinqa",
        revision="rev-1",
        relative_path="VCB/2024/Consolidated/report.txt",
        company_code="VCB",
        report_year=2024,
        statement_scope="consolidated",
        sha256=digest,
        file_size_bytes=1,
        encoding="utf-8",
        inventory_status="ready",
    )
    table_id = stable_table_id(document.doc_id, 3, 6)
    title = "Báo cáo kết quả hoạt động kinh doanh"
    metric = "Doanh thu thuần về bán hàng và cung cấp dịch vụ"
    source_cells = (
        (0, 0, "Chỉ tiêu", None, "Chỉ tiêu", 4),
        (0, 1, "2023", None, "2023", 4),
        (0, 2, "2024", None, "2024", 4),
        (1, 0, metric, metric, "Chỉ tiêu", 5),
        (1, 1, "(1.500)", metric, "2023", 5),
        (1, 2, "2.000", metric, "2024", 5),
    )
    cells = tuple(
        CellRecord(
            cell_id=stable_cell_id(table_id, row_idx, col_idx),
            table_id=table_id,
            row_idx=row_idx,
            col_idx=col_idx,
            row_label_raw=row_label,
            row_label_canonical=None,
            column_label_raw=column_label,
            column_label_canonical=None,
            value_raw=value,
            value_numeric=None,
            period=None,
            unit=None,
            source_line_start=line,
            source_line_end=line,
            extraction_confidence=1.0,
        )
        for row_idx, col_idx, value, row_label, column_label, line in source_cells
    )
    table = ExtractedTable(
        table=TableRecord(
            table_id=table_id,
            doc_id=document.doc_id,
            title_raw=title,
            statement_type=None,
            unit_raw="Đơn vị tính: triệu đồng",
            unit_normalized=None,
            line_start=3,
            line_end=6,
            row_count=2,
            column_count=3,
            quality_score=1.0,
            csv_path=None,
        ),
        cells=cells,
        placements=tuple(
            CellPlacement(row_idx=cell.row_idx, col_idx=cell.col_idx, cell_id=cell.cell_id)
            for cell in cells
        ),
        evidence=("html_table_marker",),
    )
    return document, ExtractionResult(
        doc_id=document.doc_id,
        blocks=(),
        tables=(table,),
        rejected=(),
    )
```

- [ ] **Step 2: Write the service behavior test**

```python
def test_normalize_extraction_populates_canonical_fields_and_preserves_raw() -> None:
    document, result = _extraction_fixture()

    normalized = normalize_extraction(document, result)

    table = normalized.extraction.tables[0]
    assert table.table.statement_type == "income_statement"
    assert table.table.unit_raw == "Đơn vị tính: triệu đồng"
    assert table.table.unit_normalized == "VND_million"
    values = {cell.value_raw: cell for cell in table.cells}
    assert values["(1.500)"].value_numeric == Decimal("-1500")
    assert values["(1.500)"].unit == "VND_million"
    assert values["(1.500)"].period == "2023"
    assert values["(1.500)"].row_label_canonical == "net_revenue"
    assert values["(1.500)"].row_label_raw == (
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ"
    )
    assert normalized.extraction.blocks == result.blocks
    assert normalized.extraction.rejected == result.rejected
    assert normalized.normalization_fingerprint == normalize_extraction(
        document, result
    ).normalization_fingerprint
```

Also test mismatched document IDs, unknown metric issue de-duplication to one issue per logical row, invalid number issues per source cell, unit conflicts, deterministic issue ordering, equal inputs, and frozen input equality after the call.

- [ ] **Step 3: Implement service grouping and immutable copies**

For each table:

1. derive company, statement, and table-unit decisions;
2. group cells by `row_idx` and normalize each distinct raw row label once;
3. cache period decisions by raw column label;
4. classify a value cell as a normalization candidate only when
   `row_label_raw is not None` and `value_raw != row_label_raw`; preserve header and row
   label source cells without number/unit issues;
5. parse each candidate source cell value;
6. resolve the cell unit from parser hint, column raw, and table raw;
7. collect the explicit canonical fields in `canonical_fields` and create a new
   `CellRecord` with `model_copy(update=canonical_fields)`;
8. create a new `TableRecord`, then `ExtractedTable`, preserving placements/evidence;
9. emit one metric issue for the lowest `(col_idx, cell_id)` cell in an unknown logical row;
10. sort all issues with `issue_sort_key()`.

Create issues through one helper so field/code pairings cannot drift:

```python
def _issue(
    *,
    code: NormalizationIssueCode,
    field: NormalizationField,
    document: DocumentRecord,
    table_id: str | None,
    cell_id: str | None,
    raw_value: str | None,
) -> NormalizationIssue:
    return NormalizationIssue(
        code=code,
        field=field,
        doc_id=document.doc_id,
        table_id=table_id,
        cell_id=cell_id,
        raw_value=raw_value,
    )
```

- [ ] **Step 4: Implement canonical fingerprinting**

Build the fingerprint payload from `document.doc_id`, the normalized `ExtractionResult`, sorted issues, and `RULESET_VERSION`. Serialize Pydantic values with `model_dump(mode="json")` and `orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)`; hash with SHA-256. Do not include the fingerprint field itself.

```python
payload = {
    "doc_id": document.doc_id,
    "extraction": normalized_extraction.model_dump(mode="json"),
    "issues": [issue.model_dump(mode="json") for issue in issues],
    "ruleset_version": RULESET_VERSION,
}
fingerprint = hashlib.sha256(
    orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
).hexdigest()
```

- [ ] **Step 5: Export the narrow public API and run checks**

`normalization/__init__.py` must export only:

```python
__all__ = (
    "RULESET_VERSION",
    "convert_scale",
    "economic_value",
    "normalize_extraction",
)
```

Run:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization tests/unit/schemas/test_normalization.py
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization tests/unit/normalization
uv run --frozen --no-sync mypy src/financial_report_qa/normalization tests/unit/normalization
```

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/financial_report_qa/normalization/__init__.py src/financial_report_qa/normalization/service.py tests/unit/normalization/test_service.py
git commit -m "feat: normalize extracted tables without losing provenance"
```

---

### Task 7: Strict Manifest Reader and Explicit Arrow Serialization

**Files:**
- Modify: `src/financial_report_qa/data/manifests.py`
- Create: `src/financial_report_qa/data/dataset_builder.py`
- Modify: `tests/unit/data/test_manifests.py`
- Create: `tests/unit/data/test_dataset_builder.py`

**Interfaces:**
- Consumes: the Day 2 JSONL manifest and `tuple[NormalizedDocument, ...]`.
- Produces: `ManifestSnapshot`, `read_manifest()`, `DatasetBuildConfig`, `DatasetBuildResult`, explicit Arrow schemas, flattened rows, and verified payload artifacts.

- [ ] **Step 1: Add strict manifest-read tests**

Add tests proving:

```python
def test_read_manifest_returns_models_and_exact_byte_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    result = InventoryResult(
        documents=(_document("VCB/2024/Consolidated/a.txt", "a" * 64),),
        issues=(),
    )
    write_manifest(result, path)

    snapshot = read_manifest(path)

    assert snapshot.inventory == result
    assert snapshot.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "line",
    [
        "{}\n",
        '{"record_type":"unknown"}\n',
        '{"record_type":"document","unexpected":true}\n',
        "not-json\n",
    ],
)
def test_read_manifest_rejects_invalid_rows_with_safe_line_number(
    tmp_path: Path, line: str
) -> None:
    path = tmp_path / "documents.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(DatasetBuildError, match="manifest line 1"):
        read_manifest(path)
```

Implement `ManifestSnapshot(inventory: InventoryResult, sha256: Sha256Digest)` as a frozen Pydantic model. Read bytes once, hash exact bytes, require UTF-8, require a final newline for a non-empty manifest, validate every `record_type`, reject duplicate `relative_path`, reject more than one `ready` record for the same `doc_id`, and preserve file order inside `InventoryResult`. Multiple records may share a content-addressed `doc_id` when all but one have `inventory_status="duplicate"`.

- [ ] **Step 2: Write explicit schema and flattening tests**

Assert exact schemas rather than inferred types:

```python
def test_cell_schema_uses_fixed_decimal_and_no_absolute_paths() -> None:
    assert CELL_SCHEMA.field("value_numeric").type == pa.decimal128(38, 10)
    assert CELL_SCHEMA.field("source_line_start").type == pa.int32()
    assert "absolute_path" not in CELL_SCHEMA.names


def test_flattened_rows_have_stable_order() -> None:
    def normalized_document(suffix: Literal["a", "b"]) -> NormalizedDocument:
        digest = suffix * 64
        document = DocumentRecord(
            doc_id=stable_document_id(digest),
            repo_id="org/vifinqa",
            revision="rev-1",
            relative_path=f"VCB/2024/Consolidated/{suffix}.txt",
            company_code="VCB",
            report_year=2024,
            statement_scope="consolidated",
            sha256=digest,
            file_size_bytes=1,
            encoding="utf-8",
            inventory_status="ready",
        )
        extraction = ExtractionResult(
            doc_id=document.doc_id, blocks=(), tables=(), rejected=()
        )
        return NormalizedDocument(
            document=document,
            extraction=extraction,
            issues=(),
            ruleset_version=RULESET_VERSION,
            normalization_fingerprint=digest,
        )

    rows = flatten_normalized_documents(
        (normalized_document("b"), normalized_document("a"))
    )
    assert [row["relative_path"] for row in rows.documents] == sorted(
        row["relative_path"] for row in rows.documents
    )
    assert rows.cells == tuple(
        sorted(rows.cells, key=lambda row: (
            row["table_id"], row["row_idx"], row["col_idx"], row["cell_id"]
        ))
    )
```

- [ ] **Step 3: Define builder models and Arrow schemas**

In `dataset_builder.py`, define frozen Pydantic `DatasetBuildConfig` with `snapshot_root`, `manifest_path`, `processed_root`, and non-empty `schema_version`; define `DatasetBuildResult` with release path, dataset/source fingerprints, row counts, and issue counts.

Declare complete schemas in source field order:

```python
CELL_SCHEMA = pa.schema(
    [
        ("cell_id", pa.string()),
        ("table_id", pa.string()),
        ("row_idx", pa.int32()),
        ("col_idx", pa.int32()),
        ("row_label_raw", pa.string()),
        ("row_label_canonical", pa.string()),
        ("column_label_raw", pa.string()),
        ("column_label_canonical", pa.string()),
        pa.field("value_raw", pa.string(), nullable=False),
        ("value_numeric", pa.decimal128(38, 10)),
        ("period", pa.string()),
        ("unit", pa.string()),
        pa.field("source_line_start", pa.int32(), nullable=False),
        pa.field("source_line_end", pa.int32(), nullable=False),
        pa.field("extraction_confidence", pa.float64(), nullable=False),
    ]
)
```

Declare equally explicit document, table, and issue schemas containing every corresponding Pydantic field. Convert tuple `notes` to Arrow list values. Quantize no Decimal silently: reject values that cannot be represented as decimal128(38, 10) with `DatasetBuildError` containing `cell_id`.

- [ ] **Step 4: Implement stable flattening and payload writing**

Create `FlattenedDataset` as a frozen dataclass of four row tuples. Sort exactly as the design specifies. Write:

- `documents.parquet`;
- `tables.parquet`;
- `cells.parquet`;
- `normalization_issues.parquet`;
- canonical `quality-summary.json`.

Use `pq.write_table()` with `pa.Table.from_pylist(rows, schema=declared_schema)`. For zero rows use `pa.Table.from_pylist([], schema=declared_schema)` so schema remains present. Write JSON as UTF-8 with sorted keys and a final newline.

- [ ] **Step 5: Implement payload verification and hashing**

Re-open each Parquet file, compare its schema and row count with the expected declaration, and compute SHA-256 by streaming bytes. Parse the quality summary back as JSON and compare it with its expected object. Return payload hashes sorted by artifact name. Never include `dataset-metadata.json` or `current.json` in payload hashes.

- [ ] **Step 6: Run Task 7 checks**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/data/test_manifests.py tests/unit/data/test_dataset_builder.py
uv run --frozen --no-sync ruff check src/financial_report_qa/data/manifests.py src/financial_report_qa/data/dataset_builder.py tests/unit/data/test_manifests.py tests/unit/data/test_dataset_builder.py
uv run --frozen --no-sync mypy src/financial_report_qa/data/manifests.py src/financial_report_qa/data/dataset_builder.py tests/unit/data/test_manifests.py tests/unit/data/test_dataset_builder.py
```

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/financial_report_qa/data/manifests.py src/financial_report_qa/data/dataset_builder.py tests/unit/data/test_manifests.py tests/unit/data/test_dataset_builder.py
git commit -m "feat: serialize canonical dataset artifacts"
```

---

### Task 8: Reproducible Builder and Atomic Release Pointer

**Files:**
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Modify: `src/financial_report_qa/data/__init__.py`
- Modify: `tests/unit/data/test_dataset_builder.py`
- Create: `tests/integration/test_build_dataset.py`
- Create: `tests/integration/fixtures/normalization/VCB/2024/Consolidated/report.txt`

**Interfaces:**
- Consumes: `DatasetBuildConfig`, `read_manifest()`, `extract_document()`, and `normalize_extraction()`.
- Produces: `build_dataset(config) -> DatasetBuildResult`, immutable `releases/<fingerprint>/`, and atomic `current.json`.

- [ ] **Step 1: Add the committed synthetic snapshot**

The UTF-8 fixture must contain one table that exercises title, table unit, period, metric, negative parentheses, and raw Unicode:

```html
Báo cáo kết quả hoạt động kinh doanh
Đơn vị tính: triệu đồng
<table>
<tr><th>Chỉ tiêu</th><th>2023</th><th>2024</th></tr>
<tr><td>Doanh thu thuần về bán hàng và cung cấp dịch vụ</td><td>(1.500)</td><td>2.000</td></tr>
</table>
```

- [ ] **Step 2: Write reproducibility integration test**

```python
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "normalization"


def test_build_dataset_is_reproducible_and_current_points_to_complete_release(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshot"
    shutil.copytree(FIXTURE_ROOT, snapshot_root)
    manifest_path = tmp_path / "documents.jsonl"
    write_manifest(
        build_inventory(snapshot_root, repo_id="org/vifinqa", revision="fixture-v1"),
        manifest_path,
    )
    first_root = tmp_path / "processed-a"
    second_root = tmp_path / "processed-b"

    first = build_dataset(
        DatasetBuildConfig(
            snapshot_root=snapshot_root,
            manifest_path=manifest_path,
            processed_root=first_root,
            schema_version="1",
        )
    )
    second = build_dataset(
        DatasetBuildConfig(
            snapshot_root=snapshot_root,
            manifest_path=manifest_path,
            processed_root=second_root,
            schema_version="1",
        )
    )

    assert first.dataset_fingerprint == second.dataset_fingerprint
    artifact_names = (
        "documents.parquet",
        "tables.parquet",
        "cells.parquet",
        "normalization_issues.parquet",
        "quality-summary.json",
        "dataset-metadata.json",
    )
    assert {
        name: (first.release_path / name).read_bytes() for name in artifact_names
    } == {
        name: (second.release_path / name).read_bytes() for name in artifact_names
    }
    pointer = json.loads((first_root / "current.json").read_text(encoding="utf-8"))
    assert pointer == {
        "dataset_fingerprint": first.dataset_fingerprint,
        "release": f"releases/{first.dataset_fingerprint}",
    }
    assert pq.read_table(first.release_path / "cells.parquet").num_rows > 0
```

- [ ] **Step 3: Write failure-injection tests before publication code**

Cover four boundaries with monkeypatch:

1. Parquet write raises: no release and no pointer are created.
2. verification raises: no release and the existing pointer remains byte-identical.
3. release rename raises: existing pointer remains byte-identical.
4. `os.replace` for the pointer raises: the new immutable release may exist, but the previous pointer remains byte-identical and readable.

Also test that a pre-existing matching release is verified and reused, a corrupt same-fingerprint release raises `DatasetPublicationError`, and temporary cleanup refuses paths outside `processed_root` or without prefix `.dataset-build-`.

- [ ] **Step 4: Implement deterministic orchestration**

`build_dataset()` must:

1. resolve and validate safe config paths;
2. call `read_manifest()` and select only `inventory_status == "ready"` documents;
3. stable-sort documents by `(relative_path.casefold(), relative_path, doc_id)`;
4. call `extract_document(snapshot_root, document)` then `normalize_extraction(document, extraction)`;
5. flatten/write/verify payloads below a `tempfile.mkdtemp(prefix=".dataset-build-", dir=processed_root)` directory;
6. compute the dataset fingerprint from source fingerprint, schema version, ruleset version, path-independent build config, and ordered payload hashes;
7. write and verify `dataset-metadata.json`;
8. rename the complete temp directory to `releases/<fingerprint>` or verify/reuse an existing equal release;
9. atomically replace `current.json` only after release verification;
10. return counts and fingerprints from verified artifacts.

Compute the fingerprint from canonical JSON:

```python
fingerprint_payload = {
    "source_manifest_sha256": manifest.sha256,
    "schema_version": config.schema_version,
    "ruleset_version": RULESET_VERSION,
    "build_config": {"ready_documents_only": True},
    "payload_sha256": dict(sorted(payload_hashes.items())),
}
dataset_fingerprint = hashlib.sha256(
    orjson.dumps(fingerprint_payload, option=orjson.OPT_SORT_KEYS)
).hexdigest()
```

- [ ] **Step 5: Implement safe immutable publication**

Use `Path.resolve()` and `is_relative_to()` before cleanup or rename. `release_path` must equal `processed_root / "releases" / dataset_fingerprint`. Write the pointer with `NamedTemporaryFile` in `processed_root`, call `flush()` and `os.fsync()`, then `os.replace(temp_pointer, processed_root / "current.json")`. Pointer JSON contains only the two fields asserted by the integration test, sorted with a final newline.

Export `DatasetBuildConfig`, `DatasetBuildResult`, and `build_dataset` from `data/__init__.py`.

- [ ] **Step 6: Run unit, integration, and upstream regression tests**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data tests/unit/ingestion tests/unit/normalization tests/golden/extraction
uv run --frozen --no-sync ruff check src/financial_report_qa/data src/financial_report_qa/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync mypy src/financial_report_qa/data src/financial_report_qa/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
```

- [ ] **Step 7: Commit Task 8**

```powershell
git add src/financial_report_qa/data/__init__.py src/financial_report_qa/data/dataset_builder.py tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py tests/integration/fixtures/normalization/VCB/2024/Consolidated/report.txt
git commit -m "feat: publish reproducible canonical dataset releases"
```

---

### Task 9: Product CLI, Documentation, and Final Quality Gate

**Files:**
- Create: `scripts/build_dataset.py`
- Modify: `src/financial_report_qa/data/dataset_builder.py`
- Modify: `src/financial_report_qa/cli.py`
- Modify: `tests/unit/test_cli.py`
- Modify: `README.md`
- Modify: `docs/data-download.md`

**Interfaces:**
- Consumes: `financial-report-qa build-dataset` arguments.
- Produces: exit code `0` and a concise fingerprint/count summary on success; exit code `2` and a safe error on expected failure.

- [ ] **Step 1: Write CLI dispatch and command tests**

Add to `tests/unit/test_cli.py`:

```python
def test_build_dataset_forwards_arguments() -> None:
    received: list[str] = []

    def fake_build_main(argv: Sequence[str] | None = None) -> int:
        received.extend(argv or ())
        return 0

    exit_code = main(
        ["build-dataset", "--manifest", "data/manifests/documents.jsonl"],
        build_main_fn=fake_build_main,
    )

    assert exit_code == 0
    assert received == ["--manifest", "data/manifests/documents.jsonl"]
```

Add dataset-builder command tests for required `--snapshot-root`, default manifest `data/manifests/documents.jsonl`, default processed root `data/processed`, default schema version `1`, success summary, and an `error: simulated failure` message that contains no absolute path.

- [ ] **Step 2: Implement the command and thin script**

Add a parser and `main(argv)` to `data/dataset_builder.py`. Print only release path relative to processed root, dataset fingerprint, document/table/cell/issue counts, and issue counts by code. Catch `FinancialReportQAError`, `OSError`, `ValueError`, and Pydantic validation errors; never print an absolute source path.

Add `build-dataset` to `financial_report_qa.cli` using the same dependency-injected dispatch pattern as existing commands. Create the wrapper:

```python
from financial_report_qa.data.dataset_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Document the exact workflow and layout**

Add this command to `README.md` and `docs/data-download.md`:

```powershell
uv run --frozen --no-sync financial-report-qa build-dataset `
  --snapshot-root data/raw/vifinqa/financial_statement `
  --manifest data/manifests/documents.jsonl `
  --processed-root data/processed
```

Document:

```text
data/processed/
├── current.json
└── releases/
    └── <dataset_fingerprint>/
        ├── documents.parquet
        ├── tables.parquet
        ├── cells.parquet
        ├── normalization_issues.parquet
        ├── quality-summary.json
        └── dataset-metadata.json
```

State that consumers must resolve `current.json`, releases are immutable, and cleanup of old releases is manual and outside this task.

- [ ] **Step 4: Run the complete quality gate**

```powershell
uv run --frozen --no-sync pytest -q tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync pytest -q tests/unit/schemas tests/unit/data tests/unit/ingestion tests/golden/extraction
uv run --frozen --no-sync ruff check src/financial_report_qa/normalization src/financial_report_qa/data/dataset_builder.py src/financial_report_qa/schemas/normalization.py scripts/build_dataset.py tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
uv run --frozen --no-sync ruff format --check src tests scripts
uv run --frozen --no-sync mypy src/financial_report_qa/normalization src/financial_report_qa/data/dataset_builder.py src/financial_report_qa/schemas/normalization.py scripts/build_dataset.py tests/unit/normalization tests/unit/data/test_dataset_builder.py tests/integration/test_build_dataset.py
```

Expected: every command exits `0`; pytest reports no failures, Ruff reports no issues or formatting changes, and mypy reports success.

- [ ] **Step 5: Confirm reproducibility and worktree scope**

Run the integration reproducibility test once more, then inspect:

```powershell
uv run --frozen --no-sync pytest -q tests/integration/test_build_dataset.py::test_build_dataset_is_reproducible_and_current_points_to_complete_release
git status --short
git diff --check
```

Confirm no raw data, generated release, cache, or unrelated pre-existing worktree file is staged.

- [ ] **Step 6: Commit Task 9**

```powershell
git add scripts/build_dataset.py src/financial_report_qa/data/dataset_builder.py src/financial_report_qa/cli.py tests/unit/test_cli.py README.md docs/data-download.md
git commit -m "feat: expose canonical dataset build command"
```

---

## Final Verification Checklist

- [ ] Point every design requirement to Tasks 1-9; no requirement remains unassigned.
- [ ] Scan this plan for incomplete markers, vague error handling, and undefined public interfaces; expect no findings.
- [ ] Confirm all later-task names match earlier definitions: `Decision`, `NormalizationIssue`, `NormalizedDocument`, `RULESET_VERSION`, `normalize_extraction`, `DatasetBuildConfig`, `DatasetBuildResult`, and `build_dataset`.
- [ ] Run `git diff --check` and confirm the implementation contains no whitespace errors.
- [ ] Run the entire project test suite with `uv run --frozen --no-sync pytest -q` when local runtime permits.
- [ ] Use `superpowers:verification-before-completion` before claiming Task 4/Day 5-6 complete.
