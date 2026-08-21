# Submission Compliance & Retrieval Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Làm cho mọi câu trong `submission.zip` vừa hợp lệ theo thể lệ (đáp án tính thật từ CSV, không hardcode) vừa chạy được, đồng thời tách `relevant_tables` khỏi nhánh trả lời để thu hồi 50% điểm truy hồi.

**Architecture:** Ba thay đổi cắt ngang `submission/`: (1) mở rộng schema CSV để `pandas_query` tham chiếu được mọi cột nó dùng; (2) biến "xuất bảng nguồn thật" từ đường ưu tiên thành đường bắt buộc, xoá hẳn nhánh sinh dòng tổng hợp; (3) lấy `relevant_docs`/`relevant_tables` từ đầu ra retrieval thay vì từ `compiled.evidence`. Một module linter mới chạy như chốt chặn cứng trước khi zip.

**Tech Stack:** Python 3.11, pandas, duckdb, pydantic v2, pytest, uv

## Global Constraints

- Mọi model dùng trong hệ thống phải **< 14B tham số** (BTC công bố; áp dụng cả embedding và reranker). Plan này không thêm model nào.
- **Không re-ingest corpus.** Mọi thay đổi phải giữ nguyên `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`, nếu không mọi baseline đã pin sẽ vô hiệu (ADR 0004 §1.7).
- `SubmissionItem` giữ nguyên đúng 7 trường, `extra="forbid"` (plan.md §2.4). Plan này **không** đổi contract JSON.
- ZIP phải tất định: `_FIXED_ZIP_DATE_TIME`, entries sort theo tên (plan.md §2.4 rule 8).
- Submission phải phủ **đủ 1012 id**; thiếu một id là fail toàn bộ ZIP (plan.md §2.4 rule 1).
- Release dir dùng cho mọi task: `data/processed/release_v2_422df141c935`
- Chạy test bằng: `.venv/Scripts/python.exe -m pytest`

## Baseline đo được (2026-08-21, trên `artifacts/submissions/v2gaps_full.zip`)

| Chỉ số | Giá trị |
|---|---:|
| Câu **vừa hợp lệ vừa chạy được** | **44 / 1012 (4.3%)** |
| CSV chỉ có 1 dòng, và dòng đó là đáp án | 964 |
| `pandas_query` fail replay trên CSV kèm theo | 106 |
| Trong đó `KeyError: 'table_id'` / `'row_idx'` | 84 |
| Câu answered | 186 |

**Tiêu chí thành công của plan:** chỉ số đầu tiên đạt **1012/1012**.

---

## File Structure

| File | Trách nhiệm | Thao tác |
|---|---|---|
| `src/financial_report_qa/submission/compliance.py` | Kiểm tra 7 bất biến C1–C7 trên một bundle đã dựng. Thuần hàm, không I/O ngoài sandbox replay. | **Tạo** |
| `tests/unit/submission/test_compliance.py` | Unit test cho từng bất biến | **Tạo** |
| `src/financial_report_qa/submission/exporter.py` | `_render_csv_bytes` (schema CSV), `_real_table_evidence_rows` (bắt buộc), `_relevant_docs_and_tables` (tách khỏi answering) | **Sửa** |
| `tests/unit/submission/test_submission_exporter.py` | Test hồi quy cho các thay đổi trên | **Sửa** |
| `src/financial_report_qa/submission/backstop_answer.py` | Xuất bảng nguồn thật thay vì dòng tổng hợp | **Sửa** |
| `tests/unit/submission/test_backstop_answer.py` | Test hồi quy backstop | **Sửa** |
| `src/financial_report_qa/submission/cli.py` | Gọi linter như chốt chặn cứng trước khi zip | **Sửa** |
| `src/financial_report_qa/retrieval/retrieval_scoring.py` | Đo F2 macro + MRR5 theo k trên tập gold (TABLES/DOCS) | **Tạo** |
| `tests/unit/retrieval/test_retrieval_scoring.py` | Unit test công thức F2 và MRR5 | **Tạo** |

---

## Task 1: Compliance linter

Module thuần kiểm tra, chưa gắn vào pipeline. Viết trước để Task 2–4 có thước đo.

**Files:**
- Create: `src/financial_report_qa/submission/compliance.py`
- Test: `tests/unit/submission/test_compliance.py`

**Interfaces:**
- Consumes: `SubmissionItem` từ `submission/contracts.py`; `replay_in_sandbox` từ `execution/sandbox.py`
- Produces:
  - `ComplianceViolation(question_id: int, code: str, detail: str)` — frozen dataclass
  - `check_item(item: SubmissionItem, frame: pd.DataFrame, *, timeout_seconds: float) -> tuple[ComplianceViolation, ...]`
  - `check_bundle(items: Sequence[SubmissionItem], csv_rows: Mapping[str, Sequence[Mapping[str, object]]], *, timeout_seconds: float) -> tuple[ComplianceViolation, ...]`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/submission/test_compliance.py`:

```python
"""Kiểm tra 7 bất biến chống hardcode của bundle nộp bài."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_report_qa.submission.compliance import check_item
from financial_report_qa.submission.contracts import SubmissionEvidence, SubmissionItem


def _item(*, answer: float, query: str) -> SubmissionItem:
    return SubmissionItem.model_validate(
        {
            "id": 1,
            "question": "Doanh thu thuần năm 2023?",
            "answer": answer,
            "relevant_docs": ("VNM_2023",),
            "relevant_tables": ("VNM_2023|100",),
            "evidence": (SubmissionEvidence(variable="df1", csv_path="data/q000001_df1.csv"),),
            "pandas_query": query,
        }
    )


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["period"] = frame["period"].astype("Int64")
    return frame


_GOOD_ROWS = [
    {"company_code": "VNM", "row_label_raw": "Doanh thu thuần", "column_label": "2023",
     "period": 2023, "value": 1200.0},
    {"company_code": "VNM", "row_label_raw": "Lợi nhuận sau thuế", "column_label": "2023",
     "period": 2023, "value": 120.0},
]
_GOOD_QUERY = 'df1[(df1.row_label_raw == "Doanh thu thuần") & (df1.period == 2023)]["value"].iloc[0]'


def test_compliant_item_has_no_violations() -> None:
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert violations == ()


def test_c1_single_row_csv_is_a_violation() -> None:
    rows = [_GOOD_ROWS[0]]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C1" in {v.code for v in violations}


def test_c2_answer_equal_to_only_value_is_a_violation() -> None:
    rows = [_GOOD_ROWS[0]]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C2" in {v.code for v in violations}


def test_c3_answer_named_column_is_a_violation() -> None:
    rows = [dict(row, answer=row["value"]) for row in _GOOD_ROWS]
    violations = check_item(
        _item(answer=1200.0, query=_GOOD_QUERY), _frame(rows), timeout_seconds=5
    )
    assert "C3" in {v.code for v in violations}


def test_c4_answer_literal_in_query_is_a_violation() -> None:
    violations = check_item(
        _item(answer=1200.0, query="1200.0"), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C4" in {v.code for v in violations}


def test_c5_query_referencing_no_csv_column_is_a_violation() -> None:
    violations = check_item(
        _item(answer=1200.0, query="1200.0"), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C5" in {v.code for v in violations}


def test_c7_query_that_cannot_replay_is_a_violation() -> None:
    query = 'df1.loc[(df1.table_id == "t1") & (df1.row_idx == 3), "value"].iloc[0]'
    violations = check_item(
        _item(answer=1200.0, query=query), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C7" in {v.code for v in violations}


def test_c7_wrong_replay_value_is_a_violation() -> None:
    violations = check_item(
        _item(answer=999.0, query=_GOOD_QUERY), _frame(_GOOD_ROWS), timeout_seconds=5
    )
    assert "C7" in {v.code for v in violations}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_compliance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'financial_report_qa.submission.compliance'`

- [ ] **Step 3: Write the implementation**

Tạo `src/financial_report_qa/submission/compliance.py`:

```python
"""Chốt chặn chống hardcode cho bundle nộp bài.

Thể lệ Stage 2 quy định: "Kết quả của mỗi pandas query phải được tính toán
trực tiếp từ dữ liệu có trong các bảng CSV tại thời điểm thực thi. Không được
gán cứng, mã hóa hoặc lưu sẵn kết quả dưới bất kỳ hình thức nào. Các câu hỏi
vi phạm quy định này sẽ không được tính điểm." Mục VIII còn liệt kê "Hardcode
đáp án benchmark" là hành vi có thể bị loại đội thi.

Module này biến quy định đó thành phép kiểm mechanical. Nó chỉ đọc, không sửa
gì: `submission/cli.py` gọi nó và fail build khi có vi phạm.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.submission.contracts import SubmissionItem

_ANSWER_LIKE_COLUMNS = frozenset({"answer", "result", "ans", "expected"})
_NUMBER_LITERAL_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
_VALUE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ComplianceViolation:
    """Một vi phạm cụ thể, gắn với đúng một câu hỏi."""

    question_id: int
    code: str
    detail: str


def _numbers_in(query: str) -> list[float]:
    out: list[float] = []
    for token in _NUMBER_LITERAL_PATTERN.findall(query):
        try:
            out.append(float(token))
        except ValueError:  # pragma: no cover -- regex chỉ khớp số hợp lệ
            continue
    return out


def check_item(
    item: SubmissionItem, frame: pd.DataFrame, *, timeout_seconds: float
) -> tuple[ComplianceViolation, ...]:
    """Trả về mọi vi phạm của một câu. Rỗng nghĩa là hợp lệ."""
    violations: list[ComplianceViolation] = []
    query = item.pandas_query

    def add(code: str, detail: str) -> None:
        violations.append(ComplianceViolation(question_id=item.id, code=code, detail=detail))

    # C1: CSV phải là lát cắt bảng thật, không phải một ô dựng ngược từ đáp án.
    if len(frame) < 2:
        add("C1", f"CSV chỉ có {len(frame)} dòng dữ liệu (cần >= 2)")

    # C2: đáp án không được là giá trị duy nhất nằm sẵn trong CSV.
    if len(frame) == 1 and "value" in frame.columns:
        only = frame["value"].iloc[0]
        if isinstance(only, (int, float)) and math.isfinite(float(only)):
            if abs(float(only) - item.answer) <= _VALUE_TOLERANCE:
                add("C2", f"answer {item.answer} là giá trị duy nhất trong CSV")

    # C3: không được có cột mang sẵn đáp án.
    named = _ANSWER_LIKE_COLUMNS.intersection(str(c).lower() for c in frame.columns)
    if named:
        add("C3", f"CSV chứa cột mang sẵn đáp án: {sorted(named)}")

    # C4: đáp án không được xuất hiện dưới dạng hằng số trong query.
    for literal in _numbers_in(query):
        if abs(literal - item.answer) <= _VALUE_TOLERANCE:
            add("C4", f"pandas_query chứa literal {literal} trùng answer")
            break

    # C5: query phải thực sự đọc từ CSV.
    referenced = [str(c) for c in frame.columns if re.search(rf"\b{re.escape(str(c))}\b", query)]
    if not referenced:
        add("C5", "pandas_query không tham chiếu cột nào của CSV")

    # C6: nhãn dòng nêu trong query phải tồn tại trong CSV.
    for label_column in ("row_label_raw", "row_label_canonical"):
        if label_column not in frame.columns:
            continue
        quoted = re.findall(rf"{label_column}\s*==\s*\"([^\"]+)\"", query)
        present = {str(v) for v in frame[label_column].dropna().tolist()}
        for label in quoted:
            if label not in present:
                add("C6", f"{label_column}=={label!r} không có trong CSV")

    # C7: bằng chứng quyết định -- đáp án phải replay được từ chính CSV này.
    result = replay_in_sandbox(query, frame, timeout_seconds=timeout_seconds)
    if result.error_code is not None:
        add("C7", f"replay lỗi: {result.error_code}: {result.error_message}")
    elif result.value is None:
        add("C7", "replay không trả về giá trị")
    elif abs(float(result.value) - item.answer) > _VALUE_TOLERANCE:
        add("C7", f"replay ra {float(result.value)} nhưng answer là {item.answer}")

    return tuple(violations)


def check_bundle(
    items: Sequence[SubmissionItem],
    csv_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    timeout_seconds: float,
) -> tuple[ComplianceViolation, ...]:
    """Kiểm tra toàn bộ bundle. Trả về mọi vi phạm, sắp theo question_id."""
    violations: list[ComplianceViolation] = []
    for item in items:
        csv_path = item.evidence[0].csv_path
        rows = csv_rows.get(csv_path)
        if rows is None:
            violations.append(
                ComplianceViolation(
                    question_id=item.id, code="C0", detail=f"thiếu CSV {csv_path}"
                )
            )
            continue
        frame = pd.DataFrame(list(rows))
        if "period" in frame.columns:
            frame["period"] = frame["period"].astype("Int64")
        violations.extend(check_item(item, frame, timeout_seconds=timeout_seconds))
    return tuple(sorted(violations, key=lambda v: (v.question_id, v.code)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_compliance.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Xác nhận linter bắt được bundle cũ (test hồi quy thực địa)**

Chạy:

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import zipfile, json, io, csv, collections
from financial_report_qa.submission.compliance import check_bundle
from financial_report_qa.submission.contracts import SubmissionItem
z = zipfile.ZipFile('artifacts/submissions/v2gaps_full.zip')
items = [SubmissionItem.model_validate(d) for d in json.loads(z.read('submission.json'))]
rows = {n: list(csv.DictReader(io.StringIO(z.read(n).decode('utf-8'))))
        for n in z.namelist() if n.endswith('.csv')}
for path in rows:
    for r in rows[path]:
        r['value'] = float(r['value']); r['period'] = int(r['period'])
v = check_bundle(items, rows, timeout_seconds=5)
c = collections.Counter(x.code for x in v)
print('tong vi pham:', len(v))
for k, n in sorted(c.items()): print(f'  {k}: {n}')
print('cau sach:', 1012 - len({x.question_id for x in v}))
"
```

Expected: `cau sach: 44`, với `C1` ≈ 964, `C2` ≈ 964, `C7` ≈ 106.
Nếu `cau sach` khác 44, dừng lại và điều tra trước khi đi tiếp — con số này là baseline của cả plan.

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/submission/compliance.py tests/unit/submission/test_compliance.py
git commit -m "feat(submission): add anti-hardcode compliance linter (C1-C7)"
```

---

## Task 2: Mở rộng schema CSV để query chạy được

84 query dùng `df1.table_id` / `df1.row_idx` nhưng CSV không có hai cột đó → `KeyError` → 106 câu fail replay. `build_cell_frame` đã trả về sẵn cả ba cột (`cell_frame.py:76-80`); chỉ `_render_csv_bytes` vứt bỏ chúng.

**Files:**
- Modify: `src/financial_report_qa/submission/exporter.py:665-700` (`_render_csv_bytes`)
- Modify: `src/financial_report_qa/submission/exporter.py:151-161` (`_real_table_evidence_rows` — thêm 3 khoá vào dict)
- Test: `tests/unit/submission/test_submission_exporter.py`

**Interfaces:**
- Consumes: `ComplianceViolation`, `check_item` từ Task 1
- Produces: CSV schema 9 cột — `table_id, row_idx, col_idx, company_code, row_label_canonical, row_label_raw, column_label, period, value`. Task 3 và 4 đều ghi theo schema này.

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/submission/test_submission_exporter.py`:

```python
def test_render_csv_bytes_includes_position_columns() -> None:
    """84 query dùng df1.table_id/df1.row_idx; CSV phải mang theo hai cột đó."""
    from financial_report_qa.submission.exporter import _render_csv_bytes

    rows = [
        {
            "table_id": "tbl_abc",
            "row_idx": 19,
            "col_idx": 2,
            "company_code": "VNM",
            "row_label_canonical": None,
            "row_label_raw": "Doanh thu thuần",
            "column_label": "2023",
            "period": 2023,
            "value": 1200.0,
        }
    ]
    header = _render_csv_bytes(rows).decode("utf-8").splitlines()[0]
    assert header.split(",") == [
        "table_id",
        "row_idx",
        "col_idx",
        "company_code",
        "row_label_canonical",
        "row_label_raw",
        "column_label",
        "period",
        "value",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_exporter.py::test_render_csv_bytes_includes_position_columns -v`
Expected: FAIL — `AssertionError`, header hiện tại chỉ có 6 cột

- [ ] **Step 3: Sửa `_render_csv_bytes`**

Trong `src/financial_report_qa/submission/exporter.py`, thay toàn bộ thân `_render_csv_bytes`:

```python
_CSV_COLUMNS = (
    "table_id",
    "row_idx",
    "col_idx",
    "company_code",
    "row_label_canonical",
    "row_label_raw",
    "column_label",
    "period",
    "value",
)


def _render_csv_bytes(rows: Sequence[CsvRow]) -> bytes:
    """Ghi lát cắt bảng nguồn theo schema cố định.

    `table_id`/`row_idx`/`col_idx` bắt buộc phải có: `pandas_query.py`
    `_position_clauses` sinh predicate tham chiếu trực tiếp `df1.table_id` và
    `df1.row_idx`, và trước Day 27 hai cột đó bị bỏ khỏi CSV -- khiến 84 câu
    ném `KeyError` khi replay trên chính CSV đóng gói kèm.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(_CSV_COLUMNS))
    for row in rows:
        writer.writerow(
            ["" if row.get(column) is None else row.get(column) for column in _CSV_COLUMNS]
        )
    return buffer.getvalue().encode("utf-8")
```

- [ ] **Step 4: Sửa `_real_table_evidence_rows` để mang theo 3 cột mới**

Trong `src/financial_report_qa/submission/exporter.py`, trong `_real_table_evidence_rows`, thay dict comprehension dựng `rows` (dòng ~151-161):

```python
    rows: tuple[CsvRow, ...] = tuple(
        {
            "table_id": record["table_id"],
            "row_idx": record["row_idx"],
            "col_idx": record["col_idx"],
            "company_code": record["company_code"],
            "row_label_canonical": record["row_label_canonical"],
            "row_label_raw": record["row_label_raw"],
            "column_label": record["column_label"],
            "period": record["period"],
            "value": record["value"],
        }
        for record in frame.to_dict(orient="records")
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ -v`
Expected: PASS. Nếu test cũ nào assert header 6 cột, cập nhật nó sang 9 cột — đó là thay đổi có chủ đích.

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/submission/exporter.py tests/unit/submission/test_submission_exporter.py
git commit -m "fix(submission): carry table_id/row_idx/col_idx into evidence CSV

84 position-bound queries referenced df1.table_id and df1.row_idx, which the
CSV writer dropped -- every one of them raised KeyError when replayed against
its own packaged CSV. build_cell_frame already returned all three columns."
```

---

## Task 3: Bắt buộc dùng bảng nguồn thật, xoá nhánh sinh dòng tổng hợp

`_real_table_evidence_rows` trả `None` khi replay không khớp, và caller rơi về `_replay_rows_to_csv_rows` — chính là nhánh sinh CSV một dòng. Sau Task 2, nguyên nhân `KeyError` đã hết; phần còn lại là nhập nhằng predicate (31.58% với `company+row_label+period`). Thay vì fallback sang CSV tổng hợp, câu không replay được phải bị coi là **thất bại execution** và đẩy sang backstop (Task 4 sẽ làm backstop cũng hợp lệ).

**Files:**
- Modify: `src/financial_report_qa/submission/exporter.py:520-535` (call site)
- Test: `tests/unit/submission/test_submission_exporter.py`

**Interfaces:**
- Consumes: `_real_table_evidence_rows` (Task 2 đã mở rộng schema)
- Produces: bất biến "mọi `evidence_rows` trả từ `_run_one_question` đều đến từ `build_cell_frame`" — Task 5 dựa vào đây.

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/submission/test_submission_exporter.py`:

```python
def test_answered_path_never_emits_synthesized_single_row(tmp_path) -> None:
    """Bất biến BI-1: evidence CSV luôn là lát cắt bảng thật.

    Nếu bảng thật không replay đúng, câu phải rơi xuống backstop chứ không
    được đóng gói một dòng dựng ngược từ đáp án.
    """
    from financial_report_qa.submission import exporter

    calls: list[str] = []

    def _fake_real_rows(compiled, release_dir, *, timeout_seconds):
        calls.append("called")
        return None

    original = exporter._real_table_evidence_rows
    exporter._real_table_evidence_rows = _fake_real_rows
    try:
        assert not hasattr(exporter, "_replay_rows_to_csv_rows"), (
            "_replay_rows_to_csv_rows là nhánh sinh CSV một dòng -- phải bị xoá"
        )
    finally:
        exporter._real_table_evidence_rows = original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_exporter.py::test_answered_path_never_emits_synthesized_single_row -v`
Expected: FAIL — `AssertionError: _replay_rows_to_csv_rows là nhánh sinh CSV một dòng`

- [ ] **Step 3: Đổi call site thành thất bại execution**

Trong `src/financial_report_qa/submission/exporter.py`, thay khối tại dòng ~524-535:

```python
    # BI-1 (design §5.1): evidence CSV luôn là lát cắt bảng nguồn thật. Khi
    # bảng thật không replay ra đúng đáp án, câu này KHÔNG được đóng gói một
    # dòng dựng ngược từ đáp án -- đó chính là mẫu `result = df["answer"]
    # .iloc[0]` mà thể lệ cấm. Trả về thất bại execution để backstop (đã hợp
    # lệ từ Task 4) tiếp quản.
    evidence_rows = _real_table_evidence_rows(
        compiled, release_dir, timeout_seconds=execution_settings.timeout_seconds
    )
    if evidence_rows is None:
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "error",
                    "stage": "execution",
                    "code": "evidence_frame_replay_mismatch",
                    "plan_source": plan_source,
                }
            ),
            None,
            None,
        )
```

- [ ] **Step 4: Xoá hàm `_replay_rows_to_csv_rows`**

Xoá định nghĩa `_replay_rows_to_csv_rows` khỏi `exporter.py`. Nếu `ExecutionIssueCode` là Literal đóng, thêm `"evidence_frame_replay_mismatch"` vào `execution/contracts.py`.

Run: `.venv/Scripts/python.exe -m mypy src/financial_report_qa/submission/exporter.py`
Expected: no errors. Nếu mypy báo `evidence_frame_replay_mismatch` không hợp lệ, thêm nó vào Literal của `ExecutionIssueCode`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/submission/exporter.py src/financial_report_qa/execution/contracts.py tests/unit/submission/test_submission_exporter.py
git commit -m "fix(submission): make real-table evidence mandatory, drop synthetic single-row CSV

The synthesized fallback packaged one cell whose value WAS the answer, with a
query that merely re-fetched it -- functionally result = df['answer'].iloc[0],
which the rules name as invalid. 964/1012 questions shipped this shape."
```

---

## Task 4: Backstop xuất bảng nguồn thật

`build_backstop_item` nhận `candidate_table_ids` (10 bảng đã retrieve) nhưng chỉ dùng để lấy một ô, rồi dựng `row` thủ công và emit **một** bảng. Áp dụng cho 826/1012 câu. Viết lại: xuất trọn bảng nguồn, chọn một ô **định vị được duy nhất**, và emit đủ danh sách bảng đã retrieve.

**Files:**
- Modify: `src/financial_report_qa/submission/backstop_answer.py`
- Test: `tests/unit/submission/test_backstop_answer.py`

**Interfaces:**
- Consumes: `build_cell_frame`, schema CSV 9 cột (Task 2)
- Produces: `build_backstop_item(raw_question, candidate_table_ids, release_dir) -> tuple[SubmissionItem, tuple[CsvRow, ...]]` — chữ ký giữ nguyên, hành vi đổi: `rows` là trọn bảng, `relevant_tables` là mọi bảng đã retrieve.

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/submission/test_backstop_answer.py`:

```python
def test_backstop_emits_full_source_table_not_one_row(release_dir, sample_table_ids) -> None:
    """BI-4: backstop không được tổng hợp dòng nào."""
    from financial_report_qa.submission.backstop_answer import build_backstop_item
    from financial_report_qa.submission.contracts import RawQuestion

    question = RawQuestion(id=1, question="Doanh thu thuần năm 2023?")
    item, rows = build_backstop_item(question, sample_table_ids, release_dir)

    assert len(rows) >= 2, "backstop phải xuất trọn bảng, không phải một dòng"
    assert {"table_id", "row_idx", "col_idx"} <= set(rows[0].keys())


def test_backstop_answer_replays_from_its_own_csv(release_dir, sample_table_ids) -> None:
    """C7: đáp án phải tính được từ chính CSV kèm theo."""
    import pandas as pd

    from financial_report_qa.submission.backstop_answer import build_backstop_item
    from financial_report_qa.submission.compliance import check_item
    from financial_report_qa.submission.contracts import RawQuestion

    question = RawQuestion(id=1, question="Doanh thu thuần năm 2023?")
    item, rows = build_backstop_item(question, sample_table_ids, release_dir)
    frame = pd.DataFrame(list(rows))
    frame["period"] = frame["period"].astype("Int64")

    violations = check_item(item, frame, timeout_seconds=5)
    assert violations == (), f"backstop vẫn vi phạm: {violations}"
```

Thêm fixtures vào cùng file (nếu chưa có):

```python
import pytest


@pytest.fixture
def release_dir():
    from pathlib import Path

    path = Path("data/processed/release_v2_422df141c935")
    if not path.exists():
        pytest.skip("release chưa có sẵn trên máy này")
    return path


@pytest.fixture
def sample_table_ids(release_dir):
    import duckdb

    connection = duckdb.connect(":memory:")
    frame = connection.execute(
        "SELECT DISTINCT table_id FROM read_parquet(?) WHERE value_numeric IS NOT NULL "
        "AND period IS NOT NULL LIMIT 1",
        [str(release_dir / "cells.parquet")],
    ).fetchdf()
    connection.close()
    return tuple(frame["table_id"].tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_backstop_answer.py -v -k "full_source_table or replays_from"`
Expected: FAIL — `assert len(rows) >= 2`, rows hiện có đúng 1 phần tử

- [ ] **Step 3: Viết lại `build_backstop_item`**

Trong `src/financial_report_qa/submission/backstop_answer.py`, thay `_pick_backstop_cell` và `build_backstop_item`:

```python
def _uniquely_addressable_row(frame: pd.DataFrame) -> pd.Series:
    """Chọn một ô mà predicate ngữ nghĩa định vị được duy nhất trong bảng.

    Đo trên corpus: `(row_label_raw, column_label, period)` trong phạm vi một
    bảng còn nhập nhằng 4.37%. Ưu tiên ô không nhập nhằng để `pandas_query`
    replay được mà không cần tie-break vị trí.
    """
    usable = frame[frame["period"].notna() & frame["row_label_raw"].notna()]
    if usable.empty:
        raise RuntimeError("bảng ứng viên không có ô nào định vị được")
    counts = usable.groupby(["row_label_raw", "column_label", "period"], dropna=False)[
        "value"
    ].transform("nunique")
    unique = usable[counts == 1]
    return (unique if not unique.empty else usable).iloc[0]


def build_backstop_item(
    raw_question: RawQuestion,
    candidate_table_ids: Sequence[str],
    release_dir: Path,
) -> tuple[SubmissionItem, tuple[CsvRow, ...]]:
    """Tầng cuối: luôn trả về `SubmissionItem` hợp lệ và HỢP QUY.

    Khác bản trước ở hai điểm quyết định:

    1. CSV là trọn bảng nguồn (`build_cell_frame`), không phải một dòng dựng
       ngược từ đáp án. Bản cũ vi phạm quy định "không được gán cứng, mã hóa
       hoặc lưu sẵn kết quả" -- 826/1012 câu đi qua đây.
    2. `relevant_tables` là MỌI bảng đã retrieve, không phải một bảng suy ra
       từ ô được chọn. Thể lệ chấm truy hồi (50% điểm, F2 macro) độc lập với
       việc trả lời đúng hay sai, nên vứt bỏ danh sách đã retrieve là mất
       điểm không cần thiết.

    Đáp án vẫn là best-effort và thường sai -- đó là đánh đổi chấp nhận được
    (Answer Accuracy tính trên tổng số câu, sai và bỏ trống đều bằng 0).
    """
    if not candidate_table_ids:
        raise RuntimeError(f"không có bảng ứng viên cho câu {raw_question.id}")

    table_ids = tuple(dict.fromkeys(candidate_table_ids))
    frame = build_cell_frame(release_dir, table_ids)
    chosen = _uniquely_addressable_row(frame)
    table_id = str(chosen["table_id"])

    # CSV thu về đúng bảng chứa ô đã chọn: predicate ngữ nghĩa chỉ duy nhất
    # trong phạm vi một bảng (4.37% nhập nhằng), không duy nhất giữa 10 bảng.
    table_frame = frame[frame["table_id"] == table_id]
    rows: tuple[CsvRow, ...] = tuple(
        {
            "table_id": record["table_id"],
            "row_idx": record["row_idx"],
            "col_idx": record["col_idx"],
            "company_code": record["company_code"],
            "row_label_canonical": record["row_label_canonical"],
            "row_label_raw": record["row_label_raw"],
            "column_label": record["column_label"],
            "period": record["period"],
            "value": record["value"],
        }
        for record in table_frame.to_dict(orient="records")
    )

    clauses = [
        f"(df1.row_label_raw == {json.dumps(str(chosen['row_label_raw']), ensure_ascii=False)})",
        f"(df1.period == {int(chosen['period'])})",
    ]
    if chosen["column_label"] is not None and not pd.isna(chosen["column_label"]):
        clauses.append(
            f"(df1.column_label == {json.dumps(str(chosen['column_label']), ensure_ascii=False)})"
        )
    query = f'df1[{" & ".join(clauses)}]["value"].iloc[0]'

    cell_ids = [str(record["cell_id"]) for record in table_frame.to_dict(orient="records")]
    lookup = build_citation_lookup(release_dir, cell_ids)
    docs: dict[str, None] = {}
    tables: dict[str, None] = {}
    for cell_id in cell_ids:
        provenance = lookup[cell_id]
        report_id = str(provenance["doc_relative_path"]).rsplit("/", 1)[-1]
        if report_id.endswith(".txt"):
            report_id = report_id[: -len(".txt")]
        docs.setdefault(report_id, None)
        tables.setdefault(f"{report_id}|{provenance['source_line_start']}", None)

    item = SubmissionItem.model_validate(
        {
            "id": raw_question.id,
            "question": raw_question.question,
            "answer": float(chosen["value"]),
            "relevant_docs": tuple(docs),
            "relevant_tables": tuple(tables),
            "evidence": (
                SubmissionEvidence(
                    variable="df1", csv_path=f"data/q{raw_question.id:06d}_df1.csv"
                ),
            ),
            "pandas_query": query,
        }
    )
    return item, rows
```

Xoá `_UNIVERSAL_FALLBACK_QUERY`, `_hardened_connection`, `_pick_any_corpus_cell`, `_pick_backstop_cell` — không còn dùng. Thêm `import pandas as pd` nếu chưa có.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_backstop_answer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/submission/backstop_answer.py tests/unit/submission/test_backstop_answer.py
git commit -m "fix(submission): backstop emits the real source table, not a synthesized row

826/1012 questions went through this tier shipping a one-cell CSV whose value
WAS the declared answer. Also stops discarding the retrieved table list, which
the rules score independently at 50% weight."
```

---

## Task 5: Gắn linter làm chốt chặn cứng trong export

**Files:**
- Modify: `src/financial_report_qa/submission/cli.py`
- Test: `tests/unit/submission/test_submission_cli.py`

**Interfaces:**
- Consumes: `check_bundle` (Task 1)
- Produces: `submission export` trả exit code `2` khi có vi phạm, và ghi `compliance-violations.json` vào report dir.

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/submission/test_submission_cli.py`:

```python
def test_export_fails_build_when_bundle_has_violations(monkeypatch, tmp_path) -> None:
    """Bundle vi phạm không bao giờ được ghi ra ZIP."""
    from financial_report_qa.submission import cli as submission_cli
    from financial_report_qa.submission.compliance import ComplianceViolation

    monkeypatch.setattr(
        submission_cli,
        "check_bundle",
        lambda items, csv_rows, *, timeout_seconds: (
            ComplianceViolation(question_id=1, code="C1", detail="CSV chỉ có 1 dòng"),
        ),
    )
    output_zip = tmp_path / "out.zip"
    code = submission_cli.main([...])  # tham số như test export hiện có
    assert code == 2
    assert not output_zip.exists(), "ZIP vi phạm không được ghi ra đĩa"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_cli.py -k violations -v`
Expected: FAIL — `AttributeError: module has no attribute 'check_bundle'`

- [ ] **Step 3: Gắn linter vào CLI**

Trong `src/financial_report_qa/submission/cli.py`, thêm import và chèn kiểm tra **trước** lời gọi `write_submission_zip`:

```python
from financial_report_qa.submission.compliance import check_bundle
```

```python
    # Chốt chặn cứng (design §5.3): thể lệ ghi "Các câu hỏi vi phạm quy định
    # này sẽ không được tính điểm", và mục VIII liệt kê hardcode đáp án là căn
    # cứ loại đội thi. Không bao giờ ghi ra ZIP một bundle vi phạm.
    violations = check_bundle(
        items, csv_rows, timeout_seconds=execution_settings.timeout_seconds
    )
    if violations:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "compliance-violations.json").write_text(
            json.dumps(
                [
                    {"id": v.question_id, "code": v.code, "detail": v.detail}
                    for v in violations
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        affected = len({v.question_id for v in violations})
        print(
            f"COMPLIANCE FAIL: {len(violations)} vi phạm trên {affected} câu. "
            f"Chi tiết: {report_dir / 'compliance-violations.json'}"
        )
        return 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/submission/cli.py tests/unit/submission/test_submission_cli.py
git commit -m "feat(submission): fail the export build when compliance violations exist"
```

---

## Task 6: Tách `relevant_docs`/`relevant_tables` khỏi nhánh trả lời, giữ đúng thứ tự rank

Dashboard chấm điểm công khai có 10 cột: `EXECUTION ACCURACY, TABLES F2-MACRO, DOCS
F2-MACRO, TABLES PRECISION, TABLES RECALL, TABLES MRR5, DOCS PRECISION, DOCS RECALL,
DOCS MRR5, ANSWER ACCURACY`. Hai điều đó thay đổi so với phần tóm tắt "F2 macro" trong
PDF thể lệ:

1. **DOCS và TABLES chấm riêng** — tối ưu tables không tự động tối ưu docs.
2. **MRR5 đo thứ hạng, không phải tập hợp.** `relevant_docs`/`relevant_tables` phải là
   mảng theo đúng **thứ tự retrieval-rank** (điểm cao nhất trước). F2/Precision/Recall
   coi chúng là tập không thứ tự nên không phát hiện lỗi thứ tự; MRR5 thì có.

Hiện `_relevant_docs_and_tables` lấy bảng từ `compiled.evidence` — chỉ bảng đã dùng để
tính, không phải tập đã retrieve. **Cạm bẫy cần tránh khi sửa:** nguồn thay thế hiển
nhiên là `build_cell_frame()`, nhưng hàm đó `ORDER BY table_id` (alphabet) — dùng nó sẽ
sửa được F2 nhưng âm thầm phá MRR5, vì bảng đúng có thể tụt từ hạng 1 xuống hạng 8 mà
không có test nào tự nhiên bắt được (F2 với cùng tập bảng vẫn ra cùng một số).

**Files:**
- Modify: `src/financial_report_qa/submission/exporter.py:172-188` và call site dòng ~510
- Test: `tests/unit/submission/test_submission_exporter.py`

**Interfaces:**
- Consumes: `retrieved: Sequence[str]` — đã có sẵn trong `_run_one_question`, sinh bởi
  `retrieve_candidate_table_ids()` (`retrieval/live_query.py:21`), docstring của hàm đó
  ghi rõ *"in retrieval-rank order"* — đây là nguồn thứ tự đúng duy nhất trong codebase.
- Produces: `_relevant_docs_and_tables(retrieved_table_ids, release_dir) -> tuple[tuple[str, ...], tuple[str, ...]]` — đổi chữ ký, không còn nhận `compiled`. **Bất biến bắt buộc:** phần tử thứ `i` của tuple trả về tương ứng bảng có rank `i` trong `retrieved_table_ids` (sau khử trùng lặp) — không được sắp lại theo bất kỳ tiêu chí nào khác.

- [ ] **Step 1: Write the failing test**

```python
def test_relevant_tables_come_from_retrieval_not_from_evidence(release_dir) -> None:
    """Điểm truy hồi (50%) được chấm độc lập với việc trả lời đúng hay sai,
    nên danh sách bảng phải phản ánh retrieval, không phải bảng đã tính."""
    from financial_report_qa.submission.exporter import _relevant_docs_and_tables
    import inspect

    signature = inspect.signature(_relevant_docs_and_tables)
    assert "compiled" not in signature.parameters, (
        "hàm này không được phụ thuộc kết quả execution"
    )
    assert "retrieved_table_ids" in signature.parameters


def test_relevant_tables_preserve_retrieval_rank_order(release_dir, two_ranked_table_ids) -> None:
    """MRR5 (dashboard cột TABLES MRR5) đo thứ hạng, không phải tập hợp.

    Bảng hạng 1 trong input phải là phần tử đầu tiên của output -- kể cả khi
    nó không đứng đầu theo alphabet table_id (build_cell_frame ORDER BY
    table_id sẽ sắp sai nếu bị dùng làm nguồn thứ tự)."""
    from financial_report_qa.submission.exporter import _relevant_docs_and_tables

    rank1_table_id, rank2_table_id = two_ranked_table_ids
    # Cố tình chọn cặp mà rank1 > rank2 theo alphabet, để phép test không thể
    # tình cờ pass do trùng thứ tự.
    assert rank1_table_id > rank2_table_id, (
        "fixture phải cho rank1 đứng SAU rank2 theo alphabet để test có ý nghĩa"
    )
    _docs, tables = _relevant_docs_and_tables((rank1_table_id, rank2_table_id), release_dir)
    first_table_id = tables[0].split("|", 1)[0]
    assert rank1_table_id.startswith(first_table_id) or first_table_id in rank1_table_id, (
        f"bảng hạng 1 ({rank1_table_id}) phải xuất hiện trước trong output, "
        f"nhưng output bắt đầu bằng {tables[0]!r}"
    )
```

Thêm fixture `two_ranked_table_ids` vào `tests/unit/submission/test_submission_exporter.py` (hoặc `conftest.py` cùng thư mục nếu đã có fixture `release_dir` ở đó):

```python
@pytest.fixture
def two_ranked_table_ids(release_dir):
    """Hai table_id thật trong release, cố tình lấy sao cho phần tử đầu
    đứng SAU phần tử hai theo alphabet -- nếu code dùng nhầm build_cell_frame
    (ORDER BY table_id) làm nguồn thứ tự, test này sẽ đảo ngược và fail."""
    import duckdb

    connection = duckdb.connect(":memory:")
    frame = connection.execute(
        "SELECT DISTINCT table_id FROM read_parquet(?) "
        "WHERE value_numeric IS NOT NULL AND period IS NOT NULL "
        "ORDER BY table_id DESC LIMIT 2",
        [str(release_dir / "cells.parquet")],
    ).fetchdf()
    connection.close()
    ids = frame["table_id"].tolist()
    return ids[0], ids[1]  # ids[0] > ids[1] theo alphabet (ORDER BY DESC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_exporter.py -k "relevant_tables_come_from or preserve_retrieval_rank" -v`
Expected: FAIL — `relevant_tables_come_from_retrieval_not_from_evidence` fails với `AssertionError` (chữ ký chưa đổi); `preserve_retrieval_rank_order` fails với `ModuleNotFoundError`/`AttributeError` (hàm chưa tồn tại theo chữ ký mới)

- [ ] **Step 3: Viết lại hàm — giữ đúng thứ tự đầu vào**

```python
def _relevant_docs_and_tables(
    retrieved_table_ids: Sequence[str], release_dir: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Danh sách bảng báo cáo cho chấm truy hồi.

    Dashboard chấm 8 chỉ số truy hồi (TABLES/DOCS x Precision/Recall/F2/MRR5),
    ĐỘC LẬP với Answer/Execution Accuracy. Trước Day 27 hàm này lấy bảng từ
    `compiled.evidence` -- chỉ những bảng thực sự dùng để tính -- nên câu nào
    không trả lời được thì cũng mất luôn điểm truy hồi, dù retriever đã tìm
    đúng bảng. Retriever chỉ trượt 42/1012 câu, nhưng submission chỉ báo cáo
    đúng cho ~18%.

    Bất biến bắt buộc: thứ tự phần tử trong tuple trả về PHẢI khớp thứ tự
    trong `retrieved_table_ids` (retrieval-rank, điểm cao nhất trước) -- dashboard
    chấm MRR5 (vị trí kết quả đúng đầu tiên trong top-5), không chỉ tập hợp.
    Vì vậy hàm này KHÔNG được lấy danh sách bảng qua `build_cell_frame()`
    (nó `ORDER BY table_id` -- alphabet, không phải rank) rồi duyệt theo thứ
    tự đó; chỉ được dùng nó để tra citation của MỘT table_id đã biết trước.
    """
    if not retrieved_table_ids:
        return ((), ())

    ordered_table_ids = tuple(dict.fromkeys(retrieved_table_ids))
    frame = build_cell_frame(release_dir, ordered_table_ids)

    # Một cell bất kỳ mỗi bảng là đủ để tra citation (doc + source line đều
    # ổn định trong phạm vi một table_id) -- không cần toàn bộ cell.
    first_cell_by_table: dict[str, str] = {}
    for record in frame.to_dict(orient="records"):
        table_id = str(record["table_id"])
        first_cell_by_table.setdefault(table_id, str(record["cell_id"]))

    lookup = build_citation_lookup(release_dir, tuple(first_cell_by_table.values()))

    docs: dict[str, None] = {}
    tables: dict[str, None] = {}
    for table_id in ordered_table_ids:  # <-- thứ tự rank, KHÔNG phải thứ tự frame
        cell_id = first_cell_by_table.get(table_id)
        if cell_id is None:
            continue  # bảng đã retrieve nhưng không còn cell số nào (hiếm, an toàn để bỏ qua)
        provenance = lookup[cell_id]
        report_id = PurePosixPath(str(provenance["doc_relative_path"])).name
        if report_id.endswith(".txt"):
            report_id = report_id[: -len(".txt")]
        docs.setdefault(report_id, None)
        tables.setdefault(f"{report_id}|{provenance['source_line_start']}", None)
    return tuple(docs), tuple(tables)
```

Sửa call site (dòng ~510) thành:

```python
    relevant_docs, relevant_tables = _relevant_docs_and_tables(retrieved, release_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/submission/exporter.py tests/unit/submission/test_submission_exporter.py
git commit -m "fix(submission): report retrieved tables in rank order, not just the ones used to compute

Retrieval is scored independently of answer accuracy across 8 metrics
(TABLES/DOCS x Precision/Recall/F2/MRR5). Deriving relevant_tables from
compiled.evidence tied it to the answering path, so 826 unanswered questions
reported garbage. The fix also preserves retrieval-rank order end to end --
build_cell_frame's ORDER BY table_id would have silently broken MRR5 while
leaving F2 unchanged."
```

---

## Task 7: Đo F2 macro và MRR5 theo k

Dashboard chấm 8 chỉ số truy hồi: `TABLES/DOCS × Precision/Recall/F2/MRR5`. F2 coi
`relevant_tables` là tập không thứ tự; MRR5 đo vị trí kết quả đúng đầu tiên trong top-5
— hai chỉ số có thể tối ưu ở `k` khác nhau (F2 phạt precision khi `k` lớn, MRR5 gần như
không đổi khi `k > 5`). Cần đo cả hai để chọn `k`, không chỉ F2.

**Files:**
- Create: `src/financial_report_qa/retrieval/retrieval_scoring.py`
- Test: `tests/unit/retrieval/test_retrieval_scoring.py`

**Interfaces:**
- Consumes: gold tables từ `data/qa/week1_pilot_422df141c935/expected-tables.csv`
- Produces:
  - `f2_score(predicted: Sequence[str], gold: Sequence[str]) -> float`
  - `mrr5_score(predicted_ranked: Sequence[str], gold: Sequence[str]) -> float`
  - `macro_f2(predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]) -> float`
  - `macro_mrr5(predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]) -> float`
  - `sweep_k(ranked, gold, ks=...) -> dict[int, dict[str, float]]` — mỗi `k` map tới `{"f2": ..., "mrr5": ...}`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/retrieval/test_retrieval_scoring.py`:

```python
"""F2 macro và MRR5: công thức chấm truy hồi thật của dashboard (8 cột
TABLES/DOCS x Precision/Recall/F2/MRR5), không chỉ F2 tóm tắt trong PDF thể lệ."""

from __future__ import annotations

import pytest

from financial_report_qa.retrieval.retrieval_scoring import (
    f2_score,
    macro_f2,
    macro_mrr5,
    mrr5_score,
    sweep_k,
)


def test_perfect_retrieval_scores_f2_one() -> None:
    assert f2_score(["a"], ["a"]) == pytest.approx(1.0)


def test_no_overlap_scores_f2_zero() -> None:
    assert f2_score(["b"], ["a"]) == pytest.approx(0.0)


def test_empty_prediction_scores_f2_zero() -> None:
    assert f2_score([], ["a"]) == pytest.approx(0.0)


def test_f2_weights_recall_four_times_precision() -> None:
    """1 gold, dự đoán 10 bảng có chứa gold: P=0.1, R=1.0
    F2 = 5*0.1*1.0 / (4*0.1 + 1.0) = 0.5/1.4 = 0.357"""
    predicted = ["a"] + [f"x{i}" for i in range(9)]
    assert f2_score(predicted, ["a"]) == pytest.approx(0.5 / 1.4, abs=1e-6)


def test_f2_duplicates_do_not_inflate_score() -> None:
    assert f2_score(["a", "a"], ["a"]) == pytest.approx(1.0)


def test_mrr5_scores_one_when_gold_is_first() -> None:
    assert mrr5_score(["a", "x", "y"], ["a"]) == pytest.approx(1.0)


def test_mrr5_scores_by_reciprocal_rank() -> None:
    """Gold ở vị trí 3 (1-indexed) -> 1/3."""
    assert mrr5_score(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


def test_mrr5_ignores_hits_beyond_top_five() -> None:
    predicted = ["x1", "x2", "x3", "x4", "x5", "a"]
    assert mrr5_score(predicted, ["a"]) == pytest.approx(0.0)


def test_mrr5_uses_best_rank_when_multiple_gold_present() -> None:
    predicted = ["x", "a", "b"]  # a ở rank 2, b không xuất hiện
    assert mrr5_score(predicted, ["a", "b"]) == pytest.approx(0.5)


def test_mrr5_order_sensitive_unlike_f2() -> None:
    """Cùng TẬP bảng, khác THỨ TỰ -- F2 phải bằng nhau, MRR5 phải khác nhau.
    Đây là bất biến mà bug 'dùng build_cell_frame làm nguồn thứ tự' (Task 6)
    sẽ vi phạm mà không bị F2 phát hiện."""
    gold = ["a"]
    first_ranked = ["a", "b", "c"]
    last_ranked = ["c", "b", "a"]
    assert f2_score(first_ranked, gold) == pytest.approx(f2_score(last_ranked, gold))
    assert mrr5_score(first_ranked, gold) != pytest.approx(mrr5_score(last_ranked, gold))


def test_macro_f2_averages_across_questions() -> None:
    predictions = {1: ["a"], 2: ["b"]}
    gold = {1: ["a"], 2: ["x"]}
    assert macro_f2(predictions, gold) == pytest.approx(0.5)


def test_macro_mrr5_averages_across_questions() -> None:
    predictions = {1: ["a", "x"], 2: ["x", "b"]}
    gold = {1: ["a"], 2: ["b"]}
    assert macro_mrr5(predictions, gold) == pytest.approx((1.0 + 0.5) / 2)


def test_sweep_k_returns_both_metrics_per_k() -> None:
    ranked = {1: ["a", "b", "c", "d", "e", "f"]}
    gold = {1: ["a"]}
    result = sweep_k(ranked, gold, ks=(1, 5))
    assert set(result[1]) == {"f2", "mrr5"}
    assert result[1]["f2"] == pytest.approx(1.0)
    assert result[1]["mrr5"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_retrieval_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'financial_report_qa.retrieval.retrieval_scoring'`

- [ ] **Step 3: Write the implementation**

Tạo `src/financial_report_qa/retrieval/retrieval_scoring.py`:

```python
"""Đo truy hồi theo đúng công thức dashboard chấm điểm công khai của cuộc thi.

Dashboard có 10 cột: `EXECUTION ACCURACY, TABLES F2-MACRO, DOCS F2-MACRO,
TABLES PRECISION, TABLES RECALL, TABLES MRR5, DOCS PRECISION, DOCS RECALL,
DOCS MRR5, ANSWER ACCURACY`. Module này cài hai công thức chung cho cả nhánh
TABLES và nhánh DOCS (gọi hai lần với hai tập gold khác nhau):

    Precision = |đúng| / |đã truy hồi|
    Recall    = |đúng| / |liên quan|
    F2        = 5*P*R / (4*P + R)                    -- không phân biệt thứ tự
    MRR5      = 1 / rank(kết quả đúng đầu tiên trong top-5), 0 nếu không có

F2 nghiêng recall gấp 4 lần precision, nhưng precision giảm theo 1/k trong khi
recall bão hoà -- k lớn thường lợi cho recall, hại cho F2. MRR5 chỉ quan tâm
vị trí kết quả đúng ĐẦU TIÊN trong 5 phần tử đầu, nên gần như không đổi khi
k > 5. Hai chỉ số có thể đòi hỏi k khác nhau; `sweep_k` đo cả hai để người
dùng tự cân bằng, không chỉ tối ưu một chỉ số.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_MRR_DEPTH = 5


def f2_score(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """F2 cho một truy vấn. Không phân biệt thứ tự; trùng lặp trong
    `predicted` không được tính hai lần."""
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not gold_set or not predicted_set:
        return 0.0
    hits = len(predicted_set & gold_set)
    if hits == 0:
        return 0.0
    precision = hits / len(predicted_set)
    recall = hits / len(gold_set)
    return 5 * precision * recall / (4 * precision + recall)


def mrr5_score(predicted_ranked: Sequence[str], gold: Sequence[str]) -> float:
    """MRR@5 cho một truy vấn: 1/hạng của kết quả đúng ĐẦU TIÊN trong 5 phần
    tử đầu của `predicted_ranked` (hạng 1-indexed). 0 nếu không có kết quả
    đúng nào trong top-5.

    Khác `f2_score`: thứ tự của `predicted_ranked` quyết định điểm số. Đầu
    vào PHẢI đã ở đúng thứ tự retrieval-rank (điểm cao nhất trước) -- một
    danh sách đúng tập nhưng sai thứ tự cho điểm sai mà không có cách nào
    phát hiện qua F2.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, table_id in enumerate(predicted_ranked[:_MRR_DEPTH], start=1):
        if table_id in gold_set:
            return 1.0 / rank
    return 0.0


def macro_f2(
    predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]
) -> float:
    """Trung bình F2 trên mọi câu có gold. Câu thiếu dự đoán tính 0."""
    if not gold:
        return 0.0
    total = sum(f2_score(predictions.get(qid, ()), tables) for qid, tables in gold.items())
    return total / len(gold)


def macro_mrr5(
    predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]
) -> float:
    """Trung bình MRR@5 trên mọi câu có gold. Câu thiếu dự đoán tính 0."""
    if not gold:
        return 0.0
    total = sum(mrr5_score(predictions.get(qid, ()), tables) for qid, tables in gold.items())
    return total / len(gold)


def sweep_k(
    ranked: Mapping[int, Sequence[str]],
    gold: Mapping[int, Sequence[str]],
    ks: Sequence[int] = (1, 2, 3, 5, 8, 10, 15),
) -> dict[int, dict[str, float]]:
    """F2 và MRR5 macro khi cắt danh sách đã xếp hạng ở từng k.

    `ranked` phải đã ở đúng thứ tự retrieval-rank cho mỗi câu -- kết quả
    MRR5 vô nghĩa nếu không.
    """
    result: dict[int, dict[str, float]] = {}
    for k in ks:
        truncated = {qid: list(tables)[:k] for qid, tables in ranked.items()}
        result[k] = {
            "f2": macro_f2(truncated, gold),
            "mrr5": macro_mrr5(truncated, gold),
        }
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/retrieval/test_retrieval_scoring.py -v`
Expected: PASS — 13 passed

- [ ] **Step 5: Xác minh gold có đủ độ phủ để sweep**

```bash
.venv/Scripts/python.exe -c "
import csv, collections
rows = list(csv.DictReader(open('data/qa/week1_pilot_422df141c935/expected-tables.csv', encoding='utf-8')))
print('dòng:', len(rows))
print('cột:', rows[0].keys() if rows else 'rỗng')
qids = {r.get('question_id') for r in rows}
print('số câu có gold:', len(qids))
print('số bảng gold/câu:', collections.Counter(collections.Counter(r.get('question_id') for r in rows).values()).most_common(5))
"
```

Nếu số câu có gold < 100, dừng lại: cần gán nhãn tay ~100 câu trước khi sweep có ý nghĩa (đã ghi trong design §9 rủi ro).

- [ ] **Step 6: Chạy sweep thật và ghi lại k đã chọn**

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import csv, json
from financial_report_qa.retrieval.retrieval_scoring import sweep_k

rows = list(csv.DictReader(open('data/qa/week1_pilot_422df141c935/expected-tables.csv', encoding='utf-8')))
gold: dict[int, list[str]] = {}
for r in rows:
    qid = int(r['question_id'])
    gold.setdefault(qid, []).append(r['table_id'])

# ranked: cần chạy retrieve_candidate_table_ids(question, service, k=15) cho
# mỗi câu có gold rồi map question_id -> tuple(table_id đã xếp hạng).
# Điền vào đây bằng service thật trước khi chạy.
ranked: dict[int, list[str]] = {}  # TODO: nạp từ retrieval service thật

result = sweep_k(ranked, gold)
print(json.dumps(result, indent=2))
"
```

Ghi lại `k` cân bằng tốt nhất giữa `f2` và `mrr5` vào `configs/base.yaml` (`retrieval.final_top_k`).
Đây là bước thực nghiệm — không có "expected output" cố định vì phụ thuộc dữ liệu gold thật.

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/retrieval/retrieval_scoring.py tests/unit/retrieval/test_retrieval_scoring.py
git commit -m "feat(retrieval): add F2 macro + MRR5 scorer matching the live scoring dashboard

The rules PDF only summarizes F2; the live dashboard scores 8 retrieval
metrics (TABLES/DOCS x Precision/Recall/F2/MRR5). MRR5 is order-sensitive --
mrr5_score requires its input already in retrieval-rank order, unlike f2_score
which treats predictions as a set."
```

---

## Task 8: Chạy full export và xác nhận đạt mục tiêu

- [ ] **Step 1: Chạy full export**

```bash
.venv/Scripts/python.exe _run_full_export.py
```

Expected: exit code 0. Nếu exit code 2, mở `artifacts/evaluations/v2gaps_full/compliance-violations.json` và sửa nguyên nhân — **không** nới lỏng linter.

- [ ] **Step 2: Xác nhận baseline đã đảo chiều**

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import zipfile, json, io, csv, collections
z = zipfile.ZipFile('artifacts/submissions/v2gaps_full.zip')
items = json.loads(z.read('submission.json'))
sizes = collections.Counter()
for it in items:
    rows = list(csv.reader(io.StringIO(z.read(it['evidence'][0]['csv_path']).decode('utf-8'))))
    sizes[len(rows) - 1] += 1
print('CSV 1 dòng:', sizes.get(1, 0), '(mục tiêu: 0)')
print('số câu:', len(items), '(mục tiêu: 1012)')
print('trung bình bảng/câu:', sum(len(i['relevant_tables']) for i in items) / len(items))
"
```

Expected: `CSV 1 dòng: 0`, `số câu: 1012`.

- [ ] **Step 3: Xác nhận `relevant_tables` giữ đúng thứ tự retrieval-rank trên toàn bộ export**

```bash
.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0,'src')
import json, zipfile
from pathlib import Path
from financial_report_qa.retrieval.live_query import retrieve_candidate_table_ids
from financial_report_qa.retrieval.service import RetrievalService  # dùng đúng service đã cấu hình cho export

z = zipfile.ZipFile('artifacts/submissions/v2gaps_full.zip')
items = json.loads(z.read('submission.json'))
release_dir = Path('data/processed/release_v2_422df141c935')

# Kiểm tra một mẫu nhỏ (ví dụ 20 câu) thay vì toàn bộ 1012 để chạy nhanh:
# với mỗi câu, phần đầu report_id trong relevant_tables[0] phải khớp bảng
# hạng 1 mà retrieval trả về khi chạy lại trực tiếp.
mismatches = 0
for item in items[:20]:
    if not item['relevant_tables']:
        continue
    first_report_id = item['relevant_tables'][0].split('|', 1)[0]
    # so khớp thủ công theo report_id đã biết -- xem log export để đối chiếu
    # thứ tự thật; đây là bước xác minh thủ công, không phải assert tự động.
    print(item['id'], '->', first_report_id)
print('kiểm tra xong: đối chiếu thủ công với log export dòng', 'retrieval-rank order')
"
```

Đây là bước xác minh thủ công (đối chiếu bằng mắt với log retrieval), không phải assert tự động — assert tự động về thứ tự rank đã có ở Task 6 Step 1 (`test_relevant_tables_preserve_retrieval_rank_order`) và chạy trong CI mỗi lần. Bước này chỉ để phát hiện regression không lường trước được khi chạy trên toàn bộ 1012 câu thật.

- [ ] **Step 4: Commit kết quả đo**

```bash
git add artifacts/evaluations/v2gaps_full/
git commit -m "chore(eval): record post-compliance-fix export baseline"
```

---

## Self-Review

**Spec coverage:**

| Mục design | Task |
|---|---|
| §5.1 BI-1 (CSV là lát cắt bảng thật) | Task 3, Task 4 |
| §5.1 BI-2 (giữ trọn bảng nguồn) | Task 4 |
| §5.1 BI-3 (phép tính trong query) | Đã thoả sẵn — `render_pandas_query` sinh `(end - start)/abs(start)`; Task 1 `C4` canh gác hồi quy |
| §5.1 BI-4 (backstop không tổng hợp) | Task 4 |
| §5.2 (sửa predicate) | Task 2 (thêm cột) + Task 4 (`_uniquely_addressable_row`) |
| §5.3 (linter C1–C7) | Task 1, Task 5 |
| §5.4 (kiểm chứng) | Task 1 Step 5, Task 8 |
| §2.3bis (10 cột dashboard, MRR5 order-sensitive) | Task 6 (giữ thứ tự), Task 7 (`mrr5_score`) |
| §6.1 (tách relevant_tables, giữ thứ tự rank) | Task 6 |
| §6.2 (sweep k theo F2 và MRR5) | Task 7 |
| §6.3 (Qwen3-Embedding) | **Ngoài phạm vi plan này** — cần plan riêng, phụ thuộc Colab |
| §7 (P2 answer accuracy) | **Ngoài phạm vi plan này** — plan riêng sau khi P0/P1 land |

**Ghi chú:** §6.3 và §7 cố tình để lại. Plan này khép kín ở milestone "submission hợp lệ + thu hồi điểm truy hồi", tự nó đã là phần mềm chạy được và đo được.

**Type consistency:** `CsvRow = Mapping[str, object]` dùng nhất quán ở Task 2/3/4. Schema 9 cột `_CSV_COLUMNS` định nghĩa một lần ở Task 2, Task 4 ghi theo đúng thứ tự đó. `ComplianceViolation` cùng chữ ký ở Task 1 và Task 5. Module Task 7 đổi tên từ `f2_sweep.py` sang `retrieval_scoring.py` (phạm vi giờ gồm cả MRR5, tên module cũ không còn phản ánh đúng nội dung).

**Bất biến xuyên suốt Task 6/7 cần người thực thi không phá:** mọi hàm nhận danh sách `table_id`/`predicted` với giả định "đã ở đúng thứ tự retrieval-rank" — `_relevant_docs_and_tables`, `mrr5_score`, `sweep_k`. Bất kỳ chỗ nào chèn thêm bước sort/dedup bằng `set()` hoặc duyệt qua kết quả DuckDB không có `ORDER BY` khớp rank sẽ âm thầm phá MRR5 mà F2 không phát hiện được — đây là lớp lỗi cụ thể mà Task 6 Step 1 (`test_relevant_tables_preserve_retrieval_rank_order`) và Task 7 Step 1 (`test_mrr5_order_sensitive_unlike_f2`) được viết để bắt.

**Điểm cần người duyệt chú ý:** Task 5 Step 1 để `submission_cli.main([...])` chưa đầy đủ tham số — người thực thi phải copy bộ tham số từ test export sẵn có trong cùng file. Đây là chỗ duy nhất plan không đưa code hoàn chỉnh, vì tham số phụ thuộc fixture hiện có mà plan không nên đoán.
