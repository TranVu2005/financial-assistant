# LLM Row Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay bước chọn dòng rule-based bằng LLM (Qwen3-8B, batch offline) cho cả 1012 câu, và cho phép tie-break tất định thay vì abstain ở ba điểm nhập nhằng của `locate()`.

**Architecture:** Hai nhánh độc lập. Nhánh A (Task 1–2) bật tie-break trong `execution/locator.py` — thuần local, không cần GPU, land được ngay. Nhánh B (Task 3–7) đưa LLM vào vai trò chọn dòng: sinh batch ứng viên từ `row_fusion` đã có, chạy Qwen3-8B offline trên Colab, đọc file quyết định về và biến nó thành `MetricSelector` position-bound. LLM chỉ trả về **một chỉ số nguyên**, không bao giờ trả giá trị đáp án — compiler/sandbox/compliance linter giữ nguyên.

**Tech Stack:** Python 3.11, pandas, duckdb, pydantic v2, pytest, uv, Qwen3-8B q4_K_M qua Colab T4

## Global Constraints

- Mọi model dùng trong hệ thống phải **< 14B tham số**. Qwen3-8B = 8.2B ✓. **Không** dùng Qwen3-14B/Qwen2.5-14B-Instruct (~14.7B, vi phạm).
- **Không re-ingest corpus.** Giữ nguyên `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
- `SubmissionItem` giữ đúng 7 trường, `extra="forbid"`. Plan này **không** đổi contract JSON.
- `ExecutionSettings` có `model_config = ConfigDict(extra="forbid")` — thêm khóa YAML mới **bắt buộc** phải thêm field tương ứng vào model, nếu không sẽ ném lỗi validate.
- **`TableId` bị ràng buộc định dạng** `^tbl_[0-9a-f]{64}$` (`retrieval/contracts.py:14`). Mọi `MetricSelector`, `RowFusedCandidate`, `RowMetadata` trong test **phải** dùng id hợp lệ (vd. `"tbl_" + "a" * 64`); một chuỗi như `"t1"` sẽ ném `ValidationError`. DataFrame thuần pandas không qua pydantic thì không bị ràng buộc này.
- LLM **không bao giờ** được trả về giá trị đáp án. File quyết định chỉ mang `{question_id, chosen_index}`.
- Compliance linter (C1–C7) phải vẫn báo **0 vi phạm** sau mọi thay đổi.
- Release dir: `data/processed/release_v2_422df141c935`
- Chạy test bằng: `.venv/Scripts/python.exe -m pytest`
- Ruff giới hạn dòng **100 ký tự**.

## Baseline đo được (2026-08-22, `artifacts/evaluations/v2gaps_full/submission-export-422df141c935.json`)

| Chỉ số | Giá trị |
|---|---:|
| `answered_count` | **53** |
| `metric_not_found` | 491 |
| `evidence_frame_replay_mismatch` | 133 |
| `cell_ambiguous` | 50 |
| `period_unresolved` | 49 |
| `unit_missing` | 31 |

**Tiêu chí thành công:** `answered_count > 53`; ba mã `cell_ambiguous`/`period_unresolved`/`unit_missing` giảm về gần 0; compliance vẫn 0 vi phạm.

---

## File Structure

| File | Trách nhiệm | Thao tác |
|---|---|---|
| `src/financial_report_qa/execution/tiebreak.py` | Ba hàm thuần chọn ứng viên tất định. Không I/O, không phụ thuộc config. | **Tạo** (Task 1) |
| `tests/unit/execution/test_tiebreak.py` | Unit test ba hàm trên | **Tạo** (Task 1) |
| `src/financial_report_qa/execution/locator.py` | Gọi tie-break tại 3 điểm abstain | **Sửa** (Task 2) |
| `src/financial_report_qa/execution/compiler.py` | Luồng cờ từ `ExecutionSettings` xuống `locate()` | **Sửa** (Task 2) |
| `src/financial_report_qa/core/config.py` | Thêm field `resolve_ambiguity_by_priority` | **Sửa** (Task 2) |
| `configs/submission_maximize_correct.yaml` | Bật cờ cho bản nộp | **Sửa** (Task 2) |
| `src/financial_report_qa/planning/row_choice_batch.py` | Dựng payload batch từ `RowFusedCandidate` | **Tạo** (Task 3) |
| `src/financial_report_qa/submission/cli.py` | Subcommand `row-batches`; cờ `--row-choice-decisions` | **Sửa** (Task 3, 6) |
| `src/financial_report_qa/planning/row_choice_decision.py` | Đọc quyết định, map index → `MetricSelector` position-bound | **Tạo** (Task 4) |
| `src/financial_report_qa/planning/cell_grounding.py` | Attempt 0 = quyết định LLM; xóa nhánh rule dictionary | **Sửa** (Task 5) |
| `notebooks/colab_row_choice_qwen3_8b.ipynb` | Notebook Colab chạy Qwen3-8B | **Tạo** (Task 7) |

---

## Task 1: Ba hàm tie-break thuần

Viết trước, chưa gắn vào `locate()`, để Task 2 có sẵn công cụ đã được test.

**Files:**
- Create: `src/financial_report_qa/execution/tiebreak.py`
- Test: `tests/unit/execution/test_tiebreak.py`

**Interfaces:**
- Consumes: chỉ `pandas`. Frame đầu vào là frame do `build_cell_frame` sinh, có các cột `table_id, row_idx, col_idx, value, unit, period, statutory_code`.
- Produces (Task 2 gọi cả ba):
  - `nearest_period_rows(metric_rows: pd.DataFrame, period: int) -> pd.DataFrame`
  - `dominant_value_rows(rows: pd.DataFrame) -> pd.DataFrame`
  - `infer_unit_from_table(frame: pd.DataFrame, table_id: str) -> str | None`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/execution/test_tiebreak.py`:

```python
"""Tie-break tất định thay cho abstain (design 2026-08-22 §6).

Mỗi hàm phải tất định: cùng input luôn cho cùng output, kể cả khi có hòa.
Đó là điều kiện để giải trình được lựa chọn, khác với `.iloc[0]` tuỳ ý.
"""

from __future__ import annotations

import pandas as pd

from financial_report_qa.execution.tiebreak import (
    dominant_value_rows,
    infer_unit_from_table,
    nearest_period_rows,
)


def _rows(records: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    frame["period"] = frame["period"].astype("Int64")
    return frame


def _row(
    *,
    period: int | None = 2023,
    value: float = 100.0,
    unit: str | None = "VND",
    table_id: str = "t1",
    row_idx: int = 1,
    col_idx: int = 1,
    statutory_code: str | None = None,
) -> dict[str, object]:
    return {
        "table_id": table_id,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "period": period,
        "value": value,
        "unit": unit,
        "statutory_code": statutory_code,
    }


def test_nearest_period_picks_closest() -> None:
    rows = _rows([_row(period=2020, value=1.0), _row(period=2023, value=2.0)])
    picked = nearest_period_rows(rows, 2022)
    assert picked["period"].unique().tolist() == [2023]


def test_nearest_period_tie_prefers_the_later_period() -> None:
    """2021 và 2023 cách đều 2022 -- phải chọn 2023 (muộn hơn), tất định."""
    rows = _rows([_row(period=2021, value=1.0), _row(period=2023, value=2.0)])
    picked = nearest_period_rows(rows, 2022)
    assert picked["period"].unique().tolist() == [2023]


def test_nearest_period_returns_empty_when_no_period_at_all() -> None:
    rows = _rows([_row(period=None, value=1.0)])
    assert nearest_period_rows(rows, 2022).empty


def test_dominant_value_keeps_single_value_untouched() -> None:
    rows = _rows([_row(value=5.0, col_idx=1), _row(value=5.0, col_idx=2)])
    assert len(dominant_value_rows(rows)) == 2


def test_dominant_value_picks_the_most_frequent_pair() -> None:
    rows = _rows(
        [
            _row(value=5.0, col_idx=1),
            _row(value=5.0, col_idx=2),
            _row(value=9.0, col_idx=3),
        ]
    )
    picked = dominant_value_rows(rows)
    assert picked["value"].unique().tolist() == [5.0]


def test_dominant_value_tie_breaks_on_position_deterministically() -> None:
    """Hai giá trị cùng tần suất -- phải chọn cái xuất hiện trước theo
    (table_id, row_idx, col_idx), và phải cho kết quả giống nhau bất kể
    thứ tự dòng trong input."""
    forward = _rows([_row(value=5.0, col_idx=1), _row(value=9.0, col_idx=2)])
    reverse = _rows([_row(value=9.0, col_idx=2), _row(value=5.0, col_idx=1)])
    assert dominant_value_rows(forward)["value"].unique().tolist() == [5.0]
    assert dominant_value_rows(reverse)["value"].unique().tolist() == [5.0]


def test_infer_unit_returns_most_common_unit_of_that_table() -> None:
    frame = _rows(
        [
            _row(table_id="t1", col_idx=1, unit="VND"),
            _row(table_id="t1", col_idx=2, unit="VND"),
            _row(table_id="t1", col_idx=3, unit="trieu_VND"),
            _row(table_id="t2", col_idx=1, unit="USD"),
        ]
    )
    assert infer_unit_from_table(frame, "t1") == "VND"


def test_infer_unit_returns_none_when_table_has_no_unit() -> None:
    frame = _rows([_row(table_id="t1", unit=None)])
    assert infer_unit_from_table(frame, "t1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/execution/test_tiebreak.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'financial_report_qa.execution.tiebreak'`

- [ ] **Step 3: Write the implementation**

Tạo `src/financial_report_qa/execution/tiebreak.py`:

```python
"""Chọn ứng viên tất định khi `locate()` gặp nhập nhằng.

Thiết kế 2026-08-22 §6. Answer Accuracy của cuộc thi tính theo
correct/**total**, nên một câu bỏ trống và một câu trả lời sai đều được 0
điểm -- abstain không mua được gì. Ba hàm ở đây biến ba điểm abstain của
`locator.py` thành lựa chọn có luật, giải trình được.

Mọi hàm phải **tất định**: cùng một tập dòng phải luôn cho cùng một kết quả
bất kể thứ tự dòng trong DataFrame đầu vào. Đó là khác biệt giữa "chọn theo
luật" và `.iloc[0]` tuỳ ý mà thiết kế này thay thế.
"""

from __future__ import annotations

import pandas as pd

_POSITION_COLUMNS = ["table_id", "row_idx", "col_idx"]


def nearest_period_rows(metric_rows: pd.DataFrame, period: int) -> pd.DataFrame:
    """Các dòng ở kỳ gần `period` nhất. Hòa thì ưu tiên kỳ **muộn hơn**.

    Kỳ muộn hơn được ưu tiên vì câu hỏi tài chính thường hỏi số liệu mới
    nhất; khi hệ thống đã không khớp đúng kỳ, đoán về phía gần hiện tại là
    lựa chọn ít sai hơn.
    """
    available = metric_rows["period"].dropna().unique()
    if len(available) == 0:
        return metric_rows.iloc[0:0]
    best = min(available, key=lambda value: (abs(int(value) - period), -int(value)))
    return metric_rows[metric_rows["period"] == best]


def dominant_value_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Thu hẹp một tập dòng xung đột về đúng một cặp `(value, unit)`.

    Chọn cặp xuất hiện nhiều nhất; hòa thì chọn cặp có vị trí
    `(table_id, row_idx, col_idx)` nhỏ nhất. Sắp xếp trước khi gom nhóm để
    kết quả không phụ thuộc thứ tự dòng đầu vào.
    """
    if rows["value"].nunique() <= 1:
        return rows
    ordered = rows.sort_values(_POSITION_COLUMNS, kind="stable")
    counts = ordered.groupby(["value", "unit"], dropna=False, sort=False).size()
    value, unit = counts.idxmax()
    matches = ordered["value"] == value
    matches &= ordered["unit"].isna() if pd.isna(unit) else ordered["unit"] == unit
    return ordered[matches]


def infer_unit_from_table(frame: pd.DataFrame, table_id: str) -> str | None:
    """Đơn vị phổ biến nhất trong cùng bảng, hoặc `None` nếu bảng không ghi
    đơn vị ở bất kỳ ô nào.

    Một ô thiếu `unit` gần như luôn là lỗi trích xuất chứ không phải bảng
    thật sự không có đơn vị -- các ô còn lại cùng bảng là bằng chứng tốt
    nhất sẵn có. Hòa thì chọn theo thứ tự chữ cái để tất định.
    """
    same_table = frame[frame["table_id"] == table_id]
    units = same_table["unit"].dropna()
    if units.empty:
        return None
    return str(units.mode().sort_values().iloc[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/execution/test_tiebreak.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/execution/tiebreak.py tests/unit/execution/test_tiebreak.py
git commit -m "feat(execution): add deterministic tie-break helpers for ambiguous cells"
```

---

## Task 2: Gắn tie-break vào `locate()` và luồng cờ config

`locate()` (`execution/locator.py:156`) có đúng ba điểm abstain sau khi dòng đã được định vị: `period_unresolved` (dòng 186), `cell_ambiguous` (dòng 209), `unit_missing` (dòng 224). `_prefer_statutory_rows` (dòng 116) **đã tồn tại** nhưng bị khóa sau `prefer_statutory_rows: bool = False` mà không caller nào bật.

**Cạm bẫy bắt buộc tránh:** `compiler.py` gọi `locate()` ở **10 chỗ** (dòng 237–339), nhiều dòng đã dài ~97 ký tự. Thêm một keyword vào từng chỗ sẽ vượt giới hạn ruff 100 ký tự. Dùng hàm cục bộ trong `_dispatch` như Step 5 mô tả.

**Files:**
- Modify: `src/financial_report_qa/core/config.py:48-72` (`ExecutionSettings`)
- Modify: `src/financial_report_qa/execution/locator.py:156-231` (`locate`)
- Modify: `src/financial_report_qa/execution/compiler.py:110-171, 230-345`
- Modify: `configs/submission_maximize_correct.yaml`
- Test: `tests/unit/execution/test_locator.py`

**Interfaces:**
- Consumes: `nearest_period_rows`, `dominant_value_rows`, `infer_unit_from_table` (Task 1)
- Produces: `locate(..., resolve_ambiguity_by_priority: bool = False)`; `ExecutionSettings.resolve_ambiguity_by_priority: bool = False`

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/execution/test_locator.py`:

```python
def test_period_unresolved_falls_back_to_nearest_period_when_enabled() -> None:
    """design §6: kỳ gần nhất thay vì abstain."""
    import pandas as pd

    from financial_report_qa.execution.locator import locate
    from financial_report_qa.planning.plan_contracts import MetricSelector

    table_id = "tbl_" + "a" * 64  # TableId bị ràng buộc ^tbl_[0-9a-f]{64}$
    frame = pd.DataFrame(
        [
            {
                "table_id": table_id, "row_idx": 1, "col_idx": 1, "company_code": "VNM",
                "row_label_raw": "Doanh thu", "row_label_canonical": None,
                "column_label": "2023", "period": 2023, "value": 100.0,
                "unit": "VND", "cell_id": "c1", "statutory_code": None,
            }
        ]
    )
    frame["period"] = frame["period"].astype("Int64")
    selector = MetricSelector(raw_text="Doanh thu", table_id=table_id, row_index=1)

    off = locate(frame, selector, 2021)
    assert off.error_code == "period_unresolved"

    on = locate(frame, selector, 2021, resolve_ambiguity_by_priority=True)
    assert on.error_code is None
    assert on.match is not None


def test_unit_missing_infers_unit_from_the_same_table_when_enabled() -> None:
    import pandas as pd

    from financial_report_qa.execution.locator import locate
    from financial_report_qa.planning.plan_contracts import MetricSelector

    table_id = "tbl_" + "a" * 64  # TableId bị ràng buộc ^tbl_[0-9a-f]{64}$

    def _cell(row_idx: int, col_idx: int, unit: str | None, value: float) -> dict[str, object]:
        return {
            "table_id": table_id, "row_idx": row_idx, "col_idx": col_idx,
            "company_code": "VNM", "row_label_raw": "Doanh thu",
            "row_label_canonical": None, "column_label": "2023", "period": 2023,
            "value": value, "unit": unit, "cell_id": f"c{row_idx}{col_idx}",
            "statutory_code": None,
        }

    frame = pd.DataFrame([_cell(1, 1, None, 100.0), _cell(2, 1, "VND", 200.0)])
    frame["period"] = frame["period"].astype("Int64")
    selector = MetricSelector(raw_text="Doanh thu", table_id=table_id, row_index=1)

    off = locate(frame, selector, 2023)
    assert off.error_code == "unit_missing"

    on = locate(frame, selector, 2023, resolve_ambiguity_by_priority=True)
    assert on.error_code is None
    assert on.match is not None


def test_execution_settings_accepts_the_new_flag() -> None:
    """ExecutionSettings có extra='forbid' -- khóa YAML mới bắt buộc phải có
    field tương ứng, nếu không mọi config bật cờ sẽ ném lỗi validate."""
    from financial_report_qa.core.config import ExecutionSettings

    settings = ExecutionSettings.model_validate(
        {
            "timeout_seconds": 5.0,
            "max_rows": 100,
            "allow_operations": ("lookup",),
            "resolve_ambiguity_by_priority": True,
        }
    )
    assert settings.resolve_ambiguity_by_priority is True


def test_execution_settings_defaults_the_new_flag_to_false() -> None:
    from financial_report_qa.core.config import ExecutionSettings

    settings = ExecutionSettings.model_validate(
        {"timeout_seconds": 5.0, "max_rows": 100, "allow_operations": ("lookup",)}
    )
    assert settings.resolve_ambiguity_by_priority is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/execution/test_locator.py -k "nearest_period or infers_unit or new_flag" -v`
Expected: FAIL — `TypeError: locate() got an unexpected keyword argument 'resolve_ambiguity_by_priority'` và `ValidationError: Extra inputs are not permitted`

- [ ] **Step 3: Thêm field vào `ExecutionSettings`**

Trong `src/financial_report_qa/core/config.py`, ngay sau `default_statement_scope` (dòng 66):

```python
    # Thiết kế 2026-08-22 §6: bật tie-break tất định tại ba điểm abstain của
    # `locate()` (`period_unresolved`, `cell_ambiguous`, `unit_missing`) thay
    # vì bỏ trống. Answer Accuracy tính correct/TOTAL nên abstain và sai đều
    # bằng 0 -- cùng lập luận đã dùng cho `default_statement_scope` trong
    # `configs/submission_maximize_correct.yaml`. Mặc định tắt để phép đo
    # chất lượng nội bộ vẫn giữ tín hiệu precision trung thực.
    resolve_ambiguity_by_priority: bool = False
```

- [ ] **Step 4: Sửa `locate()`**

Trong `src/financial_report_qa/execution/locator.py`, thêm import ở đầu file:

```python
from financial_report_qa.execution.tiebreak import (
    dominant_value_rows,
    infer_unit_from_table,
    nearest_period_rows,
)
```

Đổi chữ ký `locate` (dòng 156-163) thành:

```python
def locate(
    frame: pd.DataFrame,
    selector: MetricSelector,
    period: int,
    *,
    company_code: str | None = None,
    prefer_statutory_rows: bool = False,
    resolve_ambiguity_by_priority: bool = False,
) -> LocateResult:
```

Thay khối `period_unresolved` (dòng 184-192) thành:

```python
    period_rows = metric_rows[metric_rows["period"] == period]
    if period_rows.empty and resolve_ambiguity_by_priority:
        period_rows = nearest_period_rows(metric_rows, period)
    if period_rows.empty:
        return LocateResult(
            match=None,
            error_code="period_unresolved",
            error_message=(
                f"metric '{_selector_label(selector)}' has no cell resolved to period {period}"
            ),
        )
```

Thay khối `cell_ambiguous` (dòng 194-216) thành:

```python
    if prefer_statutory_rows or resolve_ambiguity_by_priority:
        period_rows = _prefer_statutory_rows(period_rows)
    distinct = period_rows.drop_duplicates(subset=["value", "unit"])
    known_units = distinct["unit"].dropna().drop_duplicates()
    if len(distinct) > 1:
        distinct_values = distinct["value"].drop_duplicates()
        # ADR 0010 decision C1: `(X, None)` và `(X, "VND")` không phải xung
        # đột thật khi mọi dòng đồng ý về giá trị -- NULL chỉ nghĩa là một ô
        # vật lý không ghi đơn vị. Chỉ >=2 đơn vị ĐÃ BIẾT mà khác nhau mới là
        # xung đột thật (đo được: 859/868 dương tính giả là ca NULL này).
        if not (len(distinct_values) == 1 and len(known_units) <= 1):
            if resolve_ambiguity_by_priority:
                period_rows = dominant_value_rows(period_rows)
                distinct = period_rows.drop_duplicates(subset=["value", "unit"])
                known_units = distinct["unit"].dropna().drop_duplicates()
            if len(distinct) > 1:
                candidates = ", ".join(
                    f"{row.value} {row.unit} (cell_id={row.cell_id})"
                    for row in distinct.itertuples()
                )
                return LocateResult(
                    match=None,
                    error_code="cell_ambiguous",
                    error_message=(
                        f"metric '{_selector_label(selector)}' at period {period} "
                        f"has conflicting values: {candidates}"
                    ),
                )
```

Thay khối `unit_missing` (dòng 218-230) thành:

```python
    resolved_unit = known_units.iloc[0] if len(known_units) == 1 else distinct["unit"].iloc[0]
    if pd.isna(resolved_unit) and resolve_ambiguity_by_priority:
        inferred = infer_unit_from_table(frame, str(period_rows["table_id"].iloc[0]))
        if inferred is not None:
            resolved_unit = inferred
    if pd.isna(resolved_unit):
        # ADR 0009 decision C1: thiếu đơn vị là lỗi khác với đơn vị không
        # tương thích. `str(resolved_unit)` ở đây sẽ ra 'None' hoặc 'nan' --
        # không cái nào là CanonicalUnit thật.
        return LocateResult(
            match=None,
            error_code="unit_missing",
            error_message=(
                f"metric '{_selector_label(selector)}' at period {period} has no recorded unit"
            ),
        )
```

- [ ] **Step 5: Luồng cờ qua `compiler.py`**

Trong `src/financial_report_qa/execution/compiler.py`, đổi chữ ký `_dispatch` (dòng 230-232):

```python
def _dispatch(
    plan: FinancialQueryPlan,
    frame: pd.DataFrame,
    period: int | None,
    *,
    resolve_ambiguity_by_priority: bool = False,
) -> tuple[tuple[CellMatch, ...], list[dict[str, object]], Decimal, CanonicalUnit]:
    company = plan.companies[0]

    def _cell(selector: MetricSelector, at_period: int, *, company_code: str) -> CellMatch:
        """Một chỗ duy nhất mang cờ tie-break xuống `locate()`.

        `_dispatch` gọi `locate()` ở 10 nhánh; thêm keyword vào từng chỗ sẽ
        đẩy nhiều dòng vượt giới hạn 100 ký tự của ruff.
        """
        return _require(
            locate(
                frame,
                selector,
                at_period,
                company_code=company_code,
                resolve_ambiguity_by_priority=resolve_ambiguity_by_priority,
            )
        )
```

Thay **mọi** lời gọi `_require(locate(frame, X, Y, company_code=Z))` trong `_dispatch` bằng `_cell(X, Y, company_code=Z)`. Có 10 chỗ ở các dòng 237, 247, 248, 269, 270, 288, 289, 310, 311, 339.

Thêm import `MetricSelector` vào `compiler.py` nếu chưa có:

```python
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
```

Tại chỗ `compile_plan` gọi `_dispatch` (dòng 171):

```python
        evidence, replay_rows, answer, unit = _dispatch(
            plan,
            frame,
            period,
            resolve_ambiguity_by_priority=execution_settings.resolve_ambiguity_by_priority,
        )
```

- [ ] **Step 6: Bật cờ trong overlay nộp bài**

Thêm vào cuối `configs/submission_maximize_correct.yaml`, dưới khóa `execution:` đã có:

```yaml
  # Thiết kế 2026-08-22 §6. Cùng lập luận correct/TOTAL như
  # `default_statement_scope` ở trên: 130 câu (cell_ambiguous 50 +
  # period_unresolved 49 + unit_missing 31) hiện bỏ trống dù hệ thống đã tìm
  # thấy dữ liệu. Bỏ trống và sai đều 0 điểm, nên đoán theo luật tất định
  # (statutory -> kỳ gần nhất -> đơn vị phổ biến) chỉ có thể tăng điểm.
  resolve_ambiguity_by_priority: true
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/execution/ -v`
Expected: PASS. Nếu test cũ nào assert `cell_ambiguous`/`unit_missing` với cờ mặc định, nó vẫn phải pass — cờ mặc định là `False` nên hành vi cũ không đổi.

Run: `.venv/Scripts/python.exe -m ruff check src/financial_report_qa/execution/`
Expected: no new E501.

- [ ] **Step 8: Commit**

```bash
git add src/financial_report_qa/execution/locator.py src/financial_report_qa/execution/compiler.py src/financial_report_qa/core/config.py configs/submission_maximize_correct.yaml tests/unit/execution/test_locator.py
git commit -m "feat(execution): resolve ambiguous cells by priority instead of abstaining

130 questions (cell_ambiguous 50, period_unresolved 49, unit_missing 31)
found their data but refused to commit. Answer Accuracy is correct/TOTAL, so
an abstention and a wrong answer both score zero -- the same argument already
applied to default_statement_scope. Off by default; the submission overlay
turns it on. _prefer_statutory_rows already existed but no caller ever passed
prefer_statutory_rows=True."
```

---

## Task 3: Sinh batch ứng viên cho LLM

`fusion_rows` đã được tính sẵn tại `exporter.py:208` bằng `row_fusion.retrieve_rows(...)`. Task này thêm một subcommand chạy **chỉ** retrieval và dump ứng viên ra JSONL, không chạy planning/execution.

**Files:**
- Create: `src/financial_report_qa/planning/row_choice_batch.py`
- Modify: `src/financial_report_qa/submission/cli.py`
- Test: `tests/unit/planning/test_row_choice_batch.py`

**Interfaces:**
- Consumes: `RowFusedCandidate` (`retrieval/row_fusion_contracts.py:44`), `RowMetadata` (`retrieval/row_documents.py:20`)
- Produces (Task 4 và Task 7 đều dựa vào):
  - `build_batch_payload(question_id: int, question: str, candidates: Sequence[RowFusedCandidate]) -> dict[str, object]`
  - Định dạng JSONL: `{"question_id": int, "question": str, "candidates": [{"index": int, "row_label": str, "row_group_context": str | None, "statement_type": str | None, "table_title": str | None, "periods": list[str], "units": list[str]}]}`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/planning/test_row_choice_batch.py`:

```python
"""Payload batch gửi cho LLM chọn dòng (thiết kế 2026-08-22 §5.3)."""

from __future__ import annotations

from financial_report_qa.planning.row_choice_batch import build_batch_payload
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


# TableId bị ràng buộc ^tbl_[0-9a-f]{64}$ -- "t1" sẽ ném ValidationError.
_TABLE_ID = "tbl_" + "a" * 64


def _candidate(*, rank: int, row_idx: int, label: str) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=_TABLE_ID,
            row_idx=row_idx,
            row_label_raw=label,
            row_group_context_raw="IV. Tài sản ngắn hạn khác",
            statement_type="balance_sheet",
            title="BẢNG CÂN ĐỐI KẾ TOÁN",
            periods=("2023", "2022"),
            units=("VND",),
        ),
        snippet=label,
    )


def test_payload_has_question_and_indexed_candidates() -> None:
    payload = build_batch_payload(
        795,
        "Chi phí trả trước ngắn hạn khác cuối 2023?",
        [_candidate(rank=1, row_idx=3, label="Chi phí trả trước ngắn hạn khác")],
    )
    assert payload["question_id"] == 795
    assert payload["question"] == "Chi phí trả trước ngắn hạn khác cuối 2023?"
    assert payload["candidates"][0]["index"] == 0
    assert payload["candidates"][0]["row_label"] == "Chi phí trả trước ngắn hạn khác"


def test_candidate_indices_are_contiguous_and_follow_rank_order() -> None:
    """Index phải khớp vị trí trong danh sách -- Task 4 map ngược index về
    candidate bằng chính thứ tự này."""
    candidates = [
        _candidate(rank=1, row_idx=3, label="A"),
        _candidate(rank=2, row_idx=7, label="B"),
        _candidate(rank=3, row_idx=9, label="C"),
    ]
    payload = build_batch_payload(1, "câu hỏi", candidates)
    assert [c["index"] for c in payload["candidates"]] == [0, 1, 2]
    assert [c["row_label"] for c in payload["candidates"]] == ["A", "B", "C"]


def test_payload_never_leaks_a_cell_value() -> None:
    """Bất biến của thiết kế: LLM không được thấy giá trị số. Nó chỉ chọn
    dòng; đáp án luôn được tính lại từ CSV trong sandbox."""
    payload = build_batch_payload(
        1, "câu hỏi", [_candidate(rank=1, row_idx=3, label="A")]
    )
    serialized = repr(payload)
    assert "value" not in serialized
    assert "fused_score" not in serialized


def test_empty_candidates_produce_an_empty_list_not_an_error() -> None:
    payload = build_batch_payload(1, "câu hỏi", [])
    assert payload["candidates"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_batch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'financial_report_qa.planning.row_choice_batch'`

- [ ] **Step 3: Write the implementation**

Tạo `src/financial_report_qa/planning/row_choice_batch.py`:

```python
"""Dựng payload batch cho bước LLM chọn dòng (thiết kế 2026-08-22 §5.3).

LLM nhận câu hỏi + danh sách dòng ứng viên đã đánh số, và chỉ trả về **một
chỉ số nguyên**. Nó không bao giờ thấy giá trị số của ô, và không bao giờ
trả về giá trị -- đáp án luôn được tính lại từ CSV bằng pandas trong sandbox.
Đó là điều giữ cho compiler deterministic và compliance linter còn ý nghĩa.

Thứ tự phần tử trong `candidates` **là** hợp đồng: `row_choice_decision.py`
map `chosen_index` ngược về ứng viên bằng chính vị trí này.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


def _candidate_payload(index: int, candidate: RowFusedCandidate) -> dict[str, object]:
    metadata = candidate.metadata
    return {
        "index": index,
        "row_label": metadata.row_label_raw or "",
        "row_group_context": metadata.row_group_context_raw,
        "statement_type": metadata.statement_type,
        "table_title": metadata.title,
        "periods": list(metadata.periods),
        "units": list(metadata.units),
    }


def build_batch_payload(
    question_id: int,
    question: str,
    candidates: Sequence[RowFusedCandidate],
) -> dict[str, object]:
    """Một dòng JSONL: câu hỏi kèm các dòng ứng viên đã đánh số từ 0.

    `candidates` phải đã ở đúng thứ tự retrieval-rank (điểm cao nhất trước) --
    `RowFusionService.retrieve_rows` trả về đúng thứ tự đó. Không sắp lại.
    """
    return {
        "question_id": question_id,
        "question": question,
        "candidates": [
            _candidate_payload(index, candidate) for index, candidate in enumerate(candidates)
        ],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_batch.py -v`
Expected: PASS — 4 passed

- [ ] **Step 5: Thêm subcommand `row-batches` vào CLI**

Trong `src/financial_report_qa/submission/cli.py`, thêm parser mới trong `_parser()` (dòng 43) ngay sau khối tham số của parser `export`. Đối tượng subparsers ở file này tên là **`commands`** (dòng 47: `export = commands.add_parser("export")`), không phải `subparsers`:

```python
    batches = commands.add_parser(
        "row-batches",
        help=(
            "Chạy retrieval + row fusion cho mọi câu hỏi và ghi ứng viên ra JSONL "
            "để LLM chọn dòng offline (thiết kế 2026-08-22 §5.2)."
        ),
    )
    batches.add_argument("--release-lock", type=Path, required=True)
    batches.add_argument("--bm25-index", type=Path, required=True)
    batches.add_argument("--questions-path", type=Path, required=True)
    batches.add_argument("--output-dir", type=Path, required=True)
    batches.add_argument("--k", type=int, default=10, help="Số bảng ứng viên mỗi câu.")
    batches.add_argument(
        "--rows-per-question", type=int, default=20, help="Số dòng ứng viên mỗi câu."
    )
    batches.add_argument("--batch-size", type=int, default=64, help="Số câu mỗi file batch.")
```

Thêm nhánh xử lý trong `main()`, ngay sau nhánh `if args.command == "export":`:

```python
        if args.command == "row-batches":
            root = Path.cwd()
            release = resolve_retrieval_release(args.release_lock, repo_root=root)
            index = load_bm25_index(args.bm25_index)
            if index.manifest.dataset_fingerprint != release.dataset_fingerprint:
                raise SubmissionError(
                    "--bm25-index dataset_fingerprint does not match --release-lock"
                )
            service = RetrievalService(index)
            questions = load_raw_questions(args.questions_path)
            row_fusion = _build_row_fusion(args, release)
            if row_fusion is None:
                raise SubmissionError(
                    f"không tìm thấy row index tại {args.bm25_index.parent}/"
                    f"{args.bm25_index.name}_row -- không thể sinh ứng viên dòng"
                )

            args.output_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for batch_number, start in enumerate(
                range(0, len(questions), args.batch_size)
            ):
                chunk = questions[start : start + args.batch_size]
                lines: list[str] = []
                for raw_question in chunk:
                    retrieved = retrieve_candidate_table_ids(
                        raw_question.question, service, k=args.k
                    )
                    fused = row_fusion.retrieve_rows(
                        raw_question.question,
                        candidate_table_ids=retrieved,
                        k=args.rows_per_question,
                    ).results
                    payload = build_batch_payload(
                        raw_question.id, raw_question.question, fused
                    )
                    lines.append(json.dumps(payload, ensure_ascii=False))
                    written += 1
                target = args.output_dir / f"batch_{batch_number:03d}.jsonl"
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"đã ghi {written} câu vào {args.output_dir}")
            return 0
```

Thêm import ở đầu file:

```python
from financial_report_qa.planning.row_choice_batch import build_batch_payload
```

- [ ] **Step 6: Tách hàm dựng row fusion để dùng chung**

Khối dựng `RowFusionService` hiện nằm inline trong nhánh `export` (`cli.py:219-253`). Trích nó thành hàm module-level `_build_row_fusion(args, release) -> RowFusionService | None` giữ nguyên logic (kể cả `try/except` in cảnh báo và trả `None`), rồi thay khối inline trong nhánh `export` bằng `row_fusion = _build_row_fusion(args, release)`.

Đây là điều kiện để nhánh `row-batches` và nhánh `export` không phân kỳ về cách dựng ứng viên — hai nhánh phải thấy **cùng một** danh sách dòng, nếu không `chosen_index` sẽ trỏ sai.

- [ ] **Step 7: Run tests and lint**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ tests/unit/planning/ -v`
Expected: PASS

Run: `.venv/Scripts/python.exe -m ruff check src/financial_report_qa/`
Expected: no new findings

- [ ] **Step 8: Commit**

```bash
git add src/financial_report_qa/planning/row_choice_batch.py tests/unit/planning/test_row_choice_batch.py src/financial_report_qa/submission/cli.py
git commit -m "feat(planning): emit row-candidate batches for offline LLM row selection

The batch carries labels, context and periods but never a cell value: the LLM
returns only an index, so the answer is still computed from the CSV by pandas
in the sandbox. Extracts _build_row_fusion so the batch command and the export
command build candidates identically -- a divergence there would make
chosen_index point at the wrong row."
```

---

## Task 4: Đọc quyết định LLM và dựng selector position-bound

**Files:**
- Create: `src/financial_report_qa/planning/row_choice_decision.py`
- Test: `tests/unit/planning/test_row_choice_decision.py`

**Interfaces:**
- Consumes: `RowFusedCandidate` (Task 3), `MetricSelector` (`planning/plan_contracts.py:74`)
- Produces (Task 5 gọi cả hai):
  - `load_decisions(path: Path) -> dict[int, int]`
  - `selector_for(question_id, candidates, decisions) -> tuple[MetricSelector | None, str]` — phần tử thứ hai là nguồn quyết định: `"llm"` hoặc `"fallback_rank1"` hoặc `"no_candidates"`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/planning/test_row_choice_decision.py`:

```python
"""Đọc quyết định LLM và biến nó thành selector position-bound."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_report_qa.planning.row_choice_decision import load_decisions, selector_for
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


# TableId bị ràng buộc ^tbl_[0-9a-f]{64}$ -- "t1" sẽ ném ValidationError.
_TABLE_ID = "tbl_" + "a" * 64


def _candidate(*, rank: int, row_idx: int, label: str) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(table_id=_TABLE_ID, row_idx=row_idx, row_label_raw=label),
        snippet=label,
    )


_CANDIDATES = [
    _candidate(rank=1, row_idx=3, label="A"),
    _candidate(rank=2, row_idx=7, label="B"),
]


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    target = tmp_path / "decisions.jsonl"
    target.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return target


def test_load_decisions_maps_question_id_to_index(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"question_id": 795, "chosen_index": 1}])
    assert load_decisions(path) == {795: 1}


def test_load_decisions_ignores_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "decisions.jsonl"
    target.write_text('\n{"question_id": 1, "chosen_index": 0}\n\n', encoding="utf-8")
    assert load_decisions(target) == {1: 0}


def test_selector_uses_the_llm_choice(tmp_path: Path) -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: 1})
    assert source == "llm"
    assert selector is not None
    assert selector.is_position_bound
    assert selector.table_id == _TABLE_ID
    assert selector.row_index == 7
    assert selector.raw_text == "B"


def test_missing_decision_falls_back_to_rank_one() -> None:
    """§7.1: fallback tất định, giải trình được -- không gọi LLM lần hai."""
    selector, source = selector_for(795, _CANDIDATES, {})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_out_of_range_index_falls_back_to_rank_one() -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: 99})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_negative_index_falls_back_to_rank_one() -> None:
    selector, source = selector_for(795, _CANDIDATES, {795: -1})
    assert source == "fallback_rank1"
    assert selector is not None
    assert selector.row_index == 3


def test_no_candidates_yields_no_selector() -> None:
    selector, source = selector_for(795, [], {795: 0})
    assert selector is None
    assert source == "no_candidates"


def test_candidate_without_a_label_yields_no_selector() -> None:
    """MetricSelector đòi đúng một trong canonical/raw_text; một ứng viên
    không có nhãn không dựng được selector hợp lệ."""
    blank = RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_5",
        table_id=_TABLE_ID,
        row_idx=5,
        rank=1,
        fused_score=1.0,
        metadata=RowMetadata(table_id=_TABLE_ID, row_idx=5, row_label_raw=None),
        snippet="",
    )
    selector, source = selector_for(1, [blank], {1: 0})
    assert selector is None
    assert source == "no_candidates"


def test_load_decisions_raises_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_decisions(tmp_path / "khong-ton-tai.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_decision.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'financial_report_qa.planning.row_choice_decision'`

- [ ] **Step 3: Write the implementation**

Tạo `src/financial_report_qa/planning/row_choice_decision.py`:

```python
"""Đọc quyết định chọn dòng của LLM và biến nó thành `MetricSelector`.

Thiết kế 2026-08-22 §5.3/§7.1. File quyết định chỉ mang
`{question_id, chosen_index}` -- không mang nhãn, không mang giá trị. Mọi
thông tin khác được tra lại từ danh sách ứng viên dựng ở local, nên một file
quyết định cũ hay hỏng không thể bơm dữ liệu mâu thuẫn với corpus vào bundle.

Selector trả về luôn là **position-bound** (`table_id` + `row_index`), nên
`locator._metric_mask` đi nhánh `_position_mask` -- không so khớp nhãn ở thời
điểm truy vấn. Theo bảng đo trong thiết kế 2026-08-21 §2.2, predicate theo vị
trí nhập nhằng 0.00% so với 31.58% của predicate ngữ nghĩa.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from financial_report_qa.planning.plan_contracts import MetricSelector
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DecisionSource = str


def load_decisions(path: Path) -> dict[int, int]:
    """Đọc JSONL quyết định thành `{question_id: chosen_index}`.

    Ném `FileNotFoundError` khi thiếu file: rơi âm thầm về đường khác sẽ
    khiến cả một lần chạy 3 tiếng dùng nhầm nguồn quyết định mà không ai
    biết.
    """
    text = path.read_text(encoding="utf-8")
    decisions: dict[int, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record = json.loads(stripped)
        decisions[int(record["question_id"])] = int(record["chosen_index"])
    return decisions


def _selector_from(candidate: RowFusedCandidate) -> MetricSelector | None:
    label = candidate.metadata.row_label_raw
    if label is None or not label.strip():
        return None
    return MetricSelector(
        raw_text=label,
        table_id=candidate.table_id,
        row_index=candidate.row_idx,
    )


def selector_for(
    question_id: int,
    candidates: Sequence[RowFusedCandidate],
    decisions: Mapping[int, int],
) -> tuple[MetricSelector | None, DecisionSource]:
    """Selector cho một câu, kèm nguồn quyết định để ghi log.

    Nguồn có ba giá trị: `"llm"` (dùng chỉ số LLM trả về), `"fallback_rank1"`
    (thiếu quyết định hoặc chỉ số không hợp lệ -- chọn ứng viên hạng 1, tất
    định và tận dụng được retrieval), `"no_candidates"` (không dựng được
    selector; caller phải rơi xuống backstop).
    """
    if not candidates:
        return None, "no_candidates"

    index = decisions.get(question_id)
    source: DecisionSource = "llm"
    if index is None or not 0 <= index < len(candidates):
        index = 0
        source = "fallback_rank1"

    selector = _selector_from(candidates[index])
    if selector is None:
        return None, "no_candidates"
    return selector, source
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_decision.py -v`
Expected: PASS — 9 passed

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/planning/row_choice_decision.py tests/unit/planning/test_row_choice_decision.py
git commit -m "feat(planning): turn LLM row decisions into position-bound selectors

The decision file carries only {question_id, chosen_index}; every label and
value is re-derived locally from the candidate list, so a stale or corrupt
decision file cannot inject data that contradicts the corpus. An invalid or
missing index falls back to the rank-1 candidate deterministically rather
than calling the LLM a second time."
```

---

## Task 5: LLM quyết định thay nhánh rule dictionary trong grounding

`ground_with_recovery` (`planning/cell_grounding.py:133`) hiện chạy Attempt 0 bằng `ground_raw_metric` (rule dictionary), rồi mới tới LLM. Task này biến quyết định LLM thành Attempt 0 và xóa nhánh rule khỏi đường chính.

**Files:**
- Modify: `src/financial_report_qa/planning/cell_grounding.py:133-260`
- Modify: `src/financial_report_qa/submission/exporter.py` (truyền quyết định xuống)
- Test: `tests/unit/planning/test_cell_grounding.py`

**Interfaces:**
- Consumes: `selector_for`, `load_decisions` (Task 4)
- Produces: `ground_with_recovery(..., row_decisions: Mapping[int, int] | None = None, question_id: int | None = None)`; `GroundingResult.plan_source` nhận thêm hai giá trị `"llm_row_choice"` và `"row_choice_fallback_rank1"`

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/planning/test_cell_grounding.py`:

```python
def test_ground_with_recovery_uses_the_llm_row_decision_first() -> None:
    """Attempt 0 giờ là quyết định LLM, không phải rule dictionary."""
    import inspect

    from financial_report_qa.planning import cell_grounding

    signature = inspect.signature(cell_grounding.ground_with_recovery)
    assert "row_decisions" in signature.parameters
    assert "question_id" in signature.parameters


def test_rule_dictionary_grounding_is_gone_from_the_primary_path() -> None:
    """`ground_raw_metric` là nhánh so khớp nhãn bằng luật -- nguyên nhân
    gốc của 491 câu `metric_not_found`. Nó không được còn trong đường chính."""
    import inspect

    from financial_report_qa.planning import cell_grounding

    source = inspect.getsource(cell_grounding.ground_with_recovery)
    assert "ground_raw_metric" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/test_cell_grounding.py -k "row_decision or dictionary_grounding_is_gone" -v`
Expected: FAIL — `AssertionError` ở cả hai (chữ ký chưa đổi; `ground_raw_metric` vẫn còn trong nguồn)

- [ ] **Step 3: Đổi Attempt 0 sang quyết định LLM**

Trong `src/financial_report_qa/planning/cell_grounding.py`, thêm import:

```python
from financial_report_qa.planning.row_choice_decision import selector_for
```

Đổi chữ ký `ground_with_recovery` (dòng 133-143) — thêm hai tham số cuối:

```python
def ground_with_recovery(
    question: str,
    entities: QueryEntities,
    retrieved: Sequence[str],
    row_labels: Sequence[str],
    fusion_rows: Sequence[RowFusedCandidate],
    release_dir: Path,
    execution_settings: ExecutionSettings,
    llm_client: ChatCompletionClient | None = None,
    max_grounding_rank: int = DEFAULT_MAX_GROUNDING_RANK,
    row_decisions: Mapping[int, int] | None = None,
    question_id: int | None = None,
) -> GroundingResult:
```

Thay khối "Attempt 0: Normal Grounding" (dòng 153-173, từ `plan_result = RulePlanResult(...)` tới hết khối `if raw_metric is not None:`) bằng:

```python
    # Attempt 0: quyết định chọn dòng của LLM (thiết kế 2026-08-22 §5).
    #
    # Trước đây chỗ này là `ground_raw_metric` -- so khớp nhãn bằng từ điển
    # luật. Đo trên lần export 2026-08-22: 491/1012 câu chết ở
    # `metric_not_found` vì câu hỏi tiếng Việt diễn đạt tự do hiếm khi trùng
    # khít `row_label_raw` trong báo cáo. Việc "dòng nào trong 20 dòng ứng
    # viên trả lời câu hỏi này" là phân loại có ràng buộc -- dạng bài model
    # 8B làm tốt, khác hẳn sinh code tự do mà nó làm kém.
    plan_result = RulePlanResult(abstain_codes=("operation_unknown",))
    plan_source = "rule"
    recovery_attempts = 0

    if question_id is not None and fusion_rows:
        selector, decision_source = selector_for(
            question_id, fusion_rows, row_decisions or {}
        )
        if selector is not None and selector.raw_text is not None:
            labelled = QueryEntities.model_validate(
                {
                    **entities.model_dump(mode="python"),
                    "metrics": (selector.raw_text,),
                    "ambiguity": tuple(
                        code for code in entities.ambiguity if code != "metric_unknown"
                    ),
                }
            )
            chosen = build_plan(
                labelled,
                candidate_table_ids=retrieved,
                known_table_ids=frozenset(retrieved),
            )
            if chosen.plan is not None:
                plan_result = _bind_metric_to_position(chosen, selector)
                plan_source = (
                    "llm_row_choice"
                    if decision_source == "llm"
                    else "row_choice_fallback_rank1"
                )
```

Thêm hàm phụ trợ ngay trên `ground_with_recovery`:

```python
def _bind_metric_to_position(
    plan_result: RulePlanResult, selector: MetricSelector
) -> RulePlanResult:
    """Gắn `table_id`/`row_index` đã chọn vào `metric` của plan.

    `build_plan` dựng selector từ nhãn nên nó chưa position-bound. Ghim vị trí
    ở đây để `locator._metric_mask` đi nhánh `_position_mask`, bỏ hẳn việc so
    khớp nhãn ở thời điểm truy vấn.
    """
    plan = plan_result.plan
    if plan is None or plan.metric is None:
        return plan_result
    bound = plan.metric.model_copy(
        update={"table_id": selector.table_id, "row_index": selector.row_index}
    )
    return plan_result.model_copy(update={"plan": plan.model_copy(update={"metric": bound})})
```

Thêm import `Mapping` và `MetricSelector` ở đầu file nếu chưa có:

```python
from collections.abc import Mapping, Sequence

from financial_report_qa.planning.plan_contracts import MetricSelector
```

- [ ] **Step 4: Xóa `ground_raw_metric` khỏi đường chính**

Xóa import `ground_raw_metric` khỏi `cell_grounding.py`. Giữ nguyên
`raw_metric_grounding.py` (các hàm `candidate_row_labels`, `candidate_column_labels`
vẫn được dùng bởi Day 26 column refinement) nhưng xóa hàm `ground_raw_metric` và
`plan_with_raw_grounding_fallback` nếu sau khi xóa import không còn caller nào.

Kiểm tra trước khi xóa:

```bash
grep -rn "ground_raw_metric\|plan_with_raw_grounding_fallback" src/ tests/
```

Chỉ xóa những hàm không còn caller nào ngoài chính file định nghĩa. Nếu test cũ còn gọi, cập nhật test đó — hành vi mất đi là có chủ đích.

- [ ] **Step 5: Truyền quyết định xuống từ exporter**

Trong `src/financial_report_qa/submission/exporter.py`:

Thêm tham số vào `export_submission` và `_run_one_question` (cùng khuôn mẫu `row_fusion` đã có):

```python
    row_decisions: Mapping[int, int] | None = None,
```

Tại lời gọi `ground_with_recovery` (dòng ~372-381), thêm hai đối số:

```python
        grounding_res = ground_with_recovery(
            question=question,
            entities=entities,
            retrieved=retrieved,
            row_labels=row_labels,
            fusion_rows=fusion_rows,
            release_dir=release_dir,
            execution_settings=execution_settings,
            llm_client=llm_client,
            row_decisions=row_decisions,
            question_id=raw_question.id,
        )
```

**Lưu ý bắt buộc:** điều kiện bao quanh lời gọi hiện là `if llm_client is not None and needs_grounding_recovery:`. Đổi thành:

```python
    if needs_grounding_recovery and (llm_client is not None or row_decisions is not None):
```

Quyết định LLM đến từ file offline, không cần `llm_client` sống — giữ nguyên điều kiện cũ sẽ khiến toàn bộ Task 5 không bao giờ chạy khi export không truyền `--llm-config`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/planning/ tests/unit/submission/ -v`
Expected: PASS. Test nào assert `plan_source == "rule_raw_grounded"` sẽ fail — cập nhật sang giá trị mới, đó là thay đổi có chủ đích.

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/planning/cell_grounding.py src/financial_report_qa/planning/raw_metric_grounding.py src/financial_report_qa/submission/exporter.py tests/unit/planning/test_cell_grounding.py
git commit -m "feat(planning): make the LLM row decision the primary grounding step

Rule-based dictionary matching was Attempt 0 and accounted for 491/1012
metric_not_found failures: free-form Vietnamese questions rarely match
row_label_raw verbatim. The LLM decision now grounds first and binds the
selector to a physical (table_id, row_idx), so the locator uses positional
extraction -- 0.00% ambiguous versus 31.58% for the semantic predicate."
```

---

## Task 6: Cờ CLI `--row-choice-decisions`

**Files:**
- Modify: `src/financial_report_qa/submission/cli.py`
- Test: `tests/unit/submission/test_submission_cli.py`

**Interfaces:**
- Consumes: `load_decisions` (Task 4), `export_submission(..., row_decisions=...)` (Task 5)
- Produces: `submission export --row-choice-decisions PATH`

- [ ] **Step 1: Write the failing test**

Thêm vào `tests/unit/submission/test_submission_cli.py`:

```python
def test_export_accepts_a_row_choice_decisions_path() -> None:
    from financial_report_qa.submission import cli as submission_cli

    parser = submission_cli._parser()
    args = parser.parse_args(
        [
            "export",
            "--release-lock", "a.json",
            "--bm25-index", "b",
            "--questions-path", "c.jsonl",
            "--execution-config", "d.yaml",
            "--output-zip", "e.zip",
            "--report-dir", "f",
            "--row-choice-decisions", "g.jsonl",
        ]
    )
    assert args.row_choice_decisions == Path("g.jsonl")


def test_export_defaults_row_choice_decisions_to_none() -> None:
    from financial_report_qa.submission import cli as submission_cli

    parser = submission_cli._parser()
    args = parser.parse_args(
        [
            "export",
            "--release-lock", "a.json",
            "--bm25-index", "b",
            "--questions-path", "c.jsonl",
            "--execution-config", "d.yaml",
            "--output-zip", "e.zip",
            "--report-dir", "f",
        ]
    )
    assert args.row_choice_decisions is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_cli.py -k row_choice -v`
Expected: FAIL — `SystemExit: 2`, argparse báo `unrecognized arguments: --row-choice-decisions`

- [ ] **Step 3: Thêm cờ và nạp quyết định**

Trong `_parser()`, thêm vào parser `export`:

```python
    export.add_argument(
        "--row-choice-decisions",
        type=Path,
        default=None,
        help=(
            "File JSONL {question_id, chosen_index} do Qwen3-8B sinh offline "
            "(xem subcommand `row-batches`). Bỏ qua để dùng fallback hạng 1."
        ),
    )
```

Trong `main()`, trước lời gọi `export_submission`:

```python
            row_decisions = (
                load_decisions(args.row_choice_decisions)
                if args.row_choice_decisions is not None
                else None
            )
```

Truyền `row_decisions=row_decisions` vào **cả hai** lời gọi `export_submission` (nhánh có `--llm-config` ở dòng ~258 và nhánh không có ở dòng ~270).

Thêm import:

```python
from financial_report_qa.planning.row_choice_decision import load_decisions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/submission/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/submission/cli.py tests/unit/submission/test_submission_cli.py
git commit -m "feat(submission): accept --row-choice-decisions on export"
```

---

## Task 7: Notebook Colab chạy Qwen3-8B

**Files:**
- Create: `notebooks/colab_row_choice_qwen3_8b.ipynb`

**Interfaces:**
- Consumes: `batch_*.jsonl` (Task 3)
- Produces: `decisions.jsonl` — `{"question_id": int, "chosen_index": int}` mỗi dòng (Task 4 đọc)

- [ ] **Step 1: Tạo notebook**

Tạo `notebooks/colab_row_choice_qwen3_8b.ipynb` với các cell sau (mỗi mục là một cell).

Cell 1 — markdown:

```markdown
# ViFinQA — LLM chọn dòng (Qwen3-8B)

Chạy trên Colab T4. Đọc `batch_*.jsonl`, ghi `decisions.jsonl`.

**Ràng buộc:** model phải < 14B tham số. Qwen3-8B = 8.2B ✓.
KHÔNG đổi sang Qwen3-14B (~14.7B) — vi phạm thể lệ.

**Bất biến:** model chỉ trả về `chosen_index`. Không bao giờ trả giá trị đáp án.
```

Cell 2 — cài đặt:

```python
!pip -q install "vllm==0.6.3" "huggingface_hub>=0.24"
```

Cell 3 — upload batch:

```python
from google.colab import files
import pathlib

pathlib.Path("batches").mkdir(exist_ok=True)
print("Chọn toàn bộ batch_*.jsonl đã sinh bằng `submission row-batches`:")
uploaded = files.upload()
for name in uploaded:
    pathlib.Path("batches", name).write_bytes(uploaded[name])
print("đã nhận:", sorted(p.name for p in pathlib.Path("batches").glob("*.jsonl")))
```

Cell 4 — nạp model:

```python
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-8B"  # 8.2B < 14B. KHÔNG đổi sang 14B.
llm = LLM(model=MODEL, dtype="half", gpu_memory_utilization=0.90, max_model_len=8192)
sampling = SamplingParams(temperature=0.0, max_tokens=16)
```

Cell 5 — dựng prompt và chạy:

```python
import json, pathlib, re

PROMPT = """Bạn là trợ lý phân tích báo cáo tài chính.

Câu hỏi: {question}

Các dòng ứng viên:
{candidates}

Chọn ĐÚNG MỘT dòng trả lời câu hỏi. Chỉ trả về JSON: {{"chosen_index": <số>}}"""


def render(payload):
    lines = []
    for c in payload["candidates"]:
        parts = [f'[{c["index"]}] {c["row_label"]}']
        if c.get("row_group_context"):
            parts.append(f'(mục: {c["row_group_context"]})')
        if c.get("table_title"):
            parts.append(f'(bảng: {c["table_title"]})')
        if c.get("periods"):
            parts.append(f'(kỳ: {", ".join(c["periods"])})')
        lines.append(" ".join(parts))
    return PROMPT.format(question=payload["question"], candidates="\n".join(lines))


def parse(text, limit):
    match = re.search(r'"chosen_index"\s*:\s*(-?\d+)', text)
    if match is None:
        match = re.search(r"-?\d+", text)
    if match is None:
        return 0
    value = int(match.group(1) if match.lastindex else match.group(0))
    return value if 0 <= value < limit else 0


out = pathlib.Path("decisions.jsonl")
done = set()
if out.exists():  # chạy lại sau timeout chỉ tốn phần còn thiếu
    done = {json.loads(l)["question_id"] for l in out.read_text("utf-8").splitlines() if l.strip()}
    print("đã có sẵn:", len(done))

with out.open("a", encoding="utf-8") as sink:
    for batch in sorted(pathlib.Path("batches").glob("batch_*.jsonl")):
        payloads = [json.loads(l) for l in batch.read_text("utf-8").splitlines() if l.strip()]
        payloads = [p for p in payloads if p["question_id"] not in done and p["candidates"]]
        if not payloads:
            continue
        outputs = llm.generate([render(p) for p in payloads], sampling)
        for payload, output in zip(payloads, outputs):
            index = parse(output.outputs[0].text, len(payload["candidates"]))
            sink.write(json.dumps({"question_id": payload["question_id"],
                                   "chosen_index": index}) + "\n")
        sink.flush()
        print(batch.name, "xong", len(payloads), "câu")
```

Cell 6 — tải kết quả về:

```python
from google.colab import files
files.download("decisions.jsonl")
```

- [ ] **Step 2: Xác minh notebook là JSON hợp lệ**

Run:

```bash
.venv/Scripts/python.exe -c "import json; json.load(open('notebooks/colab_row_choice_qwen3_8b.ipynb', encoding='utf-8')); print('notebook hợp lệ')"
```

Expected: `notebook hợp lệ`

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_row_choice_qwen3_8b.ipynb
git commit -m "feat(notebooks): add Colab notebook running Qwen3-8B for row selection

Qwen3-8B (8.2B) satisfies the competition's <14B limit; the earlier design's
Qwen3-14B suggestion (~14.7B) did not. Appends to decisions.jsonl per batch so
a Colab timeout only costs the unfinished batches."
```

---

## Task 8: Chạy end-to-end và đo

Không có code mới. Đây là bước xác nhận, chạy thủ công.

- [ ] **Step 1: Sinh batch (local, ~vài phút, không cần GPU)**

```bash
.venv/Scripts/python.exe -m financial_report_qa.cli submission row-batches --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --bm25-index data/indexes/bm25-v4/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --questions-path data/raw/ViFinQA/questions/questions.jsonl --output-dir data/qa/row_choice_batches --k 10 --rows-per-question 20 --batch-size 64
```

Expected: `đã ghi 1012 câu vào data/qa/row_choice_batches`, 16 file `batch_000.jsonl`–`batch_015.jsonl`.

- [ ] **Step 2: Chạy Colab**

Mở `notebooks/colab_row_choice_qwen3_8b.ipynb` trên Colab, Runtime → T4 GPU, chạy lần lượt các cell, upload toàn bộ `data/qa/row_choice_batches/batch_*.jsonl`, tải `decisions.jsonl` về.

Đặt file tải về vào: `data/qa/row_choice_decisions.jsonl`

- [ ] **Step 3: Kiểm tra độ phủ quyết định**

```bash
.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from financial_report_qa.planning.row_choice_decision import load_decisions; d = load_decisions(Path('data/qa/row_choice_decisions.jsonl')); print('so quyet dinh:', len(d))"
```

Expected: gần 1012. Nếu thiếu nhiều, chạy lại Cell 5 trên Colab — nó bỏ qua câu đã có.

- [ ] **Step 4: Chạy full export**

Sửa `_run_full_export.py`, thêm hai dòng vào danh sách tham số:

```python
    "--row-choice-decisions", "data/qa/row_choice_decisions.jsonl",
```

Rồi chạy:

```bash
.venv/Scripts/python.exe _run_full_export.py
```

Expected: exit code 0. Nếu exit 2, mở `artifacts/evaluations/v2gaps_full/compliance-violations.json` và sửa nguyên nhân — **không** nới lỏng linter.

- [ ] **Step 5: Đo kết quả so với baseline**

```bash
.venv/Scripts/python.exe -c "
import json, collections
r = json.load(open('artifacts/evaluations/v2gaps_full/submission-export-422df141c935.json', encoding='utf-8'))
print('answered_count:', r['answered_count'], '(baseline 53)')
codes = collections.Counter((o.get('stage'), o.get('code')) for o in r['outcomes'] if o.get('status') != 'answered')
for (stage, code), n in codes.most_common(10):
    print(f'{n:5d}  {stage}: {code}')
"
```

Expected theo tiêu chí thành công: `answered_count > 53`; `metric_not_found` giảm mạnh so với 491; `cell_ambiguous`/`period_unresolved`/`unit_missing` gần 0.

- [ ] **Step 6: Commit kết quả đo**

```bash
git add artifacts/evaluations/v2gaps_full/ _run_full_export.py
git commit -m "chore(eval): record post-LLM-row-selection export baseline"
```

---

## Self-Review

**Spec coverage:**

| Mục spec | Task |
|---|---|
| §3 chọn Qwen3-8B (<14B) | Task 7 (notebook ghim `Qwen/Qwen3-8B`), Global Constraints |
| §4 kiến trúc (LLM chỉ trả index) | Task 3 (`test_payload_never_leaks_a_cell_value`), Task 4 |
| §5.1 phạm vi: thay chọn dòng, giữ compiler | Task 5 |
| §5.2 batch offline | Task 3, Task 7 |
| §5.3 định dạng file | Task 3 (batch), Task 4 (quyết định) |
| §5.4 prompt constrained | Task 7 Cell 5 |
| §6 tie-break tại 3 điểm abstain | Task 1 (hàm), Task 2 (gắn vào `locate`) |
| §6 kích hoạt qua config | Task 2 Step 3/6 |
| §7 xử lý lỗi | Task 4 (`selector_for` ba nguồn), Task 5 Step 5 |
| §7.1 fallback rank=1 | Task 4 |
| §8 cấu trúc file | File Structure |
| §9 kiểm thử không cần LLM/Colab | Task 1, 3, 4 test thuần |
| §10 quy trình vận hành | Task 8 |
| §13 tiêu chí thành công | Task 8 Step 5 |

**Ngoài phạm vi có chủ đích:** §1.1 (quan sát `evidence_frame_replay_mismatch`) được đo ở Task 8 Step 5 nhưng không có tiêu chí pass/fail — đúng như spec ghi "cần đo, không cam kết".

**Type consistency:** `MetricSelector` dùng `table_id`/`row_index` (không phải `row_idx`) — đúng tên field tại `plan_contracts.py:99-100`; `RowFusedCandidate` dùng `row_idx`. Task 4 chuyển đổi đúng chiều tại `_selector_from`. `DecisionSource` là `str` với ba giá trị cố định, dùng nhất quán ở Task 4 và Task 5. `build_batch_payload` (Task 3) và `load_decisions`/`selector_for` (Task 4) khớp định dạng JSONL hai chiều.

**Điểm cần người thực thi chú ý:**
1. Task 2 Step 5 — 10 lời gọi `locate()` phải đổi hết sang `_cell(...)`; bỏ sót một chỗ sẽ khiến nhánh đó âm thầm không có tie-break.
2. Task 5 Step 5 — điều kiện `if llm_client is not None and needs_grounding_recovery` **bắt buộc** phải nới; giữ nguyên sẽ khiến cả Task 5 không chạy khi export không có `--llm-config`.
3. Task 5 Step 4 — chỉ xóa hàm sau khi `grep` xác nhận không còn caller; `raw_metric_grounding.py` vẫn giữ `candidate_row_labels`/`candidate_column_labels`.
