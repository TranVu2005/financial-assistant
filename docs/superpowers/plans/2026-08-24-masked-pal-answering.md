# Masked PAL Answering Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay nhánh answering `LLM chọn operation → compiler tất định` bằng masked PAL: LLM chọn ô (chỉ số) và sinh một biểu thức số học trên placeholder `[NUM_i]`, binding tất định từ CSV, chạy trong sandbox số học riêng, rồi hai lớp verification độc lập.

**Architecture:** Thêm một chuỗi module thuần mới trong `execution/` và `planning/`, không sửa gì đang chạy cho tới Task 10. Ngôn ngữ LLM sinh ra nhỏ tới mức guard bằng AST là đủ: bốn phép tính, `abs`, và tên `NUM_<i>`. Mọi `ast.Constant` bị từ chối — đó là cách thi hành N4′ bằng máy. Cùng một cây AST được render hai lần: ra `Decimal` để tính đáp án, và ra chuỗi pandas tra cứu CSV để nộp (linter C5/C7 đòi điều đó).

**Tech Stack:** Python 3.11, pydantic v2 (frozen contracts), `ast` (stdlib, không `eval`/`exec`), `Decimal`, pandas/DuckDB (release Parquet), pytest.

## Global Constraints

Sao chép nguyên văn từ `docs/superpowers/specs/2026-08-24-masked-pal-answering-design.md`:

- **N4′ — Chương trình do LLM sinh ra không được chứa literal số.** Không `* 100`, không `/ 1000`, không `round(x, 2)`. Bất kỳ node `ast.Constant` nào cũng là vi phạm.
- **N6 — Một đường đi duy nhất.** Sinh lại là retry của đúng một bước, **tối đa 1 lần**, cùng prompt, chỉ khác nhiệt độ. Vẫn lệch → `low_confidence = true`, đi tiếp. Không có lần thứ ba, không có đường thứ hai.
- **N7 — Quyết định LLM không mang giá trị số.** File quyết định chỉ chứa chỉ số vào danh sách ứng viên dựng lại được ở local, cộng chuỗi biểu thức.
- **N1 — Hai nhánh độc lập.** Nhánh answering hỏng không được ghi đè đầu ra nhánh retrieval. `retrieved` và danh sách narrow-theo-scope giữ hai tên biến tách biệt ở `exporter.py`.
- **N2 — Evidence CSV luôn là lát cắt bảng nguồn**, không bao giờ tổng hợp ngược từ đáp án.
- Ô không có `value_numeric` **không bao giờ** vào danh sách ứng viên.
- Không test nào được tải model thật hay gọi mạng.
- Repo dùng `uv`; test `uv run pytest`; lint `uv run ruff check`; type `uv run mypy src`.
- Console Windows cần `PYTHONIOENCODING=utf-8` khi in tiếng Việt.

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `src/financial_report_qa/core/errors.py` | Sửa | Thêm `ProgramError` và ba lớp con |
| `src/financial_report_qa/execution/program_contracts.py` | Tạo | `CellCandidate`, `UseClaim`, `ProgramDecision`, `BoundValue`, `ExecutedProgram`, `ScaleName` |
| `src/financial_report_qa/execution/masked_program.py` | Tạo | AST guard + interpreter số học + `apply_scale` |
| `src/financial_report_qa/execution/program_binding.py` | Tạo | Bind chỉ số → `BoundValue`; render cùng AST ra chuỗi pandas |
| `src/financial_report_qa/planning/cell_candidates.py` | Tạo | Dựng danh sách ô đánh số từ cell frame + row candidates |
| `src/financial_report_qa/planning/row_choice_batch.py` | Sửa | Thêm `build_program_batch_payload` (ứng viên **ô**) |
| `src/financial_report_qa/planning/program_decisions.py` | Tạo | Đọc/validate file quyết định JSONL |
| `src/financial_report_qa/verification/use_checks.py` | Tạo | Verify B — `uses` đối chiếu ô đã bind |
| `src/financial_report_qa/verification/explanation_check.py` | Tạo | Verify A — nối `numeric_guard` vào `ExecutedProgram` |
| `src/financial_report_qa/execution/program_pipeline.py` | Tạo | Ghép guard → bind → eval → verify → sinh lại |
| `src/financial_report_qa/submission/compliance.py` | Sửa | Thêm C8; nới C4 cho hệ số scale của renderer |
| `src/financial_report_qa/submission/exporter.py` | Sửa | Gọi pipeline mới thay `build_plan → compile_plan` |
| `src/financial_report_qa/submission/cli.py` | Sửa | Cờ `--program-decisions` |
| `tests/unit/execution/test_masked_program.py` | Tạo | Guard + eval |
| `tests/unit/execution/test_program_binding.py` | Tạo | Bind + render hai đường |
| `tests/unit/planning/test_cell_candidates.py` | Tạo | Dựng ứng viên |
| `tests/unit/planning/test_program_batch.py` | Tạo | Payload + loader quyết định |
| `tests/unit/verification/test_use_checks.py` | Tạo | Verify B |
| `tests/unit/verification/test_explanation_check.py` | Tạo | Verify A |
| `tests/unit/execution/test_program_pipeline.py` | Tạo | Vòng sinh lại |
| `tests/unit/submission/test_compliance_c8.py` | Tạo | C8 |

**Thứ tự phụ thuộc:** 1 → 2 → 3 → 4 → 5 → {6, 7} → 8 → 9 → 10 → 11.

### Ba chỗ plan này lệch spec, có chủ ý

1. **Spec §4.5/§5.2 nói dùng lại `_cell_expr`.** Không dùng được: `_cell_expr` nhận `MetricSelector`, mà `MetricSelector` nằm trong `plan_contracts.py` — file §5.4 xoá. Task 3 viết `render_cell_lookup` riêng, dùng lại `_lit` (giữ nguyên) và **đúng hình dạng biểu thức đã được C5/C6/C7 chấp nhận**.
2. **Spec §4.7a nói chuẩn hoá bằng `metric_aliases.py`.** Không cần: `row_label_canonical` đã nằm sẵn trên từng ô trong cell frame, nên luật 2 so thẳng với trường đó, không phải dựng lexicon từ release. Kết quả tương đương, không thêm phụ thuộc I/O vào một hàm kiểm tra thuần.
3. **Spec §5.5 liệt kê `csv_path` trên `ExecutedProgram`.** Bỏ: đường dẫn CSV do `exporter.py` quyết định lúc ghi ZIP, pipeline không biết và không nên biết. `ExecutedProgram` mang `table_ids`; exporter dựng CSV từ đó.

---

### Task 1: Contracts và error types

**Files:**
- Modify: `src/financial_report_qa/core/errors.py` (cuối file, sau `ExecutionReplayMismatchError`)
- Create: `src/financial_report_qa/execution/program_contracts.py`
- Test: `tests/unit/execution/test_program_contracts.py`

**Interfaces:**
- Consumes: `_FrozenModel`, `NonEmptyString`, `TableId` từ `financial_report_qa.retrieval.contracts`; `ExecutionError` từ `financial_report_qa.core.errors`.
- Produces:
  - `ScaleName = Literal["none", "percent", "thousand", "million", "billion"]`
  - `ProgramFailureCode` — 9 mã ở §6 của spec
  - `CellCandidate(index, table_id, company_code, row_idx, col_idx, row_path, row_label_raw, row_label_canonical, col_path, period, statement_type, unit)` — **không có trường value**
  - `UseClaim(num, row, col)`
  - `ProgramDecision(question_id, cells, program, uses, scale)`
  - `BoundValue(num_index, candidate_index, table_id, row_idx, col_idx, row_path, row_label_raw, row_label_canonical, col_path, period, value: Decimal, unit)`
  - `ExecutedProgram(question_id, program, scale, bindings, answer: Decimal, pandas_query, table_ids, regenerated, low_confidence, failure_code)`
  - `ProgramError`, `ProgramGuardError`, `ProgramBindingError`, `ProgramEvalError`

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/execution/test_program_contracts.py`:

```python
from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_report_qa.execution.program_contracts import (
    BoundValue,
    CellCandidate,
    ExecutedProgram,
    ProgramDecision,
    UseClaim,
)

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(**overrides: object) -> CellCandidate:
    defaults: dict[str, object] = {
        "index": 0,
        "table_id": _TABLE_ID,
        "row_idx": 3,
        "col_idx": 2,
        "row_path": "Doanh thu > Doanh thu thuần",
        "row_label_raw": "Doanh thu thuần",
        "col_path": "Năm_2023",
        "period": 2023,
    }
    return CellCandidate(**{**defaults, **overrides})  # type: ignore[arg-type]


def _bound(**overrides: object) -> BoundValue:
    defaults: dict[str, object] = {
        "num_index": 0,
        "candidate_index": 0,
        "table_id": _TABLE_ID,
        "row_idx": 3,
        "col_idx": 2,
        "row_path": "Doanh thu > Doanh thu thuần",
        "row_label_raw": "Doanh thu thuần",
        "col_path": "Năm_2023",
        "period": 2023,
        "value": Decimal("100"),
    }
    return BoundValue(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_cell_candidate_has_no_value_field() -> None:
    # N7: model chọn ô không bao giờ thấy một con số nào.
    assert "value" not in CellCandidate.model_fields
    with pytest.raises(ValidationError):
        _candidate(value=Decimal("1"))


def test_contracts_are_frozen() -> None:
    candidate = _candidate()
    with pytest.raises(ValidationError):
        candidate.index = 1  # type: ignore[misc]


def test_decision_requires_at_least_one_cell() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(), program="[NUM_0]")


def test_decision_rejects_a_negative_candidate_index() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(-1,), program="[NUM_0]")


def test_decision_defaults_to_no_scale() -> None:
    decision = ProgramDecision(question_id=1, cells=(4,), program="[NUM_0]")
    assert decision.scale == "none"
    assert decision.uses == ()


def test_decision_rejects_an_unknown_scale() -> None:
    with pytest.raises(ValidationError):
        ProgramDecision(question_id=1, cells=(4,), program="[NUM_0]", scale="dozen")


def test_use_claim_rejects_a_blank_row() -> None:
    with pytest.raises(ValidationError):
        UseClaim(num=0, row="  ", col="Năm 2023")


def test_executed_program_requires_a_binding() -> None:
    with pytest.raises(ValidationError):
        ExecutedProgram(
            question_id=1,
            program="[NUM_0]",
            scale="none",
            bindings=(),
            answer=Decimal("1"),
            pandas_query="df1[...]",
            table_ids=(_TABLE_ID,),
        )


def test_executed_program_defaults_are_confident() -> None:
    program = ExecutedProgram(
        question_id=1,
        program="[NUM_0]",
        scale="none",
        bindings=(_bound(),),
        answer=Decimal("100"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )
    assert program.regenerated is False
    assert program.low_confidence is False
    assert program.failure_code is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/execution/test_program_contracts.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.execution.program_contracts'`

- [ ] **Step 3: Thêm error types**

Trong `src/financial_report_qa/core/errors.py`, thêm ngay sau `ExecutionReplayMismatchError`:

```python
class ProgramError(ExecutionError):
    """Masked-PAL program handling failed (spec 2026-08-24)."""


class ProgramGuardError(ProgramError):
    """A generated program violated the N4' AST whitelist."""


class ProgramBindingError(ProgramError):
    """A candidate index could not be bound to a real cell."""


class ProgramEvalError(ProgramError):
    """A guarded program failed at evaluation time."""
```

- [ ] **Step 4: Viết contracts**

Tạo `src/financial_report_qa/execution/program_contracts.py`:

```python
"""Frozen contracts for the masked-PAL answering branch (spec 2026-08-24).

`CellCandidate` deliberately has no value field: the model that picks cells
never sees a number, which is what makes N7 hold at the cell level. A value
appears for the first time on `BoundValue`, and only deterministic binding
from the release can produce one -- never the model.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from financial_report_qa.retrieval.contracts import (
    NonEmptyString,
    TableId,
    _FrozenModel,
)

ScaleName = Literal["none", "percent", "thousand", "million", "billion"]

ProgramFailureCode = Literal[
    "decision_unparseable",
    "candidate_index_out_of_range",
    "numeric_literal_in_program",
    "program_node_not_allowed",
    "division_by_zero",
    "non_finite_result",
    "use_binding_mismatch",
    "explanation_number_not_grounded",
    "no_cell_candidates",
]


class CellCandidate(_FrozenModel):
    """One numbered cell offered to the model. Carries no value (N7)."""

    index: int = Field(ge=0)
    table_id: TableId
    company_code: str | None = None
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    row_path: NonEmptyString
    row_label_raw: NonEmptyString
    row_label_canonical: str | None = None
    col_path: str
    period: int | None = None
    statement_type: str | None = None
    unit: str | None = None


class UseClaim(_FrozenModel):
    """What the model says `[NUM_<num>]` is, checked against the real binding."""

    num: int = Field(ge=0)
    row: NonEmptyString
    col: NonEmptyString


class ProgramDecision(_FrozenModel):
    """One offline decision: indices plus an expression, never a value."""

    question_id: int
    cells: tuple[int, ...] = Field(min_length=1)
    program: NonEmptyString
    uses: tuple[UseClaim, ...] = ()
    scale: ScaleName = "none"

    @field_validator("cells")
    @classmethod
    def validate_non_negative(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError("cells must be non-negative candidate indices")
        return values


class BoundValue(_FrozenModel):
    """One `[NUM_i]` after deterministic binding to a real cell."""

    num_index: int = Field(ge=0)
    candidate_index: int = Field(ge=0)
    table_id: TableId
    row_idx: int = Field(ge=0)
    col_idx: int = Field(ge=0)
    row_path: NonEmptyString
    row_label_raw: NonEmptyString
    row_label_canonical: str | None = None
    col_path: str
    period: int | None = None
    value: Decimal
    unit: str | None = None


class ExecutedProgram(_FrozenModel):
    """One finished question: what ran, on which cells, and how confident."""

    question_id: int
    program: NonEmptyString
    scale: ScaleName
    bindings: tuple[BoundValue, ...] = Field(min_length=1)
    answer: Decimal
    pandas_query: NonEmptyString
    table_ids: tuple[TableId, ...] = Field(min_length=1)
    regenerated: bool = False
    low_confidence: bool = False
    failure_code: ProgramFailureCode | None = None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/execution/test_program_contracts.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Lint + type**

Run: `uv run ruff check src/financial_report_qa/execution/program_contracts.py src/financial_report_qa/core/errors.py && uv run mypy src/financial_report_qa/execution/program_contracts.py`
Expected: không lỗi

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/execution/program_contracts.py src/financial_report_qa/core/errors.py tests/unit/execution/test_program_contracts.py
git commit -m "feat(execution): add frozen contracts for masked-PAL programs"
```

---

### Task 2: AST guard và interpreter số học

**Files:**
- Create: `src/financial_report_qa/execution/masked_program.py`
- Test: `tests/unit/execution/test_masked_program.py`

**Interfaces:**
- Consumes: `ProgramGuardError`, `ProgramEvalError` (Task 1); `ScaleName` (Task 1).
- Produces:
  - `substitute_placeholders(program: str) -> str` — `[NUM_0]` → `NUM_0`
  - `NAME_PATTERN: re.Pattern[str]` — `^NUM_(\d+)$`, dùng lại ở Task 3
  - `parse_program(program: str, *, value_count: int) -> ast.Expression`
  - `evaluate(tree: ast.Expression, values: Sequence[Decimal]) -> Decimal`
  - `run_program(program: str, values: Sequence[Decimal]) -> Decimal`
  - `apply_scale(value: Decimal, scale: ScaleName) -> Decimal`
  - `SCALE_SUFFIX: dict[str, str]` — hậu tố pandas tương ứng, dùng lại ở Task 3

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/execution/test_masked_program.py`:

```python
from decimal import Decimal

import pytest

from financial_report_qa.core.errors import ProgramEvalError, ProgramGuardError
from financial_report_qa.execution.masked_program import (
    apply_scale,
    parse_program,
    run_program,
    substitute_placeholders,
)


def test_placeholders_become_parseable_identifiers() -> None:
    assert substitute_placeholders("([NUM_1] - [NUM_0]) / [NUM_0]") == "(NUM_1 - NUM_0) / NUM_0"


def test_a_bare_lookup_is_a_valid_program() -> None:
    # Câu tra cứu thuần là biểu thức ngắn nhất, không phải một operation riêng.
    assert run_program("[NUM_0]", [Decimal("4500")]) == Decimal("4500")


def test_growth_rate_evaluates_exactly() -> None:
    result = run_program("([NUM_1] - [NUM_0]) / [NUM_0]", [Decimal("4500"), Decimal("5310")])
    assert result == Decimal("0.18")


def test_abs_is_allowed() -> None:
    assert run_program("abs([NUM_0] - [NUM_1])", [Decimal("3"), Decimal("10")]) == Decimal("7")


def test_unary_minus_is_allowed() -> None:
    assert run_program("-[NUM_0]", [Decimal("7")]) == Decimal("-7")


@pytest.mark.parametrize(
    "program",
    [
        "[NUM_0] * 100",
        "[NUM_0] / 1000",
        "[NUM_0] + 0",
        "abs([NUM_0] - 1)",
        "[NUM_0] * 1e3",
        "-1 * [NUM_0]",
    ],
)
def test_every_numeric_literal_is_rejected(program: str) -> None:
    # N4': không ngoại lệ nào, kể cả hệ số đổi thang hay số 0 vô hại.
    with pytest.raises(ProgramGuardError):
        parse_program(program, value_count=1)


@pytest.mark.parametrize(
    "program",
    [
        "[NUM_0] ** [NUM_1]",
        "[NUM_0] // [NUM_1]",
        "[NUM_0] % [NUM_1]",
        "round([NUM_0])",
        "sum([[NUM_0]])",
        "[NUM_0].real",
        "[NUM_0][0]",
        "lambda: [NUM_0]",
        "[NUM_0] if [NUM_1] else [NUM_0]",
        "[NUM_0] > [NUM_1]",
        "__import__('os')",
        "df1",
    ],
)
def test_nodes_outside_the_whitelist_are_rejected(program: str) -> None:
    with pytest.raises(ProgramGuardError):
        parse_program(program, value_count=2)


def test_a_placeholder_beyond_the_bound_values_is_rejected() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("[NUM_2]", value_count=2)


def test_a_syntax_error_is_reported_as_a_guard_error() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("([NUM_0] - ", value_count=1)


def test_a_statement_is_not_an_expression() -> None:
    with pytest.raises(ProgramGuardError):
        parse_program("ans = [NUM_0]", value_count=1)


def test_division_by_zero_is_reported_with_its_own_code() -> None:
    with pytest.raises(ProgramEvalError, match="division_by_zero"):
        run_program("[NUM_0] / [NUM_1]", [Decimal("1"), Decimal("0")])


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        ("none", Decimal("0.18")),
        ("percent", Decimal("18.00")),
        ("thousand", Decimal("0.00018")),
        ("million", Decimal("0.00000018")),
        ("billion", Decimal("0.00000000018")),
    ],
)
def test_scale_factors(scale: str, expected: Decimal) -> None:
    assert apply_scale(Decimal("0.18"), scale) == expected  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/execution/test_masked_program.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.execution.masked_program'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/execution/masked_program.py`:

```python
"""Masked-PAL program: AST guard plus an arithmetic interpreter.

N4' of the 2026-08-24 spec says a generated program may not contain a
numeric literal. The guard enforces that by rejecting *every* `ast.Constant`
-- there is no harmless one, because the moment a coefficient is allowed the
invariant stops being checkable by a machine. Scaling a result into percent
or into millions is a presentation concern, applied afterwards by
`apply_scale` from a closed enum the model chooses, never written by the
model into the program itself.

Values reach the interpreter only through `values[i]`, bound deterministically
from the release. Like `pandas_query.py`, this module never calls
`eval`/`exec` and denies by default.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from decimal import Decimal, DivisionByZero, InvalidOperation

from financial_report_qa.core.errors import ProgramEvalError, ProgramGuardError
from financial_report_qa.execution.program_contracts import ScaleName

_PLACEHOLDER_PATTERN = re.compile(r"\[NUM_(\d+)\]")
NAME_PATTERN = re.compile(r"^NUM_(\d+)$")

#: Bounded like `pandas_query.py`'s budgets: a legitimate financial formula
#: is short, so anything long is a sign the model wandered off the grammar.
_MAX_PROGRAM_LENGTH = 512
_MAX_NODE_COUNT = 200

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)

_SCALE_FACTORS: dict[str, Decimal] = {
    "none": Decimal(1),
    "percent": Decimal(100),
    "thousand": Decimal(1) / Decimal(1000),
    "million": Decimal(1) / Decimal(1000000),
    "billion": Decimal(1) / Decimal(1000000000),
}

#: Hậu tố pandas tương ứng từng scale. Phải khớp `_SCALE_FACTORS` -- Task 3
#: có test ghim rằng hai đường cho cùng kết quả.
SCALE_SUFFIX: dict[str, str] = {
    "none": "",
    "percent": " * 100",
    "thousand": " / 1000",
    "million": " / 1000000",
    "billion": " / 1000000000",
}


def substitute_placeholders(program: str) -> str:
    """Rewrite `[NUM_0]` to `NUM_0` so the expression parses as Python."""
    return _PLACEHOLDER_PATTERN.sub(r"NUM_\1", program)


def parse_program(program: str, *, value_count: int) -> ast.Expression:
    """Parse and guard one program, or raise `ProgramGuardError`."""
    if len(program) > _MAX_PROGRAM_LENGTH:
        raise ProgramGuardError(
            f"program exceeds max length {_MAX_PROGRAM_LENGTH}: {len(program)} chars"
        )
    source = substitute_placeholders(program)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ProgramGuardError(f"program is not a single expression: {error}") from error

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_NODE_COUNT:
        raise ProgramGuardError(f"program exceeds max node count {_MAX_NODE_COUNT}: {node_count}")

    _guard(tree.body, value_count=value_count)
    return tree


def _guard(node: ast.AST, *, value_count: int) -> None:
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ProgramGuardError(f"operator not allowed: {type(node.op).__name__}")
        _guard(node.left, value_count=value_count)
        _guard(node.right, value_count=value_count)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise ProgramGuardError(f"unary operator not allowed: {type(node.op).__name__}")
        _guard(node.operand, value_count=value_count)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "abs":
            raise ProgramGuardError("only abs() may be called")
        if len(node.args) != 1 or node.keywords:
            raise ProgramGuardError("abs() takes exactly one positional argument")
        _guard(node.args[0], value_count=value_count)
        return
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        if match is None:
            raise ProgramGuardError(f"name not allowed: {node.id}")
        index = int(match.group(1))
        if not 0 <= index < value_count:
            raise ProgramGuardError(
                f"[NUM_{index}] is out of range for {value_count} bound value(s)"
            )
        return
    if isinstance(node, ast.Constant):
        raise ProgramGuardError(f"literal not allowed: {node.value!r}")
    raise ProgramGuardError(f"node not allowed: {type(node).__name__}")


def evaluate(tree: ast.Expression, values: Sequence[Decimal]) -> Decimal:
    """Evaluate a guarded expression over already-bound values."""
    return _evaluate(tree.body, values)


def _evaluate(node: ast.AST, values: Sequence[Decimal]) -> Decimal:
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        assert match is not None  # guarded
        return values[int(match.group(1))]
    if isinstance(node, ast.UnaryOp):
        return -_evaluate(node.operand, values)
    if isinstance(node, ast.Call):
        return abs(_evaluate(node.args[0], values))
    assert isinstance(node, ast.BinOp)  # guarded
    left = _evaluate(node.left, values)
    right = _evaluate(node.right, values)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    try:
        return left / right
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as error:
        raise ProgramEvalError("division_by_zero") from error


def run_program(program: str, values: Sequence[Decimal]) -> Decimal:
    """Guard, then evaluate, one program over its bound values."""
    tree = parse_program(program, value_count=len(values))
    result = evaluate(tree, values)
    if not result.is_finite():
        raise ProgramEvalError("non_finite_result")
    return result


def apply_scale(value: Decimal, scale: ScaleName) -> Decimal:
    """Scale a raw result for presentation. The model picks the enum, never
    the coefficient -- that is what keeps N4' absolute."""
    return value * _SCALE_FACTORS[scale]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/execution/test_masked_program.py -v`
Expected: PASS (toàn bộ, gồm 6 case literal và 12 case node)

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/financial_report_qa/execution/masked_program.py && uv run mypy src/financial_report_qa/execution/masked_program.py`
Expected: không lỗi

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/execution/masked_program.py tests/unit/execution/test_masked_program.py
git commit -m "feat(execution): guard and evaluate masked-PAL programs without eval"
```

---

### Task 3: Binding tất định và render hai đường

**Files:**
- Create: `src/financial_report_qa/execution/program_binding.py`
- Test: `tests/unit/execution/test_program_binding.py`

**Interfaces:**
- Consumes: `NAME_PATTERN`, `SCALE_SUFFIX`, `parse_program`, `evaluate`, `apply_scale` (Task 2); `BoundValue`, `CellCandidate`, `ProgramDecision`, `ScaleName` (Task 1); `_lit` từ `financial_report_qa.execution.pandas_query`; `ProgramBindingError` (Task 1).
- Produces:
  - `values_by_position(frame: pd.DataFrame) -> dict[tuple[str, int, int], Decimal]`
  - `bind_values(decision, candidates: Sequence[CellCandidate], values: Mapping[tuple[str, int, int], Decimal]) -> tuple[BoundValue, ...]`
  - `render_cell_lookup(bound: BoundValue) -> str`
  - `render_program_pandas(program: str, bindings: Sequence[BoundValue], scale: ScaleName) -> str`

**Vì sao render hai lần:** linter C5 đòi `pandas_query` tham chiếu ít nhất một cột CSV và C7 đòi nó replay được ra đúng `answer`. `([NUM_0] - [NUM_1]) / [NUM_1]` không thoả cả hai. Cùng một cây AST được render ra `Decimal` (Task 2) và ra chuỗi tra cứu CSV (task này), và C7 chính là chỗ kiểm hai đường khớp nhau.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/execution/test_program_binding.py`:

```python
from decimal import Decimal

import pandas as pd
import pytest

from financial_report_qa.core.errors import ProgramBindingError
from financial_report_qa.execution.masked_program import run_program
from financial_report_qa.execution.pandas_query import replay_pandas_query
from financial_report_qa.execution.program_binding import (
    bind_values,
    render_cell_lookup,
    render_program_pandas,
    values_by_position,
)
from financial_report_qa.execution.program_contracts import CellCandidate, ProgramDecision

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(index: int, row_idx: int, col_idx: int, label: str, period: int) -> CellCandidate:
    return CellCandidate(
        index=index,
        table_id=_TABLE_ID,
        company_code="VCB",
        row_idx=row_idx,
        col_idx=col_idx,
        row_path=f"Doanh thu > {label}",
        row_label_raw=label,
        row_label_canonical="doanh_thu_thuan",
        col_path=f"Năm_{period}",
        period=period,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 1,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": "doanh_thu_thuan",
                "column_label": "Năm 2022",
                "period": 2022,
                "unit": "triệu VND",
                "value": 4500.0,
            },
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 2,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": "doanh_thu_thuan",
                "column_label": "Năm 2023",
                "period": 2023,
                "unit": "triệu VND",
                "value": 5310.0,
            },
        ]
    )


def _candidates() -> tuple[CellCandidate, ...]:
    return (
        _candidate(0, 3, 1, "Doanh thu thuần", 2022),
        _candidate(1, 3, 2, "Doanh thu thuần", 2023),
    )


def test_placeholder_index_follows_the_order_of_cells_not_candidate_order() -> None:
    # `cells=[1, 0]` nghĩa là [NUM_0] là ứng viên 1, [NUM_1] là ứng viên 0.
    decision = ProgramDecision(question_id=7, cells=(1, 0), program="[NUM_0] - [NUM_1]")

    bindings = bind_values(decision, _candidates(), values_by_position(_frame()))

    assert [bound.num_index for bound in bindings] == [0, 1]
    assert [bound.candidate_index for bound in bindings] == [1, 0]
    assert [bound.value for bound in bindings] == [Decimal("5310.0"), Decimal("4500.0")]


def test_binding_rejects_an_index_outside_the_candidate_list() -> None:
    decision = ProgramDecision(question_id=7, cells=(9,), program="[NUM_0]")

    with pytest.raises(ProgramBindingError, match="candidate_index_out_of_range"):
        bind_values(decision, _candidates(), values_by_position(_frame()))


def test_binding_rejects_a_candidate_with_no_value_in_the_frame() -> None:
    decision = ProgramDecision(question_id=7, cells=(0,), program="[NUM_0]")
    orphan = (_candidate(0, 99, 99, "Doanh thu thuần", 2022),)

    with pytest.raises(ProgramBindingError):
        bind_values(decision, orphan, values_by_position(_frame()))


def test_cell_lookup_names_the_row_so_the_query_explains_itself() -> None:
    decision = ProgramDecision(question_id=7, cells=(1,), program="[NUM_0]")
    bound = bind_values(decision, _candidates(), values_by_position(_frame()))[0]

    rendered = render_cell_lookup(bound)

    assert "row_label_canonical" in rendered
    assert "df1.row_idx == 3" in rendered
    assert "df1.col_idx == 2" in rendered
    assert rendered.endswith('["value"].iloc[0]')


def test_both_renderings_of_the_same_program_agree() -> None:
    # Đây là bất biến trung tâm của Task 3: một cây AST, hai cách đọc.
    decision = ProgramDecision(
        question_id=7, cells=(1, 0), program="([NUM_0] - [NUM_1]) / [NUM_1]", scale="percent"
    )
    frame = _frame()
    bindings = bind_values(decision, _candidates(), values_by_position(frame))

    from financial_report_qa.execution.masked_program import apply_scale

    direct = apply_scale(run_program(decision.program, [b.value for b in bindings]), "percent")
    query = render_program_pandas(decision.program, bindings, "percent")
    replayed = replay_pandas_query(query, frame)

    assert replayed == pytest.approx(float(direct))


def test_scale_none_appends_no_suffix() -> None:
    decision = ProgramDecision(question_id=7, cells=(1,), program="[NUM_0]")
    bindings = bind_values(decision, _candidates(), values_by_position(_frame()))

    assert not render_program_pandas(decision.program, bindings, "none").endswith("100")


def test_abs_and_unary_minus_survive_the_round_trip() -> None:
    decision = ProgramDecision(question_id=7, cells=(0, 1), program="abs(-[NUM_0] + [NUM_1])")
    frame = _frame()
    bindings = bind_values(decision, _candidates(), values_by_position(frame))

    query = render_program_pandas(decision.program, bindings, "none")
    replayed = replay_pandas_query(query, frame)

    assert replayed == pytest.approx(810.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/execution/test_program_binding.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.execution.program_binding'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/execution/program_binding.py`:

```python
"""Bind `[NUM_i]` to real cells, then render the same program two ways.

The arithmetic path (`masked_program.evaluate`) produces the answer. The
pandas path here produces the `pandas_query` the submission carries, because
compliance C5 requires the query to reference a CSV column and C7 requires it
to replay to the same answer -- neither of which a bare `[NUM_0] - [NUM_1]`
can satisfy. Both readings walk the identical guarded AST, so C7 doubles as a
free third consistency check between them.

The lookup shape keeps a semantic clause (`row_label_*`) alongside the
positional ones: the positional clauses make the cell unique, the semantic
clause is what makes the emitted query explain which line the answer came
from. Compliance already strips `row_idx`/`col_idx`/`period` comparisons
before its C4 literal scan, so the positional clauses cannot be mistaken for
a hardcoded answer.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from decimal import Decimal

import pandas as pd

from financial_report_qa.core.errors import ProgramBindingError
from financial_report_qa.execution.masked_program import (
    NAME_PATTERN,
    SCALE_SUFFIX,
    parse_program,
)
from financial_report_qa.execution.pandas_query import _lit
from financial_report_qa.execution.program_contracts import (
    BoundValue,
    CellCandidate,
    ProgramDecision,
    ScaleName,
)

_BINOP_SYMBOL: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


def values_by_position(frame: pd.DataFrame) -> dict[tuple[str, int, int], Decimal]:
    """Index the cell frame by `(table_id, row_idx, col_idx)`.

    Values go through `str()` before `Decimal` so a long significand keeps
    every digit -- the same round-trip hazard `compliance.check_bundle`
    documents for `pd.read_csv`.
    """
    return {
        (str(row.table_id), int(row.row_idx), int(row.col_idx)): Decimal(str(row.value))
        for row in frame.itertuples()
    }


def bind_values(
    decision: ProgramDecision,
    candidates: Sequence[CellCandidate],
    values: Mapping[tuple[str, int, int], Decimal],
) -> tuple[BoundValue, ...]:
    """Resolve every `cells[i]` to a real cell. `[NUM_i]` is `cells[i]`."""
    by_index = {candidate.index: candidate for candidate in candidates}
    bindings: list[BoundValue] = []
    for num_index, candidate_index in enumerate(decision.cells):
        candidate = by_index.get(candidate_index)
        if candidate is None:
            raise ProgramBindingError(
                f"candidate_index_out_of_range: {candidate_index} "
                f"is not one of {len(candidates)} candidates"
            )
        position = (candidate.table_id, candidate.row_idx, candidate.col_idx)
        if position not in values:
            raise ProgramBindingError(f"no numeric value at {position} for candidate {candidate_index}")
        bindings.append(
            BoundValue(
                num_index=num_index,
                candidate_index=candidate_index,
                table_id=candidate.table_id,
                row_idx=candidate.row_idx,
                col_idx=candidate.col_idx,
                row_path=candidate.row_path,
                row_label_raw=candidate.row_label_raw,
                row_label_canonical=candidate.row_label_canonical,
                col_path=candidate.col_path,
                period=candidate.period,
                value=values[position],
                unit=candidate.unit,
            )
        )
    return tuple(bindings)


def render_cell_lookup(bound: BoundValue) -> str:
    """Render one bound cell as a unique, self-explaining CSV lookup."""
    if bound.row_label_canonical is not None:
        label_clause = f"(df1.row_label_canonical == {_lit(bound.row_label_canonical)})"
    else:
        label_clause = f"(df1.row_label_raw == {_lit(bound.row_label_raw)})"
    clauses = [
        label_clause,
        f"(df1.table_id == {_lit(bound.table_id)})",
        f"(df1.row_idx == {bound.row_idx})",
        f"(df1.col_idx == {bound.col_idx})",
    ]
    return f'df1[{" & ".join(clauses)}]["value"].iloc[0]'


def render_program_pandas(
    program: str, bindings: Sequence[BoundValue], scale: ScaleName
) -> str:
    """Render the guarded program with every `[NUM_i]` replaced by a lookup."""
    tree = parse_program(program, value_count=len(bindings))
    lookups = [render_cell_lookup(bound) for bound in bindings]
    return _render(tree.body, lookups) + SCALE_SUFFIX[scale]


def _render(node: ast.AST, lookups: Sequence[str]) -> str:
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        assert match is not None  # guarded
        return lookups[int(match.group(1))]
    if isinstance(node, ast.UnaryOp):
        return f"-({_render(node.operand, lookups)})"
    if isinstance(node, ast.Call):
        return f"abs({_render(node.args[0], lookups)})"
    assert isinstance(node, ast.BinOp)  # guarded
    symbol = _BINOP_SYMBOL[type(node.op)]
    return f"({_render(node.left, lookups)} {symbol} {_render(node.right, lookups)})"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/execution/test_program_binding.py -v`
Expected: PASS (7 tests)

Nếu `test_both_renderings_of_the_same_program_agree` fail vì `replay_pandas_query` từ chối grammar, đọc thông báo lỗi: nó chỉ ra node nào ngoài whitelist của replayer. Hình dạng lookup ở trên cố ý giống hệt thứ `_cell_expr` sinh ra, nên nếu lệch thì lệch ở dấu ngoặc — sửa `_render`, không nới whitelist của replayer.

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/financial_report_qa/execution/program_binding.py && uv run mypy src/financial_report_qa/execution/program_binding.py`
Expected: không lỗi

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/execution/program_binding.py tests/unit/execution/test_program_binding.py
git commit -m "feat(execution): bind masked placeholders and render the program as a CSV query"
```

---

### Task 4: Dựng danh sách ô ứng viên

**Files:**
- Create: `src/financial_report_qa/planning/cell_candidates.py`
- Test: `tests/unit/planning/test_cell_candidates.py`

**Interfaces:**
- Consumes: `CellCandidate` (Task 1); `RowFusedCandidate` từ `financial_report_qa.retrieval.row_fusion_contracts` (trường dùng tới: `.table_id`, `.row_idx`, `.rank`, `.metadata.row_group_context_raw`); cell frame của `financial_report_qa.execution.cell_frame.build_cell_frame` (cột dùng tới: `table_id`, `company_code`, `row_idx`, `col_idx`, `row_label_raw`, `row_label_canonical`, `column_label`, `period`, `unit`, `statement_type`, `value`).
- Produces: `build_cell_candidates(frame, row_candidates, *, periods=(), max_candidates=200) -> tuple[CellCandidate, ...]`

`build_cell_frame` đã lọc sẵn `value_numeric IS NOT NULL` và `col_idx > 0`, nên ràng buộc "ô rỗng không vào danh sách" được thoả từ nguồn — task này không lọc lại, chỉ không được thêm ô nào ngoài frame.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/planning/test_cell_candidates.py`:

```python
import pandas as pd

from financial_report_qa.planning.cell_candidates import build_cell_candidates
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

_TABLE_ID = "tbl_" + "a" * 64


def _row_candidate(row_idx: int, rank: int, group: str | None) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{_TABLE_ID}|row_{row_idx}",
        table_id=_TABLE_ID,
        row_idx=row_idx,
        rank=rank,
        fused_score=1.0 - rank / 10,
        metadata=RowMetadata(
            table_id=_TABLE_ID,
            row_idx=row_idx,
            company_code="VCB",
            row_label_raw="Doanh thu thuần" if row_idx == 3 else "Giá vốn",
            row_group_context_raw=group,
        ),
        snippet="x",
    )


def _frame() -> pd.DataFrame:
    rows = []
    for row_idx, label in ((3, "Doanh thu thuần"), (4, "Giá vốn")):
        for col_idx, period in ((1, 2022), (2, 2023)):
            rows.append(
                {
                    "table_id": _TABLE_ID,
                    "company_code": "VCB",
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "row_label_raw": label,
                    "row_label_canonical": None,
                    "column_label": f"Năm {period}",
                    "period": period,
                    "statement_type": "income_statement",
                    "unit": "triệu VND",
                    "value": 100.0 * row_idx + col_idx,
                }
            )
    return pd.DataFrame(rows)


def test_candidates_are_numbered_from_zero_in_row_rank_then_column_order() -> None:
    candidates = build_cell_candidates(
        _frame(), (_row_candidate(4, 1, None), _row_candidate(3, 2, "Doanh thu"))
    )

    assert [candidate.index for candidate in candidates] == [0, 1, 2, 3]
    # Dòng hạng 1 (row_idx 4) đứng trước dòng hạng 2 (row_idx 3).
    assert [candidate.row_idx for candidate in candidates] == [4, 4, 3, 3]
    assert [candidate.col_idx for candidate in candidates] == [1, 2, 1, 2]


def test_row_path_carries_the_group_prefix_when_there_is_one() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, "Doanh thu"),))

    assert candidates[0].row_path == "Doanh thu > Doanh thu thuần"


def test_row_path_is_the_bare_label_without_a_group() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(4, 1, None),))

    assert candidates[0].row_path == "Giá vốn"


def test_col_path_comes_from_the_column_label() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),))

    assert candidates[0].col_path == "Năm 2022"
    assert candidates[0].period == 2022


def test_periods_filter_narrows_the_columns() -> None:
    candidates = build_cell_candidates(
        _frame(), (_row_candidate(3, 1, None),), periods=("2023",)
    )

    assert [candidate.period for candidate in candidates] == [2023]


def test_an_empty_periods_filter_keeps_every_column() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),), periods=())

    assert len(candidates) == 2


def test_no_candidate_carries_a_value() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(3, 1, None),))

    assert all(not hasattr(candidate, "value") for candidate in candidates)


def test_max_candidates_truncates_from_the_lowest_ranked_row() -> None:
    candidates = build_cell_candidates(
        _frame(),
        (_row_candidate(4, 1, None), _row_candidate(3, 2, None)),
        max_candidates=3,
    )

    assert len(candidates) == 3
    assert [candidate.index for candidate in candidates] == [0, 1, 2]
    assert candidates[-1].row_idx == 3


def test_a_row_candidate_with_no_cells_in_the_frame_is_skipped() -> None:
    candidates = build_cell_candidates(_frame(), (_row_candidate(99, 1, None),))

    assert candidates == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_cell_candidates.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.planning.cell_candidates'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/planning/cell_candidates.py`:

```python
"""Build the numbered cell list the masked-PAL decision step chooses from.

Order is the contract: `ProgramDecision.cells` are positions in this list, so
two runs that produce a different order produce different decisions. Cells are
emitted in row-fusion rank order, then by `col_idx`, so the highest-ranked row
gets the lowest indices and truncation drops the least-likely rows first.

No candidate carries a value. `build_cell_frame` already filters out cells
with no `value_numeric`, so anything in the frame is bindable; this module
must not add a cell that is not in the frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DEFAULT_MAX_CANDIDATES = 200


def _row_path(label: str, group: str | None) -> str:
    if group and group.strip():
        return f"{group.strip()} > {label}"
    return label


def _optional_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text else None


def build_cell_candidates(
    frame: pd.DataFrame,
    row_candidates: Sequence[RowFusedCandidate],
    *,
    periods: Sequence[str] = (),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[CellCandidate, ...]:
    """Number every numeric cell of the ranked rows, best-ranked row first."""
    wanted_periods = {int(period) for period in periods if str(period).isdigit()}
    candidates: list[CellCandidate] = []
    for row_candidate in sorted(row_candidates, key=lambda item: item.rank):
        rows = frame[
            (frame["table_id"] == row_candidate.table_id)
            & (frame["row_idx"] == row_candidate.row_idx)
        ].sort_values("col_idx")
        for row in rows.itertuples():
            period = None if pd.isna(row.period) else int(row.period)
            if wanted_periods and period not in wanted_periods:
                continue
            if len(candidates) >= max_candidates:
                return tuple(candidates)
            label_raw = str(row.row_label_raw)
            candidates.append(
                CellCandidate(
                    index=len(candidates),
                    table_id=str(row.table_id),
                    company_code=_optional_str(row.company_code),
                    row_idx=int(row.row_idx),
                    col_idx=int(row.col_idx),
                    row_path=_row_path(
                        label_raw, row_candidate.metadata.row_group_context_raw
                    ),
                    row_label_raw=label_raw,
                    row_label_canonical=_optional_str(row.row_label_canonical),
                    col_path=str(row.column_label or ""),
                    period=period,
                    statement_type=_optional_str(getattr(row, "statement_type", None)),
                    unit=_optional_str(row.unit),
                )
            )
    return tuple(candidates)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_cell_candidates.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/financial_report_qa/planning/cell_candidates.py && uv run mypy src/financial_report_qa/planning/cell_candidates.py`
Expected: không lỗi

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/planning/cell_candidates.py tests/unit/planning/test_cell_candidates.py
git commit -m "feat(planning): number candidate cells for the masked decision step"
```

---

### Task 5: Payload batch và file quyết định

**Files:**
- Modify: `src/financial_report_qa/planning/row_choice_batch.py` (thêm hàm mới, giữ nguyên hàm cũ)
- Create: `src/financial_report_qa/planning/program_decisions.py`
- Test: `tests/unit/planning/test_program_batch.py`

**Interfaces:**
- Consumes: `CellCandidate`, `ProgramDecision` (Task 1); `QueryEntities` từ `financial_report_qa.planning.entity_contracts`; `PlanningArtifactError` từ `financial_report_qa.core.errors`.
- Produces:
  - `build_program_batch_payload(question_id: int, question: str, entities: QueryEntities, candidates: Sequence[CellCandidate]) -> dict[str, object]`
  - `load_program_decisions(path: Path) -> dict[int, ProgramDecision]`

Hàm cũ `build_batch_payload` **không đụng tới** — nó còn caller cho tới Task 11.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/planning/test_program_batch.py`:

```python
import json
from pathlib import Path

import pytest

from financial_report_qa.core.errors import PlanningArtifactError
from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.program_decisions import load_program_decisions
from financial_report_qa.planning.row_choice_batch import build_program_batch_payload

_TABLE_ID = "tbl_" + "a" * 64


def _candidate(index: int) -> CellCandidate:
    return CellCandidate(
        index=index,
        table_id=_TABLE_ID,
        company_code="VCB",
        row_idx=3,
        col_idx=index + 1,
        row_path="Doanh thu > Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path=f"Năm_{2022 + index}",
        period=2022 + index,
        unit="triệu VND",
    )


def _entities() -> QueryEntities:
    return QueryEntities(company_codes=("VCB",), periods=("2022", "2023"))


def test_payload_never_carries_a_value_or_a_score() -> None:
    payload = build_program_batch_payload(7, "Tăng trưởng?", _entities(), (_candidate(0),))

    serialized = json.dumps(payload, ensure_ascii=False)
    assert "value" not in serialized
    assert "score" not in serialized


def test_payload_numbers_candidates_in_the_given_order() -> None:
    payload = build_program_batch_payload(
        7, "Tăng trưởng?", _entities(), (_candidate(0), _candidate(1))
    )

    assert [item["index"] for item in payload["candidates"]] == [0, 1]


def test_payload_carries_the_parsed_companies_and_periods() -> None:
    payload = build_program_batch_payload(7, "Tăng trưởng?", _entities(), (_candidate(0),))

    assert payload["question_id"] == 7
    assert payload["companies"] == ["VCB"]
    assert payload["periods"] == ["2022", "2023"]


def test_decisions_load_keyed_by_question_id(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps(
            {"question_id": 7, "cells": [1, 0], "program": "[NUM_0] - [NUM_1]",
             "uses": [{"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},
                      {"num": 1, "row": "Doanh thu thuần", "col": "Năm 2022"}],
             "scale": "none"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = load_program_decisions(path)

    assert decisions[7].cells == (1, 0)
    assert decisions[7].uses[0].col == "Năm 2023"


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]"})
        + "\n\n",
        encoding="utf-8",
    )

    assert list(load_program_decisions(path)) == [1]


def test_a_duplicate_question_id_is_rejected(tmp_path: Path) -> None:
    line = json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]"})
    path = tmp_path / "decisions.jsonl"
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")

    with pytest.raises(PlanningArtifactError, match="duplicate"):
        load_program_decisions(path)


def test_malformed_json_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(PlanningArtifactError, match="line 1"):
        load_program_decisions(path)


def test_a_decision_carrying_a_numeric_value_field_is_rejected(tmp_path: Path) -> None:
    # N7: file quyết định không được mang giá trị số. `extra="forbid"` là chốt.
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"question_id": 1, "cells": [0], "program": "[NUM_0]", "value": 4500}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PlanningArtifactError):
        load_program_decisions(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PlanningArtifactError):
        load_program_decisions(tmp_path / "absent.jsonl")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/planning/test_program_batch.py -v`
Expected: FAIL với `ImportError: cannot import name 'build_program_batch_payload'`

- [ ] **Step 3: Thêm payload builder**

Thêm vào cuối `src/financial_report_qa/planning/row_choice_batch.py`:

```python
def _cell_candidate_payload(candidate: CellCandidate) -> dict[str, object]:
    return {
        "index": candidate.index,
        "company_code": candidate.company_code,
        "row_path": candidate.row_path,
        "col_path": candidate.col_path,
        "period": candidate.period,
        "statement_type": candidate.statement_type,
        "unit": candidate.unit,
    }


def build_program_batch_payload(
    question_id: int,
    question: str,
    entities: QueryEntities,
    candidates: Sequence[CellCandidate],
) -> dict[str, object]:
    """Một dòng JSONL cho bước sinh chương trình masked (spec 2026-08-24 §4.3).

    Không trường nào mang giá trị ô hay điểm fusion (N7). Thứ tự `candidates`
    **là** hợp đồng: `ProgramDecision.cells` là vị trí trong chính danh sách
    này, nên sắp lại ở đây là làm sai mọi quyết định đã sinh.
    """
    return {
        "question_id": question_id,
        "question": question,
        "companies": list(entities.company_codes),
        "periods": list(entities.periods),
        "candidates": [_cell_candidate_payload(candidate) for candidate in candidates],
    }
```

Thêm import ở đầu file:

```python
from financial_report_qa.execution.program_contracts import CellCandidate
```

- [ ] **Step 4: Viết loader**

Tạo `src/financial_report_qa/planning/program_decisions.py`:

```python
"""Read the offline masked-PAL decision file.

One JSONL line per question. `ProgramDecision` forbids extra fields, so a
line that smuggles in a numeric value is rejected rather than ignored -- that
rejection is what makes N7 enforceable on a file someone edited by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import PlanningArtifactError
from financial_report_qa.execution.program_contracts import ProgramDecision


def load_program_decisions(path: Path) -> dict[int, ProgramDecision]:
    """Load every decision, keyed by `question_id`, in file order."""
    if not path.is_file():
        raise PlanningArtifactError(f"program decision file not found: {path}")
    decisions: dict[int, ProgramDecision] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise PlanningArtifactError(f"{path}: invalid JSON on line {number}") from error
        try:
            decision = ProgramDecision.model_validate(payload)
        except ValidationError as error:
            raise PlanningArtifactError(f"{path}: invalid decision on line {number}") from error
        if decision.question_id in decisions:
            raise PlanningArtifactError(
                f"{path}: duplicate question_id {decision.question_id} on line {number}"
            )
        decisions[decision.question_id] = decision
    return decisions
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/planning/test_program_batch.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Chắc chắn không phá hàm cũ**

Run: `uv run pytest tests/unit/planning/ -v`
Expected: PASS toàn bộ, gồm cả test của `build_batch_payload` cũ

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/planning/row_choice_batch.py src/financial_report_qa/planning/program_decisions.py tests/unit/planning/test_program_batch.py
git commit -m "feat(planning): build the masked cell batch payload and load its decisions"
```

---

### Task 6: Verify B — `uses` đối chiếu ô đã bind

**Files:**
- Create: `src/financial_report_qa/verification/use_checks.py`
- Test: `tests/unit/verification/test_use_checks.py`

**Interfaces:**
- Consumes: `BoundValue`, `UseClaim` (Task 1).
- Produces:
  - `UseCheckResult(matched: bool, mismatches: tuple[str, ...])` — dataclass frozen
  - `check_use_bindings(uses: Sequence[UseClaim], bindings: Sequence[BoundValue]) -> UseCheckResult`

**Cái này bắt gì:** trượt chỉ số — model mô tả đúng ô nó muốn nhưng ghi sai số thứ tự trong danh sách đánh số. Không lớp nào khác bắt được, vì đáp án vẫn là một con số hợp lệ đọc từ bảng. Nó **không** bắt được trường hợp model thật sự tin dòng sai là dòng đúng.

Ba luật so nhãn dòng, dừng ở luật đầu tiên khớp:
1. bằng nhau sau chuẩn hoá (casefold + gộp khoảng trắng + bỏ dấu câu ở hai đầu),
2. bằng nhau sau chuẩn hoá với `row_label_canonical`,
3. `uses[i].row` là hậu tố của `row_path` sau chuẩn hoá.

Cột chỉ so **năm** trích từ `uses[i].col`, không so toàn chuỗi: `col_path` thật (`Tổng_cộng_31/12/2022`) và cách model thuật lại (`Năm 2022`) khác hình thức nhưng cùng một kỳ.

Không dùng fuzzy match có ngưỡng: ngưỡng là tham số phải chỉnh tay và sai ở đó là sai âm thầm.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/verification/test_use_checks.py`:

```python
from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, UseClaim
from financial_report_qa.verification.use_checks import check_use_bindings

_TABLE_ID = "tbl_" + "a" * 64


def _bound(num_index: int, period: int, *, canonical: str | None = None) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=num_index,
        table_id=_TABLE_ID,
        row_idx=3,
        col_idx=num_index + 1,
        row_path="Doanh thu > Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        row_label_canonical=canonical,
        col_path=f"Tổng_cộng_31/12/{period}",
        period=period,
        value=Decimal("100"),
    )


def test_matching_claims_pass() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="Năm 2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True
    assert result.mismatches == ()


def test_swapped_indices_are_caught() -> None:
    # Đây chính là kịch bản trượt chỉ số: `uses` giữ nguyên, `cells` hoán vị.
    bindings = (_bound(0, 2022), _bound(1, 2023))
    uses = (
        UseClaim(num=0, row="Doanh thu thuần", col="Năm 2023"),
        UseClaim(num=1, row="Doanh thu thuần", col="Năm 2022"),
    )

    result = check_use_bindings(uses, bindings)

    assert result.matched is False
    assert len(result.mismatches) == 2


def test_case_and_spacing_differences_still_match() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="  DOANH   THU  THUẦN ", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_the_canonical_label_is_accepted() -> None:
    bound = _bound(0, 2023, canonical="doanh thu bán hàng và cung cấp dịch vụ")
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu bán hàng và cung cấp dịch vụ", col="2023"),), (bound,)
    )

    assert result.matched is True


def test_the_child_label_alone_matches_a_grouped_row_path() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_a_wrong_row_label_is_caught() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Giá vốn hàng bán", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is False


def test_a_wrong_year_is_caught() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="Năm 2021"),), (_bound(0, 2023),)
    )

    assert result.matched is False


def test_a_column_claim_with_no_year_only_checks_the_row() -> None:
    result = check_use_bindings(
        (UseClaim(num=0, row="Doanh thu thuần", col="cột cuối"),), (_bound(0, 2023),)
    )

    assert result.matched is True


def test_missing_claims_are_a_mismatch() -> None:
    result = check_use_bindings((), (_bound(0, 2023),))

    assert result.matched is False


def test_a_claim_for_an_unknown_placeholder_is_a_mismatch() -> None:
    result = check_use_bindings(
        (UseClaim(num=5, row="Doanh thu thuần", col="2023"),), (_bound(0, 2023),)
    )

    assert result.matched is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/verification/test_use_checks.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.verification.use_checks'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/verification/use_checks.py`:

```python
"""Verify B: what the model said it used against what it actually bound.

This is the only check that catches index slippage -- the model describing
the right cell while emitting the wrong position in a numbered list. Nothing
downstream can catch it, because the resulting answer is still a legitimate
number read out of a real table.

It does NOT catch a model that genuinely believes the wrong row answers the
question. That limit is deliberate and stated in §9 of the spec.

Every rule is exact after normalization. No fuzzy threshold: a threshold is a
knob someone has to tune, and being wrong about it fails silently.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from financial_report_qa.execution.program_contracts import BoundValue, UseClaim

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(19|20)\d{2}")


def _normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip edge punctuation. Accents are kept
    -- stripping them would merge Vietnamese terms that are genuinely
    different."""
    folded = unicodedata.normalize("NFC", text).casefold()
    return _WHITESPACE.sub(" ", folded).strip(" .,:;-–—()[]")


def _year_of(text: str) -> int | None:
    match = _YEAR.search(text)
    return int(match.group(0)) if match else None


def _row_matches(claim: str, bound: BoundValue) -> bool:
    claimed = _normalize(claim)
    if claimed == _normalize(bound.row_label_raw):
        return True
    if claimed == _normalize(bound.row_path):
        return True
    if bound.row_label_canonical is not None and claimed == _normalize(
        bound.row_label_canonical
    ):
        return True
    return _normalize(bound.row_path).endswith(claimed)


def _column_matches(claim: str, bound: BoundValue) -> bool:
    claimed_year = _year_of(claim)
    if claimed_year is None:
        # Model không nêu năm -> không có gì để bác bỏ; hàng đã kiểm riêng.
        return True
    return claimed_year == bound.period


@dataclass(frozen=True)
class UseCheckResult:
    """Whether every placeholder's claim agrees with its actual binding."""

    matched: bool
    mismatches: tuple[str, ...]


def check_use_bindings(
    uses: Sequence[UseClaim], bindings: Sequence[BoundValue]
) -> UseCheckResult:
    """Compare each `UseClaim` to the cell its placeholder really bound to."""
    claims = {claim.num: claim for claim in uses}
    mismatches: list[str] = []
    for bound in bindings:
        claim = claims.pop(bound.num_index, None)
        if claim is None:
            mismatches.append(f"[NUM_{bound.num_index}] has no use claim")
            continue
        if not _row_matches(claim.row, bound):
            mismatches.append(
                f"[NUM_{bound.num_index}] claims row {claim.row!r} "
                f"but bound {bound.row_path!r}"
            )
        elif not _column_matches(claim.col, bound):
            mismatches.append(
                f"[NUM_{bound.num_index}] claims column {claim.col!r} "
                f"but bound period {bound.period}"
            )
    for leftover in sorted(claims):
        mismatches.append(f"use claim for [NUM_{leftover}] has no binding")
    return UseCheckResult(matched=not mismatches, mismatches=tuple(mismatches))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/verification/test_use_checks.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/financial_report_qa/verification/use_checks.py && uv run mypy src/financial_report_qa/verification/use_checks.py`
Expected: không lỗi

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/verification/use_checks.py tests/unit/verification/test_use_checks.py
git commit -m "feat(verification): catch index slippage by checking claims against bindings"
```

---

### Task 7: Verify A — giải thích lượt hai

**Files:**
- Create: `src/financial_report_qa/verification/explanation_check.py`
- Test: `tests/unit/verification/test_explanation_check.py`

**Interfaces:**
- Consumes: `ExecutedProgram` (Task 1); `build_number_whitelist` **không dùng được** (nó nhận `FinancialQueryPlan` + `CompiledQuery`); dùng `guard_generated_text` và `NumericGuardResult` từ `financial_report_qa.verification.numeric_guard`.
- Produces:
  - `program_number_whitelist(executed: ExecutedProgram) -> frozenset[Decimal]`
  - `check_explanation(explanation: str, executed: ExecutedProgram) -> NumericGuardResult`

Lượt hai là lượt **duy nhất** thấy giá trị thật, và nó **không có quyền sửa đáp án** — chỉ có quyền báo lệch. Masking ở lượt sinh chương trình vẫn nguyên vẹn.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/verification/test_explanation_check.py`:

```python
from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.verification.explanation_check import (
    check_explanation,
    program_number_whitelist,
)

_TABLE_ID = "tbl_" + "a" * 64


def _bound(num_index: int, period: int, value: str) -> BoundValue:
    return BoundValue(
        num_index=num_index,
        candidate_index=num_index,
        table_id=_TABLE_ID,
        row_idx=3,
        col_idx=num_index + 1,
        row_path="Doanh thu thuần",
        row_label_raw="Doanh thu thuần",
        col_path=f"Năm_{period}",
        period=period,
        value=Decimal(value),
    )


def _executed() -> ExecutedProgram:
    return ExecutedProgram(
        question_id=7,
        program="([NUM_0] - [NUM_1]) / [NUM_1]",
        scale="percent",
        bindings=(_bound(0, 2023, "5310"), _bound(1, 2022, "4500")),
        answer=Decimal("18"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )


def test_whitelist_holds_the_answer_the_values_and_the_periods() -> None:
    whitelist = program_number_whitelist(_executed())

    assert Decimal("18") in whitelist
    assert Decimal("5310") in whitelist
    assert Decimal("4500") in whitelist
    assert Decimal("2023") in whitelist
    assert Decimal("2022") in whitelist


def test_an_explanation_using_only_grounded_numbers_passes() -> None:
    text = "Doanh thu thuần tăng từ 4500 năm 2022 lên 5310 năm 2023, tức 18%."

    assert check_explanation(text, _executed()).allowed is True


def test_an_invented_number_is_rejected() -> None:
    text = "Doanh thu thuần tăng từ 4500 lên 5310, tương đương 810 tỷ và 20%."

    result = check_explanation(text, _executed())

    assert result.allowed is False
    assert "20" in result.disallowed_numbers


def test_an_explanation_with_no_numbers_passes() -> None:
    assert check_explanation("Doanh thu thuần tăng so với năm trước.", _executed()).allowed


def test_the_raw_unscaled_result_is_not_whitelisted() -> None:
    # 0.18 là kết quả trước khi áp `scale`; giải thích chỉ được nêu đáp án đã
    # scale, nếu không thì con số người chấm thấy khác con số bài nộp.
    result = check_explanation("Tỷ lệ là 0.18", _executed())

    assert result.allowed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/verification/test_explanation_check.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.verification.explanation_check'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/verification/explanation_check.py`:

```python
"""Verify A: the second, unmasked pass may describe -- never decide.

The program-generation pass is blind to values, so its own text cannot be
compared against anything. This pass runs after binding and execution, sees
the real numbers, and is checked with the same whitelist posture
`numeric_guard` already applies to the paraphrase path: any number outside
{answer, bound values, periods} is a fabrication, not a warning.

Nothing here can change the answer. It can only report a divergence, which
the pipeline turns into at most one regeneration (N6).
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.program_contracts import ExecutedProgram
from financial_report_qa.verification.numeric_guard import (
    NumericGuardResult,
    guard_generated_text,
)


def program_number_whitelist(executed: ExecutedProgram) -> frozenset[Decimal]:
    """Every number an explanation of `executed` is allowed to mention."""
    whitelist: set[Decimal] = {executed.answer}
    for bound in executed.bindings:
        whitelist.add(bound.value)
        if bound.period is not None:
            whitelist.add(Decimal(bound.period))
    return frozenset(whitelist)


def check_explanation(explanation: str, executed: ExecutedProgram) -> NumericGuardResult:
    """Reject an explanation that mentions a number not grounded in the run."""
    return guard_generated_text(explanation, whitelist=program_number_whitelist(executed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/verification/test_explanation_check.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/verification/explanation_check.py tests/unit/verification/test_explanation_check.py
git commit -m "feat(verification): ground the second-pass explanation in the executed run"
```

---

### Task 8: Pipeline và vòng sinh lại

**Files:**
- Create: `src/financial_report_qa/execution/program_pipeline.py`
- Test: `tests/unit/execution/test_program_pipeline.py`

**Interfaces:**
- Consumes: mọi thứ của Task 1–7.
- Produces:
  - `class DecisionSource(Protocol)` — `decide(question_id: int, attempt: int) -> ProgramDecision`
  - `class ExplanationSource(Protocol)` — `explain(executed: ExecutedProgram) -> str`
  - `PipelineResult(executed: ExecutedProgram | None, failure_code: ProgramFailureCode | None)` — dataclass frozen
  - `run_question(question_id, candidates, frame, decisions, *, explanations=None) -> PipelineResult`
  - `MAX_ATTEMPTS = 2` — hằng số module, **không** phải tham số

**Bất biến N6:** `max_attempts = 2` nghĩa là một lần chạy cộng **đúng một** lần sinh lại. Không có tham số nào cho phép nhiều hơn ở đường live; test ghim điều đó.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/execution/test_program_pipeline.py`:

```python
from decimal import Decimal

import pandas as pd

from financial_report_qa.execution.program_contracts import CellCandidate, ProgramDecision
from financial_report_qa.execution.program_pipeline import run_question

_TABLE_ID = "tbl_" + "a" * 64


def _candidates() -> tuple[CellCandidate, ...]:
    return tuple(
        CellCandidate(
            index=index,
            table_id=_TABLE_ID,
            company_code="VCB",
            row_idx=3,
            col_idx=index + 1,
            row_path="Doanh thu thuần",
            row_label_raw="Doanh thu thuần",
            col_path=f"Năm_{2022 + index}",
            period=2022 + index,
        )
        for index in range(2)
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 1,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": None,
                "column_label": "Năm 2022",
                "period": 2022,
                "unit": "triệu VND",
                "value": 4500.0,
            },
            {
                "table_id": _TABLE_ID,
                "company_code": "VCB",
                "row_idx": 3,
                "col_idx": 2,
                "row_label_raw": "Doanh thu thuần",
                "row_label_canonical": None,
                "column_label": "Năm 2023",
                "period": 2023,
                "unit": "triệu VND",
                "value": 5310.0,
            },
        ]
    )


class _Decisions:
    """Trả về từng quyết định theo lần thử, đếm số lần được hỏi."""

    def __init__(self, *decisions: ProgramDecision) -> None:
        self._decisions = decisions
        self.attempts: list[int] = []

    def decide(self, question_id: int, attempt: int) -> ProgramDecision:
        self.attempts.append(attempt)
        return self._decisions[min(attempt, len(self._decisions) - 1)]


def _good() -> ProgramDecision:
    return ProgramDecision(
        question_id=7,
        cells=(1,),
        program="[NUM_0]",
        uses=({"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},),  # type: ignore[arg-type]
    )


def _bad_literal() -> ProgramDecision:
    return ProgramDecision(question_id=7, cells=(1,), program="[NUM_0] * 100")


def test_a_clean_first_attempt_is_not_marked_regenerated() -> None:
    source = _Decisions(_good())

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.answer == Decimal("5310.0")
    assert result.executed.regenerated is False
    assert result.executed.low_confidence is False
    assert source.attempts == [0]


def test_a_bad_first_attempt_is_retried_exactly_once_and_recovers() -> None:
    source = _Decisions(_bad_literal(), _good())

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.regenerated is True
    assert result.executed.low_confidence is False
    assert source.attempts == [0, 1]


def test_two_bad_attempts_still_produce_an_answer_marked_low_confidence() -> None:
    # Bỏ trống chắc chắn 0 điểm; sai thì cũng chỉ có thể 0 điểm.
    bad_uses = ProgramDecision(
        question_id=7,
        cells=(1,),
        program="[NUM_0]",
        uses=({"num": 0, "row": "Giá vốn hàng bán", "col": "Năm 2023"},),  # type: ignore[arg-type]
    )
    source = _Decisions(bad_uses, bad_uses)

    result = run_question(7, _candidates(), _frame(), source)

    assert result.executed is not None
    assert result.executed.low_confidence is True
    assert result.executed.failure_code == "use_binding_mismatch"
    assert source.attempts == [0, 1]


def test_the_retry_never_runs_a_third_time() -> None:
    source = _Decisions(_bad_literal(), _bad_literal(), _good())

    result = run_question(7, _candidates(), _frame(), source)

    assert source.attempts == [0, 1]
    # Không bind được lần nào -> không có đáp án, nhưng phải nói rõ vì sao.
    assert result.executed is None
    assert result.failure_code == "numeric_literal_in_program"


def test_an_empty_candidate_list_fails_before_asking_the_model() -> None:
    source = _Decisions(_good())

    result = run_question(7, (), _frame(), source)

    assert result.executed is None
    assert result.failure_code == "no_cell_candidates"
    assert source.attempts == []


def test_a_fabricated_number_in_the_explanation_triggers_the_retry() -> None:
    source = _Decisions(_good(), _good())
    explanations = iter(["Doanh thu là 9999.", "Doanh thu là 5310."])

    result = run_question(
        7, _candidates(), _frame(), source,
        explanations=lambda executed: next(explanations),
    )

    assert result.executed is not None
    assert result.executed.regenerated is True
    assert result.executed.low_confidence is False


def test_division_by_zero_is_reported_by_its_code() -> None:
    zero_frame = _frame()
    zero_frame.loc[zero_frame["col_idx"] == 1, "value"] = 0.0
    decision = ProgramDecision(
        question_id=7,
        cells=(1, 0),
        program="[NUM_0] / [NUM_1]",
        uses=(
            {"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},  # type: ignore[arg-type]
            {"num": 1, "row": "Doanh thu thuần", "col": "Năm 2022"},  # type: ignore[arg-type]
        ),
    )
    source = _Decisions(decision, decision)

    result = run_question(7, _candidates(), zero_frame, source)

    assert result.executed is None
    assert result.failure_code == "division_by_zero"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/execution/test_program_pipeline.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.execution.program_pipeline'`

- [ ] **Step 3: Viết module**

Tạo `src/financial_report_qa/execution/program_pipeline.py`:

```python
"""One question, one straight line: decide -> guard -> bind -> run -> verify.

N6 allows exactly one retry of the decision step and nothing else: no second
route, no alternate strategy, no third attempt. `MAX_ATTEMPTS = 2` is that
rule written down, and the live path has no parameter to raise it.

When both attempts diverge but one of them still produced a number, the number
is submitted with `low_confidence` set. Leaving it blank scores zero for
certain; a wrong answer can only also score zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from financial_report_qa.core.errors import ProgramError, ProgramGuardError
from financial_report_qa.execution.masked_program import apply_scale, run_program
from financial_report_qa.execution.program_binding import (
    bind_values,
    render_program_pandas,
    values_by_position,
)
from financial_report_qa.execution.program_contracts import (
    CellCandidate,
    ExecutedProgram,
    ProgramDecision,
    ProgramFailureCode,
)
from financial_report_qa.verification.explanation_check import check_explanation
from financial_report_qa.verification.use_checks import check_use_bindings

#: N6: một lần chạy cộng đúng một lần sinh lại. Không nới ở đường live.
MAX_ATTEMPTS = 2


class DecisionSource(Protocol):
    def decide(self, question_id: int, attempt: int) -> ProgramDecision: ...


ExplanationSource = Callable[[ExecutedProgram], str]


@dataclass(frozen=True)
class PipelineResult:
    """Either an executed program, or the code that stopped every attempt."""

    executed: ExecutedProgram | None
    failure_code: ProgramFailureCode | None


def _failure_code(error: Exception) -> ProgramFailureCode:
    message = str(error)
    if "candidate_index_out_of_range" in message:
        return "candidate_index_out_of_range"
    if "division_by_zero" in message:
        return "division_by_zero"
    if "non_finite_result" in message:
        return "non_finite_result"
    if isinstance(error, ProgramGuardError) and "literal not allowed" in message:
        return "numeric_literal_in_program"
    return "program_node_not_allowed"


def _execute(
    question_id: int,
    decision: ProgramDecision,
    candidates: Sequence[CellCandidate],
    frame: pd.DataFrame,
    *,
    regenerated: bool,
) -> ExecutedProgram:
    bindings = bind_values(decision, candidates, values_by_position(frame))
    raw = run_program(decision.program, [bound.value for bound in bindings])
    answer = apply_scale(raw, decision.scale)
    query = render_program_pandas(decision.program, bindings, decision.scale)
    return ExecutedProgram(
        question_id=question_id,
        program=decision.program,
        scale=decision.scale,
        bindings=bindings,
        answer=answer,
        pandas_query=query,
        table_ids=tuple(sorted({bound.table_id for bound in bindings})),
        regenerated=regenerated,
    )


def run_question(
    question_id: int,
    candidates: Sequence[CellCandidate],
    frame: pd.DataFrame,
    decisions: DecisionSource,
    *,
    explanations: ExplanationSource | None = None,
) -> PipelineResult:
    """Answer one question, retrying the decision step at most once."""
    if not candidates:
        return PipelineResult(executed=None, failure_code="no_cell_candidates")

    last_executed: ExecutedProgram | None = None
    last_code: ProgramFailureCode | None = None

    for attempt in range(MAX_ATTEMPTS):
        regenerated = attempt > 0
        try:
            decision = decisions.decide(question_id, attempt)
            executed = _execute(
                question_id, decision, candidates, frame, regenerated=regenerated
            )
        except ProgramError as error:
            last_code = _failure_code(error)
            continue

        use_result = check_use_bindings(decision.uses, executed.bindings)
        if not use_result.matched:
            last_executed, last_code = executed, "use_binding_mismatch"
            continue

        if explanations is not None:
            guard = check_explanation(explanations(executed), executed)
            if not guard.allowed:
                last_executed = executed
                last_code = "explanation_number_not_grounded"
                continue

        return PipelineResult(executed=executed, failure_code=None)

    if last_executed is not None:
        return PipelineResult(
            executed=last_executed.model_copy(
                update={"low_confidence": True, "failure_code": last_code}
            ),
            failure_code=last_code,
        )
    return PipelineResult(executed=None, failure_code=last_code)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/execution/test_program_pipeline.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint + type**

Run: `uv run ruff check src/financial_report_qa/execution/program_pipeline.py && uv run mypy src/financial_report_qa/execution/program_pipeline.py`
Expected: không lỗi

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/execution/program_pipeline.py tests/unit/execution/test_program_pipeline.py
git commit -m "feat(execution): run one question end to end with a single bounded retry"
```

---

### Task 9: Linter C8 và nới C4 cho hệ số scale

**Files:**
- Modify: `src/financial_report_qa/submission/compliance.py`
- Modify: `src/financial_report_qa/submission/contracts.py` (thêm `program` lên `SubmissionItem`)
- Test: `tests/unit/submission/test_compliance_c8.py`

**Interfaces:**
- Consumes: `parse_program` (Task 2); `SubmissionItem`.
- Produces: `check_program_literals(program: str) -> str | None` — trả về chi tiết vi phạm hoặc `None`; C8 trong `check_item`.

**Hai thay đổi, hai lý do:**
- **C8** thi hành N4′ ở chốt chặn cuối: chương trình đã lưu phải parse qua guard sạch. Đây là chỗ một quyết định bị sửa tay bị chặn.
- **C4** hiện strip `row_idx|col_idx|period == N` và `.iloc[N]`. Hệ số scale `* 100` do renderer của ta nối vào **không** được strip, nên một đáp án tình cờ bằng 100 sẽ bị báo sai. Thêm một pattern strip đúng các hậu tố scale hợp lệ.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/submission/test_compliance_c8.py`:

```python
import pytest

from financial_report_qa.submission.compliance import (
    _strip_structural_tokens,
    check_program_literals,
)


def test_a_clean_program_passes() -> None:
    assert check_program_literals("([NUM_0] - [NUM_1]) / [NUM_1]") is None


def test_a_bare_lookup_passes() -> None:
    assert check_program_literals("[NUM_0]") is None


@pytest.mark.parametrize(
    "program", ["[NUM_0] * 100", "[NUM_0] + 0", "abs([NUM_0] - 1)", "round([NUM_0])"]
)
def test_a_literal_or_forbidden_call_is_reported(program: str) -> None:
    detail = check_program_literals(program)

    assert detail is not None


def test_the_scale_suffix_is_stripped_before_the_c4_literal_scan() -> None:
    query = 'df1[(df1.row_idx == 3)]["value"].iloc[0] * 100'

    assert "100" not in _strip_structural_tokens(query)


@pytest.mark.parametrize("suffix", [" * 100", " / 1000", " / 1000000", " / 1000000000"])
def test_every_scale_suffix_is_stripped(suffix: str) -> None:
    query = f'df1[(df1.row_idx == 3)]["value"].iloc[0]{suffix}'

    stripped = _strip_structural_tokens(query)

    assert not any(character.isdigit() for character in stripped.split("]")[-1])


def test_a_real_literal_elsewhere_is_still_visible_to_c4() -> None:
    query = 'df1[(df1.row_idx == 3)]["value"].iloc[0] - 4500'

    assert "4500" in _strip_structural_tokens(query)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/submission/test_compliance_c8.py -v`
Expected: FAIL với `ImportError: cannot import name 'check_program_literals'`

- [ ] **Step 3: Thêm trường `program` lên `SubmissionItem`**

Trong `src/financial_report_qa/submission/contracts.py`, thêm vào `SubmissionItem` (sau `pandas_query`):

```python
    #: Biểu thức masked-PAL đã sinh ra `pandas_query`. Rỗng cho câu đi
    #: backstop. C8 parse trường này để thi hành N4'.
    program: str = ""
```

Và vào `QuestionOutcome` (cùng file, `contracts.py:110`), để báo cáo export
nói được câu nào phải sinh lại và câu nào vẫn lệch:

```python
    regenerated: bool = False
    low_confidence: bool = False
```

Cả bốn trường đều có mặc định, nên mọi call site hiện có tiếp tục chạy không
sửa.

- [ ] **Step 4: Thêm C8 và pattern strip vào `compliance.py`**

Thêm import ở đầu file:

```python
from financial_report_qa.core.errors import ProgramGuardError
from financial_report_qa.execution.masked_program import parse_program
```

Thêm pattern ngay sau `_POSITIONAL_INDEX_PATTERN`:

```python
# Hệ số đổi thang do `program_binding.render_program_pandas` nối vào cuối
# truy vấn (spec 2026-08-24 §4.5). Nó là hằng số của renderer, không phải của
# model, và không bao giờ là chỗ giấu đáp án -- nhưng một đáp án tình cờ bằng
# 100 sẽ trượt C4 nếu không strip. Danh sách đóng, khớp đúng `SCALE_SUFFIX`.
_SCALE_SUFFIX_PATTERN = re.compile(r"(?:\*\s*100|/\s*1000(?:000)?(?:000)?)\s*$")
```

Thêm hàm strip dùng chung và C8 checker trước `check_item`:

```python
def _strip_structural_tokens(query: str) -> str:
    """Bỏ mọi token cấu trúc trước khi quét literal cho C4."""
    stripped = _QUOTED_STRING_PATTERN.sub(" ", query)
    stripped = _SCALE_SUFFIX_PATTERN.sub(" ", stripped)
    stripped = _STRUCTURAL_COMPARISON_PATTERN.sub(" ", stripped)
    return _POSITIONAL_INDEX_PATTERN.sub(" ", stripped)


def check_program_literals(program: str) -> str | None:
    """C8: chương trình đã lưu phải qua được guard N4'. `None` là hợp lệ."""
    if not program:
        return None
    try:
        parse_program(program, value_count=_MAX_NUM_PLACEHOLDERS)
    except ProgramGuardError as error:
        return str(error)
    return None
```

và hằng số cạnh `_VALUE_TOLERANCE`:

```python
# C8 chỉ kiểm hình dạng chương trình, không kiểm số ứng viên -- việc đó thuộc
# về binding lúc chạy. Giới hạn rộng để `[NUM_i]` hợp lệ nào cũng qua được.
_MAX_NUM_PLACEHOLDERS = 1000
```

Trong `_numbers_in`, thay phần strip hiện có bằng lời gọi `_strip_structural_tokens(query)` để một chỗ duy nhất định nghĩa "token cấu trúc".

Trong `check_item`, thêm ngay sau khối C7:

```python
    # C8: N4' -- chương trình do LLM sinh ra không được chứa literal số.
    program_detail = check_program_literals(item.program)
    if program_detail is not None:
        add("C8", f"program vi phạm N4': {program_detail}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/submission/test_compliance_c8.py -v`
Expected: PASS

- [ ] **Step 6: Chắc chắn C1–C7 không đổi hành vi**

Run: `uv run pytest tests/unit/submission/ -v`
Expected: PASS toàn bộ. `program` mặc định `""` nên mọi item cũ đi qua C8 không vi phạm.

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/submission/compliance.py src/financial_report_qa/submission/contracts.py tests/unit/submission/test_compliance_c8.py
git commit -m "feat(submission): enforce the no-numeric-literal rule at the final gate"
```

---

### Task 10: Nối vào exporter và CLI

**Files:**
- Modify: `src/financial_report_qa/submission/exporter.py`
- Modify: `src/financial_report_qa/submission/cli.py`
- Test: `tests/unit/submission/test_exporter_program_path.py` (tạo)

**Interfaces:**
- Consumes: `run_question`, `PipelineResult` (Task 8); `build_cell_candidates` (Task 4); `load_program_decisions` (Task 5); `build_cell_frame` (đã có).
- Produces: `submission export --program-decisions <path>` chạy đường masked PAL; không truyền cờ thì giữ nguyên đường cũ, nguyên vẹn.

**N1:** biến `retrieved` (dùng cho `relevant_docs`/`relevant_tables`) **không** được thay bằng `executed.table_ids`. Hai tên biến tách biệt, y như hiện nay.

- [ ] **Step 1: Write the failing test**

Tạo `tests/unit/submission/test_exporter_program_path.py`:

```python
from decimal import Decimal

from financial_report_qa.execution.program_contracts import BoundValue, ExecutedProgram
from financial_report_qa.submission.exporter import build_item_from_executed

_TABLE_ID = "tbl_" + "a" * 64
_OTHER_TABLE_ID = "tbl_" + "b" * 64


def _executed() -> ExecutedProgram:
    return ExecutedProgram(
        question_id=7,
        program="[NUM_0]",
        scale="none",
        bindings=(
            BoundValue(
                num_index=0,
                candidate_index=0,
                table_id=_TABLE_ID,
                row_idx=3,
                col_idx=2,
                row_path="Doanh thu thuần",
                row_label_raw="Doanh thu thuần",
                col_path="Năm_2023",
                period=2023,
                value=Decimal("5310"),
            ),
        ),
        answer=Decimal("5310"),
        pandas_query='df1[(df1.row_idx == 3)]["value"].iloc[0]',
        table_ids=(_TABLE_ID,),
    )


def test_the_item_carries_the_program_for_c8() -> None:
    item = build_item_from_executed(
        _executed(), retrieved=(_OTHER_TABLE_ID, _TABLE_ID), relevant_docs=("doc.txt",)
    )

    assert item.program == "[NUM_0]"
    assert item.answer == 5310.0


def test_relevant_tables_keep_retrieval_rank_order_not_the_executed_tables() -> None:
    # N1 + bất biến MRR5: nhánh retrieval không bị nhánh answering ghi đè.
    item = build_item_from_executed(
        _executed(), retrieved=(_OTHER_TABLE_ID, _TABLE_ID), relevant_docs=("doc.txt",)
    )

    assert item.relevant_tables == (_OTHER_TABLE_ID, _TABLE_ID)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/submission/test_exporter_program_path.py -v`
Expected: FAIL với `ImportError: cannot import name 'build_item_from_executed'`

- [ ] **Step 3: Thêm builder vào `exporter.py`**

Thêm import:

```python
from financial_report_qa.execution.program_contracts import ExecutedProgram
```

Thêm hàm (đặt cạnh các hàm dựng `SubmissionItem` hiện có):

```python
def build_item_from_executed(
    executed: ExecutedProgram,
    *,
    retrieved: tuple[str, ...],
    relevant_docs: tuple[str, ...],
) -> SubmissionItem:
    """Dựng một `SubmissionItem` từ kết quả masked-PAL.

    `retrieved` đến từ nhánh 1 và đi thẳng vào `relevant_tables` theo đúng
    thứ tự retrieval-rank. Nó KHÔNG được thay bằng `executed.table_ids`:
    dashboard chấm MRR5 theo vị trí, và nhánh answering không có quyền ghi đè
    đầu ra nhánh retrieval (N1).
    """
    return SubmissionItem(
        id=executed.question_id,
        answer=float(executed.answer),
        pandas_query=executed.pandas_query,
        program=executed.program,
        relevant_docs=relevant_docs,
        relevant_tables=retrieved,
        evidence=(),
    )
```

Nếu `SubmissionItem` đòi thêm trường bắt buộc nào khác, đọc `submission/contracts.py:64` và điền đúng trường đó — không thêm trường mới, không bỏ trường cũ.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/submission/test_exporter_program_path.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Thêm cờ CLI**

Trong `_parser()` của `src/financial_report_qa/submission/cli.py`, thêm vào parser `export`, ngay sau `--row-choice-decisions`:

```python
    export.add_argument(
        "--program-decisions",
        type=Path,
        default=None,
        help=(
            "File JSONL quyết định masked-PAL (spec 2026-08-24 §4.3). Bỏ qua "
            "thì chạy đường cũ. Thay --row-choice-decisions khi có mặt."
        ),
    )
```

Trong nhánh `if args.command == "export":`, trước khi gọi exporter:

```python
            program_decisions = (
                load_program_decisions(args.program_decisions)
                if args.program_decisions is not None
                else None
            )
```

và truyền `program_decisions=program_decisions` xuống hàm export. Trong exporter, khi `program_decisions` khác `None`, mỗi câu đi:

```python
                frame = build_cell_frame(release_dir, retrieved)
                candidates = build_cell_candidates(frame, row_candidates, periods=entities.periods)
                result = run_question(question.id, candidates, frame, source)
```

`source` là adapter dưới đây, thêm vào `exporter.py`:

```python
@dataclass(frozen=True)
class FileDecisionSource:
    """Serve one question's decision from the offline file.

    The file is produced offline, so a live "regeneration" can only mean a
    second decision the file already carries. When it carries only one, the
    retry re-runs the identical decision and fails the identical way -- which
    is correct: `low_confidence` then records that nothing better was
    available, instead of pretending a second opinion existed.
    """

    decisions: Mapping[int, ProgramDecision]
    retries: Mapping[int, ProgramDecision] = field(default_factory=dict)

    def decide(self, question_id: int, attempt: int) -> ProgramDecision:
        if attempt > 0 and question_id in self.retries:
            return self.retries[question_id]
        try:
            return self.decisions[question_id]
        except KeyError as error:
            raise ProgramBindingError(
                f"no program decision for question {question_id}"
            ) from error
```

Thêm import ở đầu `exporter.py`:

```python
from collections.abc import Mapping
from dataclasses import dataclass, field

from financial_report_qa.core.errors import ProgramBindingError
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.program_contracts import ProgramDecision
from financial_report_qa.execution.program_pipeline import run_question
from financial_report_qa.planning.cell_candidates import build_cell_candidates
```

Thêm import ở đầu `submission/cli.py`:

```python
from financial_report_qa.planning.program_decisions import load_program_decisions
```

- [ ] **Step 6: Verify đường cũ không đổi**

Run: `uv run pytest tests/unit/submission/ tests/integration/ -v`
Expected: PASS toàn bộ — không truyền `--program-decisions` thì không nhánh nào mới chạy.

- [ ] **Step 7: Lint + type**

Run: `uv run ruff check src/financial_report_qa/submission/ && uv run mypy src/financial_report_qa/submission/exporter.py`
Expected: không lỗi

- [ ] **Step 8: Commit**

```bash
git add src/financial_report_qa/submission/exporter.py src/financial_report_qa/submission/cli.py tests/unit/submission/test_exporter_program_path.py
git commit -m "feat(submission): wire the masked-PAL path behind --program-decisions"
```

---

### Task 11: Đo trên gold, rồi xoá theo cổng

**Files:**
- Create: `artifacts/evaluations/masked-pal-vs-compiler.md`
- Delete (ngay): `src/financial_report_qa/planning/table_context_rendering.py` + test của nó
- Delete (chỉ khi cổng đạt): 9 file ở §5.4 của spec

**Interfaces:**
- Consumes: toàn bộ Task 1–10.
- Produces: báo cáo so sánh, và một cây mã không còn hai đường answering.

- [ ] **Step 1: Xoá tầng dự phòng ngoài kiến trúc**

`planning/table_context_rendering.py` tự khai trong docstring là "Day 23 last-resort tier… grounded LLM fallback" — đúng thứ N6 cấm, và §8.1 của spec 2026-08-23 đã liệt kê vào diện xoá.

```bash
git rm src/financial_report_qa/planning/table_context_rendering.py
git grep -n "table_context_rendering"
```

Xoá mọi import còn lại mà `git grep` chỉ ra, cùng file test tương ứng.

Run: `uv run pytest -q`
Expected: PASS toàn bộ

```bash
git commit -am "refactor(planning): drop the last-resort table-context tier"
```

- [ ] **Step 2: Chạy đường cũ trên gold, ghi lại baseline**

```bash
PYTHONIOENCODING=utf-8 uv run financial-report-qa submission export --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --questions data/qa/answer-gold-v1.jsonl --output artifacts/evaluations/compiler-baseline
```

Ghi lại Answer Accuracy và số câu không trả lời được. Đây là con số §8.1 và §8.2 của spec so với.

- [ ] **Step 3: Sinh file quyết định masked-PAL cho tập gold**

Không viết script mới: subcommand `submission row-batches` **đã** chạy đúng
chuỗi retrieval → row fusion → payload cho mọi câu hỏi
([cli.py:404](src/financial_report_qa/submission/cli.py:404)). Thêm chế độ ô
vào nó, đừng dựng đường thứ hai.

Trong `_parser()`, thêm vào parser `batches`:

```python
    batches.add_argument(
        "--program",
        action="store_true",
        help=(
            "Sinh payload ứng viên Ô cho masked PAL (spec 2026-08-24 §4.3) "
            "thay vì ứng viên dòng. Cần --release-dir."
        ),
    )
    batches.add_argument(
        "--release-dir",
        type=Path,
        default=None,
        help="Thư mục release Parquet, để dựng cell frame khi bật --program.",
    )
```

Trong nhánh `if args.command == "row-batches":`, thay lời gọi
`build_batch_payload(...)` bằng:

```python
                    if args.program:
                        if args.release_dir is None:
                            raise SubmissionError("--program cần --release-dir")
                        payload = build_program_batch_payload(
                            raw_question.id,
                            raw_question.question,
                            parse_query_entities(raw_question.question),
                            build_question_cell_candidates(
                                args.release_dir, raw_question.question, retrieved, fused
                            ),
                        )
                    else:
                        payload = build_batch_payload(
                            raw_question.id,
                            raw_question.question,
```

- [ ] **Step 3a: Một hàm duy nhất dựng danh sách ứng viên**

**Đây là ràng buộc quan trọng nhất của cả plan.** `ProgramDecision.cells` là
vị trí trong danh sách ứng viên. Nếu danh sách lúc sinh payload khác danh sách
lúc export dù chỉ một phần tử, **mọi chỉ số lệch đi** — và lệch theo cách
Verify B bắt được nhưng chỉ sau khi đã hỏng cả tập.

Hai đường hiện tại **không** giống nhau: `export` thu hẹp theo scope trước khi
gọi row fusion (`_scope_candidate_tables`,
[exporter.py:242](src/financial_report_qa/submission/exporter.py:242)), còn
`row-batches` gọi thẳng trên `retrieved` chưa thu hẹp
([cli.py:432](src/financial_report_qa/submission/cli.py:432)). Chênh lệch này
đã tồn tại sẵn cho chỉ số **dòng** của đường cũ; với đường mới nó là lỗi chết
người.

Thêm vào `submission/exporter.py` một hàm dùng chung, và bắt **cả hai** đường
gọi đúng nó:

```python
def build_question_cell_candidates(
    release_dir: Path,
    question: str,
    retrieved: Sequence[str],
    fusion_rows: Sequence[RowFusedCandidate],
) -> tuple[CellCandidate, ...]:
    """Dựng danh sách ô đánh số cho một câu, một cách duy nhất.

    `ProgramDecision.cells` là vị trí trong danh sách này, nên lúc sinh payload
    và lúc export phải cho ra danh sách y hệt. Đó là lý do hàm này tồn tại
    thay vì hai lời gọi `build_cell_candidates` song song ở hai file.
    """
    entities = parse_query_entities(question)
    frame = build_cell_frame(release_dir, list(retrieved))
    return build_cell_candidates(frame, fusion_rows, periods=entities.periods)
```

Trong `_run_one_question`, đường `--program-decisions` phải gọi **chính hàm
này** với **chính `retrieved`** (không phải `answerable`), rồi mới thu hẹp
scope ở bước sau nếu cần. Ghi một test ghim:

```python
def test_batch_time_and_export_time_candidate_lists_are_identical() -> None:
    frame = _frame()
    rows = (_row_candidate(4, 1, None), _row_candidate(3, 2, "Doanh thu"))

    first = build_cell_candidates(frame, rows, periods=("2023",))
    second = build_cell_candidates(frame, rows, periods=("2023",))

    assert [c.index for c in first] == [c.index for c in second]
    assert [(c.table_id, c.row_idx, c.col_idx) for c in first] == [
        (c.table_id, c.row_idx, c.col_idx) for c in second
    ]
```

- [ ] **Step 3b: Chạy sinh payload**

```bash
PYTHONIOENCODING=utf-8 uv run financial-report-qa submission row-batches --program --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --release-dir data/processed/release_v2_422df141c935 --bm25-index data/indexes/bm25-v4/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --questions-path data/qa/answer-gold-v1.jsonl --output-dir artifacts/batches/program-gold
```

Expected: một file batch mỗi 64 câu, mỗi dòng có `candidates` với `index` chạy
từ 0 và **không trường nào tên `value`**.

Gọi LLM offline theo batch trên các file này, rồi ghi kết quả thành JSONL đúng
schema `ProgramDecision`. File quyết định phải **không chứa một giá trị số
nào** ngoài `question_id`, `cells`, `num`, và chuỗi `program`/`uses`/`scale` —
`extra="forbid"` của contract là chốt chặn cho điều đó.

Kiểm nhanh trước khi dùng:

```bash
uv run python -c "from pathlib import Path; from financial_report_qa.planning.program_decisions import load_program_decisions; d = load_program_decisions(Path('data/decisions/program-decisions-gold.jsonl')); print(len(d), 'decisions')"
```

Expected: in ra số quyết định bằng số câu gold. Bất kỳ `PlanningArtifactError` nào cũng chỉ đúng dòng hỏng.

- [ ] **Step 4: Chạy đường mới trên cùng tập gold**

```bash
PYTHONIOENCODING=utf-8 uv run financial-report-qa submission export --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --questions data/qa/answer-gold-v1.jsonl --program-decisions data/decisions/program-decisions-gold.jsonl --output artifacts/evaluations/masked-pal
```

- [ ] **Step 5: Viết báo cáo so sánh**

Tạo `artifacts/evaluations/masked-pal-vs-compiler.md` với đúng bốn số cho mỗi đường: Answer Accuracy, số câu không trả lời được, số câu `low_confidence`, số vi phạm C1–C8. Không diễn giải thêm; số nói.

- [ ] **Step 6: Cổng**

Đọc báo cáo:

- **Answer Accuracy đường mới ≥ đường cũ** và **số câu không trả lời được < 41%** → sang Step 7.
- Không đạt → **dừng**. Không xoá gì. Phân tích: xem `failure_code` nào chiếm đa số, sửa đúng chỗ đó, chạy lại từ Step 4. `compiler.py` là baseline duy nhất đang có; xoá nó trước khi có số là vứt mất thước đo.

- [ ] **Step 7: Xoá đường cũ (chỉ khi Step 6 đạt)**

```bash
git rm src/financial_report_qa/execution/compiler.py src/financial_report_qa/execution/operations.py src/financial_report_qa/execution/locator.py src/financial_report_qa/execution/tiebreak.py src/financial_report_qa/execution/contracts.py src/financial_report_qa/planning/plan_contracts.py src/financial_report_qa/planning/plan_validator.py src/financial_report_qa/planning/question_plan.py src/financial_report_qa/planning/cell_grounding.py
git grep -ln "compile_plan\|FinancialQueryPlan\|MetricSelector\|CompiledQuery\|load_decisions"
```

Với mỗi file `git grep` chỉ ra, gỡ import và nhánh code tương ứng. Bốn chỗ biết trước:

1. `execution/pandas_query.py` — bỏ `render_pandas_query` và mọi hàm `_*_expr` nhận `MetricSelector`; **giữ** `_lit` và `replay_pandas_query`.
2. `verification/checks.py` — bỏ `check_period_inferred_warning` và `check_scope_inferred`; bốn check còn lại đổi sang nhận `ExecutedProgram`. `verification/builder.py::build_answer_package` đổi tham số theo; `verification/contracts.py::AnswerPackage` thêm ba trường mặc định `program: str = ""`, `regenerated: bool = False`, `low_confidence: bool = False`.
3. `verification/numeric_guard.py` — bỏ `build_number_whitelist` (Task 7 đã thay bằng `program_number_whitelist`); **giữ** `extract_number_tokens` và `guard_generated_text`.
4. `submission/cli.py` — bỏ cờ `--row-choice-decisions` và import `load_decisions`; `planning/row_choice_batch.py` bỏ `build_batch_payload` và `_candidate_payload` cũ.

Xoá mọi file test của các module đã xoá.

Run: `uv run pytest -q && uv run ruff check && uv run mypy src`
Expected: PASS toàn bộ, không lỗi lint/type

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove the operation-enum answering path in favour of masked PAL"
```

---

## Ngoài phạm vi

Theo §11 của spec:

- Binder (LLM sinh pandas trên bảng đã mask).
- Self-consistency (sinh N chương trình, bỏ phiếu theo giá trị).
- LLM critic phán đúng/sai.
- Graph/Tree table representation.
- Reranker cho row retrieval.
- Fine-tune bất kỳ model nào.
- **Nhánh 1 (retrieval).** Task 5–8 của `2026-08-23-retrieval-rerank-pipeline.md` là plan riêng và phải xong trước — nó chiếm 50% điểm với chi phí thấp hơn nhiều.
