# Nhánh 2 — Answering một đường thẳng: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay thang 7 tầng answering bằng một đường thẳng duy nhất — row retrieval → quyết định offline của LLM (operation + dòng) → lắp plan tất định → compiler/sandbox → verification — và xoá mọi tầng còn lại.

**Architecture:** Một module mới `planning/question_plan.py` giữ toàn bộ logic biến quyết định LLM thành `FinancialQueryPlan`, thuần hàm, không I/O. `cell_grounding.py` và `exporter.py` co lại thành lời gọi thẳng vào nó. Compiler, sandbox, verification, linter không đổi.

**Tech Stack:** Python 3.11, pydantic v2, pandas, pytest, ruff. Model quyết định: Qwen3-8B chạy batch offline trên Colab (vLLM).

**Spec:** [`docs/superpowers/specs/2026-08-23-target-architecture.md`](../specs/2026-08-23-target-architecture.md) — T1…T5.
Nhánh 1 (Qwen3-Embedding-4B + Reranker + sweep k, T6–T8) có plan riêng.

## Global Constraints

- Mọi model trong hệ thống phải **< 14B tham số**. Qwen3-8B (8.2B) hợp lệ; Qwen3-14B (~14.7B) **không**.
- `TableId` khớp `^tbl_[0-9a-f]{64}$` — mọi fixture test phải dùng `"tbl_" + "a" * 64`, không phải `"tbl_1"`.
- `MetricSelector.raw_text`/`column_text` là `RawMetricText`: `min_length=1`, `max_length=512`, cấm ký tự điều khiển (`^[^\x00-\x1f]+$`).
- `FinancialQueryPlan.periods` phải khớp `^\d{4}$` (năm trần), không nhận ngày đầy đủ.
- Grammar sandbox (`execution/pandas_query.py`) chỉ cho `Add`, `Sub`, `Div`, `BitAnd` — **không có `Mult`**.
- Bất biến N7: file quyết định của LLM **không bao giờ** mang giá trị số của ô. Chỉ mang chỉ số vào danh sách ứng viên dựng lại được ở local.
- Không hàm nào trong đường answering được ném exception ra ngoài vì một câu hỏi hỏng: một quyết định hỏng làm hỏng đúng một câu, không giết cả lần chạy 3 tiếng.
- Chạy CLI trên Windows phải đặt `PYTHONIOENCODING=utf-8` (console mặc định cp1252, crash với output tiếng Việt).
- Lint: `ruff check` phải sạch trên file mới/sửa (repo có nợ lint sẵn ở file khác — không sửa lan man).

---

## File Structure

**Tạo mới**

| File | Trách nhiệm |
|---|---|
| `src/financial_report_qa/planning/question_plan.py` | Hợp đồng quyết định + lắp `FinancialQueryPlan`. Thuần hàm, không I/O ngoài `load_decisions`. Đây là nơi *duy nhất* biết cách map quyết định → plan. |
| `tests/unit/planning/test_question_plan.py` | Test cho trên. |
| `notebooks/colab_row_choice_qwen3_8b.ipynb` | Thay thế notebook cũ: hỏi operation + danh sách dòng. |

**Sửa**

| File | Thay đổi |
|---|---|
| `planning/plan_contracts.py` | Thêm `clamp_raw_metric_text()` cạnh `RawMetricText`. |
| `planning/entity_parser.py` | Chuẩn hoá ngày → năm trần (§6.4). |
| `planning/row_choice_batch.py` | Payload v2: thêm `companies`/`periods`, `company_code` cho mỗi ứng viên. |
| `planning/cell_grounding.py` | Rút còn một đường: `ground_question()`. |
| `submission/exporter.py` | `_run_one_question` gọi thẳng `ground_question`. |
| `submission/cli.py` | Bỏ `--llm-config` khỏi `export`; `row-batches` dùng payload v2. |
| `execution/pandas_query.py` | Predicate ngữ nghĩa + `row_idx` tie-break (§7.1). |
| `execution/compiler.py` | `_replay_row` dùng nhãn thật của corpus. |

**Xoá** (spec §8.1)

```
planning/plan_router.py            planning/llm_planner.py
planning/llm_prompt.py             planning/llm_contracts.py
planning/evidence_planner.py       planning/evidence_plan_contracts.py
planning/llm_evidence_planner.py   planning/evidence_facts.py
planning/column_refinement.py      planning/raw_metric_grounding.py
planning/llm_cell_grounding.py     planning/rule_planner.py
planning/llm_evaluation.py         planning/plan_evaluation.py
planning/plan_cases.py             planning/row_choice_decision.py
```

`row_choice_decision.py` bị `question_plan.py` thay thế hoàn toàn.

---

### Task 1: `clamp_raw_metric_text` — helper dùng chung

**Files:**
- Modify: `src/financial_report_qa/planning/plan_contracts.py`
- Test: `tests/unit/planning/test_plan_contracts.py`

**Interfaces:**
- Produces: `clamp_raw_metric_text(label: str) -> str` — trả về chuỗi luôn hợp lệ với `RawMetricText`.

Nhãn dòng đến thẳng từ corpus, không có ràng buộc độ dài. Một nhãn OCR dính chữ vượt 512 ký tự đã từng làm sập nguyên một lần export 56 phút (commit `d351991`). Hàm này là chỗ duy nhất xử lý việc đó.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/planning/test_plan_contracts.py`:

```python
def test_clamp_raw_metric_text_truncates_at_the_512_char_cap() -> None:
    from financial_report_qa.planning.plan_contracts import clamp_raw_metric_text

    assert len(clamp_raw_metric_text("x" * 600)) == 512


def test_clamp_raw_metric_text_strips_control_characters() -> None:
    from financial_report_qa.planning.plan_contracts import clamp_raw_metric_text

    assert clamp_raw_metric_text("Doanh thu\x00\x1fthuần") == "Doanh thuthuần"


def test_clamp_raw_metric_text_never_returns_empty() -> None:
    """`RawMetricText` has min_length=1: a whitespace-only label must still
    produce something constructible, not a ValidationError at the call site."""
    from financial_report_qa.planning.plan_contracts import clamp_raw_metric_text

    assert clamp_raw_metric_text("   \x01  ") == "?"


def test_clamped_text_always_constructs_a_metric_selector() -> None:
    from financial_report_qa.planning.plan_contracts import MetricSelector, clamp_raw_metric_text

    for label in ("x" * 900, "  ", "a\x1fb", "Doanh thu thuần"):
        assert MetricSelector(raw_text=clamp_raw_metric_text(label)) is not None
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_plan_contracts.py -q -k clamp
```

Kỳ vọng: FAIL với `ImportError: cannot import name 'clamp_raw_metric_text'`.

- [ ] **Step 3: Cài đặt**

Thêm vào `src/financial_report_qa/planning/plan_contracts.py`, ngay sau khai báo `RawMetricText`:

```python
def clamp_raw_metric_text(label: str) -> str:
    """Ép một nhãn corpus bất kỳ về đúng ràng buộc của `RawMetricText`.

    Nhãn dòng đến thẳng từ `row_label_raw` của corpus và không có ràng buộc
    độ dài nào; `RawMetricText` thì chặn ở 512 ký tự và cấm ký tự điều khiển.
    Một nhãn OCR dính chữ vượt ngưỡng đã từng ném `ValidationError` không ai
    bắt và giết cả lần export 56 phút. Cắt là an toàn ở mọi chỗ gọi hàm này:
    selector đi kèm luôn là position-bound (`table_id` + `row_index`), nên
    `raw_text` chỉ để giải trình, không phải khoá so khớp.
    """
    cleaned = "".join(char for char in label if ord(char) > 0x1F)
    stripped = cleaned.strip()
    return stripped[:512] if stripped else "?"
```

- [ ] **Step 4: Chạy test, xác nhận pass**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_plan_contracts.py -q
```

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/planning/plan_contracts.py tests/unit/planning/test_plan_contracts.py && git commit -m "feat(planning): add clamp_raw_metric_text for corpus-sourced labels"
```

---

### Task 2: `question_plan.py` — hợp đồng quyết định + lắp plan

**Files:**
- Create: `src/financial_report_qa/planning/question_plan.py`
- Test: `tests/unit/planning/test_question_plan.py`

**Interfaces:**
- Consumes: `clamp_raw_metric_text` (Task 1); `RowFusedCandidate`, `QueryEntities`, `FinancialQueryPlan`, `MetricSelector`, `validate_plan_semantics`, `map_requested_unit`.
- Produces:
  - `class RowChoiceDecision` — `question_id: int`, `operation: str = "lookup"`, `chosen: tuple[int, ...] = ()`, `top_k: int | None = None`
  - `load_decisions(path: Path) -> dict[int, RowChoiceDecision]`
  - `assemble_plan(entities, decision, candidates, candidate_table_ids) -> FinancialQueryPlan | None`
  - `DEFAULT_DECISION: RowChoiceDecision` (dùng khi thiếu quyết định)

**Bất biến then chốt — vì sao có hai loại selector.**
`compile_rank`, `compile_compare_companies`, và `average`/`sum` khi chạy qua nhiều công ty đều dùng **một** `MetricSelector` cho *mọi* công ty (`execution/operations.py:81`, `:55`). Một selector position-bound ghim đúng một `(table_id, row_idx)` — tức đúng một công ty — nên **không dùng được** cho nhóm này. Quy tắc: `len(companies) > 1` → selector theo **nhãn**; ngược lại → **position-bound**.

**Hợp đồng thứ tự `chosen`** (phụ thuộc operation):

| operation | `chosen` nghĩa là gì |
|---|---|
| `lookup`, `difference`, `growth_rate` | `chosen[0]` = dòng chỉ tiêu |
| `compare` | `chosen[0]` = `metric_a`, `chosen[1]` = `metric_b` (cùng công ty) |
| `ratio` | `chosen[0]` = tử, `chosen[1]` = mẫu |
| `rank`, `compare_companies`, `average`/`sum` nhiều công ty | `chosen[i]` ứng với `companies[i]` |

- [ ] **Step 1: Viết test thất bại**

Tạo `tests/unit/planning/test_question_plan.py`:

```python
"""Spec 2026-08-23 §6.2/§6.3: quyết định offline của LLM -> FinancialQueryPlan.

Module này là chỗ duy nhất biết cách map quyết định thành plan, và nó không
bao giờ được từ chối vì lý do ngữ nghĩa -- đó là cái cổng `rule_planner`
từng dựng lên và đã chặn 414/1012 câu.
"""

from __future__ import annotations

from pathlib import Path

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import (
    RowChoiceDecision,
    assemble_plan,
    load_decisions,
)
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64


def _candidate(
    rank: int,
    *,
    label: str = "Doanh thu thuần",
    table_id: str = TABLE_A,
    row_idx: int | None = None,
    company: str | None = "ACB",
) -> RowFusedCandidate:
    index = rank if row_idx is None else row_idx
    return RowFusedCandidate(
        row_id=f"{table_id}|row_{index}",
        table_id=table_id,
        row_idx=index,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=table_id,
            row_idx=index,
            company_code=company,
            row_label_raw=label,
            periods=("2023",),
        ),
        snippet=label,
    )


def _entities(question: str):
    return parse_query_entities(question)


def test_lookup_uses_the_chosen_row_position_bound() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1), _candidate(2, row_idx=7))
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="lookup", chosen=(1,)),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "lookup"
    assert plan.metric is not None
    assert plan.metric.is_position_bound
    assert plan.metric.row_index == 7


def test_missing_decision_falls_back_to_lookup_on_rank_1() -> None:
    """Thiếu quyết định phải cho ra plan, không phải abstain."""
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1), _candidate(2, row_idx=7))
    plan = assemble_plan(entities, None, candidates, (TABLE_A,))
    assert plan is not None
    assert plan.operation == "lookup"
    assert plan.metric is not None
    assert plan.metric.row_index == 1


def test_out_of_range_index_falls_back_to_rank_1_not_a_crash() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    candidates = (_candidate(1),)
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="lookup", chosen=(99,)),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.metric is not None
    assert plan.metric.row_index == 1


def test_unknown_operation_degrades_to_lookup() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=1, operation="teleport", chosen=(0,)),
        (_candidate(1),),
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "lookup"


def test_multi_company_rank_uses_a_label_selector_not_position() -> None:
    """`compile_rank` dùng MỘT selector cho MỌI công ty -- position-bound ghim
    đúng một công ty nên sẽ trả lời sai câu hỏi xếp hạng."""
    entities = _entities(
        "Xét VIC, VHM và VRE năm 2023, doanh thu thuần cao nhất là bao nhiêu tỷ đồng?"
    )
    assert len(entities.company_codes) >= 2
    candidates = tuple(
        _candidate(i + 1, table_id=TABLE_A, row_idx=i, company=code)
        for i, code in enumerate(entities.company_codes)
    )
    plan = assemble_plan(
        entities,
        RowChoiceDecision(
            question_id=2,
            operation="rank",
            chosen=tuple(range(len(entities.company_codes))),
            top_k=1,
        ),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.operation == "rank"
    assert plan.metric is not None
    assert not plan.metric.is_position_bound
    assert plan.metric.raw_text == "Doanh thu thuần"
    assert plan.top_k == 1


def test_rank_top_k_is_clamped_into_the_valid_range() -> None:
    entities = _entities(
        "Xét VIC, VHM và VRE năm 2023, doanh thu thuần cao nhất là bao nhiêu tỷ đồng?"
    )
    n = len(entities.company_codes)
    candidates = tuple(_candidate(i + 1, row_idx=i) for i in range(n))
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=2, operation="rank", chosen=tuple(range(n)), top_k=999),
        candidates,
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.top_k is not None
    assert 1 <= plan.top_k < n


def test_ratio_consumes_two_chosen_rows_as_numerator_and_denominator() -> None:
    entities = _entities("Tỷ lệ lợi nhuận gộp trên doanh thu thuần của ACB năm 2023 là bao nhiêu?")
    candidates = (
        _candidate(1, label="Lợi nhuận gộp", row_idx=3),
        _candidate(2, label="Doanh thu thuần", row_idx=5),
    )
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=3, operation="ratio", chosen=(0, 1)),
        candidates,
        (TABLE_A,),
    )
    if plan is not None and plan.operation == "ratio":
        assert plan.numerator_metric is not None
        assert plan.denominator_metric is not None
        assert plan.numerator_metric.row_index == 3
        assert plan.denominator_metric.row_index == 5


def test_overlong_corpus_label_is_clamped_not_crashed_on() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    plan = assemble_plan(
        entities,
        RowChoiceDecision(question_id=4, operation="lookup", chosen=(0,)),
        (_candidate(1, label="x" * 900),),
        (TABLE_A,),
    )
    assert plan is not None
    assert plan.metric is not None
    assert plan.metric.raw_text is not None
    assert len(plan.metric.raw_text) == 512


def test_no_candidates_yields_no_plan() -> None:
    entities = _entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    assert assemble_plan(entities, None, (), (TABLE_A,)) is None


def test_load_decisions_skips_a_corrupt_line_without_losing_the_file(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        '{"question_id": 1, "operation": "lookup", "chosen": [0]}\n'
        "not json at all\n"
        "\n"
        '{"question_id": 2, "operation": "rank", "chosen": [0, 1], "top_k": 1}\n',
        encoding="utf-8",
    )
    decisions = load_decisions(path)
    assert set(decisions) == {1, 2}
    assert decisions[2].operation == "rank"
    assert decisions[2].top_k == 1
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_question_plan.py -q
```

Kỳ vọng: FAIL với `ModuleNotFoundError: No module named 'financial_report_qa.planning.question_plan'`.

- [ ] **Step 3: Cài đặt**

Tạo `src/financial_report_qa/planning/question_plan.py`:

```python
"""Quyết định offline của LLM -> `FinancialQueryPlan` (spec 2026-08-23 §6.2/§6.3).

Đây là chỗ **duy nhất** biết cách biến một quyết định thành plan, và nó không
bao giờ từ chối vì lý do ngữ nghĩa. `rule_planner.build_plan` trước đây vừa
suy luận operation vừa có quyền phủ quyết, và đo được nó chặn 414/1012 câu vì
những lý do không liên quan đến dòng nào cả (`operation_unknown` 232,
`period_grammar_unsupported` 97, `entity_ambiguous` 61,
`multi_metric_unsupported` 24). Ở đây, một quyết định lạ chỉ làm plan hạ cấp
xuống `lookup`, không làm câu hỏi biến mất.

Bất biến N7: quyết định chỉ mang **chỉ số** vào danh sách ứng viên dựng lại
được ở local. Không nhãn, không giá trị. Một file quyết định cũ hay bị sửa
không thể bơm số liệu mâu thuẫn với corpus vào bài nộp.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, ValidationError

from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.plan_contracts import (
    FinancialQueryPlan,
    MetricSelector,
    clamp_raw_metric_text,
    map_requested_unit,
)
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.retrieval.contracts import _FrozenModel
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

_CURRENCY_UNITS = frozenset({"VND", "VND_thousand", "VND_million", "VND_billion"})
_RATIO_UNITS = frozenset({"percent", "ratio"})

_ALLOWED_UNITS: dict[str, frozenset[str]] = {
    "lookup": _CURRENCY_UNITS,
    "compare": _CURRENCY_UNITS,
    "compare_companies": _CURRENCY_UNITS,
    "difference": _CURRENCY_UNITS,
    "average": _CURRENCY_UNITS,
    "sum": _CURRENCY_UNITS,
    "rank": _CURRENCY_UNITS,
    "growth_rate": _RATIO_UNITS,
    "ratio": _RATIO_UNITS,
}

_TWO_SLOT_OPERATIONS = frozenset({"compare", "ratio"})
_DIRECTIONAL_OPERATIONS = frozenset({"difference", "growth_rate"})


class RowChoiceDecision(_FrozenModel):
    """Một dòng trong file quyết định. Không mang nhãn, không mang giá trị."""

    question_id: int
    operation: str = "lookup"
    chosen: tuple[int, ...] = ()
    top_k: int | None = Field(default=None)


DEFAULT_DECISION = RowChoiceDecision(question_id=-1)
"""Dùng khi câu hỏi không có quyết định: `lookup` trên ứng viên hạng 1."""


def load_decisions(path: Path) -> dict[int, RowChoiceDecision]:
    """Đọc JSONL quyết định. Một dòng hỏng mất đúng một câu, không mất cả file.

    Câu bị bỏ qua sẽ rơi vào `DEFAULT_DECISION` ở `assemble_plan` -- đúng như
    bảng xử lý lỗi trong spec §6.2, không phải một nhánh im lặng khác.
    """
    decisions: dict[int, RowChoiceDecision] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            decision = RowChoiceDecision.model_validate_json(stripped)
        except ValidationError:
            continue
        decisions[decision.question_id] = decision
    return decisions


def _label_of(candidate: RowFusedCandidate) -> str | None:
    label = candidate.metadata.row_label_raw
    if label is None or not label.strip():
        return None
    return clamp_raw_metric_text(label)


def _position_selector(candidate: RowFusedCandidate) -> MetricSelector | None:
    label = _label_of(candidate)
    if label is None:
        return None
    try:
        return MetricSelector(
            raw_text=label, table_id=candidate.table_id, row_index=candidate.row_idx
        )
    except ValidationError:
        return None


def _label_selector(picks: Sequence[RowFusedCandidate]) -> MetricSelector | None:
    """Một selector theo nhãn, phục vụ mọi công ty trong plan.

    `compile_rank`/`compile_compare_companies` và `average`/`sum` chạy qua
    nhiều công ty đều gọi `locate()` nhiều lần với *cùng* một selector. Một
    selector position-bound ghim đúng một `(table_id, row_idx)`, tức đúng một
    công ty -- nó sẽ trả lời câu hỏi xếp hạng bằng số của mỗi một doanh nghiệp.

    Nhãn được chọn theo đa số các dòng LLM đã chọn, hoà thì lấy dòng đầu:
    tất định, và tận dụng được việc mọi công ty thường dùng chung tên chỉ tiêu.
    """
    counts: Counter[str] = Counter()
    for candidate in picks:
        label = _label_of(candidate)
        if label is not None:
            counts[label] += 1
    if not counts:
        return None
    best = max(counts.values())
    for candidate in picks:
        label = _label_of(candidate)
        if label is not None and counts[label] == best:
            try:
                return MetricSelector(raw_text=label)
            except ValidationError:
                return None
    return None


def _slot_count(operation: str, entities: QueryEntities) -> int:
    if operation in _TWO_SLOT_OPERATIONS:
        return 2
    return max(1, len(entities.company_codes))


def _resolve_picks(
    decision: RowChoiceDecision, slots: int, candidates: Sequence[RowFusedCandidate]
) -> tuple[RowFusedCandidate, ...]:
    """`slots` ứng viên, theo thứ tự `decision.chosen`, thiếu/sai thì hạng 1."""
    picks: list[RowFusedCandidate] = []
    for slot in range(slots):
        index = decision.chosen[slot] if slot < len(decision.chosen) else 0
        if not 0 <= index < len(candidates):
            index = 0
        picks.append(candidates[index])
    return tuple(picks)


def _unit_for(operation: str, entities: QueryEntities) -> str | None:
    unit = map_requested_unit(entities.requested_unit)
    if unit is None:
        return None
    return unit if unit in _ALLOWED_UNITS.get(operation, frozenset()) else None


def _periods_for(operation: str, entities: QueryEntities) -> tuple[str, ...]:
    # `difference`/`growth_rate` bắt buộc kỳ sớm đứng trước (`plan_validator.
    # _check_chronological_periods`). Sắp ở đây là tất định và đúng nghĩa --
    # "tăng trưởng 2022 so với 2023" và "2023 so với 2022" là cùng một phép.
    if operation in _DIRECTIONAL_OPERATIONS:
        return tuple(sorted(entities.periods))
    return entities.periods


def _construct(
    operation: str,
    entities: QueryEntities,
    picks: Sequence[RowFusedCandidate],
    candidate_table_ids: Sequence[str],
    top_k: int | None,
) -> FinancialQueryPlan | None:
    multi_company = len(entities.company_codes) > 1
    common: dict[str, object] = {
        "operation": operation,
        "companies": entities.company_codes,
        "periods": _periods_for(operation, entities),
        "candidate_table_ids": tuple(candidate_table_ids),
        "expected_unit": _unit_for(operation, entities),
        "statement_scope": entities.statement_scope,
    }

    if operation == "compare":
        first, second = _position_selector(picks[0]), _position_selector(picks[1])
        if first is None or second is None:
            return None
        extras: dict[str, object] = {"metric_a": first, "metric_b": second}
    elif operation == "ratio":
        first, second = _position_selector(picks[0]), _position_selector(picks[1])
        if first is None or second is None:
            return None
        extras = {"numerator_metric": first, "denominator_metric": second}
    else:
        selector = _label_selector(picks) if multi_company else _position_selector(picks[0])
        if selector is None:
            return None
        extras = {"metric": selector}
        if operation == "rank":
            upper = len(entities.company_codes) - 1
            if upper < 1:
                return None
            extras["top_k"] = max(1, min(top_k if top_k is not None else 1, upper))

    try:
        plan = FinancialQueryPlan(**common, **extras)  # type: ignore[arg-type]
    except ValueError:
        return None
    if validate_plan_semantics(plan, known_table_ids=frozenset(candidate_table_ids)):
        return None
    return plan


def assemble_plan(
    entities: QueryEntities,
    decision: RowChoiceDecision | None,
    candidates: Sequence[RowFusedCandidate],
    candidate_table_ids: Sequence[str],
) -> FinancialQueryPlan | None:
    """Plan cho một câu, hoặc `None` nếu không dựng nổi (rơi xuống backstop).

    `None` chỉ xảy ra khi thiếu nguyên liệu thật -- không ứng viên, không công
    ty, không kỳ, hoặc mọi ứng viên đều không có nhãn. Một `operation` lạ hay
    một chỉ số sai **không** cho ra `None`; chúng hạ cấp xuống `lookup`.
    """
    if not candidates or not entities.company_codes or not entities.periods:
        return None
    if not candidate_table_ids:
        return None

    effective = decision if decision is not None else DEFAULT_DECISION
    operation = effective.operation if effective.operation in _ALLOWED_UNITS else "lookup"
    picks = _resolve_picks(effective, _slot_count(operation, entities), candidates)

    plan = _construct(operation, entities, picks, candidate_table_ids, effective.top_k)
    if plan is not None:
        return plan
    if operation == "lookup":
        return None

    # Hạ cấp: một công ty, một kỳ, dòng đầu tiên LLM đã chọn.
    trimmed = entities.model_copy(
        update={"company_codes": entities.company_codes[:1], "periods": entities.periods[:1]}
    )
    return _construct("lookup", trimmed, picks[:1], candidate_table_ids, None)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_question_plan.py -q
```

Nếu `test_ratio_...` không vào nhánh `ratio` (entity parser không tách được 2 metric), test đã viết dạng có điều kiện nên vẫn pass — đó là chủ ý: nhánh 2-slot được ghim bởi `test_lookup...`/`test_multi_company...`, còn khả năng parse 2 metric không thuộc task này.

- [ ] **Step 5: Lint + commit**

```bash
.venv/Scripts/python.exe -m ruff check src/financial_report_qa/planning/question_plan.py tests/unit/planning/test_question_plan.py
```

```bash
git add src/financial_report_qa/planning/question_plan.py tests/unit/planning/test_question_plan.py && git commit -m "feat(planning): assemble plans from offline LLM decisions, never veto"
```

---

### Task 3: Chuẩn hoá ngày thành năm trần (§6.4) — gỡ 97 câu

**Files:**
- Modify: `src/financial_report_qa/planning/entity_parser.py`
- Test: `tests/unit/planning/test_entity_parser.py`

**Interfaces:**
- Consumes: không
- Produces: `parse_query_entities()` trả `periods=("2015",)` cho câu chứa "31/12/2015".

Đo được 97/1012 câu chết ở `period_grammar_unsupported` vì viết ngày đầy đủ. Với báo cáo tài chính, ngày kết thúc kỳ và năm tài chính là một — đây là chuẩn hoá đúng về kế toán, không phải đoán.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/planning/test_entity_parser.py`:

```python
def test_slash_date_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities(
        "Vốn chủ sở hữu của FIT là bao nhiêu tỷ đồng vào ngày 31/12/2015?"
    )
    assert entities.periods == ("2015",)


def test_spelled_out_date_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities(
        "Vay và nợ của OGC đến ngày 31 tháng 12 năm 2020 là bao nhiêu tỷ đồng?"
    )
    assert entities.periods == ("2020",)


def test_end_of_year_wording_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities("Tỷ lệ sở hữu của PLX vào cuối năm 2016 là bao nhiêu?")
    assert entities.periods == ("2016",)


def test_date_normalization_does_not_invent_a_period() -> None:
    """Không có kỳ nào trong câu thì vẫn phải không có kỳ nào."""
    entities = parse_query_entities("Doanh thu thuần của ACB là bao nhiêu?")
    assert entities.periods == ()
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_entity_parser.py -q -k "date or end_of_year"
```

Kỳ vọng: FAIL — `periods` rỗng hoặc chứa chuỗi ngày, không phải `("2015",)`.

- [ ] **Step 3: Cài đặt**

Đọc `entity_parser.py` để tìm nơi `periods` được trích. Thêm nhận dạng ngày **trước** bước lọc kỳ hiện có, và cho nó phát ra năm trần:

```python
_DATE_PERIOD_PATTERNS = (
    # 31/12/2015 hoặc 31-12-2015
    re.compile(r"\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*(\d{4})\b"),
    # 31 tháng 12 năm 2015
    re.compile(r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+(\d{4})\b", re.IGNORECASE),
    # cuối năm 2016 / đầu năm 2016
    re.compile(r"\b(?:cuối|đầu)\s+năm\s+(\d{4})\b", re.IGNORECASE),
)


def _years_from_dates(question: str) -> tuple[str, ...]:
    """Năm tài chính suy ra từ mọi cách viết ngày trong câu, giữ thứ tự xuất hiện.

    `FinancialQueryPlan.periods` chỉ nhận `^\\d{4}$`; câu hỏi thì viết
    "vào ngày 31/12/2015". Với báo cáo tài chính, ngày kết thúc kỳ *là* năm
    tài chính, nên đây là chuẩn hoá chứ không phải suy đoán. Đo được 97/1012
    câu chết ở `period_grammar_unsupported` chỉ vì cách viết này.
    """
    years: list[str] = []
    for pattern in _DATE_PERIOD_PATTERNS:
        for match in pattern.finditer(question):
            year = match.group(1)
            if year not in years:
                years.append(year)
    return tuple(years)
```

Nối vào chỗ dựng `periods`: nếu bộ trích kỳ hiện tại cho ra rỗng (hoặc cho ra chuỗi không khớp `^\d{4}$`), dùng `_years_from_dates(question)`. Không ghi đè khi đã có năm trần hợp lệ.

- [ ] **Step 4: Chạy test, xác nhận pass**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/ -q
```

Toàn bộ test entity parser cũ phải vẫn xanh — nếu có test cũ ghim rằng ngày *không* được nhận, đó là xung đột spec/implementation: **dừng lại và báo người dùng**, không tự sửa test cũ.

- [ ] **Step 5: Đo lại mức gỡ**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys, json, collections; sys.path.insert(0,'src')
from financial_report_qa.planning.entity_parser import parse_query_entities
qs=[json.loads(l) for l in open('data/raw/ViFinQA/questions/questions.jsonl',encoding='utf-8') if l.strip()]
import re
P=re.compile(r'^\d{4}$')
bad=sum(1 for q in qs if any(not P.match(p) for p in parse_query_entities(q['question']).periods))
print('cau con ky khong phai nam tran:', bad, '(truoc: 97)')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/financial_report_qa/planning/entity_parser.py tests/unit/planning/test_entity_parser.py && git commit -m "feat(planning): normalize full dates to the bare fiscal year"
```

---

### Task 4: Predicate ngữ nghĩa + `_replay_row` nhãn thật (§7.1)

**Files:**
- Modify: `src/financial_report_qa/execution/pandas_query.py:88-104`
- Modify: `src/financial_report_qa/execution/compiler.py:71-87`
- Test: `tests/unit/execution/test_execution_pandas_query.py`

**Interfaces:**
- Consumes: không
- Produces: `_position_clauses` sinh predicate có cả nhãn; `_replay_row(...)` nhận thêm `row_label_raw`/`row_label_canonical` thật của corpus.

Spec 2026-08-21 §5.2 yêu cầu predicate ngữ nghĩa + `row_idx` **chỉ** phá thế hoà, và nói rõ đây là "lựa chọn có thể giải trình, khác với lookup thuần vị trí". Code hiện làm ngược. Đo được 18/87 ca `query_rejected` là do query lọc theo text của selector (cách nói trong câu hỏi) nên không khớp gì trong bảng thật.

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/execution/test_execution_pandas_query.py`:

```python
def test_position_bound_predicate_keeps_the_semantic_row_label() -> None:
    """Spec 2026-08-21 §5.2: `row_idx` chỉ phá thế hoà, không thay ngữ nghĩa.

    Một query thuần toạ độ không giải trình được nó lấy đúng chỉ tiêu nào.
    """
    from financial_report_qa.execution.pandas_query import render_pandas_query
    from financial_report_qa.planning.plan_contracts import (
        FinancialQueryPlan,
        MetricSelector,
    )

    table_id = "tbl_" + "a" * 64
    plan = FinancialQueryPlan(
        operation="lookup",
        companies=("ACB",),
        periods=("2023",),
        candidate_table_ids=(table_id,),
        metric=MetricSelector(raw_text="Doanh thu thuần", table_id=table_id, row_index=4),
    )
    query = render_pandas_query(plan)

    assert 'row_label_raw == "Doanh thu thuần"' in query
    assert "row_idx == 4" in query
    assert f'table_id == "{table_id}"' in query
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/execution/test_execution_pandas_query.py -q -k semantic_row_label
```

Kỳ vọng: FAIL — query hiện không chứa `row_label_raw`.

Nếu `render_pandas_query` không phải tên hàm công khai, đọc `pandas_query.py` và dùng đúng điểm vào; giữ nguyên ý test.

- [ ] **Step 3: Cài đặt — predicate**

Trong `src/financial_report_qa/execution/pandas_query.py`, thay `_position_clauses`:

```python
def _position_clauses(selector: MetricSelector) -> list[str]:
    """Spec 2026-08-21 §5.2: ngữ nghĩa mang ý nghĩa, vị trí chỉ phá thế hoà.

    Phần `row_label_*` là thứ khiến truy vấn giải trình được -- nó nói rõ câu
    trả lời lấy từ chỉ tiêu nào. `table_id`/`row_idx` đi kèm để phá thế hoà
    giữa các dòng trùng nhãn (đo được 4.37% dòng trùng nhãn khác giá trị, do
    OCR lặp dòng), chứ không thay thế phần ngữ nghĩa.

    Bản trước đây bỏ hẳn nhãn và viện dẫn "plan.md §14"; §14 của `plan.md`
    hiện tại là "Rủi ro lịch trình và phương án cắt giảm" -- tham chiếu treo.
    """
    assert selector.table_id is not None and selector.row_index is not None
    column, value = _metric_column_and_value(selector)
    clauses = [
        f"(df1.{column} == {_lit(value)})",
        f"(df1.table_id == {_lit(selector.table_id)})",
        f"(df1.row_idx == {int(selector.row_index)})",
    ]
    if selector.column_text is not None:
        clauses.append(f"(df1.column_label == {_lit(selector.column_text)})")
    return clauses
```

- [ ] **Step 4: Cài đặt — `_replay_row` dùng nhãn thật**

Vấn đề: `compiler._replay_row` dựng dòng replay từ `selector.raw_text`, nên replay nội bộ **tự khớp một cách giả tạo** — query lọc theo chính chuỗi mà nó vừa ghi vào frame. Frame corpus thật mang nhãn thật, nên cổng ở `exporter` mới trượt.

Trong `src/financial_report_qa/execution/compiler.py`, đổi `_replay_row` để nhận nhãn thật từ `CellMatch`:

```python
def _replay_row(
    *,
    company_code: str,
    selector: MetricSelector,
    period: int,
    value: Decimal,
    row_label_raw: str | None = None,
    row_label_canonical: str | None = None,
    column_label: str | None = None,
) -> dict[str, object]:
    """Một dòng của frame replay nội bộ.

    Nhãn phải là nhãn **thật của corpus** (`CellMatch`), không phải
    `selector.raw_text`. Dựng từ selector khiến replay nội bộ tự khớp: query
    lọc đúng chuỗi mà chính nó vừa ghi vào frame. Frame corpus thật ở
    `exporter._real_table_evidence_rows` mang nhãn thật, nên cái tự khớp đó
    biến thành `query_rejected` (đo được 18/87 ca).
    """
    return {
        "company_code": company_code,
        "row_label_canonical": (
            row_label_canonical if row_label_canonical is not None else selector.canonical
        ),
        "row_label_raw": row_label_raw if row_label_raw is not None else selector.raw_text,
        "column_label": column_label if column_label is not None else selector.column_text,
        "period": period,
        "value": value,
        "table_id": selector.table_id,
        "row_idx": selector.row_index,
    }
```

Rồi ở mọi chỗ gọi `_replay_row` trong `_dispatch`, truyền nhãn từ `CellMatch` tương ứng (`cell.row_label_raw`, `cell.row_label_canonical`, `cell.column_label` — đọc `execution/contracts.py::CellMatch` để lấy đúng tên trường; nếu `CellMatch` chưa mang nhãn, lấy từ hàng frame mà `locate()` đã chọn).

- [ ] **Step 5: Chạy test đầy đủ**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/execution tests/unit/submission -q
```

Test hiện có ghim query thuần vị trí sẽ đỏ. Đó là **hành vi cố ý đổi theo spec §5.2** — sửa test cho khớp predicate mới, và ghi lý do vào docstring test.

- [ ] **Step 6: Kiểm chứng trên dữ liệu thật**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit -q
```

- [ ] **Step 7: Commit**

```bash
git add src/financial_report_qa/execution/pandas_query.py src/financial_report_qa/execution/compiler.py tests/unit/execution && git commit -m "fix(execution): semantic predicate with row_idx tie-break, real corpus labels in replay rows"
```

---

### Task 5: Batch payload v2 — operation + nhiều dòng

**Files:**
- Modify: `src/financial_report_qa/planning/row_choice_batch.py`
- Modify: `src/financial_report_qa/submission/cli.py` (lệnh `row-batches`)
- Test: `tests/unit/planning/test_row_choice_batch.py`

**Interfaces:**
- Consumes: `RowFusedCandidate`, `QueryEntities`
- Produces: `build_batch_payload(question_id, question, entities, candidates) -> dict[str, object]`

- [ ] **Step 1: Viết test thất bại**

Thay `tests/unit/planning/test_row_choice_batch.py` bằng:

```python
"""Payload batch v2: LLM quyết cả operation lẫn dòng (spec 2026-08-23 §6.2)."""

from __future__ import annotations

from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.row_choice_batch import build_batch_payload
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_A = "tbl_" + "a" * 64


def _candidate(rank: int, *, label: str = "Doanh thu thuần", company: str = "ACB"):
    return RowFusedCandidate(
        row_id=f"{TABLE_A}|row_{rank}",
        table_id=TABLE_A,
        row_idx=rank,
        rank=rank,
        fused_score=1.0 / rank,
        metadata=RowMetadata(
            table_id=TABLE_A,
            row_idx=rank,
            company_code=company,
            row_label_raw=label,
            title="Báo cáo KQKD",
            periods=("2023",),
        ),
        snippet=label,
    )


def test_payload_carries_companies_and_periods_for_operation_choice() -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1),))
    assert payload["question_id"] == 7
    assert payload["companies"] == ["ACB"]
    assert payload["periods"] == ["2023"]


def test_candidate_carries_company_code_so_multi_company_picks_are_possible() -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1, company="VIC"),))
    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    assert candidates[0]["company_code"] == "VIC"
    assert candidates[0]["index"] == 0


def test_payload_never_leaks_a_cell_value_or_score() -> None:
    """Bất biến N7 -- đây là thứ giữ cho cam kết chống hardcode còn đứng vững."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    payload = build_batch_payload(7, entities.question, entities, (_candidate(1),))
    blob = repr(payload)
    assert "fused_score" not in blob
    assert "value" not in blob
    for candidate in payload["candidates"]:
        assert "value" not in candidate
        assert "fused_score" not in candidate
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_batch.py -q
```

Kỳ vọng: FAIL — `build_batch_payload` chưa nhận `entities`, payload chưa có `companies`.

- [ ] **Step 3: Cài đặt**

Trong `src/financial_report_qa/planning/row_choice_batch.py`:

```python
def _candidate_payload(index: int, candidate: RowFusedCandidate) -> dict[str, object]:
    metadata = candidate.metadata
    return {
        "index": index,
        "company_code": metadata.company_code,
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
    entities: QueryEntities,
    candidates: Sequence[RowFusedCandidate],
) -> dict[str, object]:
    """Một dòng JSONL: câu hỏi + công ty/kỳ đã tách + ứng viên đánh số từ 0.

    `companies` và `periods` đi kèm để model biết câu hỏi có mấy công ty --
    đó là thứ quyết định nó phải trả về mấy chỉ số trong `chosen` và
    operation nào hợp lý (`rank`/`compare_companies` chỉ có nghĩa khi nhiều
    công ty).

    `candidates` phải đã ở đúng thứ tự retrieval-rank; thứ tự này **là** hợp
    đồng -- `question_plan.assemble_plan` map `chosen` ngược về ứng viên bằng
    chính vị trí này. Không sắp lại.

    Không trường nào mang giá trị ô hay điểm fusion (bất biến N7).
    """
    return {
        "question_id": question_id,
        "question": question,
        "companies": list(entities.company_codes),
        "periods": list(entities.periods),
        "candidates": [
            _candidate_payload(index, candidate) for index, candidate in enumerate(candidates)
        ],
    }
```

Thêm import `QueryEntities`. Cập nhật chỗ gọi trong `submission/cli.py` (lệnh `row-batches`) để truyền `parse_query_entities(question)`.

- [ ] **Step 4: Chạy test, xác nhận pass**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_row_choice_batch.py tests/unit/submission -q
```

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/planning/row_choice_batch.py src/financial_report_qa/submission/cli.py tests/unit/planning/test_row_choice_batch.py && git commit -m "feat(planning): batch payload v2 carries companies, periods and per-candidate company"
```

---

### Task 6: `cell_grounding` → một đường thẳng

**Files:**
- Modify: `src/financial_report_qa/planning/cell_grounding.py`
- Test: `tests/unit/planning/test_cell_grounding.py`

**Interfaces:**
- Consumes: `assemble_plan`, `RowChoiceDecision` (Task 2); `compile_grounded` (giữ nguyên).
- Produces: `ground_question(entities, decision, fusion_rows, candidate_table_ids, release_dir, execution_settings) -> GroundingResult`

`GroundingResult` giữ nguyên hình dạng nhưng `plan_source` chỉ còn `"llm_decision"`.

- [ ] **Step 1: Viết test thất bại**

Thay `tests/unit/planning/test_cell_grounding.py` bằng bộ test mới (xoá mọi test của thang tầng cũ — chúng ghim hành vi đã bị bỏ):

```python
"""Nhánh 2 là một đường thẳng (spec 2026-08-23 §6, nguyên tắc N6).

Không thang tầng, không candidate switching, không context expansion. Một
câu hỏi đi qua đúng một chuỗi bước; hỏng ở đâu thì hỏng rõ ở đó.
"""

from __future__ import annotations

from pathlib import Path

from financial_report_qa.planning.cell_grounding import ground_question
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.planning.question_plan import RowChoiceDecision


def test_ground_question_reports_llm_decision_as_the_only_plan_source(
    release_dir: Path, execution_settings, fusion_rows, table_ids
) -> None:
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=RowChoiceDecision(question_id=1, operation="lookup", chosen=(0,)),
        fusion_rows=fusion_rows,
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.plan_source == "llm_decision"


def test_ground_question_fails_cleanly_when_no_plan_can_be_assembled(
    release_dir: Path, execution_settings, table_ids
) -> None:
    """Không ứng viên -> thất bại có mã, không exception, không tầng thứ hai."""
    entities = parse_query_entities("Tra cứu doanh thu thuần của ACB năm 2023.")
    result = ground_question(
        entities=entities,
        decision=None,
        fusion_rows=(),
        candidate_table_ids=table_ids,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    assert result.status == "failed"
    assert result.error_code == "no_row_candidates"


def test_cell_grounding_has_no_recovery_ladder_left() -> None:
    """Ghim N6: các tầng đã bỏ không được lặng lẽ quay lại."""
    from financial_report_qa.planning import cell_grounding

    for gone in (
        "ground_with_recovery",
        "_candidate_switching",
        "_context_expansion",
        "choose_row_label",
    ):
        assert not hasattr(cell_grounding, gone), f"{gone} thuộc thang tầng đã bỏ"
```

Fixture `release_dir`/`execution_settings`/`fusion_rows`/`table_ids`: tái dùng fixture đã có trong `tests/unit/submission/test_submission_exporter.py` (`_write_release`) — chuyển chúng vào `tests/unit/conftest.py` nếu chưa dùng chung được.

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning/test_cell_grounding.py -q
```

Kỳ vọng: FAIL — `ground_question` chưa tồn tại.

- [ ] **Step 3: Cài đặt**

Viết lại `cell_grounding.py` còn đúng: `compile_grounded` (giữ nguyên), `_accepted` (giữ), `GroundingResult` (giữ, thu hẹp `plan_source`), và:

```python
def ground_question(
    *,
    entities: QueryEntities,
    decision: RowChoiceDecision | None,
    fusion_rows: Sequence[RowFusedCandidate],
    candidate_table_ids: Sequence[str],
    release_dir: Path,
    execution_settings: ExecutionSettings,
) -> GroundingResult:
    """Đường answering duy nhất: quyết định -> plan -> compile.

    Không có tầng thứ hai. Nguyên tắc N6: thang tầng cũ che giấu một tầng có
    tỷ lệ trả lời 0% trên 409 câu, và không tầng nào chịu trách nhiệm. Ở đây
    một câu hỏi hỏng có đúng một mã lỗi, chỉ về đúng một bước.
    """
    if not fusion_rows:
        return GroundingResult(
            status="failed", error_code="no_row_candidates", plan_source="llm_decision"
        )

    plan = assemble_plan(entities, decision, fusion_rows, candidate_table_ids)
    if plan is None:
        return GroundingResult(
            status="failed", error_code="plan_not_assembled", plan_source="llm_decision"
        )

    compiled_plan, compiled = compile_grounded(
        plan, fusion_rows, release_dir, execution_settings
    )
    if compiled.status != "answered":
        return GroundingResult(
            status="failed",
            error_code=compiled.error_code or "execution_failed",
            plan_source="llm_decision",
        )
    return _accepted(
        plan=compiled_plan,
        compiled=compiled,
        plan_source="llm_decision",
        fusion_rows=fusion_rows,
    )
```

Xoá khỏi file: `ground_with_recovery`, vòng candidate switching, context expansion, mọi import của `column_refinement`, `llm_cell_grounding`, `raw_metric_grounding`, `rule_planner`, `row_choice_decision`, `DEFAULT_MAX_GROUNDING_RANK`, `_bind_metric_to_position`.

- [ ] **Step 4: Chạy test, xác nhận pass**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/planning -q
```

- [ ] **Step 5: Commit**

```bash
git add src/financial_report_qa/planning/cell_grounding.py tests/unit/planning/test_cell_grounding.py && git commit -m "refactor(planning): collapse cell grounding to the single decision path"
```

---

### Task 7: `exporter` → một đường thẳng, xoá tầng thừa

**Files:**
- Modify: `src/financial_report_qa/submission/exporter.py`
- Modify: `src/financial_report_qa/submission/cli.py`
- Delete: 16 module ở mục **File Structure → Xoá**
- Test: `tests/unit/submission/test_submission_exporter.py`

**Interfaces:**
- Consumes: `ground_question` (Task 6), `load_decisions` (Task 2)
- Produces: `QuestionOutcome.plan_source` chỉ còn `"llm_decision"` | `"backstop"`

- [ ] **Step 1: Viết test thất bại**

Thêm vào `tests/unit/submission/test_submission_exporter.py`:

```python
def test_export_reports_only_the_two_allowed_plan_sources(tmp_path: Path) -> None:
    """Tiêu chí thành công §12.5: chỉ còn `llm_decision` và `backstop`."""
    release_dir = _write_release(tmp_path)
    questions = [RawQuestion(id=1, question="Tra cứu doanh thu thuần của ACB năm 2023.")]
    report, _items, _rows = export_submission(
        questions,
        _service(),
        release_dir,
        execution_settings=_ALLOW_LOOKUP,
        dataset_fingerprint="0" * 64,
        k=5,
    )
    assert {outcome.plan_source for outcome in report.outcomes} <= {"llm_decision", "backstop"}


def test_exporter_no_longer_imports_any_deleted_tier() -> None:
    """Ghim N6 ở mức module: các tầng đã bỏ không được quay lại qua import."""
    import financial_report_qa.submission.exporter as exporter_module

    source = Path(exporter_module.__file__).read_text(encoding="utf-8")
    for gone in (
        "plan_router",
        "llm_planner",
        "llm_evidence_planner",
        "evidence_planner",
        "column_refinement",
        "raw_metric_grounding",
        "llm_cell_grounding",
        "rule_planner",
    ):
        assert gone not in source, f"{gone} thuộc thang tầng đã bỏ"
```

- [ ] **Step 2: Chạy test, xác nhận fail**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit/submission/test_submission_exporter.py -q -k "allowed_plan_sources or deleted_tier"
```

- [ ] **Step 3: Cài đặt — rút `_run_one_question`**

Thay toàn bộ khối lập plan (từ `plan_result, grounded = plan_with_raw_grounding_fallback(...)` đến hết nhánh `ground_with_recovery`) bằng:

```python
    entities = parse_query_entities(question)
    fusion_rows = (
        row_fusion.retrieve_rows(
            question, candidate_table_ids=retrieved, k=DEFAULT_ROW_CANDIDATE_COUNT
        ).results
        if row_fusion is not None
        else ()
    )

    grounding = ground_question(
        entities=entities,
        decision=(row_decisions or {}).get(raw_question.id),
        fusion_rows=fusion_rows,
        candidate_table_ids=retrieved,
        release_dir=release_dir,
        execution_settings=execution_settings,
    )
    if grounding.status != "accepted":
        return (
            QuestionOutcome.model_validate(
                {
                    "id": raw_question.id,
                    "question": question,
                    "status": "abstained",
                    "stage": "planning",
                    "code": grounding.error_code,
                    "plan_source": "llm_decision",
                }
            ),
            None,
            None,
        )
    assert grounding.plan is not None and grounding.compiled is not None
    plan_result = RulePlanResult(plan=grounding.plan)  # hoặc giữ thẳng grounding.plan
    compiled = grounding.compiled
    plan_source = "llm_decision"
    low_confidence = grounding.low_confidence
```

`RulePlanResult` đến từ `rule_planner` đang bị xoá — thay bằng dùng thẳng `grounding.plan` ở phần verification/đóng gói bên dưới. Đọc phần còn lại của `_run_one_question` và thay `plan_result.plan` bằng `grounding.plan`.

Bỏ tham số `llm_client` khỏi `export_submission`/`_run_one_question`; đổi `row_decisions` sang `Mapping[int, RowChoiceDecision] | None`.

- [ ] **Step 4: Cài đặt — thu hẹp `plan_source`**

Trong `submission/contracts.py`, đổi `QuestionOutcome.plan_source` thành:

```python
    plan_source: Literal["llm_decision", "backstop"] | None = None
```

- [ ] **Step 5: Cài đặt — CLI**

Trong `submission/cli.py`: bỏ `--llm-config` khỏi `export` và mọi nhánh `LLMClient`; `--row-choice-decisions` giờ dùng `question_plan.load_decisions`. `export_submission` chỉ còn một chỗ gọi.

- [ ] **Step 6: Xoá 16 module + test của chúng**

```bash
git rm src/financial_report_qa/planning/plan_router.py src/financial_report_qa/planning/llm_planner.py src/financial_report_qa/planning/llm_prompt.py src/financial_report_qa/planning/llm_contracts.py src/financial_report_qa/planning/evidence_planner.py src/financial_report_qa/planning/evidence_plan_contracts.py src/financial_report_qa/planning/llm_evidence_planner.py src/financial_report_qa/planning/evidence_facts.py src/financial_report_qa/planning/column_refinement.py src/financial_report_qa/planning/raw_metric_grounding.py src/financial_report_qa/planning/llm_cell_grounding.py src/financial_report_qa/planning/rule_planner.py src/financial_report_qa/planning/llm_evaluation.py src/financial_report_qa/planning/plan_evaluation.py src/financial_report_qa/planning/plan_cases.py src/financial_report_qa/planning/row_choice_decision.py
```

Rồi lần theo lỗi import: `planning/cli.py`, `pipeline/evaluation.py`, `execution/evaluation.py`, `verification/evaluation.py` đều import `rule_planner`. Với mỗi chỗ: nếu module đó chỉ phục vụ thang tầng cũ thì xoá luôn subcommand tương ứng; nếu còn cần, thay bằng `question_plan.assemble_plan`.

**Nếu một module tưởng là bỏ được lại đang phục vụ mục đích còn sống ngoài thang tầng — dừng lại và báo người dùng**, đừng tự quyết mở rộng phạm vi xoá.

- [ ] **Step 7: Chạy toàn bộ test**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/unit -q
```

```bash
.venv/Scripts/python.exe -m ruff check src/financial_report_qa tests
```

- [ ] **Step 8: Commit**

```bash
git add -A src tests && git commit -m "refactor(submission): single answering path, delete the seven-tier ladder"
```

---

### Task 8: Notebook Colab v2 — hỏi operation + danh sách dòng

**Files:**
- Modify: `notebooks/colab_row_choice_qwen3_8b.ipynb`

Notebook cũ chỉ hỏi `chosen_index`. Payload v2 có `companies`, và quyết định v2 có `operation`/`chosen`/`top_k`.

- [ ] **Step 1: Sửa cell prompt/parse**

Prompt:

```
Bạn là trợ lý phân tích báo cáo tài chính.
Không giải thích, không suy luận, không dùng <think>.

Câu hỏi: {question}
Công ty trong câu: {companies}
Kỳ trong câu: {periods}

Các dòng ứng viên:
{candidates}

Chọn phép tính và dòng trả lời câu hỏi.
- operation là MỘT trong: lookup, compare, compare_companies, difference,
  growth_rate, ratio, average, sum, rank
- chosen là danh sách chỉ số dòng. Một công ty -> một chỉ số.
  Nhiều công ty (rank, compare_companies, average, sum) -> mỗi công ty một
  chỉ số, đúng thứ tự công ty ở trên.
  ratio/compare -> đúng hai chỉ số (tử trước, mẫu sau).
- top_k chỉ cần cho rank (1 = cao nhất).

Chỉ trả về JSON, không kèm gì khác:
{{"operation": "...", "chosen": [...], "top_k": null}}
```

Parse — phải chịu được `<think>` rò rỉ và JSON lẫn văn xuôi:

```python
import json, re

_OPERATIONS = {"lookup", "compare", "compare_companies", "difference",
               "growth_rate", "ratio", "average", "sum", "rank"}

def parse(text, limit, n_companies):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    operation, chosen, top_k = "lookup", [], None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if payload.get("operation") in _OPERATIONS:
                operation = payload["operation"]
            raw = payload.get("chosen")
            if isinstance(raw, list):
                chosen = [int(v) for v in raw if isinstance(v, (int, float))]
            if isinstance(payload.get("top_k"), (int, float)):
                top_k = int(payload["top_k"])
        except (ValueError, TypeError):
            pass
    chosen = [v for v in chosen if 0 <= v < limit] or [0]
    if operation in {"rank", "compare_companies"} and n_companies < 2:
        operation = "lookup"           # arity không thoả -> để local hạ cấp
    return {"operation": operation, "chosen": chosen, "top_k": top_k}
```

Ghi ra `decisions.jsonl` với `question_id` + ba trường trên.

- [ ] **Step 2: Kiểm tra notebook hợp lệ**

Notebook JSON đã từng hỏng hai lần vì newline thật lọt vào chuỗi nguồn. Bắt buộc kiểm:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import json, ast
nb = json.load(open('notebooks/colab_row_choice_qwen3_8b.ipynb', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if src.lstrip().startswith('!'):
        continue
    ast.parse(src)
    print('cell', i, 'OK')
print('notebook JSON + Python đều hợp lệ')
"
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_row_choice_qwen3_8b.ipynb && git commit -m "feat(notebooks): batch decides operation and rows, not just a row index"
```

---

### Task 9: Chạy thật và đo (thủ công)

**Files:** không sửa code.

Bước này người dùng chạy; agent không có GPU và không thể tự kiểm chứng Colab.

- [ ] **Step 1: Sinh batch**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m financial_report_qa.cli submission row-batches --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json --bm25-index data/indexes/bm25-v4/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --questions-path data/raw/ViFinQA/questions/questions.jsonl --output-dir data/qa/row_choice_batches --k 10 --rows-per-question 20 --batch-size 64
```

Xoá sạch `--output-dir` trước khi chạy — lệnh không tự dọn, batch v1 cũ lẫn vào sẽ cho quyết định sai định dạng.

- [ ] **Step 2: Colab**

Runtime T4 GPU, chạy notebook, thử 1 batch trước, rồi cả 16. Tải `decisions.jsonl` về `data/qa/row_choice_decisions.jsonl`.

- [ ] **Step 3: Kiểm tra độ phủ**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import sys, collections; sys.path.insert(0,'src')
from pathlib import Path
from financial_report_qa.planning.question_plan import load_decisions
d = load_decisions(Path('data/qa/row_choice_decisions.jsonl'))
print('so quyet dinh:', len(d))
print('phan bo operation:', collections.Counter(v.operation for v in d.values()).most_common())
"
```

- [ ] **Step 4: Full export**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe _run_full_export.py
```

`_run_full_export.py` phải bỏ `--llm-config` (đã gỡ khỏi CLI ở Task 7).

- [ ] **Step 5: Đo**

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "
import json, collections
r = json.load(open('artifacts/evaluations/v2gaps_full/submission-export-422df141c935.json', encoding='utf-8'))
print('answered_count:', r['answered_count'], '(truoc: 60, muc tieu: >186)')
print('plan_source:', collections.Counter(o.get('plan_source') for o in r['outcomes']).most_common())
codes = collections.Counter((o.get('stage'), o.get('code')) for o in r['outcomes'] if o.get('status') != 'answered')
for (stage, code), n in codes.most_common(10):
    print(f'{n:5d}  {stage}: {code}')
"
```

**Cổng dừng:** nếu `answered_count < 60`, việc rút thang tầng đã mất câu — dừng lại phân tích, **không nộp**. Rủi ro này có thật: 38/60 câu hiện tại đến từ `rule`/`rule_raw_grounded`/`llm_evidence_planner`, đều là tầng bị xoá.

---

## Self-Review

**Spec coverage.** §6.1 giữ nguyên (không task nào cần). §6.2 → Task 5 + 8. §6.3 → Task 2. §6.4 → Task 3. §6.5 giữ nguyên. §6.6 → Task 6 + 7. §7.1 → Task 4. §7.2/§7.3 đã xong trước plan này. §8.1 → Task 7 Step 6. T5 → Task 9. **Không phủ:** §5 (Nhánh 1) — cố ý, có plan riêng.

**Type consistency.** `RowChoiceDecision` dùng nhất quán ở Task 2 (định nghĩa), 6 (`decision=`), 7 (`Mapping[int, RowChoiceDecision]`), 8 (sinh JSON), 9 (đọc lại). `load_decisions` trả `dict[int, RowChoiceDecision]` ở cả Task 2 và Task 9. `plan_source` là `"llm_decision"` ở Task 6, 7 và test §12.5.

**Chỗ plan cố ý không chốt cứng.** Task 3 Step 3 và Task 4 Step 4 nói "đọc file để tìm đúng điểm nối" thay vì dán code hoàn chỉnh — vì cả hai phụ thuộc vào cấu trúc nội bộ mà tôi chưa đọc hết (`entity_parser` chỗ dựng `periods`, `CellMatch` có mang nhãn hay không). Người thực thi phải đọc trước khi sửa; nếu cấu trúc khác dự đoán, **báo lại chứ không tự bẻ hướng**.

**Rủi ro lớn nhất** đã có cổng dừng ở Task 9 Step 5: rút thang tầng có thể làm giảm `answered_count` trước khi làm tăng.
