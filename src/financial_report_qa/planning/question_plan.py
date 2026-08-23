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

import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import Field, ValidationError

from financial_report_qa.core.errors import PlanningInputError
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

    Nhưng *mọi* dòng hỏng thì không còn là "một dòng hỏng", đó là sai định
    dạng, và nó phải nổ chứ không được trả về `{}`. File quyết định v1 mang
    `chosen_index`; `RowChoiceDecision` cấm trường lạ nên mỗi dòng ném
    `ValidationError`. Bản đầu nuốt sạch: 970 quyết định biến mất không một
    cảnh báo, mọi câu rơi về `DEFAULT_DECISION` (lookup hạng 1), và một lần
    export dài một tiếng cho ra đúng baseline mà không ai biết vì sao. Bỏ qua
    100% input là chế độ hỏng tệ nhất có thể -- nó trông y hệt thành công.
    """
    decisions: dict[int, RowChoiceDecision] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        try:
            decision = RowChoiceDecision.model_validate_json(stripped)
        except ValidationError:
            continue
        decisions[decision.question_id] = decision

    if total and not decisions:
        raise PlanningInputError(
            f"{path}: đọc được 0/{total} quyết định. Kỳ vọng các trường "
            "`question_id`/`operation`/`chosen`/`top_k`; file quyết định v1 "
            "(`chosen_index`) không còn dùng được -- chạy lại notebook v2."
        )
    if len(decisions) < total:
        warnings.warn(
            f"{path}: chỉ đọc được {len(decisions)}/{total} quyết định, "
            f"{total - len(decisions)} dòng bị bỏ qua vì không hợp lệ. "
            "Các câu tương ứng sẽ dùng mặc định (lookup hạng 1).",
            UserWarning,
            stacklevel=2,
        )
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

    # Hạ cấp: một công ty, một kỳ, dòng đầu tiên LLM đã chọn.
    #
    # Áp dụng cả khi `operation` ĐÃ là `lookup`. `lookup` chỉ hợp lệ với đúng
    # một công ty và một kỳ (`plan_validator`), nên một quyết định `lookup`
    # trên câu nhiều công ty/kỳ bị `_construct` từ chối -- và bản đầu chặn
    # đường hạ cấp ở đây bằng `if operation == "lookup": return None`, làm câu
    # hỏi biến mất thay vì trả lời chiều đầu tiên. Đo trên lần export thật:
    # 264/322 ca `plan_not_assembled` có đủ công ty lẫn kỳ, và mọi ca trong đó
    # đều mang hình dạng này (129 ca `1 công ty × 2 kỳ`, 39 ca `2 công ty × 1
    # kỳ`, phần còn lại nhiều công ty hơn).
    #
    # Hạ cấp chỉ có nghĩa khi nó thật sự cắt bớt được gì đó: một `lookup` đã
    # tối giản mà vẫn hỏng thì bản hạ cấp giống hệt bản vừa thất bại.
    if operation == "lookup" and len(entities.company_codes) <= 1 and len(entities.periods) <= 1:
        return None
    trimmed = entities.model_copy(
        update={"company_codes": entities.company_codes[:1], "periods": entities.periods[:1]}
    )
    return _construct("lookup", trimmed, picks[:1], candidate_table_ids, None)
