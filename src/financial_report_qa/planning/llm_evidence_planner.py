"""plan.md §12 Evidence-Aware Planner: the model picks arithmetic, not cells.

The prompt this module builds is the whole redesign in one place. It contains:

- the question,
- the grounded candidate facts, numbered, each with its label, period, value
  and unit -- real cells read out of the release by `evidence_facts`,
- the allowed operations,
- a strict JSON schema whose only fields are `operation` and `operands`.

What it no longer contains is the reasoning chain §12 blames for the 231
`llm_plan_invalid` questions: find the metric, then the table, then the row,
then the column, then emit a whole typed plan. The model is asked for an enum
and a couple of identifiers that were printed in its own prompt, which is the
shape the organizers' own measurements say a sub-10B model can actually hit
(`llm_cell_grounding`'s module docstring has the numbers).

Every failure mode returns `None` -- unreachable model, unparseable reply,
disallowed operation, invented fact id, duplicate operands -- so the caller
falls through to its existing planner rather than handling exceptions.
`evidence_planner.build_plan_from_facts` then re-checks the chosen facts for
consistency before anything executes; nothing here is trusted downstream.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import LLMError
from financial_report_qa.planning.evidence_facts import DEFAULT_MAX_FACTS, enumerate_candidate_facts
from financial_report_qa.planning.evidence_plan_contracts import EvidencePlan
from financial_report_qa.planning.evidence_planner import build_plan_from_facts
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.plan_contracts import ExpectedUnit, FinancialQueryPlan
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

DEFAULT_ALLOWED_OPERATIONS: tuple[str, ...] = (
    "lookup",
    "difference",
    "growth_rate",
    "ratio",
    "compare",
    "sum",
    "average",
)
"""The operations an evidence plan can express. `compare_companies` and `rank`
are absent for the reason `evidence_planner` documents: their single metric
selector serves several companies and therefore cannot be position-bound."""

_SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích báo cáo tài chính Việt Nam. Bạn sẽ nhận một câu hỏi "
    "và một danh sách CÁC SỐ LIỆU ĐÃ ĐƯỢC TRÍCH SẴN từ báo cáo thật, mỗi số liệu "
    "có một mã (F1, F2, ...).\n\n"
    "Nhiệm vụ DUY NHẤT của bạn: chọn PHÉP TÍNH cần thực hiện và các mã số liệu "
    "làm toán hạng.\n\n"
    "Quy tắc:\n"
    '- Trả về DUY NHẤT một JSON object dạng {"operation": "...", "operands": ["F1", "F2"]}.\n'
    "- Chỉ dùng các mã có trong danh sách; TUYỆT ĐỐI không tự nghĩ ra mã mới.\n"
    "- Không cần nêu tên bảng, tên dòng, tên cột hay công thức: những thứ đó đã "
    "được xác định sẵn.\n\n"
    "Ý nghĩa các phép tính:\n"
    "- lookup: lấy đúng một số liệu (1 toán hạng).\n"
    "- difference: chênh lệch của cùng một chỉ tiêu giữa hai kỳ (2 toán hạng).\n"
    "- growth_rate: tốc độ tăng trưởng của cùng một chỉ tiêu giữa hai kỳ "
    "(2 toán hạng).\n"
    "- ratio: tỷ lệ giữa hai chỉ tiêu trong cùng một kỳ; toán hạng đầu là tử số "
    "(2 toán hạng).\n"
    "- compare: chênh lệch giữa hai chỉ tiêu khác nhau trong cùng một kỳ "
    "(2 toán hạng).\n"
    "- sum / average: tổng hoặc trung bình của cùng một chỉ tiêu qua nhiều kỳ."
)


def _schema(allowed_operations: Sequence[str]) -> dict[str, object]:
    """The output grammar. Two fields, and `additionalProperties: false`.

    With `json_schema_constrained` (configs/*.yaml) the server enforces this
    as a hard grammar, so "the model emitted something that is not a plan"
    stops being a failure mode rather than being parsed around.
    """
    return {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": list(allowed_operations)},
            "operands": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["operation", "operands"],
        "additionalProperties": False,
    }


def _render_fact(fact: GroundedFact) -> str:
    column = f", cột {fact.column}" if fact.column else ""
    return (
        f"{fact.fact_id}: {fact.row_label} | kỳ {fact.period}{column} "
        f"| giá trị {fact.raw_value} | đơn vị {fact.unit}"
    )


def _build_user_prompt(question: str, facts: Sequence[GroundedFact]) -> str:
    rendered = "\n".join(_render_fact(fact) for fact in facts)
    return f"Câu hỏi: {question}\n\nCác số liệu đã trích sẵn:\n{rendered}\n\nJSON kết quả:"


def _parse(content: str) -> dict[str, object] | None:
    """Read the reply as an object, tolerating surrounding prose.

    Lenient about shape, strict about content -- exactly the split
    `llm_cell_grounding._parse_choice` uses. Nothing invalid is repaired
    here; it is simply not returned.
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def choose_evidence_plan(
    question: str,
    facts: Sequence[GroundedFact],
    *,
    client: ChatCompletionClient,
    allowed_operations: Sequence[str] = DEFAULT_ALLOWED_OPERATIONS,
) -> EvidencePlan | None:
    """Ask the planner which arithmetic to run over `facts`, or `None`."""
    if not facts or not allowed_operations:
        return None

    try:
        content = client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(question, facts),
            json_schema=_schema(allowed_operations),
        )
    except LLMError:
        return None

    payload = _parse(content)
    if payload is None:
        return None
    operation = payload.get("operation")
    operands = payload.get("operands")
    if operation not in allowed_operations or not isinstance(operands, list):
        return None

    known = {fact.fact_id for fact in facts}
    normalized: list[str] = []
    for operand in operands:
        if not isinstance(operand, str):
            return None
        # A small model writes `f1` often enough that refusing it would throw
        # away correct plans over a formatting slip. Only the case is
        # forgiven -- an id that is not on the menu is still refused.
        candidate = operand.strip().upper()
        if candidate not in known:
            return None
        normalized.append(candidate)

    try:
        return EvidencePlan(operation=operation, operands=tuple(normalized))  # type: ignore[arg-type]
    except ValidationError:
        return None


def plan_with_evidence(
    question: str,
    fusion_rows: Sequence[RowFusedCandidate],
    release_dir: Path,
    *,
    client: ChatCompletionClient,
    company_code: str | None = None,
    periods: Sequence[int] = (),
    expected_unit: ExpectedUnit | None = None,
    allowed_operations: Sequence[str] = DEFAULT_ALLOWED_OPERATIONS,
    max_facts: int = DEFAULT_MAX_FACTS,
) -> FinancialQueryPlan | None:
    """The whole plan.md §12 tier: facts -> planner -> executable plan.

    Returns `None` whenever any stage declines -- no retrieved rows, no
    enumerable facts, a model that will not name an operation, or a chosen
    operand set that fails `build_plan_from_facts`'s consistency checks. The
    caller falls through to its existing planner, so this tier can only add
    answers.
    """
    facts = enumerate_candidate_facts(
        release_dir,
        fusion_rows,
        company_code=company_code,
        periods=periods,
        max_facts=max_facts,
    )
    if not facts:
        return None
    evidence_plan = choose_evidence_plan(
        question, facts, client=client, allowed_operations=allowed_operations
    )
    if evidence_plan is None:
        return None
    return build_plan_from_facts(evidence_plan, facts, expected_unit=expected_unit).plan
