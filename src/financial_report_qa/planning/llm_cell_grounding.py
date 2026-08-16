"""Day 25 LLM cell-grounding tier: the LLM picks WHICH ROW, nothing else.

Grounded in the organizers' own error analysis of 430 end-to-end failures
(`_2026__AIGuru___Finance_Tabular_QA`, p. 40): 54.7% numerical extraction
(wrong cell) + 34.7% insufficient evidence = 89.3%, against 0.9% arithmetic
and 0.5% formula errors -- "LLMs không dốt toán, vấn đề là chọn sai ô số
liệu". Cell selection is the bottleneck worth spending a model call on;
arithmetic is not.

The same deck (p. 39) measures why our earlier tiers returned 0/75: models
under 10B crash on Program-of-Thought style structured generation at up to a
99% syntax-error rate (Qwen3-4B: 4.64% PoT vs 11.56% CoT). Asking a local 7B
model for a whole typed `FinancialQueryPlan` is exactly that failing shape.

So this tier asks the smallest possible question instead: given the row
labels that genuinely exist in this question's own retrieved tables,
numbered, reply with one integer. The model cannot invent a label -- it can
only index into real ones, or decline with 0. Everything downstream
(locator, compiler, sandbox replay, verification, evidence CSV) is unchanged
and still proves the number independently.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from financial_report_qa.core.errors import LLMError
from financial_report_qa.planning.llm_client import ChatCompletionClient

# Keeps the prompt inside a small model's usable context and bounds latency.
# Candidate frames for one question rarely exceed this after company/period
# scoping; beyond it, the extra labels are almost always unrelated notes.
_MAX_LABELS = 60

_SYSTEM_PROMPT = (
    "Bạn là trợ lý đọc báo cáo tài chính Việt Nam. Bạn sẽ nhận một câu hỏi và "
    "một danh sách nhãn dòng ĐÃ ĐÁNH SỐ, trích nguyên văn từ các bảng dữ liệu "
    "thật liên quan tới câu hỏi.\n\n"
    "Nhiệm vụ DUY NHẤT của bạn: chọn nhãn dòng chứa số liệu mà câu hỏi cần.\n\n"
    "Quy tắc:\n"
    '- Trả về DUY NHẤT một JSON object dạng {"choice": <số thứ tự>}.\n'
    '- Nếu không nhãn nào đúng, dùng {"choice": 0}.\n'
    "- Không được tự nghĩ ra nhãn mới; chỉ chọn trong danh sách."
)

# One integer field: the smallest output shape that still pins the model to a
# real row. `json_schema_constrained` (configs/*.yaml) turns this into a hard
# grammar constraint on the server, so "invalid JSON" -- 73.5% of the Day 22
# failures at this tier -- stops being a failure mode at all.
_CHOICE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
    "additionalProperties": False,
}


_COLUMN_SYSTEM_PROMPT = (
    "Bạn là trợ lý đọc báo cáo tài chính Việt Nam. Bạn sẽ nhận một câu hỏi, "
    "TÊN DÒNG đã xác định, và danh sách TÊN CỘT ĐÃ ĐÁNH SỐ, trích nguyên văn "
    "từ chính bảng chứa dòng đó.\n\n"
    "Nhiệm vụ DUY NHẤT của bạn: chọn cột chứa con số mà câu hỏi cần.\n\n"
    "Lưu ý: một dòng thường có nhiều cột khác nghĩa nhau, ví dụ "
    '"Số đầu năm", "Số phát sinh trong năm", "Số đã thực nộp", "Số cuối năm". '
    "Hãy đọc kỹ câu hỏi hỏi thời điểm nào và loại số nào.\n\n"
    "Quy tắc:\n"
    '- Trả về DUY NHẤT một JSON object dạng {"choice": <số thứ tự>}.\n'
    '- Nếu không cột nào đúng, dùng {"choice": 0}.\n'
    "- Không được tự nghĩ ra tên cột; chỉ chọn trong danh sách."
)


def _build_user_prompt(question: str, labels: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {label}" for index, label in enumerate(labels, start=1))
    return f"Câu hỏi: {question}\n\nCác nhãn dòng có thật:\n{numbered}\n\nSố thứ tự:"


def _parse_choice(content: str, label_count: int) -> int | None:
    """Extract the chosen index, validated against the real range.

    Reads the JSON object when the reply is one, else falls back to the first
    integer in the text (a small model sometimes ignores the format despite
    the schema). Lenient about shape, strict about value: anything outside
    1..label_count -- including the explicit 0 "none of these" -- yields
    None, never a clamped or nearest guess.
    """
    choice: int | None = None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("choice"), int):
        choice = payload["choice"]
    else:
        match = re.search(r"-?\d+", content)
        if match is not None:
            choice = int(match.group())
    if choice is None or not 1 <= choice <= label_count:
        return None
    return choice


def choose_row_label(
    question: str,
    labels: Sequence[str],
    *,
    client: ChatCompletionClient,
) -> str | None:
    """Return the one row label the LLM picks for this question, or `None`.

    `None` on every failure mode (no labels, model unreachable, unparseable
    reply, explicit "none of these", out-of-range index) so the caller simply
    falls through to its existing abstain path.
    """
    candidates = list(dict.fromkeys(label for label in labels if label and label.strip()))
    if not candidates:
        return None
    candidates = candidates[:_MAX_LABELS]

    try:
        content = client.complete_json(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(question, candidates),
            json_schema=_CHOICE_SCHEMA,
        )
    except LLMError:
        return None

    choice = _parse_choice(content, len(candidates))
    if choice is None:
        return None
    return candidates[choice - 1]


def choose_column_label(
    question: str,
    row_label: str,
    columns: Sequence[str],
    *,
    client: ChatCompletionClient,
) -> str | None:
    """Return the one column header the LLM picks for this question, or `None`.

    The second half of identifying a cell. A Vietnamese statement row commonly
    spans several semantically distinct amounts -- the real PC1 2025 tax note
    puts "Số phải nộp đầu năm", "Số phải nộp trong năm", "Số đã thực nộp" and
    "Số phải nộp cuối năm" on one row -- and two of those can resolve to the
    same year, which is why the locator abstains as `cell_ambiguous`.

    Deterministic column grounding was measured and rejected first: of the 117
    questions still abstaining, only 5 name a column in their text at all, and
    those matches were mostly row labels that happen to recur as headers. The
    column is genuinely absent from the question and has to be inferred from
    the headers themselves, which is a reading task, not a rule.

    Same contract as `choose_row_label`: the model can only index into real
    headers, `None` on every failure mode, and the locator/compiler/sandbox
    still prove the resulting number independently.
    """
    candidates = list(dict.fromkeys(column for column in columns if column and column.strip()))
    if not candidates:
        return None
    candidates = candidates[:_MAX_LABELS]

    numbered = "\n".join(f"{index}. {column}" for index, column in enumerate(candidates, start=1))
    user_prompt = (
        f"Câu hỏi: {question}\n\n"
        f"Dòng đã xác định: {row_label}\n\n"
        f"Các tên cột có thật của bảng chứa dòng này:\n{numbered}\n\n"
        "Số thứ tự:"
    )

    try:
        content = client.complete_json(
            system_prompt=_COLUMN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=_CHOICE_SCHEMA,
        )
    except LLMError:
        return None

    choice = _parse_choice(content, len(candidates))
    if choice is None:
        return None
    return candidates[choice - 1]
