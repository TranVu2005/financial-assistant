"""Regression tests for prompt-budget row candidate ranking."""

from __future__ import annotations

import json

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.planning.llm_cell_grounding import choose_row_label
from financial_report_qa.planning.llm_client import LLMClient

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen2.5:7b",
    timeout_seconds=5.0,
    max_output_tokens=32,
    temperature=0.0,
    context_length=32768,
    json_schema_constrained=True,
)


def _client(handler: object) -> LLMClient:
    return LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)  # type: ignore[arg-type]


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def test_relevant_row_after_prompt_cap_is_ranked_into_the_menu() -> None:
    """Before the fix, candidate 61 was discarded before the model saw it.

    Candidate labels arrive in a stable corpus order, not relevance order. A
    row explicitly named by the question therefore has to be ranked before the
    60-label prompt cap is applied.
    """
    target = "Lợi nhuận sau thuế chưa phân phối"
    labels = tuple(f"Khoản mục kế toán phụ {index:02d}" for index in range(60)) + (target,)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 1})))

    question = "Lợi nhuận sau thuế chưa phân phối của ACB năm 2023 là bao nhiêu?"
    chosen = choose_row_label(question, labels, client=_client(handler))

    assert chosen == target
    prompt = captured["body"]["messages"][1]["content"]  # type: ignore[index]
    assert f"1. {target}" in prompt
    assert "Khoản mục kế toán phụ 59" not in prompt


def test_small_candidate_menu_keeps_existing_order() -> None:
    """Ranking is only a prompt-budget operation; small menus stay unchanged."""
    labels = ("Doanh thu thuần", "Giá vốn hàng bán", "Lợi nhuận sau thuế")

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][1]["content"]
        assert "1. Doanh thu thuần" in prompt
        assert "2. Giá vốn hàng bán" in prompt
        assert "3. Lợi nhuận sau thuế" in prompt
        return httpx.Response(200, json=_envelope(json.dumps({"choice": 3})))

    chosen = choose_row_label(
        "Lợi nhuận sau thuế của ACB năm 2023 là bao nhiêu?",
        labels,
        client=_client(handler),
    )
    assert chosen == "Lợi nhuận sau thuế"
