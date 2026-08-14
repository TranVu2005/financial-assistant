"""Unit tests for the Day 17 LLM planner (§17.6).

Every scenario drives a real `LLMClient` over `httpx.MockTransport` so the
retry/repair wiring between `llm_client.py` and `llm_planner.py` is exercised
end to end, without ever reaching a real socket (Day 17 plan §1.8).
"""

from __future__ import annotations

import json

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.planning.llm_planner import build_plan
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen3-4b-instruct-2507-q4_k_m",
    timeout_seconds=5.0,
    max_output_tokens=160,
    temperature=0.0,
    context_length=4096,
    json_schema_constrained=True,
)
_TABLE_IDS = ("tbl_" + "a" * 64,)
_KNOWN_TABLES = frozenset(_TABLE_IDS)
_QUESTION = "Tra cứu doanh thu thuần của NVL năm 2023."
_VALID_LOOKUP = json.dumps(
    {
        "operation": "lookup",
        "companies": ["NVL"],
        "periods": ["2023"],
        "metric": {"canonical": "net_revenue"},
    }
)


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _client(handler: object) -> LLMClient:
    return LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=2)  # type: ignore[arg-type]


def test_valid_first_response_returns_a_plan() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_VALID_LOOKUP))

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert result.abstain_codes == ()
    assert result.plan is not None
    assert isinstance(result.plan, FinancialQueryPlan)
    assert result.plan.candidate_table_ids == _TABLE_IDS
    assert result.plan.operation == "lookup"


def test_invalid_json_then_valid_repair_returns_a_plan() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_envelope("not json at all"))
        body = json.loads(request.content)
        assert "Lỗi" in body["messages"][1]["content"]
        return httpx.Response(200, json=_envelope(_VALID_LOOKUP))

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert calls["count"] == 2
    assert result.plan is not None
    assert result.plan.operation == "lookup"


def test_invalid_json_twice_abstains_with_llm_invalid_json() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_envelope("still not json"))

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert calls["count"] == 2
    assert result.plan is None
    assert result.abstain_codes == ("llm_invalid_json",)


def test_schema_invalid_plan_twice_abstains_with_llm_plan_invalid() -> None:
    invalid = json.dumps(
        {"operation": "not_a_real_operation", "companies": ["NVL"], "periods": ["2023"]}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(invalid))

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert result.plan is None
    assert result.abstain_codes == ("llm_plan_invalid",)


def test_semantically_invalid_plan_twice_abstains_with_llm_plan_invalid() -> None:
    """growth_rate forbids `expected_unit` values other than "percent" —
    valid JSON, valid `LLMPlanOutput`, but rejected by `validate_plan_semantics`."""
    bad_unit = json.dumps(
        {
            "operation": "growth_rate",
            "companies": ["NVL"],
            "periods": ["2022", "2023"],
            "metric": {"canonical": "net_revenue"},
            "expected_unit": "ratio",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(bad_unit))

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert result.plan is None
    assert result.abstain_codes == ("llm_plan_invalid",)


def test_llm_unavailable_abstains_without_attempting_repair() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="unavailable")

    result = build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    # LLMClient itself retries (max_retries=2 -> 3 attempts) before raising;
    # the planner must not layer a second logical repair round trip on top.
    assert calls["count"] == 3
    assert result.plan is None
    assert result.abstain_codes == ("llm_unavailable",)


def test_repair_prompt_references_the_original_bad_output() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(200, json=_envelope('{"operation": "lookup"}'))
        return httpx.Response(200, json=_envelope(_VALID_LOOKUP))

    build_plan(
        _QUESTION,
        client=_client(handler),
        candidate_table_ids=_TABLE_IDS,
        known_table_ids=_KNOWN_TABLES,
    )

    assert len(calls) == 2
    messages = calls[1]["messages"]
    assert isinstance(messages, list)
    repair_user_message = messages[1]["content"]
    assert '{"operation": "lookup"}' in repair_user_message
    assert calls[0] != calls[1]
