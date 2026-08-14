"""Unit tests for the Day 17 plan router (§17.7, ADR 0006 decision A1).

Rule planner runs first; the LLM planner is only invoked on abstain. The
router must never let the LLM override a plan the rule planner already
produced successfully — that guarantee is what keeps the Day 16 numbers
(operation accuracy 1.0, false-plan rate 0.0) intact after Day 17 lands.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.planning.plan_router import route_plan

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


def _entities(question: str, **kwargs: object) -> QueryEntities:
    return QueryEntities(question=question, **kwargs)  # type: ignore[arg-type]


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _client_counting(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[LLMClient, dict[str, int]]:
    calls = {"count": 0}

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return handler(request)

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(counting_handler), max_retries=1)
    return client, calls


def test_rule_planner_success_never_calls_the_llm() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM must not be called when the rule planner already succeeded")

    client, calls = _client_counting(handler)
    entities = _entities(
        "Tra cứu doanh thu thuần của NVL năm 2023.",
        company_codes=("NVL",),
        periods=("2023",),
        metrics=("net_revenue",),
    )

    routed = route_plan(
        entities, client=client, candidate_table_ids=_TABLE_IDS, known_table_ids=_KNOWN_TABLES
    )

    assert routed.source == "rule"
    assert routed.result.plan is not None
    assert routed.result.plan.operation == "lookup"
    assert calls["count"] == 0


def test_rule_planner_abstain_falls_back_to_llm() -> None:
    valid_plan = json.dumps(
        {
            "operation": "lookup",
            "companies": ["NVL"],
            "periods": ["2023"],
            "metric": {"canonical": "net_revenue"},
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(valid_plan))

    client, calls = _client_counting(handler)
    entities = _entities(
        "Câu hỏi không rõ chỉ tiêu nào.",
        company_codes=("NVL",),
        periods=("2023",),
        metrics=(),
        ambiguity=("metric_unknown",),
    )

    routed = route_plan(
        entities, client=client, candidate_table_ids=_TABLE_IDS, known_table_ids=_KNOWN_TABLES
    )

    assert routed.source == "llm"
    assert routed.result.plan is not None
    assert calls["count"] == 1


def test_both_planners_abstain_reports_llm_source_and_codes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("not json"))

    client, calls = _client_counting(handler)
    entities = _entities(
        "Câu hỏi không rõ chỉ tiêu nào.",
        company_codes=("NVL",),
        periods=("2023",),
        metrics=(),
        ambiguity=("metric_unknown",),
    )

    routed = route_plan(
        entities, client=client, candidate_table_ids=_TABLE_IDS, known_table_ids=_KNOWN_TABLES
    )

    assert routed.source == "llm"
    assert routed.result.plan is None
    assert routed.result.abstain_codes == ("llm_invalid_json",)
    assert calls["count"] == 2
