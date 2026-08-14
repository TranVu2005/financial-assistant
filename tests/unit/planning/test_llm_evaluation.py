"""Unit tests for the Day 17 LLM-planner/router evaluation harness (§17.8).

ADR 0006 decision C1: two independent measurements, never mixed —
`evaluate_llm_plan_cases` scores the LLM planner alone (bypassing the rule
planner) on `expected_operation` cases; `evaluate_router_abstain_cases` scores
the full A1 router on `expected_abstain_code` cases, where `false_plan_rate`
is the hard KPI that must stay 0.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.core.errors import LLMUnavailableError
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.planning.llm_evaluation import (
    ReplayCacheClient,
    cache_key,
    evaluate_llm_plan_cases,
    evaluate_router_abstain_cases,
    load_replay_cache,
    write_llm_plan_case_report,
    write_router_abstain_report,
)
from financial_report_qa.planning.plan_cases import PlanCase
from financial_report_qa.planning.plan_contracts import PlanOperation
from financial_report_qa.planning.rule_planner import PlanAbstainCode

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen3-4b-instruct-2507-q4_k_m",
    timeout_seconds=5.0,
    max_output_tokens=160,
    temperature=0.0,
    context_length=4096,
    json_schema_constrained=True,
)
_MODEL_IDENTITY = "qwen3-4b-instruct-2507-q4_k_m@t0.0"


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _valid_lookup(company: str, period: str, metric: str = "net_revenue") -> str:
    return json.dumps(
        {
            "operation": "lookup",
            "companies": [company],
            "periods": [period],
            "metric": {"canonical": metric},
        }
    )


# --- ReplayCacheClient -------------------------------------------------


def test_cache_key_is_stable_for_identical_inputs() -> None:
    assert cache_key(_MODEL_IDENTITY, "sys", "user") == cache_key(_MODEL_IDENTITY, "sys", "user")


def test_cache_key_differs_for_different_prompts() -> None:
    assert cache_key(_MODEL_IDENTITY, "sys", "a") != cache_key(_MODEL_IDENTITY, "sys", "b")


def test_replay_cache_client_returns_cached_content_without_calling_underlying() -> None:
    key = cache_key(_MODEL_IDENTITY, "sys", "user")
    client = ReplayCacheClient(
        cache={key: "cached-content"}, model_identity=_MODEL_IDENTITY, underlying=None
    )
    content = client.complete_json(system_prompt="sys", user_prompt="user", json_schema={})
    assert content == "cached-content"


def test_replay_cache_client_raises_unavailable_on_miss_with_no_underlying() -> None:
    client = ReplayCacheClient(cache={}, model_identity=_MODEL_IDENTITY, underlying=None)
    with pytest.raises(LLMUnavailableError):
        client.complete_json(system_prompt="sys", user_prompt="user", json_schema={})


def test_replay_cache_client_falls_through_to_underlying_and_records(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("live-content"))

    underlying = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    record_path = tmp_path / "cache.jsonl"
    client = ReplayCacheClient(
        cache={}, model_identity=_MODEL_IDENTITY, underlying=underlying, record_path=record_path
    )

    content = client.complete_json(system_prompt="sys", user_prompt="user", json_schema={})

    assert content == "live-content"
    recorded = load_replay_cache(record_path)
    assert recorded[cache_key(_MODEL_IDENTITY, "sys", "user")] == "live-content"


def test_load_replay_cache_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    assert load_replay_cache(tmp_path / "does-not-exist.jsonl") == {}


# --- evaluate_llm_plan_cases --------------------------------------------


def _plan_case(case_id: str, question: str, expected_operation: PlanOperation) -> PlanCase:
    return PlanCase(
        case_id=case_id,
        template_id="lookup_ticker",
        question=question,
        expected_operation=expected_operation,
    )


def test_evaluate_llm_plan_cases_all_correct() -> None:
    cases = (
        _plan_case("a", "Tra cứu doanh thu thuần của NVL năm 2023.", "lookup"),
        _plan_case("b", "Tra cứu doanh thu thuần của VGT năm 2022.", "lookup"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        question = body["messages"][1]["content"]
        if "NVL" in question:
            return httpx.Response(200, json=_envelope(_valid_lookup("NVL", "2023")))
        return httpx.Response(200, json=_envelope(_valid_lookup("VGT", "2022")))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_llm_plan_cases(cases, client=client, case_set_sha256="f" * 64)

    assert report.case_count == 2
    assert report.operation_accuracy == 1.0
    assert report.invalid_json_rate == 0.0
    assert report.repair_success_rate == 0.0
    assert report.failures == ()


def test_evaluate_llm_plan_cases_counts_invalid_json_rate() -> None:
    cases = (_plan_case("a", "Tra cứu doanh thu thuần của NVL năm 2023.", "lookup"),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("not json, twice"))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_llm_plan_cases(cases, client=client, case_set_sha256="f" * 64)

    assert report.operation_accuracy == 0.0
    assert report.invalid_json_rate == 1.0
    assert len(report.failures) == 1
    assert report.failures[0].outcome == "invalid_json"


def test_evaluate_llm_plan_cases_reports_repair_success_rate() -> None:
    cases = (_plan_case("a", "Tra cứu doanh thu thuần của NVL năm 2023.", "lookup"),)
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(200, json=_envelope("bad json"))
        return httpx.Response(200, json=_envelope(_valid_lookup("NVL", "2023")))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_llm_plan_cases(cases, client=client, case_set_sha256="f" * 64)

    assert report.operation_accuracy == 1.0
    assert report.repair_success_rate == 1.0


def test_write_llm_plan_case_report_round_trips(tmp_path: Path) -> None:
    cases = (_plan_case("a", "Tra cứu doanh thu thuần của NVL năm 2023.", "lookup"),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_valid_lookup("NVL", "2023")))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_llm_plan_cases(cases, client=client, case_set_sha256="f" * 64)
    json_path, markdown_path = write_llm_plan_case_report(report, tmp_path)

    assert json_path.is_file()
    assert markdown_path.is_file()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["operation_accuracy"] == 1.0


# --- evaluate_router_abstain_cases ---------------------------------------


def _abstain_case(case_id: str, question: str, expected_abstain_code: PlanAbstainCode) -> PlanCase:
    return PlanCase(
        case_id=case_id,
        template_id="missing_company",
        question=question,
        expected_abstain_code=expected_abstain_code,
    )


def test_evaluate_router_abstain_cases_zero_false_plan_rate_when_llm_also_abstains() -> None:
    """The hard KPI (Day 16, still enforced by the router in Day 17):
    a question the rule planner correctly abstains on must never become a
    plan just because the LLM fallback ran."""
    cases = (_abstain_case("a", "Doanh thu thuần năm 2023 là bao nhiêu?", "entity_ambiguous"),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("not json"))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_router_abstain_cases(cases, client=client, case_set_sha256="f" * 64)

    assert report.false_plan_rate == 0.0
    assert report.abstain_recall == 1.0


def test_evaluate_router_abstain_cases_detects_a_false_plan() -> None:
    """If the LLM fallback fabricates a plan for a question that should
    abstain, `false_plan_rate` must catch it — this is the test that proves
    the metric actually detects a violation, mirroring Day 16's own test."""
    cases = (_abstain_case("a", "Doanh thu thuần năm 2023 là bao nhiêu?", "entity_ambiguous"),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(_valid_lookup("NVL", "2023")))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_router_abstain_cases(cases, client=client, case_set_sha256="f" * 64)

    assert report.false_plan_rate == 1.0
    assert len(report.failures) == 1


def test_write_router_abstain_report_round_trips(tmp_path: Path) -> None:
    cases = (_abstain_case("a", "Doanh thu thuần năm 2023 là bao nhiêu?", "entity_ambiguous"),)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope("not json"))

    client = LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=0)
    report = evaluate_router_abstain_cases(cases, client=client, case_set_sha256="f" * 64)
    json_path, markdown_path = write_router_abstain_report(report, tmp_path)

    assert json_path.is_file()
    assert markdown_path.is_file()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["false_plan_rate"] == 0.0
