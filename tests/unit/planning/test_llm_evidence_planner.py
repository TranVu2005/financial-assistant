"""Tests for the plan.md §12 Evidence-Aware Planner's model call.

Drives a real `LLMClient` over `httpx.MockTransport` (the Day 17 pattern) so
prompt construction, schema and parsing are exercised end to end without a
socket.
"""

from __future__ import annotations

import json
from decimal import Decimal

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.llm_client import LLMClient
from financial_report_qa.planning.llm_evidence_planner import choose_evidence_plan

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:11434/v1",
    model="qwen2.5:7b",
    timeout_seconds=5.0,
    max_output_tokens=64,
    temperature=0.0,
    context_length=32768,
    json_schema_constrained=True,
)
_QUESTION = "Doanh thu 2023 tăng bao nhiêu % so với 2022?"
TABLE_ID = "tbl_" + "1" * 64


def _fact(fact_id: str, period: int, value: str) -> GroundedFact:
    return GroundedFact(
        fact_id=fact_id,
        table_id=TABLE_ID,
        row_index=14,
        row_label="Doanh thu thuần",
        column=f"Năm {period}",
        company_code="VNM",
        period=period,
        raw_value=Decimal(value),
        unit="VND_million",
        grounding_score=1.0,
    )


_FACTS = (_fact("F1", 2023, "63075"), _fact("F2", 2022, "60180"))


def _client(handler: object) -> LLMClient:
    return LLMClient(_SETTINGS, transport=httpx.MockTransport(handler), max_retries=1)  # type: ignore[arg-type]


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _responder(payload: object) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return httpx.Response(200, json=_envelope(content))

    return handler


def test_returns_the_operation_and_operands_the_model_chose() -> None:
    plan = choose_evidence_plan(
        _QUESTION,
        _FACTS,
        client=_client(_responder({"operation": "growth_rate", "operands": ["F1", "F2"]})),
    )
    assert plan is not None
    assert plan.operation == "growth_rate"
    assert plan.operands == ("F1", "F2")


def test_prompt_shows_each_fact_with_its_value_period_and_unit() -> None:
    """§12: the planner's job is only to pick an operation over facts it can
    already read -- so the facts have to actually be in the prompt."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=_envelope(json.dumps({"operation": "lookup", "operands": ["F1"]}))
        )

    choose_evidence_plan(_QUESTION, _FACTS, client=_client(handler))
    body = captured["body"]
    assert isinstance(body, dict)
    prompt = "\n".join(str(message["content"]) for message in body["messages"])  # type: ignore[index,union-attr]
    assert _QUESTION in prompt
    for token in ("F1", "F2", "Doanh thu thuần", "2023", "2022", "63075", "VND_million"):
        assert token in prompt, token


def test_prompt_never_asks_the_model_for_a_table_row_or_column() -> None:
    """The §12 redesign is defined by what leaves the prompt: the planner is
    no longer asked to locate anything."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json=_envelope(json.dumps({"operation": "lookup", "operands": ["F1"]}))
        )

    choose_evidence_plan(_QUESTION, _FACTS, client=_client(handler))
    body = captured["body"]
    assert isinstance(body, dict)
    schema = json.dumps(body.get("response_format", {}), ensure_ascii=False)
    for forbidden in ("table_id", "candidate_table_ids", "row_label", "column_label", "metric"):
        assert forbidden not in schema, forbidden


def test_only_allowed_operations_are_offered_and_accepted() -> None:
    plan = choose_evidence_plan(
        _QUESTION,
        _FACTS,
        client=_client(_responder({"operation": "growth_rate", "operands": ["F1", "F2"]})),
        allowed_operations=("lookup",),
    )
    assert plan is None


def test_an_invented_fact_id_is_refused() -> None:
    plan = choose_evidence_plan(
        _QUESTION,
        _FACTS,
        client=_client(_responder({"operation": "growth_rate", "operands": ["F1", "F7"]})),
    )
    assert plan is None


def test_lowercase_fact_ids_are_accepted() -> None:
    """A small model writes `f1` often enough that refusing it would throw
    away correct plans over a formatting slip."""
    plan = choose_evidence_plan(
        _QUESTION,
        _FACTS,
        client=_client(_responder({"operation": "growth_rate", "operands": ["f1", "f2"]})),
    )
    assert plan is not None and plan.operands == ("F1", "F2")


def test_unparseable_reply_abstains_instead_of_raising() -> None:
    plan = choose_evidence_plan(_QUESTION, _FACTS, client=_client(_responder("tôi không chắc")))
    assert plan is None


def test_duplicate_operands_abstain_rather_than_produce_a_self_comparison() -> None:
    plan = choose_evidence_plan(
        _QUESTION,
        _FACTS,
        client=_client(_responder({"operation": "growth_rate", "operands": ["F1", "F1"]})),
    )
    assert plan is None


def test_unreachable_model_abstains() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert choose_evidence_plan(_QUESTION, _FACTS, client=_client(handler)) is None


def test_no_facts_means_no_model_call_at_all() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("the planner must not be called without facts")

    assert choose_evidence_plan(_QUESTION, (), client=_client(handler)) is None
