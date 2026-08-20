"""Unit tests for the Day 17 OpenAI-compatible LLM client (§17.3).

Every test drives an `httpx.MockTransport` — no test in this module is
permitted to reach a real socket, matching the Day 17 plan's requirement that
the whole suite runs with no llama.cpp server available.
"""

from __future__ import annotations

import json

import httpx
import pytest

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.core.errors import LLMRequestError, LLMResponseError, LLMUnavailableError
from financial_report_qa.planning.llm_client import LLMClient

_SETTINGS = LLMSettings(
    base_url="http://127.0.0.1:8080/v1",
    model="qwen3-4b-instruct-2507-q4_k_m",
    timeout_seconds=5.0,
    max_output_tokens=160,
    temperature=0.0,
    context_length=4096,
    json_schema_constrained=True,
)


def _envelope(content: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _client(handler: object, *, max_retries: int = 2) -> LLMClient:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LLMClient(_SETTINGS, transport=transport, max_retries=max_retries)


def test_successful_completion_returns_content() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_envelope('{"operation":"lookup"}'))

    client = _client(handler)
    content = client.complete_json(
        system_prompt="system", user_prompt="user", json_schema={"type": "object"}
    )

    assert content == '{"operation":"lookup"}'
    assert len(captured) == 1
    payload = captured[0]
    assert payload["model"] == "qwen3-4b-instruct-2507-q4_k_m"
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 160
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    response_format = payload["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"


def test_json_schema_constrained_false_omits_response_format() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_envelope("{}"))

    settings = _SETTINGS.model_copy(update={"json_schema_constrained": False})
    client = LLMClient(settings, transport=httpx.MockTransport(handler), max_retries=0)
    client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})

    assert "response_format" not in captured[0]


def test_no_retry_on_4xx() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    client = _client(handler)
    with pytest.raises(LLMRequestError):
        client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})

    assert calls["count"] == 1


def test_retries_on_5xx_then_succeeds() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_envelope("ok"))

    client = _client(handler)
    content = client.complete_json(
        system_prompt="s", user_prompt="u", json_schema={"type": "object"}
    )

    assert content == "ok"
    assert calls["count"] == 2


def test_retries_exhausted_on_repeated_5xx_raises_unavailable() -> None:
    """A server erroring on both the constrained and the unconstrained form
    is genuinely unusable: 3 constrained attempts, then 3 more without the
    schema (see `test_constrained_5xx_falls_back_to_an_unconstrained_request`),
    then `LLMUnavailableError` -- `LLMServerError` is a subclass, so callers
    catching the base class are unaffected."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="unavailable")

    client = _client(handler, max_retries=2)
    with pytest.raises(LLMUnavailableError):
        client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})

    assert calls["count"] == 6


def test_connection_failure_retries_then_raises_unavailable() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectTimeout("no route to host", request=request)

    client = _client(handler, max_retries=1)
    with pytest.raises(LLMUnavailableError):
        client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})

    assert calls["count"] == 2


def test_malformed_response_envelope_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _client(handler)
    with pytest.raises(LLMResponseError):
        client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})


def test_constrained_5xx_falls_back_to_an_unconstrained_request() -> None:
    """llama.cpp/Ollama answer HTTP 500 when the constrained decoder cannot
    satisfy the JSON-schema grammar ("The model produced output that does not
    match the expected peg-native format"). The endpoint is healthy, so the
    only thing left to try is the same prompt without the grammar -- the
    reply is still validated against the schema by `llm_planner`. Measured on
    the plan.md §19 dev benchmark: 5/144 questions died as `llm_unavailable`
    on exactly this 500."""
    seen: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        constrained = "response_format" in json.loads(request.content)
        seen.append(constrained)
        if constrained:
            return httpx.Response(500, text="peg-native format error")
        return httpx.Response(200, json=_envelope('{"operation":"lookup"}'))

    client = _client(handler, max_retries=1)
    content = client.complete_json(
        system_prompt="s", user_prompt="u", json_schema={"type": "object"}
    )

    assert content == '{"operation":"lookup"}'
    assert seen == [True, True, False]


def test_constrained_fallback_not_attempted_when_endpoint_unreachable() -> None:
    """A transport failure is not a grammar problem: dropping the schema
    cannot help, so the offline path must not double its attempts."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectTimeout("no route to host", request=request)

    client = _client(handler, max_retries=1)
    with pytest.raises(LLMUnavailableError):
        client.complete_json(system_prompt="s", user_prompt="u", json_schema={"type": "object"})

    assert calls["count"] == 2
