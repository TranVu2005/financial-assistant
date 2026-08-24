"""Day 17 OpenAI-compatible chat-completion client (§17.3).

Deliberately thin: this module only knows how to POST one
`/chat/completions` request with bounded retry and turn the response into a
content string. It has no idea what any downstream artifact looks like —
parsing, repair, and abstain decisions belong to the callers.

Retry policy: connection failures, timeouts, and 5xx responses are retried up
to `max_retries` additional times (temperature 0 makes a retried call
deterministic in intent, if not byte-for-byte reproducible — see ADR 0006).
4xx responses are never retried: the server has already told us the request
itself is invalid, and retrying identical bytes cannot change that.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from financial_report_qa.core.config import LLMSettings
from financial_report_qa.core.errors import (
    LLMRequestError,
    LLMResponseError,
    LLMServerError,
    LLMUnavailableError,
)

_CHAT_COMPLETIONS_PATH = "/chat/completions"


class ChatCompletionClient(Protocol):
    """Structural interface `llm_planner.py` depends on.

    Lets `llm_evaluation.py` substitute a `ReplayCacheClient` (§17.8) for
    `LLMClient` without a shared base class — both just need this one method.
    """

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, object]
    ) -> str: ...


class LLMClient:
    """A bounded-retry OpenAI-compatible chat-completion client."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 2,
    ) -> None:
        self._settings = settings
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=settings.base_url, timeout=settings.timeout_seconds, transport=transport
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, json_schema: dict[str, object]
    ) -> str:
        """Send one chat-completion request and return the raw content string."""
        payload: dict[str, object] = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._settings.json_schema_constrained:
            constrained = dict(payload)
            constrained["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "financial_query_plan",
                    "schema": json_schema,
                    "strict": True,
                },
            }
            try:
                return _extract_content(self._post_with_retry(constrained))
            except LLMServerError:
                # llama.cpp/Ollama answer HTTP 500 when their constrained
                # decoder cannot satisfy the grammar ("The model produced
                # output that does not match the expected peg-native format").
                # The endpoint is healthy and the prompt is fine, so the one
                # thing left worth trying is the same request without the
                # grammar: `llm_planner` still validates the reply against the
                # same schema, so an unconstrained answer is never trusted
                # more than a constrained one -- it is only given the chance
                # to be parsed at all. Measured on the plan.md §19 dev
                # benchmark: 5/144 questions were lost to this 500 alone.
                pass
        response = self._post_with_retry(payload)
        return _extract_content(response)

    def _post_with_retry(self, payload: dict[str, object]) -> httpx.Response:
        attempts = self._max_retries + 1
        last_error: str | None = None
        server_answered = False
        for _ in range(attempts):
            try:
                response = self._client.post(_CHAT_COMPLETIONS_PATH, json=payload)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code >= 500:
                server_answered = True
                last_error = f"HTTP {response.status_code}: {response.text}"
                continue
            if response.status_code >= 400:
                raise LLMRequestError(f"llm endpoint rejected request: HTTP {response.status_code}")
            return response
        if server_answered:
            raise LLMServerError(f"llm endpoint failed after {attempts} attempt(s): {last_error}")
        raise LLMUnavailableError(
            f"llm endpoint unreachable after {attempts} attempt(s): {last_error}"
        )


def _extract_content(response: httpx.Response) -> str:
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMResponseError(f"malformed chat-completion response envelope: {exc}") from exc
    if not isinstance(content, str):
        raise LLMResponseError("chat-completion response content is not a string")
    return content
