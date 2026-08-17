"""Day 17 LLM planner (§17.6): question -> `FinancialQueryPlan` | abstain.

Reuses `RulePlanResult` from `rule_planner.py` (exactly one of `plan` or
`abstain_codes`) instead of a parallel result type, so `plan_router.py` can
treat the rule planner and the LLM planner uniformly.

Never returns a plan that has not passed `validate_plan_semantics` — the same
invariant the rule planner already holds (ADR 0005 §Hệ quả), because a
malformed or hallucinated LLM output is exactly the case that invariant
exists to catch (ADR 0006).

ADR 0006 decision D: JSON-syntax errors, `LLMPlanOutput` schema errors, and
`validate_plan_semantics` issues all go through the same single repair
attempt — plan.md permits "một lần repair JSON tối đa", not one repair per
error type. After that one repair, any remaining failure becomes an abstain;
there is never a second round trip.
"""

from __future__ import annotations

import json

from financial_report_qa.core.errors import (
    LLMError,
    LLMRequestError,
    LLMResponseError,
)
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.llm_contracts import LLMPlanOutput, to_financial_query_plan
from financial_report_qa.planning.llm_prompt import (
    build_grounded_system_prompt,
    build_grounded_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.planning.plan_validator import validate_plan_semantics
from financial_report_qa.planning.rule_planner import PlanAbstainCode, RulePlanResult
from financial_report_qa.retrieval.contracts import TableId

_LLM_PLAN_JSON_SCHEMA: dict[str, object] = LLMPlanOutput.model_json_schema()


class _InvalidJsonError(ValueError):
    """Raised when the raw LLM content is not parseable JSON at all."""


def _client_failure_code(error: LLMError) -> PlanAbstainCode:
    """Map a client failure to the abstain code that names what went wrong.

    `llm_client` already separates these three; collapsing them under one
    `except LLMError` threw that away. A Day 26 smoke run reported 8 questions
    as `llm_unavailable` while the Ollama server log showed every request
    returning HTTP 200 -- the endpoint was up and answering, so "unavailable"
    sent the investigation at the server instead of at the envelope.
    """
    if isinstance(error, LLMResponseError):
        return "llm_bad_response"
    if isinstance(error, LLMRequestError):
        return "llm_request_rejected"
    return "llm_unavailable"


def _abstain(code: PlanAbstainCode, *, repaired: bool = False) -> RulePlanResult:
    return RulePlanResult(abstain_codes=(code,), repaired=repaired)


def _parse_and_validate(
    raw_content: str,
    *,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
) -> FinancialQueryPlan:
    """Raise on any failure; the caller decides whether it is repairable.

    `_InvalidJsonError` marks unparseable content specifically, so the final
    abstain code can distinguish "never produced JSON" (`llm_invalid_json`)
    from "produced JSON that does not form a valid plan" (`llm_plan_invalid`).
    """
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise _InvalidJsonError(f"invalid JSON: {exc}") from exc
    output = LLMPlanOutput.model_validate(data)
    plan = to_financial_query_plan(output, candidate_table_ids=candidate_table_ids)
    issues = validate_plan_semantics(plan, known_table_ids=known_table_ids)
    if issues:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        raise ValueError(f"semantic validation failed: {messages}")
    return plan


def _repair_user_prompt(base_user_prompt: str, previous_output: str, error: str) -> str:
    return (
        f"{base_user_prompt}\n\n"
        f"Kết quả trước đó không hợp lệ:\n{previous_output}\n\n"
        f"Lỗi: {error}\n\n"
        "Hãy sửa lại và CHỈ trả về đúng một JSON object hợp lệ."
    )


def _run_llm_plan(
    *,
    client: ChatCompletionClient,
    system_prompt: str,
    user_prompt: str,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
) -> RulePlanResult:
    """Shared core of `build_plan`/`build_plan_grounded`: ask the LLM for a
    plan; on a repairable failure, retry once with the same system prompt;
    else abstain. Never returns a plan that has not passed
    `validate_plan_semantics` (ADR 0005 §Hệ quả)."""
    try:
        raw_content = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=_LLM_PLAN_JSON_SCHEMA,
        )
    except LLMError as exc:
        return _abstain(_client_failure_code(exc))

    try:
        plan = _parse_and_validate(
            raw_content, candidate_table_ids=candidate_table_ids, known_table_ids=known_table_ids
        )
        return RulePlanResult(plan=plan)
    except ValueError as first_error:
        first_error_message = str(first_error)

    try:
        repaired_content = client.complete_json(
            system_prompt=system_prompt,
            user_prompt=_repair_user_prompt(user_prompt, raw_content, first_error_message),
            json_schema=_LLM_PLAN_JSON_SCHEMA,
        )
    except LLMError as exc:
        return _abstain(_client_failure_code(exc), repaired=True)

    try:
        plan = _parse_and_validate(
            repaired_content,
            candidate_table_ids=candidate_table_ids,
            known_table_ids=known_table_ids,
        )
        return RulePlanResult(plan=plan, repaired=True)
    except _InvalidJsonError:
        return _abstain("llm_invalid_json", repaired=True)
    except ValueError:
        return _abstain("llm_plan_invalid", repaired=True)


def build_plan(
    question: str,
    *,
    client: ChatCompletionClient,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
) -> RulePlanResult:
    """Ask the LLM for a plan from the vocabulary-free prompt (no candidate
    table content shown -- ADR 0006 decision B1)."""
    return _run_llm_plan(
        client=client,
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(question),
        candidate_table_ids=candidate_table_ids,
        known_table_ids=known_table_ids,
    )


def build_plan_grounded(
    question: str,
    *,
    client: ChatCompletionClient,
    candidate_table_ids: tuple[TableId, ...],
    known_table_ids: frozenset[str],
    table_context: str,
) -> RulePlanResult:
    """Day 23 last-resort fallback tier: same typed-`FinancialQueryPlan`
    contract as `build_plan` -- the LLM still only ever emits a plan, never
    raw Python, and the compiler/sandbox still independently verify the
    final number -- but the prompt shows the LLM real candidate-table
    content (`table_context`, from `table_context_rendering.render_table_context`)
    so `raw_text` selectors can be copied verbatim instead of invented (Day
    22 measured 23.4% of vocabulary-free LLM plans did exactly that on a
    weak model). Called only after both the rule planner and `build_plan`
    have already abstained."""
    return _run_llm_plan(
        client=client,
        system_prompt=build_grounded_system_prompt(),
        user_prompt=build_grounded_user_prompt(question, table_context),
        candidate_table_ids=candidate_table_ids,
        known_table_ids=known_table_ids,
    )
