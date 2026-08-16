"""Unit tests for the Day 17 LLM planner prompt builder (§17.4).

Token-budget assertions use a conservative chars/token proxy (2.0 chars per
token) instead of a real tokenizer, so this suite has no model-download
dependency and runs the same with or without a network. Day 17 plan §1.4
measured the *actual* `bge-m3` ratio for this kind of content at 2.16-4.79
chars/token, so 2.0 chars/token is a deliberately pessimistic (safe) proxy —
if we pass at 2.0 chars/token we would also pass with a real tokenizer.
"""

from __future__ import annotations

import json

from financial_report_qa.planning.llm_prompt import (
    _FEW_SHOTS,
    build_grounded_system_prompt,
    build_grounded_user_prompt,
    build_system_prompt,
    build_user_prompt,
)
from financial_report_qa.planning.plan_contracts import PlanOperation

# Half of `configs/local_rtx3050.yaml`'s `context_length: 4096`, leaving room
# for the user message, few-shot answers already counted, and the completion
# itself (`max_output_tokens: 160`).
_SYSTEM_PROMPT_TOKEN_BUDGET = 2048
_CONSERVATIVE_CHARS_PER_TOKEN = 2.0

_ALL_OPERATIONS: tuple[PlanOperation, ...] = (
    "lookup",
    "compare",
    "compare_companies",
    "difference",
    "growth_rate",
    "ratio",
    "average",
    "sum",
    "rank",
)


def test_system_prompt_is_deterministic() -> None:
    assert build_system_prompt() == build_system_prompt()


def test_system_prompt_covers_every_plan_operation() -> None:
    prompt = build_system_prompt()
    for operation in _ALL_OPERATIONS:
        assert f'"{operation}"' in prompt, f"missing few-shot or guide entry for {operation}"


def test_system_prompt_does_not_dump_the_full_json_schema() -> None:
    """§1.4: `model_json_schema()` alone is 963 tokens (23.5% of the 4096
    window) — the prompt must use a hand-written compact description instead."""
    prompt = build_system_prompt()
    assert "$defs" not in prompt
    assert "additionalProperties" not in prompt
    assert "properties" not in prompt


def test_few_shots_never_include_candidate_table_ids() -> None:
    """ADR 0006 decision A/§1.5: the LLM never sees or produces table ids."""
    for example in _FEW_SHOTS:
        assert "candidate_table_ids" not in example.plan


def test_system_prompt_stays_within_conservative_token_budget() -> None:
    prompt = build_system_prompt()
    estimated_tokens = len(prompt) / _CONSERVATIVE_CHARS_PER_TOKEN
    assert estimated_tokens < _SYSTEM_PROMPT_TOKEN_BUDGET, (
        f"system prompt ~{estimated_tokens:.0f} estimated tokens exceeds budget "
        f"{_SYSTEM_PROMPT_TOKEN_BUDGET}"
    )


def test_few_shot_plans_are_valid_json_and_shape() -> None:
    for example in _FEW_SHOTS:
        rendered = json.dumps(example.plan, ensure_ascii=False)
        reloaded = json.loads(rendered)
        assert reloaded["operation"] in _ALL_OPERATIONS


def test_user_prompt_embeds_the_question_verbatim() -> None:
    question = "Tra cứu doanh thu thuần của NVL năm 2023."
    prompt = build_user_prompt(question)
    assert question in prompt


def test_grounded_system_prompt_covers_every_plan_operation() -> None:
    """Day 23 last-resort tier: same operation contract as the vocabulary-free
    prompt -- the LLM still only ever emits a typed FinancialQueryPlan."""
    prompt = build_grounded_system_prompt()
    for operation in _ALL_OPERATIONS:
        assert f'"{operation}"' in prompt


def test_grounded_system_prompt_instructs_copying_row_labels_verbatim() -> None:
    """Day 22 measured 23.4% of vocabulary-free LLM plans invented a metric
    name that did not exist in any candidate table -- the grounded prompt
    must explicitly instruct copying real row labels, not paraphrasing."""
    prompt = build_grounded_system_prompt()
    assert "nguyên văn" in prompt.lower()


def test_grounded_user_prompt_embeds_question_and_table_context() -> None:
    question = "Lãi tiền gửi của VJC năm 2018 là bao nhiêu?"
    table_context = "--- Bảng tbl_x (VJC/2018/report.txt) ---\nLãi tiền gửi | 100"
    prompt = build_grounded_user_prompt(question, table_context)
    assert question in prompt
    assert table_context in prompt
