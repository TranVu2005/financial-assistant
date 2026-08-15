"""Day 20 numeric guard (ADR 0009 decision F1).

Gate for the *optional* LLM paraphrase path (`templates.py` is the default,
LLM-free path and needs none of this). Day 20 plan Sec 1.8 measured that
every numeric token across all 70 gold70 questions is a year, so a
whitelist of {locked answer, plan periods, plan.top_k, evidence values} is
sufficient to separate legitimate numbers from ones an LLM invented.
Anything outside the whitelist is a reject-and-fall-back-to-template
signal, not a warning -- the same deny-by-default posture as the Day 19
sandbox.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan

# Day 20 plan Sec 1.8: the naive pattern `\d[\d.,]*` captures trailing
# sentence punctuation ('2023.'); this pattern stops before a trailing `.`
# or `,` that is not followed by another digit.
_NUMBER_TOKEN_PATTERN = re.compile(r"-?\d[\d.,]*\d|-?\d")


def extract_number_tokens(text: str) -> tuple[str, ...]:
    """Tokenize numbers in `text`, normalized to strip trailing punctuation."""
    return tuple(match.group(0) for match in _NUMBER_TOKEN_PATTERN.finditer(text))


def _parse_number(token: str) -> Decimal | None:
    cleaned = token.replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def build_number_whitelist(plan: FinancialQueryPlan, compiled: CompiledQuery) -> frozenset[Decimal]:
    """Every number an LLM paraphrase is allowed to mention."""
    whitelist: set[Decimal] = set()
    if compiled.answer is not None:
        whitelist.add(compiled.answer)
    for period in plan.periods:
        whitelist.add(Decimal(period))
    if plan.top_k is not None:
        whitelist.add(Decimal(plan.top_k))
    for cell in compiled.evidence:
        whitelist.add(cell.value)
    return frozenset(whitelist)


@dataclass(frozen=True)
class NumericGuardResult:
    """Whether generated text stayed within the number whitelist."""

    allowed: bool
    disallowed_numbers: tuple[str, ...]


def guard_generated_text(text: str, *, whitelist: frozenset[Decimal]) -> NumericGuardResult:
    """Reject `text` if it mentions a number outside `whitelist`."""
    disallowed = []
    for token in extract_number_tokens(text):
        value = _parse_number(token)
        if value is None or value not in whitelist:
            disallowed.append(token)
    return NumericGuardResult(allowed=not disallowed, disallowed_numbers=tuple(disallowed))
