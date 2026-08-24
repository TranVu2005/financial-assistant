"""Numeric guard (ADR 0009 decision F1).

Day 20 plan Sec 1.8 measured that every numeric token across all 70 gold70
questions is a year, so a whitelist of numbers is sufficient to separate
legitimate numbers from ones an LLM invented. Anything outside the
whitelist is a reject signal, not a warning -- the same deny-by-default
posture as the Day 19 sandbox.

The masked-PAL path builds its whitelist from the executed program via
`program_number_whitelist` (Task 7); the plan-era `build_number_whitelist`
was removed with the operation-enum answering path (spec 2026-08-24 §8.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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
