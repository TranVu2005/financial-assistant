"""Verify A: the second, unmasked pass may describe -- never decide.

The program-generation pass is blind to values, so its own text cannot be
compared against anything. This pass runs after binding and execution, sees
the real numbers, and is checked with the same whitelist posture
`numeric_guard` already applies to the paraphrase path: any number outside
{answer, bound values, periods} is a fabrication, not a warning.

Nothing here can change the answer. It can only report a divergence, which
the pipeline turns into at most one regeneration (N6).
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.program_contracts import ExecutedProgram
from financial_report_qa.verification.numeric_guard import (
    NumericGuardResult,
    guard_generated_text,
)


def program_number_whitelist(executed: ExecutedProgram) -> frozenset[Decimal]:
    """Every number an explanation of `executed` is allowed to mention."""
    whitelist: set[Decimal] = {executed.answer}
    for bound in executed.bindings:
        whitelist.add(bound.value)
        if bound.period is not None:
            whitelist.add(Decimal(bound.period))
    return frozenset(whitelist)


def check_explanation(explanation: str, executed: ExecutedProgram) -> NumericGuardResult:
    """Reject an explanation that mentions a number not grounded in the run."""
    return guard_generated_text(explanation, whitelist=program_number_whitelist(executed))
