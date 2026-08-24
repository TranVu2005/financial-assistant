"""Verification checks over an executed masked-PAL program (ADR 0009 D1).

Four pure functions, each returning `VerificationIssue | None`, all blocking.
The plan-era pair `check_period_inferred_warning`/`check_scope_inferred` and
their plan-era inputs were removed with the operation-enum answering
path (spec 2026-08-24 §8.2): the executed program binds real cells directly,
so there is no inferred period or statement scope left to warn about.
"""

from __future__ import annotations

import re
from decimal import Decimal

from financial_report_qa.execution.masked_program import apply_scale, run_program
from financial_report_qa.execution.program_contracts import ExecutedProgram
from financial_report_qa.verification.contracts import VerificationIssue

# ADR 0009 decision B1: `ratio` is the computed unit, `percent` is its
# presentation form -- either declaration is presentable for the other. The
# masked program carries the equivalent idea in `scale`.
_KNOWN_SCALES: frozenset[str] = frozenset({"none", "percent", "thousand", "million", "billion"})

# Matches a comma-grouped thousands number in full (e.g. "84,420,878"), or
# falls back to a plain digit run with an optional decimal fraction. A
# single-group pattern (`\d+(?:[.,]\d+)?`) stops after the first comma and
# was found, via a real end-to-end run on gold70 (Day 20 task 20.10), to
# wrongly reject 17/30 answered results by parsing "84,420,878 VND" as 84,420.
_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def check_recompute_mismatch(executed: ExecutedProgram) -> VerificationIssue | None:
    """Independently re-derive the answer from the bound cell values through
    the same deterministic evaluator (`run_program` + `apply_scale`) and
    compare exactly (ADR 0009 decision D1) -- catches any drift between what
    the pipeline computed and what the shipped package claims."""
    try:
        raw = run_program(executed.program, [bound.value for bound in executed.bindings])
        recomputed = apply_scale(raw, executed.scale)
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return VerificationIssue(
            code="recompute_mismatch",
            message=f"recompute raised {type(exc).__name__}: {exc}",
        )
    if recomputed != executed.answer:
        return VerificationIssue(
            code="recompute_mismatch",
            message=(
                f"recompute {recomputed!r} does not match executed answer {executed.answer!r}"
            ),
        )
    return None


def check_scale_not_presentable(executed: ExecutedProgram) -> VerificationIssue | None:
    """The declared `scale` must be a recognized presentation form before it
    may be applied to the raw result (ADR 0009 decision B1's presentability
    rule, ported from the plan-era expected-unit check: the declared display
    form must actually be derivable)."""
    if executed.scale not in _KNOWN_SCALES:
        return VerificationIssue(
            code="unit_not_presentable",
            message=f"declared scale '{executed.scale}' is not a recognized scale",
        )
    return None


def check_evidence_outside_retrieval(
    executed: ExecutedProgram, retrieved_table_ids: frozenset[str]
) -> VerificationIssue | None:
    """Day 20 plan Sec 1.7: this invariant held on gold70 by construction but
    had no assertion guarding it. Explicit here so a future caller that
    passes an unscoped table set is caught, not trusted."""
    outside = sorted(set(executed.table_ids) - retrieved_table_ids)
    if outside:
        return VerificationIssue(
            code="evidence_outside_retrieval",
            message=f"evidence cites table(s) outside the retrieved set: {', '.join(outside)}",
        )
    return None


def check_display_roundtrip_mismatch(
    answer: Decimal, display: str, *, display_precision: int
) -> VerificationIssue | None:
    """Parse the leading number out of `display` and compare to `answer`
    quantized to `display_precision`, within half a unit of that precision
    (ADR 0009 decision D1: tolerance is the declared display precision
    itself, not an arbitrary epsilon)."""
    match = _NUMBER_PATTERN.search(display)
    if match is None:
        return VerificationIssue(
            code="display_roundtrip_mismatch",
            message=f"display string has no parseable number: {display!r}",
        )
    parsed = Decimal(match.group(0).replace(",", ""))
    step = Decimal(1).scaleb(-display_precision)
    expected = answer.quantize(step)
    tolerance = step / 2
    if abs(parsed - expected) > tolerance:
        return VerificationIssue(
            code="display_roundtrip_mismatch",
            message=(
                f"display {parsed!r} does not round-trip to answer {answer!r} "
                f"within precision {display_precision}"
            ),
        )
    return None
