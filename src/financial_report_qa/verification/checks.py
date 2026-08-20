"""Day 20 verification checks (ADR 0009 decisions B1/D1; plan Sec 3 task 20.5).

Five pure functions, each returning `VerificationIssue | None`. Four are
blocking (Sec 2.D1: verifier tolerance is *exact* for recompute, *declared
precision* for display); `check_period_inferred_warning` is not (Day 20 plan
Sec 1.5: 6/30 gold70 answers rely on an inferred period -- treating that as
blocking would reject 20% of otherwise-correct answers).
"""

from __future__ import annotations

import re
from decimal import Decimal

from financial_report_qa.execution import operations
from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.verification.contracts import VerificationIssue

# ADR 0009 decision B1: `ratio` is the computed unit, `percent` is its
# presentation form -- either declaration is presentable for the other.
_UNIT_PRESENTATION_EQUIVALENTS: dict[str, frozenset[str]] = {
    "ratio": frozenset({"ratio", "percent"}),
    "percent": frozenset({"percent", "ratio"}),
}

# Matches a comma-grouped thousands number in full (e.g. "84,420,878"), or
# falls back to a plain digit run with an optional decimal fraction. A
# single-group pattern (`\d+(?:[.,]\d+)?`) stops after the first comma and
# was found, via a real end-to-end run on gold70 (Day 20 task 20.10), to
# wrongly reject 17/30 answered results by parsing "84,420,878 VND" as 84,420.
_NUMBER_PATTERN = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _presentable_units(unit: str) -> frozenset[str]:
    return _UNIT_PRESENTATION_EQUIVALENTS.get(unit, frozenset({unit}))


def check_recompute_mismatch(
    plan: FinancialQueryPlan, compiled: CompiledQuery
) -> VerificationIssue | None:
    """Independently re-derive the answer from `compiled.evidence` through
    `operations.py` and compare exactly (ADR 0009 decision D1) -- catches a
    builder bug that reorders or mis-selects evidence even when the
    `pandas_query` replay (Day 18/19) already matched."""
    if compiled.status != "answered":
        return None
    evidence = compiled.evidence
    operation = compiled.operation
    try:
        if operation == "lookup":
            answer, unit = operations.compile_lookup(evidence[0])
        elif operation == "difference":
            answer, unit = operations.compile_difference(evidence[1], evidence[0])
        elif operation == "growth_rate":
            answer, unit = operations.compile_growth_rate(evidence[1], evidence[0])
        elif operation == "compare":
            answer, unit = operations.compile_compare(evidence[0], evidence[1])
        elif operation == "compare_companies":
            answer, unit = operations.compile_compare_companies(evidence[0], evidence[1])
        elif operation == "ratio":
            answer, unit = operations.compile_ratio(evidence[0], evidence[1])
        elif operation == "average":
            answer, unit = operations.compile_average(evidence)
        elif operation == "sum":
            answer, unit = operations.compile_sum(evidence)
        elif operation == "rank":
            if plan.top_k is None:
                return VerificationIssue(
                    code="recompute_mismatch", message="rank plan is missing top_k"
                )
            answer, unit = operations.compile_rank(evidence, top_k=plan.top_k)
        else:
            return VerificationIssue(
                code="recompute_mismatch",
                message=f"no recompute rule for operation '{operation}'",
            )
    except (ValueError, ZeroDivisionError, IndexError) as exc:
        return VerificationIssue(
            code="recompute_mismatch",
            message=f"recompute raised {type(exc).__name__}: {exc}",
        )

    # `compile_plan` presents the answer in `plan.expected_unit` when that
    # differs from the evidence unit (compiler.py, "expected_unit" branch),
    # but `operations.compile_*` returns the value in the *evidence* unit.
    # Comparing the two directly makes every unit-converted answer a
    # mismatch: measured on the plan.md §19 dev benchmark, that rejected
    # 21/144 questions whose answers were correct. Mirror the compiler's
    # conversion here, then compare.
    compiled_unit = compiled.unit
    if compiled_unit is not None and unit != compiled_unit:
        try:
            answer = operations.convert_cell_value(answer, unit, compiled_unit)
        except ValueError as exc:
            return VerificationIssue(
                code="recompute_mismatch",
                message=(
                    f"recompute {answer!r} {unit!r} cannot be expressed in the "
                    f"compiled unit {compiled_unit!r}: {exc}"
                ),
            )
        unit = compiled_unit

    if answer != compiled.answer or unit != compiled.unit:
        return VerificationIssue(
            code="recompute_mismatch",
            message=(
                f"recompute {answer!r} {unit!r} does not match compiled "
                f"answer {compiled.answer!r} {compiled.unit!r}"
            ),
        )
    return None


def check_unit_not_presentable(
    plan: FinancialQueryPlan, compiled: CompiledQuery
) -> VerificationIssue | None:
    """A declared `expected_unit` must be presentable from the computed unit
    (ADR 0009 decision B1's equivalence table), not identical to it."""
    if compiled.status != "answered" or plan.expected_unit is None:
        return None
    assert compiled.unit is not None
    if plan.expected_unit not in _presentable_units(compiled.unit):
        return VerificationIssue(
            code="unit_not_presentable",
            message=(
                f"expected_unit '{plan.expected_unit}' is not presentable "
                f"from computed unit '{compiled.unit}'"
            ),
        )
    return None


def check_evidence_outside_retrieval(
    compiled: CompiledQuery, retrieved_table_ids: frozenset[str]
) -> VerificationIssue | None:
    """Day 20 plan Sec 1.7: this invariant held on gold70 by construction but
    had no assertion guarding it. Explicit here so a future caller that
    passes an unscoped table set is caught, not trusted."""
    if compiled.status != "answered":
        return None
    outside = sorted({cell.table_id for cell in compiled.evidence} - retrieved_table_ids)
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


def check_period_inferred_warning(compiled: CompiledQuery) -> VerificationIssue | None:
    """Non-blocking (Day 20 plan Sec 1.5): surfaces reliance on ADR 0007
    decision C2's n=10-cell period-inference rule instead of hiding it."""
    if compiled.status != "answered":
        return None
    if any(cell.period_inferred for cell in compiled.evidence):
        return VerificationIssue(
            code="period_inferred_warning",
            message="one or more evidence cells rely on an inferred period",
        )
    return None


def check_scope_inferred(compiled: CompiledQuery) -> VerificationIssue | None:
    """Blocking (Day 21 plan §1.5/ADR 0010 decision B1), unlike
    `check_period_inferred_warning`: `CompiledQuery.scope_inferred` is True
    only when the plan left `statement_scope` unset and
    `ExecutionSettings.default_statement_scope` resolved the candidate frame
    instead -- 92.8% of two-scope groups disagree in VALUE, so presenting
    such an answer as certain would be wrong far more often than a
    one-year-off inferred period."""
    if compiled.status != "answered":
        return None
    if compiled.scope_inferred:
        return VerificationIssue(
            code="scope_inferred",
            message="statement_scope was inferred from a default, not stated in the plan",
        )
    return None
