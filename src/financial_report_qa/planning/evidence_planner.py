"""plan.md §12: `{operation, operands}` over facts -> executable plan.

This is the half of the Evidence-Aware Planner that does not involve a model.
It takes the operation and fact ids the planner chose and turns them into an
ordinary `FinancialQueryPlan` whose selectors are already position-bound
(§9/§14), or refuses with a typed reason.

Two properties are worth stating outright, because they are why this design is
safe to bolt onto the existing stack:

1. **Nothing here reads the question.** Every decision is a consistency check
   among the chosen facts -- same issuer, right number of operands, the row and
   period relationships the operation actually means. A planner that points at
   an incoherent set of facts is rejected, never repaired by guessing.
2. **The output is an ordinary plan.** `plan_validator`, `compile_plan`, the
   sandbox replay, verification and the evidence CSV all keep working
   unchanged; §12 replaces how a plan is *decided*, not how it is proved.

`compare_companies` and `rank` are deliberately out of scope: a single
`MetricSelector` serves every company in those operations, so it cannot be
pinned to one physical row (§14). They are rejected as `operation_unsupported`
so the caller stays on the existing planner instead of receiving a plan whose
grounding regime is silently different from the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import model_validator

from financial_report_qa.planning.evidence_plan_contracts import EvidencePlan
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import (
    ExpectedUnit,
    FinancialQueryPlan,
    MetricSelector,
)
from financial_report_qa.retrieval.contracts import _FrozenModel

EvidencePlanRejectCode = Literal[
    # The planner named a fact id it was never shown.
    "operand_unknown",
    # Wrong number of operands for the operation.
    "operand_arity_invalid",
    # The operation cannot be expressed as a position-bound plan (§14).
    "operation_unsupported",
    # The chosen facts belong to different issuers.
    "operands_company_mismatch",
    # The operation needs one row over time, but the facts are different rows
    # (or needs two rows, and the facts are the same one).
    "operands_row_mismatch",
    # The operation needs distinct periods and the facts share one, or needs
    # one period and the facts span several.
    "operands_period_mismatch",
]

# `average`/`sum` are variadic over periods; everything else is fixed-arity.
_OPERAND_COUNT: dict[str, int] = {
    "lookup": 1,
    "difference": 2,
    "growth_rate": 2,
    "ratio": 2,
    "compare": 2,
}
_AGGREGATES = frozenset({"average", "sum"})
_UNSUPPORTED = frozenset({"compare_companies", "rank"})


class EvidencePlanBuild(_FrozenModel):
    """Either an executable plan, or the typed reason there is none."""

    plan: FinancialQueryPlan | None = None
    reject_code: EvidencePlanRejectCode | None = None

    @model_validator(mode="after")
    def validate_exactly_one_outcome(self) -> Self:
        if (self.plan is None) == (self.reject_code is None):
            raise ValueError("an evidence plan build is either a plan or a reject code")
        return self


def _reject(code: EvidencePlanRejectCode) -> EvidencePlanBuild:
    return EvidencePlanBuild(reject_code=code)


def _selector(facts: Sequence[GroundedFact]) -> MetricSelector:
    """One position-bound selector covering `facts`, which all share a row.

    The column predicate survives only when every fact agrees on it. A
    two-period selector normally spans "Năm 2023" and "Năm 2022", and keeping
    either one would make the other cell unfindable -- period is what
    separates them, not the header.
    """
    first = facts[0]
    columns = {fact.column for fact in facts}
    column = columns.pop() if len(columns) == 1 else None
    return MetricSelector(
        raw_text=first.row_label,
        column_text=column,
        table_id=first.table_id,
        row_index=first.row_index,
    )


def _same_row(facts: Sequence[GroundedFact]) -> bool:
    return len({(fact.table_id, fact.row_index) for fact in facts}) == 1


def _periods(facts: Sequence[GroundedFact]) -> tuple[str, ...]:
    return tuple(str(period) for period in sorted({fact.period for fact in facts}))


def build_plan_from_facts(
    evidence_plan: EvidencePlan,
    facts: Sequence[GroundedFact],
    *,
    expected_unit: ExpectedUnit | None = None,
) -> EvidencePlanBuild:
    """Resolve an `EvidencePlan` against the fact menu it was chosen from."""
    operation = evidence_plan.operation
    if operation in _UNSUPPORTED:
        return _reject("operation_unsupported")

    by_id = {fact.fact_id: fact for fact in facts}
    if any(operand not in by_id for operand in evidence_plan.operands):
        return _reject("operand_unknown")
    operands = [by_id[operand] for operand in evidence_plan.operands]

    if operation in _AGGREGATES:
        if len(operands) < 2:
            return _reject("operand_arity_invalid")
    elif len(operands) != _OPERAND_COUNT[operation]:
        return _reject("operand_arity_invalid")

    if len({fact.company_code for fact in operands}) != 1:
        return _reject("operands_company_mismatch")
    company = operands[0].company_code
    if company is None:
        return _reject("operands_company_mismatch")

    periods = _periods(operands)
    table_ids = tuple(dict.fromkeys(fact.table_id for fact in operands))
    common = {
        "companies": (company,),
        "candidate_table_ids": table_ids,
        "expected_unit": expected_unit,
    }

    if operation == "lookup":
        return EvidencePlanBuild(
            plan=FinancialQueryPlan.model_validate(
                {
                    **common,
                    "operation": operation,
                    "periods": periods,
                    "metric": _selector(operands),
                }
            )
        )

    if operation in ("difference", "growth_rate"):
        # One line item against itself over time. Two different rows is a
        # different question the operation cannot express.
        if not _same_row(operands):
            return _reject("operands_row_mismatch")
        if len(periods) != 2:
            return _reject("operands_period_mismatch")
        return EvidencePlanBuild(
            plan=FinancialQueryPlan.model_validate(
                {
                    **common,
                    "operation": operation,
                    "periods": periods,
                    "metric": _selector(operands),
                }
            )
        )

    if operation in ("ratio", "compare"):
        if len(periods) != 1:
            return _reject("operands_period_mismatch")
        if _same_row(operands):
            return _reject("operands_row_mismatch")
        numerator, denominator = operands
        pair = (
            {
                "numerator_metric": _selector([numerator]),
                "denominator_metric": _selector([denominator]),
            }
            if operation == "ratio"
            else {"metric_a": _selector([numerator]), "metric_b": _selector([denominator])}
        )
        return EvidencePlanBuild(
            plan=FinancialQueryPlan.model_validate(
                {**common, "operation": operation, "periods": periods, **pair}
            )
        )

    # `average`/`sum`: one row, several periods. The company dimension is the
    # other arity `_validate_aggregate` allows, but it needs several companies
    # and therefore an unbindable selector -- out of scope here for the same
    # reason `compare_companies` is.
    if not _same_row(operands):
        return _reject("operands_row_mismatch")
    if len(periods) != len(operands):
        return _reject("operands_period_mismatch")
    return EvidencePlanBuild(
        plan=FinancialQueryPlan.model_validate(
            {**common, "operation": operation, "periods": periods, "metric": _selector(operands)}
        )
    )
