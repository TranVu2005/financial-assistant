"""plan.md §12: the Evidence-Aware Planner's output shape.

An `EvidencePlan` is everything the LLM is allowed to decide once grounding
has already resolved the cells: which arithmetic to run, over which of the
facts it was shown. Nothing else. The fields §12 names as the things a
planner must stop inventing -- `table_name`, row locator, column locator,
metric name -- are not merely optional here, they are unrepresentable, and
`extra="forbid"` (inherited from `_FrozenModel`) makes an attempt to smuggle
one in a validation error rather than a silently ignored key.

That is the point of the redesign. The 231 `llm_plan_invalid` questions came
from asking a small model to emit a whole typed `FinancialQueryPlan` -- metric
selector, candidate tables, periods, scope -- in one shot. This shape asks for
an enum and a short list of identifiers that were printed in its own prompt.

Turning an `EvidencePlan` back into an executable `FinancialQueryPlan` is
`evidence_planner.build_plan_from_facts`'s job, and that is where per-operation
arity and fact-consistency are checked -- the same schema/semantics split
`plan_contracts` and `plan_validator` already use (ADR 0004).
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from financial_report_qa.planning.grounding_contracts import FactId
from financial_report_qa.planning.plan_contracts import PlanOperation
from financial_report_qa.retrieval.contracts import _FrozenModel

# No operation in this project reads more than four operands: `rank` and the
# aggregates are the only variadic ones, and they are excluded from evidence
# planning entirely (see `evidence_planner`). Bounding the list keeps a
# runaway generation from turning into a huge fact-consistency scan.
MAX_OPERANDS = 4


class EvidencePlan(_FrozenModel):
    """One arithmetic step over already-grounded facts."""

    operation: PlanOperation
    operands: tuple[FactId, ...]

    @model_validator(mode="after")
    def validate_operands(self) -> Self:
        if not self.operands:
            raise ValueError("operands must not be empty")
        if len(self.operands) > MAX_OPERANDS:
            raise ValueError(f"operands must have at most {MAX_OPERANDS} entries")
        if len(set(self.operands)) != len(self.operands):
            # The same fact twice is never a real two-operand computation: a
            # growth rate against itself is 0 and a ratio against itself is 1,
            # both of which look like answers.
            raise ValueError("operands must not contain duplicates")
        return self
