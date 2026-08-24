"""One question, one straight line: decide -> guard -> bind -> run -> verify.

N6 allows exactly one retry of the decision step and nothing else: no second
route, no alternate strategy, no third attempt. `MAX_ATTEMPTS = 2` is that
rule written down, and the live path has no parameter to raise it.

When both attempts diverge but one of them still produced a number, the number
is submitted with `low_confidence` set. Leaving it blank scores zero for
certain; a wrong answer can only also score zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from financial_report_qa.core.errors import ProgramError, ProgramGuardError
from financial_report_qa.execution.masked_program import apply_scale, run_program
from financial_report_qa.execution.program_binding import (
    bind_values,
    render_program_pandas,
    values_by_position,
)
from financial_report_qa.execution.program_contracts import (
    CellCandidate,
    ExecutedProgram,
    ProgramDecision,
    ProgramFailureCode,
)
from financial_report_qa.verification.explanation_check import check_explanation
from financial_report_qa.verification.use_checks import check_use_bindings

#: N6: một lần chạy cộng đúng một lần sinh lại. Không nới ở đường live.
MAX_ATTEMPTS = 2


class DecisionSource(Protocol):
    def decide(self, question_id: int, attempt: int) -> ProgramDecision: ...


ExplanationSource = Callable[[ExecutedProgram], str]


@dataclass(frozen=True)
class PipelineResult:
    """Either an executed program, or the code that stopped every attempt."""

    executed: ExecutedProgram | None
    failure_code: ProgramFailureCode | None


def _failure_code(error: Exception) -> ProgramFailureCode:
    message = str(error)
    if "candidate_index_out_of_range" in message:
        return "candidate_index_out_of_range"
    if "division_by_zero" in message:
        return "division_by_zero"
    if "non_finite_result" in message:
        return "non_finite_result"
    if isinstance(error, ProgramGuardError) and "literal not allowed" in message:
        return "numeric_literal_in_program"
    return "program_node_not_allowed"


def _execute(
    question_id: int,
    decision: ProgramDecision,
    candidates: Sequence[CellCandidate],
    frame: pd.DataFrame,
    *,
    regenerated: bool,
) -> ExecutedProgram:
    bindings = bind_values(decision, candidates, values_by_position(frame))
    raw = run_program(decision.program, [bound.value for bound in bindings])
    answer = apply_scale(raw, decision.scale)
    query = render_program_pandas(decision.program, bindings, decision.scale)
    return ExecutedProgram(
        question_id=question_id,
        program=decision.program,
        scale=decision.scale,
        bindings=bindings,
        answer=answer,
        pandas_query=query,
        table_ids=tuple(sorted({bound.table_id for bound in bindings})),
        regenerated=regenerated,
    )


def run_question(
    question_id: int,
    candidates: Sequence[CellCandidate],
    frame: pd.DataFrame,
    decisions: DecisionSource,
    *,
    explanations: ExplanationSource | None = None,
) -> PipelineResult:
    """Answer one question, retrying the decision step at most once."""
    if not candidates:
        return PipelineResult(executed=None, failure_code="no_cell_candidates")

    last_executed: ExecutedProgram | None = None
    last_code: ProgramFailureCode | None = None

    for attempt in range(MAX_ATTEMPTS):
        regenerated = attempt > 0
        try:
            decision = decisions.decide(question_id, attempt)
            executed = _execute(question_id, decision, candidates, frame, regenerated=regenerated)
        except ProgramError as error:
            last_code = _failure_code(error)
            continue

        use_result = check_use_bindings(decision.uses, executed.bindings)
        if not use_result.matched:
            last_executed, last_code = executed, "use_binding_mismatch"
            continue

        if explanations is not None:
            guard = check_explanation(explanations(executed), executed)
            if not guard.allowed:
                last_executed = executed
                last_code = "explanation_number_not_grounded"
                continue

        return PipelineResult(executed=executed, failure_code=None)

    if last_executed is not None:
        return PipelineResult(
            executed=last_executed.model_copy(
                update={"low_confidence": True, "failure_code": last_code}
            ),
            failure_code=last_code,
        )
    return PipelineResult(executed=None, failure_code=last_code)
