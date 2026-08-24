"""`build_answer_package` orchestrator (ADR 0009).

One function, never a guessed or half-verified package: every answer that
ships must pass the four verification checks in `checks.py`. The plan-era
inputs from the plan/compiler era died with the operation-enum
answering path (spec 2026-08-24 §8.2); a package is now built from the one
finished `ExecutedProgram` the masked-PAL pipeline produced.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from financial_report_qa.execution.program_contracts import ExecutedProgram
from financial_report_qa.verification import checks
from financial_report_qa.verification.contracts import AnswerPackage, is_blocking_issue


def _default_display(answer: Decimal) -> tuple[str, int]:
    """Render the answer plainly and derive the precision actually printed.

    The plan-era display templates (`templates.py`) were removed together
    with the operation enum; until a masked-PAL-specific renderer exists the
    package ships the exact Decimal in plain positional notation, and the
    declared precision is the exponent that notation used.
    """
    exponent = answer.as_tuple().exponent
    precision = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return format(answer, "f"), precision


def build_answer_package(
    *,
    question_id: str,
    question: str,
    executed: ExecutedProgram,
    retrieved_table_ids: frozenset[str],
) -> AnswerPackage:
    """Build a verified `AnswerPackage` from one executed masked-PAL program.

    Raises `ValueError` if the execution carries no answer -- there is
    nothing to verify for an error result (it is already a typed failure,
    handled upstream).
    """
    answer = executed.answer
    display, display_precision = _default_display(answer)

    issues = [
        issue
        for issue in (
            checks.check_recompute_mismatch(executed),
            checks.check_scale_not_presentable(executed),
            checks.check_evidence_outside_retrieval(executed, retrieved_table_ids),
            checks.check_display_roundtrip_mismatch(
                answer, display, display_precision=display_precision
            ),
        )
        if issue is not None
    ]
    status: Literal["verified", "rejected"] = (
        "rejected" if any(is_blocking_issue(issue.code) for issue in issues) else "verified"
    )

    return AnswerPackage.model_validate(
        {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            # The compiler-era canonical-unit declaration died with the
            # operation enum; the masked program declares its magnitude via
            # `executed.scale`, which verification covers through
            # `check_scale_not_presentable`.
            "unit": None,
            "display": display,
            "display_precision": display_precision,
            "answer_text": display,
            "retrieved_table_ids": tuple(sorted(retrieved_table_ids)),
            "pandas_query": executed.pandas_query,
            "verification_status": status,
            "verification_issues": tuple(issues),
            "program": executed.program,
            "regenerated": executed.regenerated,
            "low_confidence": executed.low_confidence,
        }
    )
