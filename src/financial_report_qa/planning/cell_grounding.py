"""Nhánh answering duy nhất: quyết định -> plan -> compile (spec 2026-08-23 §6).

Nguyên tắc N6: không thang tầng, không candidate switching, không context
expansion. Một câu hỏi đi qua đúng một chuỗi bước -- hỏng ở đâu thì hỏng rõ
ở đó, với đúng một mã lỗi chỉ về đúng một bước.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.evidence_rendering import plan_grounding_score
from financial_report_qa.planning.fact_grounding import bind_plan_to_rows, grounded_facts
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.planning.question_plan import RowChoiceDecision, assemble_plan
from financial_report_qa.retrieval.contracts import _FrozenModel
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


class GroundingResult(_FrozenModel):
    """The result of grounding one question down the single answering path."""

    status: Literal["accepted", "failed"]
    plan: FinancialQueryPlan | None = None
    compiled: CompiledQuery | None = None
    # spec 2026-08-23 §6: chỉ còn một nguồn plan -- quyết định của LLM.
    plan_source: Literal["llm_decision"] | None = None
    error_code: str | None = None
    # plan.md §9: the retrieval confidence backing the accepted plan's row
    # selector(s) -- `None` when the plan was never scored by fusion (e.g. a
    # deterministic canonical-alias match).
    grounding_score: float | None = None
    # Vestiges of the removed recovery ladder, kept so downstream consumers
    # keep their shape: on the single path they are always at their defaults.
    recovery_attempts: int = 0
    low_confidence: bool = False
    # plan.md §9: per-fact provenance for the accepted answer -- each cell it
    # was computed from, identified by `(table_id, row_index)`. Empty when
    # grounding failed, and also when the answer's cells could not be
    # attributed to a row (see `fact_grounding.grounded_facts`).
    facts: tuple[GroundedFact, ...] = ()


def compile_grounded(
    plan: FinancialQueryPlan,
    fusion_rows: Sequence[RowFusedCandidate],
    release_dir: Path,
    execution_settings: ExecutionSettings,
) -> tuple[FinancialQueryPlan, CompiledQuery]:
    """Compile `plan`, positionally if retrieval can pin its rows (plan.md §14).

    The bound attempt goes first: it is the one that extracts by
    `df.loc[row_index, column]` with no label matching left in Pandas, so it
    resolves the label collisions that make a label-based compile abstain with
    `cell_ambiguous` or answer off the wrong row. Falling back to the unbound
    plan when binding is impossible or does not compile keeps every question
    that already worked working -- binding can only add answers here, never
    remove them.
    """
    bound = bind_plan_to_rows(plan, fusion_rows)
    if bound is not None:
        compiled = compile_plan(bound, release_dir, execution_settings=execution_settings)
        if compiled.status == "answered":
            return bound, compiled
    return plan, compile_plan(plan, release_dir, execution_settings=execution_settings)


def _accepted(
    *,
    plan: FinancialQueryPlan,
    compiled: CompiledQuery,
    plan_source: str,
    fusion_rows: Sequence[RowFusedCandidate],
    recovery_attempts: int = 0,
    low_confidence: bool = False,
) -> GroundingResult:
    """One accepted result, with its §9 facts derived from the same compile."""
    score = plan_grounding_score(plan, fusion_rows)
    return GroundingResult(
        status="accepted",
        plan=plan,
        compiled=compiled,
        plan_source=plan_source,
        recovery_attempts=recovery_attempts,
        grounding_score=score,
        low_confidence=low_confidence,
        facts=grounded_facts(compiled, grounding_score=score),
    )


def ground_question(
    *,
    entities: QueryEntities,
    decision: RowChoiceDecision | None,
    fusion_rows: Sequence[RowFusedCandidate],
    candidate_table_ids: Sequence[str],
    release_dir: Path,
    execution_settings: ExecutionSettings,
) -> GroundingResult:
    """Đường answering duy nhất: quyết định -> plan -> compile.

    Không có tầng thứ hai. Nguyên tắc N6: thang tầng cũ che giấu một tầng có
    tỷ lệ trả lời 0% trên 409 câu, và không tầng nào chịu trách nhiệm. Ở đây
    một câu hỏi hỏng có đúng một mã lỗi, chỉ về đúng một bước.
    """
    if not fusion_rows:
        return GroundingResult(
            status="failed", error_code="no_row_candidates", plan_source="llm_decision"
        )

    plan = assemble_plan(entities, decision, fusion_rows, candidate_table_ids)
    if plan is None:
        return GroundingResult(
            status="failed", error_code="plan_not_assembled", plan_source="llm_decision"
        )

    compiled_plan, compiled = compile_grounded(
        plan, fusion_rows, release_dir, execution_settings
    )
    if compiled.status != "answered":
        return GroundingResult(
            status="failed",
            error_code=compiled.error_code or "execution_failed",
            plan_source="llm_decision",
        )
    return _accepted(
        plan=compiled_plan,
        compiled=compiled,
        plan_source="llm_decision",
        fusion_rows=fusion_rows,
    )
