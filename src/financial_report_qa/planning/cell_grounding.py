"""Decoupled Cell Grounding and Grounding Recovery module.

`ground_with_recovery` runs its own metric grounding (Attempt 0: raw-metric
rule grounding, then LLM row choice) and, if that still does not compile --
or compiles but the row it used ranked outside `max_grounding_rank` in
fusion (plan.md §15's confidence threshold, checked on rank, not raw score;
see `evidence_rendering.plan_grounding_rank` for why) -- walks a recovery
ladder: Attempt 1 (Candidate Switching across the retrieved row labels) then
Attempt 2 (Context Expansion, giving the LLM table/row context instead of
bare labels). Column refinement runs inside every tier whenever a compile
fails specifically with `cell_ambiguous`. A compiling-but-unconfident answer
is kept as a fallback throughout: an answer that costs exactly what an
abstention costs (this project's official scoring) but might be right beats
returning nothing just because a more confident match was never found.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.contracts import CompiledQuery
from financial_report_qa.normalization.companies import label_is_company_name
from financial_report_qa.planning.column_refinement import plan_with_column
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.planning.evidence_rendering import (
    evidence_table_context,
    plan_grounding_rank,
    plan_grounding_score,
)
from financial_report_qa.planning.fact_grounding import bind_plan_to_rows, grounded_facts
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.llm_cell_grounding import (
    choose_column_label,
    choose_row_label,
    choose_row_label_with_context,
)
from financial_report_qa.planning.llm_client import ChatCompletionClient
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.planning.raw_metric_grounding import (
    candidate_column_labels,
    ground_raw_metric,
)
from financial_report_qa.planning.rule_planner import RulePlanResult, build_plan
from financial_report_qa.planning.table_context_rendering import render_table_context
from financial_report_qa.retrieval.contracts import _FrozenModel
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


class GroundingResult(_FrozenModel):
    """The result of the cell grounding and recovery phase."""

    status: Literal["accepted", "failed"]
    plan: FinancialQueryPlan | None = None
    compiled: CompiledQuery | None = None
    plan_source: str | None = None
    error_code: str | None = None
    recovery_attempts: int = 0
    # plan.md §9: the retrieval confidence backing the accepted plan's row
    # selector(s) -- `None` when the plan was never scored by fusion (e.g. a
    # deterministic canonical-alias match).
    grounding_score: float | None = None
    # plan.md §15: True when this accepted result is the recovery ladder's
    # best-effort fallback -- every row selector it used ranked outside
    # `max_grounding_rank` in fusion, and the ladder never found (or was
    # never given the means to find) a more confidently-ranked alternative.
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


DEFAULT_MAX_GROUNDING_RANK = 3
"""plan.md §15 threshold, expressed as a fused-rank cutoff rather than a raw
score cutoff (see `evidence_rendering.plan_grounding_rank`). Top-3 mirrors
the Row Recall@3 checkpoint plan.md §20 tracks; an unvalidated starting
point (like the row-fusion branch weights), not a calibrated value -- no
gold row/column labels exist yet to tune it against."""


def ground_with_recovery(
    question: str,
    entities: QueryEntities,
    retrieved: Sequence[str],
    row_labels: Sequence[str],
    fusion_rows: Sequence[RowFusedCandidate],
    release_dir: Path,
    execution_settings: ExecutionSettings,
    llm_client: ChatCompletionClient | None = None,
    max_grounding_rank: int = DEFAULT_MAX_GROUNDING_RANK,
) -> GroundingResult:
    """Ground query metrics to verified row and column labels with recovery."""
    # Issuer-name rows (subsidiary listings, consolidation headers) sit in
    # numeric tables and therefore compile, but they carry no metric. Drop
    # them once, here, so no tier below -- LLM row choice, candidate
    # switching, context expansion -- can select one: a real run accepted
    # "▪ Tập đoàn Dệt May Việt Nam - Công ty mẹ" as the row for "Tổng cộng
    # dự phòng phải trả" (plan.md §19 dev benchmark, question 175).
    row_labels = tuple(label for label in row_labels if not label_is_company_name(label))

    # Attempt 0: Normal Grounding
    plan_result = RulePlanResult(abstain_codes=("operation_unknown",))
    plan_source = "rule"
    recovery_attempts = 0

    # Try rule-based deterministic raw metric grounding first
    raw_metric = ground_raw_metric(entities.question, retrieved, release_dir)
    if raw_metric is not None:
        labelled = QueryEntities.model_validate(
            {
                **entities.model_dump(mode="python"),
                "metrics": (raw_metric,),
                "ambiguity": tuple(code for code in entities.ambiguity if code != "metric_unknown"),
            }
        )
        plan_result = build_plan(
            labelled,
            candidate_table_ids=retrieved,
            known_table_ids=frozenset(retrieved),
        )
        plan_source = "rule_raw_grounded"

    # If rule-based grounding fails, try LLM-based grounding (Attempt 0 LLM)
    if plan_result.plan is None and llm_client is not None:
        chosen_label = choose_row_label(
            question,
            row_labels,
            client=llm_client,
        )
        if chosen_label is not None:
            labelled = QueryEntities.model_validate(
                {
                    **entities.model_dump(mode="python"),
                    "metrics": (chosen_label,),
                    "ambiguity": tuple(
                        code for code in entities.ambiguity if code != "metric_unknown"
                    ),
                }
            )
            relabelled = build_plan(
                labelled,
                candidate_table_ids=retrieved,
                known_table_ids=frozenset(retrieved),
            )
            if relabelled.plan is not None:
                plan_result, plan_source = relabelled, "llm_cell_grounded"

    # If no plan could be built at all, fail grounding
    if plan_result.plan is None:
        error_code = (
            "+".join(plan_result.abstain_codes) if plan_result.abstain_codes else "planning_failed"
        )
        return GroundingResult(
            status="failed",
            error_code=error_code,
            plan_source=plan_source,
        )

    plan, compiled = compile_grounded(
        plan_result.plan, fusion_rows, release_dir, execution_settings
    )

    # Day 26 Column Refinement on Attempt 0
    if compiled.error_code == "cell_ambiguous" and llm_client is not None:
        refined = plan_with_column(
            plan,
            lambda row_label, columns: choose_column_label(
                question, row_label, columns, client=llm_client
            ),
            columns_for=lambda row_label: candidate_column_labels(
                release_dir, plan.candidate_table_ids, row_label
            ),
        )
        if refined is not None:
            retried_plan, retried_compiled = compile_grounded(
                refined, fusion_rows, release_dir, execution_settings
            )
            if retried_compiled.status == "answered":
                plan = retried_plan
                compiled = retried_compiled
                plan_source = "llm_column_refined"

    # If Attempt 0 succeeds and is confidently ranked, accept it outright.
    # plan.md §15: a compile succeeding is not by itself enough -- a row
    # fusion ranked outside the top `max_grounding_rank` candidates is not
    # trusted just because *some* cell happened to be found for it. Keep it
    # as `fallback` and keep looking instead of returning error OR accepting
    # blindly.
    fallback: GroundingResult | None = None
    if compiled.status == "answered":
        rank = plan_grounding_rank(plan, fusion_rows)
        if rank is None or rank <= max_grounding_rank:
            return _accepted(
                plan=plan, compiled=compiled, plan_source=plan_source, fusion_rows=fusion_rows
            )
        fallback = _accepted(
            plan=plan,
            compiled=compiled,
            plan_source=plan_source,
            fusion_rows=fusion_rows,
            low_confidence=True,
        )

    # Attempt 1: Grounding Recovery (Candidate Switching)
    GROUNDING_ERROR_CODES = {
        "metric_not_found",
        "cell_ambiguous",
        "period_unresolved",
        "unit_missing",
    }
    if compiled.error_code in GROUNDING_ERROR_CODES or fallback is not None:
        tried_labels: set[str] = set()
        selectors = (
            plan.metric,
            plan.metric_a,
            plan.metric_b,
            plan.numerator_metric,
            plan.denominator_metric,
        )
        for selector in selectors:
            if selector is not None:
                if selector.raw_text:
                    tried_labels.add(selector.raw_text)
                if selector.canonical:
                    tried_labels.add(selector.canonical)

        max_recovery_attempts = 5
        for candidate_label in row_labels:
            if candidate_label in tried_labels:
                continue
            if recovery_attempts >= max_recovery_attempts:
                break
            tried_labels.add(candidate_label)
            recovery_attempts += 1

            # Rebuild plan using the candidate label
            labelled = QueryEntities.model_validate(
                {
                    **entities.model_dump(mode="python"),
                    "metrics": (candidate_label,),
                    "ambiguity": tuple(
                        code for code in entities.ambiguity if code != "metric_unknown"
                    ),
                }
            )
            relabelled = build_plan(
                labelled,
                candidate_table_ids=retrieved,
                known_table_ids=frozenset(retrieved),
            )
            if relabelled.plan is not None:
                candidate_plan, candidate_compiled = compile_grounded(
                    relabelled.plan, fusion_rows, release_dir, execution_settings
                )

                # Day 26 cell_ambiguous recovery inside candidate switching:
                if candidate_compiled.error_code == "cell_ambiguous" and llm_client is not None:
                    refined = plan_with_column(
                        candidate_plan,
                        lambda row_label, columns: choose_column_label(
                            question, row_label, columns, client=llm_client
                        ),
                        columns_for=lambda row_label: candidate_column_labels(
                            release_dir, candidate_plan.candidate_table_ids, row_label
                        ),
                    )
                    if refined is not None:
                        retried_plan, retried_compiled = compile_grounded(
                            refined, fusion_rows, release_dir, execution_settings
                        )
                        if retried_compiled.status == "answered":
                            candidate_plan = retried_plan
                            candidate_compiled = retried_compiled

                if candidate_compiled.status == "answered":
                    candidate_rank = plan_grounding_rank(candidate_plan, fusion_rows)
                    if candidate_rank is None or candidate_rank <= max_grounding_rank:
                        return _accepted(
                            plan=candidate_plan,
                            compiled=candidate_compiled,
                            plan_source="llm_cell_grounded_recovered",
                            fusion_rows=fusion_rows,
                            recovery_attempts=recovery_attempts,
                        )
                    if fallback is None:
                        fallback = _accepted(
                            plan=candidate_plan,
                            compiled=candidate_compiled,
                            plan_source="llm_cell_grounded_recovered",
                            fusion_rows=fusion_rows,
                            recovery_attempts=recovery_attempts,
                            low_confidence=True,
                        )

    # Attempt 2: Context Expansion
    if (fallback is not None or compiled.status != "answered") and llm_client is not None:
        table_context = (
            evidence_table_context(fusion_rows)
            if fusion_rows
            else render_table_context(release_dir, retrieved)
        )
        chosen_label = choose_row_label_with_context(
            question,
            row_labels,
            table_context,
            client=llm_client,
        )
        if chosen_label is not None:
            labelled = QueryEntities.model_validate(
                {
                    **entities.model_dump(mode="python"),
                    "metrics": (chosen_label,),
                    "ambiguity": tuple(
                        code for code in entities.ambiguity if code != "metric_unknown"
                    ),
                }
            )
            relabelled = build_plan(
                labelled,
                candidate_table_ids=retrieved,
                known_table_ids=frozenset(retrieved),
            )
            if relabelled.plan is not None:
                candidate_plan, candidate_compiled = compile_grounded(
                    relabelled.plan, fusion_rows, release_dir, execution_settings
                )

                if candidate_compiled.error_code == "cell_ambiguous":
                    refined = plan_with_column(
                        candidate_plan,
                        lambda row_label, columns: choose_column_label(
                            question, row_label, columns, client=llm_client
                        ),
                        columns_for=lambda row_label: candidate_column_labels(
                            release_dir, candidate_plan.candidate_table_ids, row_label
                        ),
                    )
                    if refined is not None:
                        retried_plan, retried_compiled = compile_grounded(
                            refined, fusion_rows, release_dir, execution_settings
                        )
                        if retried_compiled.status == "answered":
                            candidate_plan = retried_plan
                            candidate_compiled = retried_compiled

                if candidate_compiled.status == "answered":
                    candidate_rank = plan_grounding_rank(candidate_plan, fusion_rows)
                    if candidate_rank is None or candidate_rank <= max_grounding_rank:
                        return _accepted(
                            plan=candidate_plan,
                            compiled=candidate_compiled,
                            plan_source="llm_cell_grounded_context_expanded",
                            fusion_rows=fusion_rows,
                            recovery_attempts=recovery_attempts + 1,
                        )
                    if fallback is None:
                        fallback = _accepted(
                            plan=candidate_plan,
                            compiled=candidate_compiled,
                            plan_source="llm_cell_grounded_context_expanded",
                            fusion_rows=fusion_rows,
                            recovery_attempts=recovery_attempts + 1,
                            low_confidence=True,
                        )

    # plan.md §15: an unconfident-but-compiling answer beats returning
    # nothing -- an abstention scores the same as a wrong answer under this
    # project's official metric, so a low-confidence guess still has a
    # chance of being right where an abstention has none.
    if fallback is not None:
        return fallback

    return GroundingResult(
        status="failed",
        error_code=compiled.error_code or "execution_failed",
        plan_source=plan_source,
        recovery_attempts=recovery_attempts,
    )
