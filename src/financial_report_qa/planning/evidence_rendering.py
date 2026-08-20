"""Evidence-aware rendering: row fusion results → planner inputs.

Pure module — receives pre-computed row fusion results, never calls retrieval
or touches the filesystem. Keeps the retrieval ↔ planning module boundary
intact while giving the planner ranked evidence instead of raw label lists.

Two functions replace their unranked counterparts when row fusion is available:

- ``evidence_row_labels``  replaces ``raw_metric_grounding.candidate_row_labels``
- ``evidence_table_context`` replaces ``table_context_rendering.render_table_context``
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

if TYPE_CHECKING:
    from financial_report_qa.planning.plan_contracts import FinancialQueryPlan


def evidence_row_labels(
    fusion_results: Sequence[RowFusedCandidate],
    *,
    max_labels: int = 60,
) -> tuple[str, ...]:
    """Row labels ranked by fusion score, deduplicated.

    Unlike ``candidate_row_labels()`` which returns ALL labels from ALL tables
    unranked (then alphabetically truncated at 60), this returns labels ordered
    by evidence strength — the row fusion already scored and ranked them.

    Deduplication preserves the first (highest-scored) occurrence of each label
    so that a label appearing in multiple tables keeps its best rank position.
    """
    seen: dict[str, None] = {}
    for candidate in fusion_results:
        label = candidate.metadata.row_label_raw
        if label and label.strip() and label.strip() not in seen:
            seen[label.strip()] = None
        if len(seen) >= max_labels:
            break
    return tuple(seen)


def evidence_table_context(
    fusion_results: Sequence[RowFusedCandidate],
    *,
    max_rows: int = 20,
) -> str:
    """Compact context built from top fusion rows only.

    Instead of rendering entire tables (up to 80 rows each × N tables),
    renders only the top-ranked row snippets with their table/company context.
    Much more focused and within small-model context budgets.

    Each row is rendered as a block with its table metadata and the full
    snippet text from the row document, giving the LLM enough context to
    understand the row's meaning without needing to scan entire tables.
    """
    if not fusion_results:
        return ""

    blocks: list[str] = []
    for candidate in fusion_results[:max_rows]:
        meta = candidate.metadata
        header_parts: list[str] = []
        if meta.company_code:
            header_parts.append(f"Công ty: {meta.company_code}")
        if meta.title:
            header_parts.append(f"Bảng: {meta.title}")
        if meta.statement_type:
            header_parts.append(f"Loại: {meta.statement_type}")
        header = " | ".join(header_parts) if header_parts else f"Bảng {candidate.table_id}"

        score_parts: list[str] = [f"score={candidate.fused_score:.4f}"]
        if candidate.bm25_rank is not None:
            score_parts.append(f"bm25_rank={candidate.bm25_rank}")
        if candidate.dense_rank is not None:
            score_parts.append(f"dense_rank={candidate.dense_rank}")
        score_info = ", ".join(score_parts)

        blocks.append(f"--- [{score_info}] {header} ---\n{candidate.snippet}")

    return "\n\n".join(blocks)


def _matching_candidate(
    label: str | None,
    fusion_results: Sequence[RowFusedCandidate],
) -> RowFusedCandidate | None:
    if not label:
        return None
    for candidate in fusion_results:
        if candidate.metadata.row_label_raw == label:
            return candidate
    return None


def row_label_confidence(
    label: str | None,
    fusion_results: Sequence[RowFusedCandidate],
) -> float | None:
    """The fused retrieval score for the row whose raw label matches `label`
    exactly, or `None` when `label` is unset or was never scored by fusion
    (e.g. a canonical-dictionary match resolved without going through row
    retrieval at all -- an exact alias hit needs no confidence number)."""
    candidate = _matching_candidate(label, fusion_results)
    return candidate.fused_score if candidate is not None else None


def _plan_selectors(plan: FinancialQueryPlan) -> tuple[object, ...]:
    return (
        plan.metric,
        plan.metric_a,
        plan.metric_b,
        plan.numerator_metric,
        plan.denominator_metric,
    )


def plan_grounding_score(
    plan: FinancialQueryPlan,
    fusion_results: Sequence[RowFusedCandidate],
) -> float | None:
    """One confidence number (plan.md §9's `grounding_score`) for the whole
    plan: the weakest-linked confidence among every row selector the plan
    actually uses (`metric`, or the pair for a two-metric operation).
    `None` when no selector matched a fusion-scored row -- the plan may
    still be entirely correct (e.g. deterministic canonical lookup), it
    just has no retrieval-confidence evidence attached."""
    scores = [
        score
        for selector in _plan_selectors(plan)
        if selector is not None
        and (score := row_label_confidence(selector.raw_text, fusion_results)) is not None
    ]
    return min(scores) if scores else None


def plan_grounding_rank(
    plan: FinancialQueryPlan,
    fusion_results: Sequence[RowFusedCandidate],
) -> int | None:
    """The worst (highest-numbered) fused rank among the plan's row
    selectors that matched a fusion-scored row, or `None` when none did.

    Rank, not raw `fused_score`, is what plan.md §15's confidence threshold
    is checked against here: `fused_score` mixes branches weighted very
    differently (an exact alias hit is weighted low but is near-certainly
    correct; a rank-1 bm25 hit is weighted high but only means "best
    keyword overlap"), so raw scores are not comparable across candidates
    scored by different branch mixes. Rank *is* comparable -- it is each
    candidate's position after all branches are already fused and sorted.
    """
    ranks = [
        candidate.rank
        for selector in _plan_selectors(plan)
        if selector is not None
        and (candidate := _matching_candidate(selector.raw_text, fusion_results)) is not None
    ]
    return max(ranks) if ranks else None
