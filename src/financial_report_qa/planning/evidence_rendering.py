"""Evidence-aware rendering: row fusion results → ranked, deduplicated inputs.

Pure module -- receives pre-computed row fusion results, never calls retrieval
or touches the filesystem. Keeps the retrieval ↔ planning module boundary
intact while giving consumers ranked evidence instead of raw label lists.

The plan-era grounding scorers (`plan_grounding_score`/`plan_grounding_rank`)
were removed together with the operation-enum answering path they scored
(spec 2026-08-24 §8.2); the row-label rendering helpers below remain the one
ranked view of fusion results.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.normalization._shared import sanitize_selector_text
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


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
        # `sanitize_selector_text`, not just `.strip()`: this menu is what
        # `choose_row_label`/`choose_row_label_with_context` pick from, and
        # the choice becomes a quoted raw label, which forbids
        # control characters outright -- a real corpus label formed by
        # joining two source lines can carry an embedded newline that
        # `.strip()` alone would leave in place.
        sanitized = sanitize_selector_text(label) if label else None
        if sanitized and sanitized not in seen:
            seen[sanitized] = None
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
