"""Unit tests for evidence_rendering — pure row-fusion-to-planner transforms."""

from __future__ import annotations

from financial_report_qa.planning.evidence_rendering import (
    evidence_row_labels,
    evidence_table_context,
    plan_grounding_rank,
    plan_grounding_score,
    row_label_confidence,
)
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_A = "tbl_" + "a" * 64
TABLE_B = "tbl_" + "b" * 64


def _candidate(
    *,
    row_id: str | None = None,
    table_id: str = TABLE_A,
    row_idx: int = 0,
    rank: int = 1,
    fused_score: float = 0.5,
    bm25_rank: int | None = 1,
    bm25_score: float | None = 0.3,
    dense_rank: int | None = 1,
    dense_score: float | None = 0.7,
    row_label_raw: str | None = "Doanh thu",
    company_code: str | None = "VCB",
    statement_type: str | None = "income_statement",
    title: str | None = "Báo cáo KQKD",
    snippet: str = "company: VCB\nrow_label: Doanh thu",
) -> RowFusedCandidate:
    if row_id is None:
        row_id = f"{table_id}|row_{row_idx}"
    return RowFusedCandidate(
        row_id=row_id,
        table_id=table_id,
        row_idx=row_idx,
        rank=rank,
        fused_score=fused_score,
        bm25_rank=bm25_rank,
        bm25_score=bm25_score,
        dense_rank=dense_rank,
        dense_score=dense_score,
        metadata=RowMetadata(
            table_id=table_id,
            row_idx=row_idx,
            company_code=company_code,
            statement_type=statement_type,
            title=title,
            row_label_raw=row_label_raw,
        ),
        snippet=snippet,
    )


def test_evidence_row_labels_ranked_by_score() -> None:
    candidates = (
        _candidate(row_idx=0, rank=1, fused_score=0.9, row_label_raw="Chi phí lãi vay"),
        _candidate(row_idx=1, rank=2, fused_score=0.7, row_label_raw="Doanh thu bán hàng"),
        _candidate(table_id=TABLE_B, row_idx=0, rank=3, fused_score=0.5, row_label_raw="Cho vay khách hàng"),
        _candidate(table_id=TABLE_B, row_idx=1, rank=4, fused_score=0.3, row_label_raw="Tiền gửi NHNN"),
    )

    labels = evidence_row_labels(candidates)

    assert labels == ("Chi phí lãi vay", "Doanh thu bán hàng", "Cho vay khách hàng", "Tiền gửi NHNN")


def test_evidence_row_labels_deduplicates() -> None:
    # Same label appearing in two tables — only first (highest-scored) occurrence kept
    candidates = (
        _candidate(row_idx=0, rank=1, fused_score=0.9, row_label_raw="Doanh thu"),
        _candidate(table_id=TABLE_B, row_idx=0, rank=2, fused_score=0.7, row_label_raw="Doanh thu"),
        _candidate(row_idx=1, rank=3, fused_score=0.5, row_label_raw="Chi phí"),
    )

    labels = evidence_row_labels(candidates)

    assert labels == ("Doanh thu", "Chi phí")


def test_evidence_row_labels_respects_max() -> None:
    candidates = tuple(
        _candidate(
            table_id=TABLE_A,
            row_idx=i,
            rank=i + 1,
            fused_score=1.0 / (i + 1),
            row_label_raw=f"Label_{i}",
        )
        for i in range(100)
    )

    labels = evidence_row_labels(candidates, max_labels=5)

    assert len(labels) == 5
    assert labels == ("Label_0", "Label_1", "Label_2", "Label_3", "Label_4")


def test_evidence_row_labels_empty_fusion() -> None:
    assert evidence_row_labels(()) == ()


def test_evidence_row_labels_skips_none_labels() -> None:
    candidates = (
        _candidate(row_idx=0, rank=1, fused_score=0.9, row_label_raw=None),
        _candidate(row_idx=1, rank=2, fused_score=0.7, row_label_raw=""),
        _candidate(row_idx=2, rank=3, fused_score=0.5, row_label_raw="  "),
        _candidate(row_idx=3, rank=4, fused_score=0.3, row_label_raw="Doanh thu"),
    )

    labels = evidence_row_labels(candidates)

    assert labels == ("Doanh thu",)


def test_evidence_table_context_top_rows() -> None:
    candidates = (
        _candidate(
            row_idx=0,
            rank=1,
            fused_score=0.9,
            bm25_rank=1,
            dense_rank=2,
            row_label_raw="Doanh thu",
            company_code="VCB",
            title="Báo cáo KQKD",
            statement_type="income_statement",
            snippet="company: VCB\nrow_label: Doanh thu\nperiods: 2020",
        ),
        _candidate(
            table_id=TABLE_B,
            row_idx=1,
            rank=2,
            fused_score=0.5,
            bm25_rank=None,
            bm25_score=None,
            dense_rank=1,
            row_label_raw="Cho vay khách hàng",
            company_code="CTG",
            title="CĐKT",
            statement_type="balance_sheet",
            snippet="company: CTG\nrow_label: Cho vay khách hàng",
        ),
    )

    context = evidence_table_context(candidates)

    # Contains both rows
    assert "VCB" in context
    assert "CTG" in context
    assert "Doanh thu" in context
    assert "Cho vay khách hàng" in context
    # Contains score info
    assert "score=0.9000" in context
    assert "bm25_rank=1" in context
    assert "dense_rank=2" in context


def test_evidence_table_context_max_rows() -> None:
    candidates = tuple(
        _candidate(
            table_id=TABLE_A,
            row_idx=i,
            rank=i + 1,
            fused_score=1.0 / (i + 1),
            row_label_raw=f"Label_{i}",
            snippet=f"row {i}",
        )
        for i in range(50)
    )

    context = evidence_table_context(candidates, max_rows=3)

    assert "row 0" in context
    assert "row 1" in context
    assert "row 2" in context
    assert "row 3" not in context


def test_evidence_table_context_empty() -> None:
    assert evidence_table_context(()) == ""


def _plan(**metric_kwargs: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("VCB",),
        "periods": ("2020",),
        "candidate_table_ids": (TABLE_A,),
    }
    defaults.update(metric_kwargs)
    return FinancialQueryPlan(**defaults)


def test_row_label_confidence_matches_exact_label() -> None:
    candidates = (
        _candidate(row_idx=0, fused_score=0.42, row_label_raw="Doanh thu"),
        _candidate(row_idx=1, fused_score=0.9, row_label_raw="Chi phí"),
    )

    assert row_label_confidence("Doanh thu", candidates) == 0.42
    assert row_label_confidence("Chi phí", candidates) == 0.9


def test_row_label_confidence_no_match_returns_none() -> None:
    candidates = (_candidate(row_idx=0, fused_score=0.42, row_label_raw="Doanh thu"),)

    assert row_label_confidence("Không tồn tại", candidates) is None


def test_row_label_confidence_none_or_empty_label_returns_none() -> None:
    candidates = (_candidate(row_idx=0, fused_score=0.42, row_label_raw="Doanh thu"),)

    assert row_label_confidence(None, candidates) is None
    assert row_label_confidence("", candidates) is None


def test_plan_grounding_score_single_metric() -> None:
    plan = _plan(metric=MetricSelector(raw_text="Doanh thu"))
    candidates = (_candidate(row_idx=0, fused_score=0.77, row_label_raw="Doanh thu"),)

    assert plan_grounding_score(plan, candidates) == 0.77


def test_plan_grounding_score_takes_weakest_link_across_selectors() -> None:
    plan = _plan(
        operation="difference",
        metric_a=MetricSelector(raw_text="Doanh thu"),
        metric_b=MetricSelector(raw_text="Chi phí"),
    )
    candidates = (
        _candidate(row_idx=0, fused_score=0.9, row_label_raw="Doanh thu"),
        _candidate(row_idx=1, fused_score=0.3, row_label_raw="Chi phí"),
    )

    assert plan_grounding_score(plan, candidates) == 0.3


def test_plan_grounding_score_canonical_match_is_none() -> None:
    # A canonical-dictionary metric never went through row fusion at all.
    plan = _plan(metric=MetricSelector(canonical="revenue"))
    candidates = (_candidate(row_idx=0, fused_score=0.9, row_label_raw="Doanh thu"),)

    assert plan_grounding_score(plan, candidates) is None


def test_plan_grounding_score_no_fusion_rows_is_none() -> None:
    plan = _plan(metric=MetricSelector(raw_text="Doanh thu"))

    assert plan_grounding_score(plan, ()) is None


def test_plan_grounding_rank_single_metric() -> None:
    plan = _plan(metric=MetricSelector(raw_text="Doanh thu"))
    candidates = (_candidate(row_idx=0, rank=4, row_label_raw="Doanh thu"),)

    assert plan_grounding_rank(plan, candidates) == 4


def test_plan_grounding_rank_takes_worst_rank_across_selectors() -> None:
    plan = _plan(
        operation="difference",
        metric_a=MetricSelector(raw_text="Doanh thu"),
        metric_b=MetricSelector(raw_text="Chi phí"),
    )
    candidates = (
        _candidate(row_idx=0, rank=1, row_label_raw="Doanh thu"),
        _candidate(row_idx=1, rank=7, row_label_raw="Chi phí"),
    )

    # Worst (highest-numbered) rank wins, not the best one.
    assert plan_grounding_rank(plan, candidates) == 7


def test_plan_grounding_rank_canonical_match_is_none() -> None:
    plan = _plan(metric=MetricSelector(canonical="revenue"))
    candidates = (_candidate(row_idx=0, rank=1, row_label_raw="Doanh thu"),)

    assert plan_grounding_rank(plan, candidates) is None


def test_plan_grounding_rank_no_fusion_rows_is_none() -> None:
    plan = _plan(metric=MetricSelector(raw_text="Doanh thu"))

    assert plan_grounding_rank(plan, ()) is None
