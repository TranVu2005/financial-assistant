"""Tests for plan.md §9/§14 fact grounding: label -> position binding."""

from decimal import Decimal

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery, ReplayRow
from financial_report_qa.planning.fact_grounding import bind_plan_to_rows, grounded_facts
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.retrieval.row_documents import RowMetadata
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate

TABLE_ID = "tbl_" + "1" * 64
OTHER_TABLE_ID = "tbl_" + "2" * 64
CELL_ID = "cell_" + "a" * 64


def _candidate(
    *,
    rank: int,
    row_idx: int,
    table_id: str = TABLE_ID,
    row_label_raw: str | None = "Tiền mặt",
    row_label_canonical: str | None = None,
    company_code: str | None = "ACB",
    fused_score: float = 0.9,
) -> RowFusedCandidate:
    return RowFusedCandidate(
        row_id=f"{table_id}|row_{row_idx}",
        table_id=table_id,
        row_idx=row_idx,
        rank=rank,
        fused_score=fused_score,
        metadata=RowMetadata(
            table_id=table_id,
            row_idx=row_idx,
            company_code=company_code,
            row_label_raw=row_label_raw,
            row_label_canonical=row_label_canonical,
        ),
        snippet="Tiền mặt | 900",
    )


def _lookup_plan(**overrides: object) -> FinancialQueryPlan:
    payload: dict[str, object] = {
        "operation": "lookup",
        "companies": ("ACB",),
        "periods": ("2023",),
        "candidate_table_ids": (TABLE_ID,),
        "metric": MetricSelector(raw_text="Tiền mặt"),
    }
    payload.update(overrides)
    return FinancialQueryPlan.model_validate(payload)


def test_bind_pins_the_selector_to_the_top_ranked_matching_row() -> None:
    plan = _lookup_plan()
    bound = bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14)])
    assert bound is not None
    assert bound.metric is not None
    assert bound.metric.is_position_bound
    assert bound.metric.table_id == TABLE_ID
    assert bound.metric.row_index == 14
    # The label survives as provenance; it is simply no longer the identifier.
    assert bound.metric.raw_text == "Tiền mặt"


def test_bind_prefers_the_better_ranked_of_two_rows_sharing_a_label() -> None:
    plan = _lookup_plan()
    bound = bind_plan_to_rows(
        plan,
        [_candidate(rank=3, row_idx=3), _candidate(rank=1, row_idx=14)],
    )
    assert bound is not None and bound.metric is not None
    assert bound.metric.row_index == 14


def test_bind_matches_a_canonical_selector_against_the_canonical_row_label() -> None:
    plan = _lookup_plan(metric=MetricSelector(canonical="cash_and_cash_equivalents"))
    bound = bind_plan_to_rows(
        plan,
        [
            _candidate(
                rank=1,
                row_idx=14,
                row_label_raw="Tiền mặt",
                row_label_canonical="cash_and_cash_equivalents",
            )
        ],
    )
    assert bound is not None and bound.metric is not None
    assert bound.metric.row_index == 14


def test_bind_returns_none_when_no_candidate_row_matches_the_selector() -> None:
    """Binding never invents a position: an unmatched selector stays unbound
    and the caller keeps the label-based plan it already had."""
    plan = _lookup_plan()
    assert bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14, row_label_raw="Doanh thu")]) is (
        None
    )


def test_bind_ignores_rows_outside_the_plans_candidate_tables() -> None:
    plan = _lookup_plan()
    assert bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14, table_id=OTHER_TABLE_ID)]) is (
        None
    )


def test_bind_ignores_rows_belonging_to_another_company() -> None:
    plan = _lookup_plan()
    assert bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14, company_code="MBB")]) is None


def test_bind_declines_multi_company_plans() -> None:
    """One selector serving several companies cannot be one position: each
    company's figure lives in its own table. Generalizing that is plan.md
    §12's per-fact operand model, not something binding may fake here."""
    plan = _lookup_plan(operation="compare_companies", companies=("ACB", "MBB"))
    assert bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14)]) is None


def test_bind_pins_both_selectors_of_a_two_metric_operation() -> None:
    plan = _lookup_plan(
        operation="ratio",
        metric=None,
        numerator_metric=MetricSelector(raw_text="Tiền mặt"),
        denominator_metric=MetricSelector(raw_text="Tổng tài sản"),
    )
    bound = bind_plan_to_rows(
        plan,
        [
            _candidate(rank=1, row_idx=14),
            _candidate(rank=2, row_idx=40, row_label_raw="Tổng tài sản"),
        ],
    )
    assert bound is not None
    assert bound.numerator_metric is not None and bound.numerator_metric.row_index == 14
    assert bound.denominator_metric is not None and bound.denominator_metric.row_index == 40


def test_bind_declines_when_only_one_of_two_selectors_can_be_pinned() -> None:
    """A half-bound plan would mix positional and semantic extraction in one
    answer, which is precisely the ambiguity §14 removes."""
    plan = _lookup_plan(
        operation="ratio",
        metric=None,
        numerator_metric=MetricSelector(raw_text="Tiền mặt"),
        denominator_metric=MetricSelector(raw_text="Tổng tài sản"),
    )
    assert bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14)]) is None


def test_bind_keeps_a_grounded_column_predicate() -> None:
    plan = _lookup_plan(
        metric=MetricSelector(raw_text="Tiền mặt", column_text="Số cuối năm"),
    )
    bound = bind_plan_to_rows(plan, [_candidate(rank=1, row_idx=14)])
    assert bound is not None and bound.metric is not None
    assert bound.metric.column_text == "Số cuối năm"


def _compiled(evidence: tuple[CellMatch, ...], replay: tuple[ReplayRow, ...]) -> CompiledQuery:
    return CompiledQuery(
        operation="lookup",
        status="answered",
        answer=Decimal("900"),
        unit="VND",
        evidence=evidence,
        pandas_query="df1.loc[...]",
        error_code=None,
        error_message=None,
        replay_rows=replay,
    )


def test_grounded_facts_describe_each_evidence_cell_by_position() -> None:
    compiled = _compiled(
        (
            CellMatch(
                table_id=TABLE_ID,
                cell_ids=(CELL_ID,),
                value=Decimal("900"),
                unit="VND",
                period=2023,
                period_inferred=False,
                row_index=14,
                column_label="Năm 2023",
            ),
        ),
        (
            ReplayRow(
                company_code="ACB",
                row_label_canonical=None,
                row_label_raw="Tiền mặt",
                column_label=None,
                period=2023,
                value=Decimal("900"),
                table_id=TABLE_ID,
                row_index=14,
            ),
        ),
    )
    facts = grounded_facts(compiled, grounding_score=0.94)
    assert len(facts) == 1
    fact = facts[0]
    assert fact.fact_id == "F1"
    assert fact.table_id == TABLE_ID
    assert fact.row_index == 14
    assert fact.row_label == "Tiền mặt"
    assert fact.column == "Năm 2023"
    assert fact.period == 2023
    assert fact.raw_value == Decimal("900")
    assert fact.unit == "VND"
    assert fact.grounding_score == 0.94


def test_grounded_facts_numbers_every_fact_of_a_two_cell_answer() -> None:
    cells = tuple(
        CellMatch(
            table_id=TABLE_ID,
            cell_ids=("cell_" + character * 64,),
            value=Decimal(value),
            unit="VND",
            period=period,
            period_inferred=False,
            row_index=14,
        )
        for character, value, period in (("a", "100", 2022), ("b", "180", 2023))
    )
    replay = tuple(
        ReplayRow(
            company_code="ACB",
            row_label_canonical=None,
            row_label_raw="Tiền mặt",
            period=period,
            value=Decimal(value),
            table_id=TABLE_ID,
            row_index=14,
        )
        for value, period in (("100", 2022), ("180", 2023))
    )
    facts = grounded_facts(_compiled(cells, replay), grounding_score=None)
    assert [fact.fact_id for fact in facts] == ["F1", "F2"]
    assert [fact.period for fact in facts] == [2022, 2023]
    assert all(fact.grounding_score is None for fact in facts)


def test_grounded_facts_is_empty_for_an_unresolved_compilation() -> None:
    errored = CompiledQuery(
        operation="lookup",
        status="error",
        answer=None,
        unit=None,
        evidence=(),
        pandas_query="df1[...]",
        error_code="metric_not_found",
        error_message="no cell matches metric selector",
    )
    assert grounded_facts(errored, grounding_score=None) == ()


def test_grounded_facts_skips_evidence_that_was_never_pinned_to_a_row() -> None:
    """§9 identity is positional. A cell the locator could not attribute to a
    row index has no fact -- it is reported as absent, never as a fact whose
    provenance happens to be missing."""
    compiled = _compiled(
        (
            CellMatch(
                table_id=TABLE_ID,
                cell_ids=(CELL_ID,),
                value=Decimal("900"),
                unit="VND",
                period=2023,
                period_inferred=False,
            ),
        ),
        (
            ReplayRow(
                company_code="ACB",
                row_label_canonical=None,
                row_label_raw="Tiền mặt",
                period=2023,
                value=Decimal("900"),
            ),
        ),
    )
    assert grounded_facts(compiled, grounding_score=None) == ()
