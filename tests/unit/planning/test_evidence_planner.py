"""Tests for plan.md §12: turning `{operation, operands}` back into a plan.

The planner names facts; this module turns that naming into an executable
`FinancialQueryPlan` whose selectors are already position-bound (§14), or
rejects it with a typed reason. Nothing here consults the question text --
every decision is a consistency check between the chosen facts.
"""

from decimal import Decimal

import pytest

from financial_report_qa.planning.evidence_plan_contracts import EvidencePlan
from financial_report_qa.planning.evidence_planner import build_plan_from_facts
from financial_report_qa.planning.grounding_contracts import GroundedFact

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_MBB = "tbl_" + "2" * 64


def _fact(
    fact_id: str,
    *,
    row_index: int = 14,
    row_label: str = "Doanh thu thuan",
    period: int = 2023,
    value: str = "63075",
    column: str | None = "Năm 2023",
    company_code: str = "ACB",
    table_id: str = TABLE_ID,
) -> GroundedFact:
    return GroundedFact(
        fact_id=fact_id,
        table_id=table_id,
        row_index=row_index,
        row_label=row_label,
        column=column,
        company_code=company_code,
        period=period,
        raw_value=Decimal(value),
        unit="VND",
        grounding_score=1.0,
    )


def test_lookup_builds_a_position_bound_plan_from_one_fact() -> None:
    facts = (_fact("F1"),)
    built = build_plan_from_facts(EvidencePlan(operation="lookup", operands=("F1",)), facts)
    assert built.reject_code is None
    plan = built.plan
    assert plan is not None
    assert plan.operation == "lookup"
    assert plan.companies == ("ACB",)
    assert plan.periods == ("2023",)
    assert plan.candidate_table_ids == (TABLE_ID,)
    assert plan.metric is not None
    assert plan.metric.is_position_bound
    assert plan.metric.table_id == TABLE_ID
    assert plan.metric.row_index == 14
    assert plan.metric.raw_text == "Doanh thu thuan"
    assert plan.metric.column_text == "Năm 2023"


def test_growth_rate_builds_a_two_period_plan_over_one_row() -> None:
    """§12's own worked example: two facts of the same row at two periods."""
    facts = (_fact("F1"), _fact("F2", period=2022, value="60180", column="Năm 2022"))
    built = build_plan_from_facts(
        EvidencePlan(operation="growth_rate", operands=("F1", "F2")), facts
    )
    assert built.reject_code is None
    plan = built.plan
    assert plan is not None
    assert plan.operation == "growth_rate"
    # Chronological, as `plan_validator` requires for a directional operation.
    assert plan.periods == ("2022", "2023")
    assert plan.metric is not None and plan.metric.row_index == 14


def test_growth_rate_accepts_operands_listed_oldest_first() -> None:
    """The compiler always computes later-minus-earlier, which is what every
    "tăng bao nhiêu %" question means, so operand order carries no extra
    information worth rejecting a usable plan over."""
    facts = (_fact("F1", period=2022, value="60180", column="Năm 2022"), _fact("F2"))
    built = build_plan_from_facts(
        EvidencePlan(operation="growth_rate", operands=("F1", "F2")), facts
    )
    assert built.reject_code is None
    assert built.plan is not None and built.plan.periods == ("2022", "2023")


def test_two_period_plan_drops_a_column_the_operands_disagree_on() -> None:
    """One selector spans both periods, so a column predicate that is only
    true for one of them would make the other cell unfindable."""
    facts = (_fact("F1"), _fact("F2", period=2022, value="60180", column="Năm 2022"))
    built = build_plan_from_facts(
        EvidencePlan(operation="difference", operands=("F1", "F2")), facts
    )
    assert built.plan is not None
    assert built.plan.metric is not None
    assert built.plan.metric.column_text is None


def test_two_period_plan_keeps_a_column_the_operands_share() -> None:
    facts = (
        _fact("F1", column="So cuoi nam"),
        _fact("F2", period=2022, value="60180", column="So cuoi nam"),
    )
    built = build_plan_from_facts(
        EvidencePlan(operation="difference", operands=("F1", "F2")), facts
    )
    assert built.plan is not None
    assert built.plan.metric is not None
    assert built.plan.metric.column_text == "So cuoi nam"


def test_ratio_maps_the_operands_to_numerator_and_denominator_in_order() -> None:
    facts = (
        _fact("F1", row_index=14, row_label="Loi nhuan sau thue"),
        _fact("F2", row_index=40, row_label="Tong tai san", value="900000"),
    )
    built = build_plan_from_facts(EvidencePlan(operation="ratio", operands=("F1", "F2")), facts)
    assert built.reject_code is None
    plan = built.plan
    assert plan is not None
    assert plan.numerator_metric is not None and plan.numerator_metric.row_index == 14
    assert plan.denominator_metric is not None and plan.denominator_metric.row_index == 40
    assert plan.metric is None


def test_compare_maps_the_operands_to_metric_a_and_metric_b() -> None:
    facts = (
        _fact("F1", row_index=14, row_label="Doanh thu thuan"),
        _fact("F2", row_index=20, row_label="Gia von hang ban", value="41000"),
    )
    built = build_plan_from_facts(EvidencePlan(operation="compare", operands=("F1", "F2")), facts)
    assert built.reject_code is None
    plan = built.plan
    assert plan is not None
    assert plan.metric_a is not None and plan.metric_a.row_index == 14
    assert plan.metric_b is not None and plan.metric_b.row_index == 20


def test_sum_builds_an_aggregate_over_the_operands_periods() -> None:
    facts = (
        _fact("F1", period=2021, value="1", column="Năm 2021"),
        _fact("F2", period=2022, value="2", column="Năm 2022"),
        _fact("F3", period=2023, value="3", column="Năm 2023"),
    )
    built = build_plan_from_facts(EvidencePlan(operation="sum", operands=("F1", "F2", "F3")), facts)
    assert built.reject_code is None
    assert built.plan is not None
    assert built.plan.periods == ("2021", "2022", "2023")
    assert built.plan.companies == ("ACB",)


def test_unknown_operand_is_rejected_not_guessed() -> None:
    """A hallucinated fact id is the one failure mode this design still has
    to catch, and it must never resolve to "the nearest fact"."""
    built = build_plan_from_facts(
        EvidencePlan(operation="lookup", operands=("F9",)), (_fact("F1"),)
    )
    assert built.plan is None
    assert built.reject_code == "operand_unknown"


@pytest.mark.parametrize(
    ("operation", "operands"),
    [("lookup", ("F1", "F2")), ("growth_rate", ("F1",)), ("ratio", ("F1",))],
)
def test_wrong_operand_count_for_the_operation_is_rejected(
    operation: str, operands: tuple[str, ...]
) -> None:
    facts = (_fact("F1"), _fact("F2", period=2022, value="60180", column="Năm 2022"))
    built = build_plan_from_facts(
        EvidencePlan.model_validate({"operation": operation, "operands": operands}), facts
    )
    assert built.plan is None
    assert built.reject_code == "operand_arity_invalid"


def test_operands_from_different_companies_are_rejected() -> None:
    facts = (_fact("F1"), _fact("F2", company_code="MBB", table_id=TABLE_ID_MBB, value="99999"))
    built = build_plan_from_facts(EvidencePlan(operation="compare", operands=("F1", "F2")), facts)
    assert built.plan is None
    assert built.reject_code == "operands_company_mismatch"


def test_growth_rate_over_two_different_rows_is_rejected() -> None:
    """A growth rate compares one line item against itself over time. Two
    different rows is a different question the operation cannot express."""
    facts = (
        _fact("F1"),
        _fact("F2", row_index=40, row_label="Tong tai san", period=2022, column="Năm 2022"),
    )
    built = build_plan_from_facts(
        EvidencePlan(operation="growth_rate", operands=("F1", "F2")), facts
    )
    assert built.plan is None
    assert built.reject_code == "operands_row_mismatch"


def test_growth_rate_over_one_period_is_rejected() -> None:
    facts = (_fact("F1", column="So dau nam"), _fact("F2", column="So cuoi nam", value="70000"))
    built = build_plan_from_facts(
        EvidencePlan(operation="growth_rate", operands=("F1", "F2")), facts
    )
    assert built.plan is None
    assert built.reject_code == "operands_period_mismatch"


def test_compare_across_two_periods_is_rejected() -> None:
    facts = (
        _fact("F1", row_index=14),
        _fact("F2", row_index=20, period=2022, column="Năm 2022", value="41000"),
    )
    built = build_plan_from_facts(EvidencePlan(operation="compare", operands=("F1", "F2")), facts)
    assert built.plan is None
    assert built.reject_code == "operands_period_mismatch"


@pytest.mark.parametrize("operation", ["compare_companies", "rank"])
def test_multi_company_operations_are_out_of_scope_for_evidence_planning(
    operation: str,
) -> None:
    """plan.md §14: one `MetricSelector` serves every company of these
    operations, so it cannot be pinned to a single physical row. Rejecting
    them keeps the caller on the existing planner rather than emitting a
    plan whose grounding regime is silently different."""
    facts = (_fact("F1"), _fact("F2", company_code="MBB", table_id=TABLE_ID_MBB, value="99999"))
    built = build_plan_from_facts(
        EvidencePlan.model_validate({"operation": operation, "operands": ("F1", "F2")}), facts
    )
    assert built.plan is None
    assert built.reject_code == "operation_unsupported"


def test_built_plan_passes_the_existing_semantic_validator() -> None:
    """The whole design rests on this: an evidence plan becomes an ordinary
    `FinancialQueryPlan`, so every downstream check still applies unchanged."""
    from financial_report_qa.planning.plan_validator import validate_plan_semantics

    facts = (_fact("F1"), _fact("F2", period=2022, value="60180", column="Năm 2022"))
    for operation, operands in (
        ("lookup", ("F1",)),
        ("growth_rate", ("F1", "F2")),
        ("difference", ("F1", "F2")),
    ):
        built = build_plan_from_facts(
            EvidencePlan.model_validate({"operation": operation, "operands": operands}), facts
        )
        assert built.plan is not None, operation
        issues = validate_plan_semantics(
            built.plan, known_table_ids=frozenset(built.plan.candidate_table_ids)
        )
        assert issues == (), (operation, issues)


def test_expected_unit_is_carried_through_when_the_caller_knows_it() -> None:
    built = build_plan_from_facts(
        EvidencePlan(operation="lookup", operands=("F1",)),
        (_fact("F1"),),
        expected_unit="VND_billion",
    )
    assert built.plan is not None and built.plan.expected_unit == "VND_billion"
