"""Tests for the plan.md §12 `EvidencePlan` contract.

§12's whole point is that the planner's output shrinks to `{operation,
operands}` over facts grounding already resolved. The contract therefore has
to make it structurally impossible to name anything else -- no table, no row
locator, no column, no metric string.
"""

import pytest
from pydantic import ValidationError

from financial_report_qa.planning.evidence_plan_contracts import EvidencePlan


def test_evidence_plan_carries_only_an_operation_and_fact_ids() -> None:
    plan = EvidencePlan(operation="growth_rate", operands=("F1", "F2"))
    assert plan.operation == "growth_rate"
    assert plan.operands == ("F1", "F2")


def test_evidence_plan_forbids_naming_a_table_or_row_locator() -> None:
    """plan.md §12: `table_name`, `row locator`, `column locator` and
    `metric name` are precisely what the planner must no longer invent."""
    with pytest.raises(ValidationError):
        EvidencePlan.model_validate(
            {"operation": "lookup", "operands": ("F1",), "table_id": "tbl_" + "1" * 64}
        )


def test_evidence_plan_requires_at_least_one_operand() -> None:
    with pytest.raises(ValidationError):
        EvidencePlan(operation="lookup", operands=())


def test_evidence_plan_rejects_duplicate_operands() -> None:
    """The same fact twice is never a real two-operand computation -- a
    growth rate against itself is 0, a ratio against itself is 1."""
    with pytest.raises(ValidationError):
        EvidencePlan(operation="growth_rate", operands=("F1", "F1"))


def test_evidence_plan_rejects_malformed_fact_ids() -> None:
    with pytest.raises(ValidationError):
        EvidencePlan(operation="lookup", operands=("row 14",))


def test_evidence_plan_rejects_an_unknown_operation() -> None:
    with pytest.raises(ValidationError):
        EvidencePlan(operation="cagr", operands=("F1", "F2"))  # type: ignore[arg-type]


def test_evidence_plan_is_frozen() -> None:
    plan = EvidencePlan(operation="lookup", operands=("F1",))
    with pytest.raises(ValidationError):
        plan.operands = ("F2",)  # type: ignore[misc]
