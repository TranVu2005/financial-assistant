"""Unit tests for the Day 17 LLM-facing plan contracts (§17.5).

`LLMPlanOutput` is `FinancialQueryPlan` minus `candidate_table_ids` (ADR 0006
decision B1 + Day 17 plan §1.5: the LLM must never be asked to produce a
64-hex-char table id itself; the caller injects `candidate_table_ids` after
the fact).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from financial_report_qa.planning.llm_contracts import LLMPlanOutput, to_financial_query_plan
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

_TABLE_IDS = ("tbl_" + "a" * 64,)


def test_llm_plan_output_has_no_candidate_table_ids_field() -> None:
    with pytest.raises(ValidationError):
        LLMPlanOutput.model_validate(
            {
                "operation": "lookup",
                "companies": ["NVL"],
                "periods": ["2023"],
                "metric": {"canonical": "net_revenue"},
                "candidate_table_ids": [_TABLE_IDS[0]],
            }
        )


def test_to_financial_query_plan_injects_candidate_table_ids() -> None:
    output = LLMPlanOutput(
        operation="lookup",
        companies=("NVL",),
        periods=("2023",),
        metric=MetricSelector(canonical="net_revenue"),
    )

    plan = to_financial_query_plan(output, candidate_table_ids=_TABLE_IDS)

    assert isinstance(plan, FinancialQueryPlan)
    assert plan.candidate_table_ids == _TABLE_IDS
    assert plan.operation == "lookup"
    assert plan.companies == ("NVL",)
    assert plan.metric == MetricSelector(canonical="net_revenue")


def test_to_financial_query_plan_propagates_structural_errors() -> None:
    """An LLM output with duplicate companies is structurally invalid; the
    caller (llm_planner) must see the same `ValueError` it would from a
    hand-built `FinancialQueryPlan`, not a swallowed/different error type."""
    output = LLMPlanOutput(
        operation="compare_companies",
        companies=("NVL", "NVL"),
        periods=("2023",),
        metric=MetricSelector(canonical="net_revenue"),
    )

    with pytest.raises(ValueError, match="duplicates"):
        to_financial_query_plan(output, candidate_table_ids=_TABLE_IDS)
