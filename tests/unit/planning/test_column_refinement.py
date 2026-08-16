"""Tests for the column-refinement retry (Day 26)."""

from __future__ import annotations

from financial_report_qa.planning.column_refinement import plan_with_column
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

TABLE_A = "tbl_" + "a" * 64


def _plan() -> FinancialQueryPlan:
    return FinancialQueryPlan.model_validate(
        {
            "operation": "lookup",
            "companies": ("PC1",),
            "periods": ("2025",),
            "candidate_table_ids": (TABLE_A,),
            "metric": MetricSelector(raw_text="Thuế giá trị gia tăng"),
        }
    )


def test_plan_with_column_sets_the_chosen_column_on_the_metric_selector() -> None:
    """The retry the locator needs: same plan, one dimension added."""
    result = plan_with_column(_plan(), lambda row_label, columns: "Số phải nộpcuối năm")
    assert result is not None
    assert result.metric is not None
    assert result.metric.column_text == "Số phải nộpcuối năm"
    assert result.metric.raw_text == "Thuế giá trị gia tăng"
    assert result.operation == "lookup"
    assert result.companies == ("PC1",)


def test_plan_with_column_returns_none_when_no_column_is_chosen() -> None:
    """Declining must leave the original `cell_ambiguous` standing, not
    produce a plan filtered on a column nobody picked."""
    assert plan_with_column(_plan(), lambda row_label, columns: None) is None


def test_plan_with_column_returns_none_when_a_column_is_already_set() -> None:
    """One bounded retry only -- never a loop that keeps re-narrowing."""
    plan = _plan().model_copy(
        update={
            "metric": MetricSelector(raw_text="Thuế giá trị gia tăng", column_text="Số cuối năm")
        }
    )
    assert plan_with_column(plan, lambda row_label, columns: "Số phải nộpcuối năm") is None


def test_plan_with_column_passes_the_row_label_to_the_chooser() -> None:
    """The chooser needs the row to scope its column menu."""
    seen: list[str] = []

    def chooser(row_label: str, columns: tuple[str, ...]) -> str | None:
        seen.append(row_label)
        return None

    plan_with_column(_plan(), chooser)
    assert seen == ["Thuế giá trị gia tăng"]
