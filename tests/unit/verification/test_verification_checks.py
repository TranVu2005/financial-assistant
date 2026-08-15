"""Tests for the Day 20 verification checks (ADR 0009 decisions B1/D1/E1;
Day 20 plan Sec 3 task 20.5).

Each check is a pure function: (plan and/or compiled query and/or display
fields) -> `VerificationIssue | None`. `None` means the check passed.
"""

from __future__ import annotations

from decimal import Decimal

from financial_report_qa.execution.contracts import CellMatch, CompiledQuery
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.verification.checks import (
    check_display_roundtrip_mismatch,
    check_evidence_outside_retrieval,
    check_period_inferred_warning,
    check_recompute_mismatch,
    check_unit_not_presentable,
)

TABLE_ID = "tbl_" + "1" * 64
TABLE_ID_2 = "tbl_" + "2" * 64


def _cell(**overrides: object) -> CellMatch:
    defaults: dict[str, object] = {
        "table_id": TABLE_ID,
        "cell_ids": ("cell_" + "a" * 64,),
        "value": Decimal("100"),
        "unit": "VND",
        "period": 2023,
        "period_inferred": False,
    }
    defaults.update(overrides)
    return CellMatch.model_validate(defaults)


def _plan(**overrides: object) -> FinancialQueryPlan:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "companies": ("ACB",),
        "periods": ("2023",),
        "candidate_table_ids": (TABLE_ID,),
        "metric": MetricSelector(canonical="cash_and_cash_equivalents"),
    }
    defaults.update(overrides)
    return FinancialQueryPlan(**defaults)  # type: ignore[arg-type]


def _compiled(**overrides: object) -> CompiledQuery:
    defaults: dict[str, object] = {
        "operation": "lookup",
        "status": "answered",
        "answer": Decimal("100"),
        "unit": "VND",
        "evidence": (_cell(),),
        "pandas_query": 'df1[(df1.period == 2023)]["value"].iloc[0]',
        "error_code": None,
        "error_message": None,
    }
    defaults.update(overrides)
    return CompiledQuery.model_validate(defaults)


# --- check_recompute_mismatch -------------------------------------------------


def test_recompute_mismatch_none_when_answer_matches_lookup() -> None:
    plan = _plan()
    compiled = _compiled()
    assert check_recompute_mismatch(plan, compiled) is None


def test_recompute_mismatch_flags_wrong_answer() -> None:
    plan = _plan()
    compiled = _compiled(answer=Decimal("999"))
    issue = check_recompute_mismatch(plan, compiled)
    assert issue is not None
    assert issue.code == "recompute_mismatch"


def test_recompute_mismatch_recomputes_difference_correctly() -> None:
    start = _cell(value=Decimal("2"), unit="VND_million", period=2022)
    end = _cell(value=Decimal("1000000"), unit="VND", period=2023)
    plan = _plan(
        operation="difference",
        periods=("2022", "2023"),
    )
    compiled = _compiled(
        operation="difference", evidence=(start, end), answer=Decimal("-1000000"), unit="VND"
    )
    assert check_recompute_mismatch(plan, compiled) is None


def test_recompute_mismatch_flags_growth_rate_division_by_zero_evidence() -> None:
    """A recompute that itself raises (e.g. zero base) is a mismatch, not a
    crash -- the check must fail closed."""
    start = _cell(value=Decimal("0"), period=2022)
    end = _cell(value=Decimal("100"), period=2023)
    plan = _plan(operation="growth_rate", periods=("2022", "2023"))
    compiled = _compiled(
        operation="growth_rate", evidence=(start, end), answer=Decimal("1"), unit="ratio"
    )
    issue = check_recompute_mismatch(plan, compiled)
    assert issue is not None
    assert issue.code == "recompute_mismatch"


def test_recompute_mismatch_none_for_error_status() -> None:
    plan = _plan()
    compiled = CompiledQuery.model_validate(
        {
            "operation": "lookup",
            "status": "error",
            "answer": None,
            "unit": None,
            "evidence": (),
            "pandas_query": "<plan rejected before rendering>",
            "error_code": "metric_not_found",
            "error_message": "no match",
        }
    )
    assert check_recompute_mismatch(plan, compiled) is None


# --- check_unit_not_presentable ------------------------------------------------


def test_unit_not_presentable_none_when_expected_unit_unset() -> None:
    plan = _plan()
    compiled = _compiled()
    assert check_unit_not_presentable(plan, compiled) is None


def test_unit_not_presentable_accepts_ratio_and_percent_as_equivalent() -> None:
    """ADR 0009 decision B1: 'ratio' (computed) and 'percent' (presented) are
    the same underlying unit -- either declaration is presentable."""
    plan = _plan(
        operation="growth_rate",
        periods=("2022", "2023"),
        expected_unit="percent",
    )
    compiled = _compiled(operation="growth_rate", unit="ratio", answer=Decimal("0.05"))
    assert check_unit_not_presentable(plan, compiled) is None


def test_unit_not_presentable_flags_incompatible_declaration() -> None:
    plan = _plan(operation="lookup", expected_unit="VND_million")
    compiled = _compiled(unit="percent", answer=Decimal("5"))
    issue = check_unit_not_presentable(plan, compiled)
    assert issue is not None
    assert issue.code == "unit_not_presentable"


# --- check_evidence_outside_retrieval ------------------------------------------


def test_evidence_outside_retrieval_none_when_all_evidence_inside() -> None:
    compiled = _compiled()
    assert check_evidence_outside_retrieval(compiled, frozenset({TABLE_ID})) is None


def test_evidence_outside_retrieval_flags_table_not_retrieved() -> None:
    compiled = _compiled(evidence=(_cell(table_id=TABLE_ID_2),))
    issue = check_evidence_outside_retrieval(compiled, frozenset({TABLE_ID}))
    assert issue is not None
    assert issue.code == "evidence_outside_retrieval"
    assert TABLE_ID_2 in issue.message


# --- check_display_roundtrip_mismatch -------------------------------------------


def test_display_roundtrip_mismatch_none_when_display_matches() -> None:
    issue = check_display_roundtrip_mismatch(Decimal("100"), "100 VND", display_precision=0)
    assert issue is None


def test_display_roundtrip_mismatch_none_for_multi_group_thousands_separator() -> None:
    """Real end-to-end run on gold70 (Day 20 task 20.10) found this: a naive
    number regex stops after the first comma group, so '84,420,878 VND'
    parses as 84,420 instead of 84420878 -- 17/30 answered results were
    wrongly rejected before this fix."""
    issue = check_display_roundtrip_mismatch(
        Decimal("84420878"), "84,420,878 VND", display_precision=0
    )
    assert issue is None


def test_display_roundtrip_mismatch_flags_wrong_display_value() -> None:
    issue = check_display_roundtrip_mismatch(Decimal("100"), "999 VND", display_precision=0)
    assert issue is not None
    assert issue.code == "display_roundtrip_mismatch"


def test_display_roundtrip_mismatch_within_declared_precision() -> None:
    """Day 20 plan Sec 1.4: rounding a 28-digit ratio to 4 places loses
    ~2.8e-5 -- that loss must be within tolerance, not flagged."""
    raw = Decimal("-0.01932846513079090948136258551")
    rounded = round(raw, 4)
    issue = check_display_roundtrip_mismatch(raw, f"{rounded}", display_precision=4)
    assert issue is None


def test_display_roundtrip_mismatch_flags_unparseable_display() -> None:
    issue = check_display_roundtrip_mismatch(Decimal("100"), "no number here", display_precision=0)
    assert issue is not None
    assert issue.code == "display_roundtrip_mismatch"


# --- check_period_inferred_warning ----------------------------------------------


def test_period_inferred_warning_none_when_no_evidence_inferred() -> None:
    compiled = _compiled()
    assert check_period_inferred_warning(compiled) is None


def test_period_inferred_warning_flags_inferred_evidence() -> None:
    compiled = _compiled(evidence=(_cell(period_inferred=True),))
    issue = check_period_inferred_warning(compiled)
    assert issue is not None
    assert issue.code == "period_inferred_warning"
