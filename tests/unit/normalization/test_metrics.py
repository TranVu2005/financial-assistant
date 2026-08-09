import pytest

from financial_report_qa.normalization.metrics import (
    DERIVED_METRICS,
    METRIC_ALIASES,
    SOURCE_METRICS_BY_STATEMENT,
    is_non_metric_label,
    normalize_metric,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Doanh thu bán hàng và cung cấp dịch vụ", "revenue"),
        ("  LNST  ", "profit_after_tax"),
        ("Lợi nhuận thuần từ HĐKD", "operating_profit"),
        ("Chi phí thuế TNDN hoãn lại", "deferred_income_tax_expense"),
        ("Basic EPS", "basic_eps"),
        ("Tổng tài sản", "total_assets"),
        ("TSCĐ hữu hình", "property_plant_equipment"),
        ("VCSH", "equity"),
        ("Tổng nguồn vốn", "total_equity_and_liabilities"),
        ("Tiền thuần từ hoạt động đầu tư", "investing_cash_flow"),
        ("Thuế TNDN đã nộp", "income_tax_paid"),
        ("Capital expenditure", "capital_expenditure"),
    ],
)
def test_normalize_metric_supports_core_financial_statement_aliases(
    raw: str, canonical: str
) -> None:
    decision = normalize_metric(raw)

    assert decision.value == canonical
    assert decision.issue_code is None


def test_every_registered_alias_resolves_to_its_declared_metric() -> None:
    for alias, canonical in METRIC_ALIASES.items():
        decision = normalize_metric(alias)
        assert decision.value == canonical, alias
        assert decision.issue_code is None, alias


@pytest.mark.parametrize("raw", [None, "", "  ", "Chỉ tiêu", "Mã số", "Tổng cộng"])
def test_normalize_metric_ignores_structural_labels(raw: str | None) -> None:
    decision = normalize_metric(raw)

    assert decision.value is None
    assert decision.issue_code is None


@pytest.mark.parametrize(
    "raw",
    ["Doanh thu", "Tiền", "EBIT", "EBITDA", "ROE", "Biên lợi nhuận ròng"],
)
def test_normalize_metric_does_not_guess_ambiguous_or_derived_labels(raw: str) -> None:
    decision = normalize_metric(raw)

    assert decision.value is None
    assert decision.issue_code == "metric_unknown"


def test_derived_metrics_are_separate_from_source_aliases() -> None:
    source_metrics = set().union(*SOURCE_METRICS_BY_STATEMENT.values())

    assert {"ebit", "ebitda", "return_on_equity", "free_cash_flow"} <= DERIVED_METRICS
    assert DERIVED_METRICS.isdisjoint(source_metrics)
    assert DERIVED_METRICS.isdisjoint(METRIC_ALIASES.values())


def test_source_taxonomy_contains_all_three_primary_statements() -> None:
    assert set(SOURCE_METRICS_BY_STATEMENT) == {
        "income_statement",
        "balance_sheet",
        "cash_flow_statement",
    }
    assert all(SOURCE_METRICS_BY_STATEMENT.values())


def test_total_sources_keep_distinct_accounting_provenance() -> None:
    assets = normalize_metric("Tổng tài sản")
    funding = normalize_metric("Tổng nguồn vốn")

    assert assets.value == "total_assets"
    assert funding.value == "total_equity_and_liabilities"
    assert assets.value != funding.value


@pytest.mark.parametrize(
    "raw",
    [
        "Năm hiện hành",
        "2.",
        "5.",
        "3. Cam kết cho vay không hủy ngang",
        "6. Outstanding shares",
    ],
)
def test_reviewed_structural_rows_are_not_metrics(raw: str) -> None:
    assert is_non_metric_label(raw) is True
    assert normalize_metric(raw).issue_code is None


def test_ordinal_prefix_does_not_hide_supported_metric() -> None:
    assert is_non_metric_label("1. Doanh thu thuần") is False
