"""Unit tests for the deterministic Day 10 entity parser."""

from __future__ import annotations

from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import parse_query_entities


def test_bare_ticker_is_extracted() -> None:
    entities = parse_query_entities("Tra cứu tổng tài sản của DBC năm 2023.")
    assert entities.company_codes == ("DBC",)
    assert entities.periods == ("2023",)
    assert entities.metrics == ("total_assets",)
    assert entities.ambiguity == ()


def test_bare_uppercase_word_that_is_not_a_registered_ticker_is_ignored() -> None:
    entities = parse_query_entities("GDP tăng trưởng thế nào trong năm 2023?")
    assert entities.company_codes == ()
    assert "company_missing" in entities.ambiguity


def test_full_legal_name_resolves_to_ticker() -> None:
    entities = parse_query_entities(
        "Tra cứu tổng tài sản của Ngân hàng TMCP Á Châu năm 2023."
    )
    assert entities.company_codes == ("ACB",)


def test_two_distinct_bare_tickers_are_both_kept_without_ambiguity() -> None:
    """Two companies named together is a legitimate multi-entity question,
    not a contradiction — RetrievalFilters treats multiple values as OR."""
    entities = parse_query_entities("So sánh tổng tài sản giữa DBC và KHG năm 2023.")
    assert entities.company_codes == ("DBC", "KHG")
    assert not any(code.startswith("company_") for code in entities.ambiguity)


def test_explicit_ticker_contradicting_a_different_company_name_is_a_conflict() -> None:
    entities = parse_query_entities(
        "Tra cứu tổng tài sản của Ngân hàng TMCP Á Châu (mã CK: KHG) năm 2023."
    )
    assert entities.company_codes == ()
    assert entities.ambiguity == ("company_conflict",)


def test_ticker_embedded_in_another_companys_name_is_not_a_spurious_match() -> None:
    """"FPT" is a registered ticker but is also a substring of FTS's display
    name ("CTCP Chứng khoán FPT"); only FTS should be extracted here."""
    entities = parse_query_entities(
        "Tra cứu lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023."
    )
    assert entities.company_codes == ("FTS",)


def test_missing_company_is_flagged_not_guessed() -> None:
    entities = parse_query_entities("Doanh thu thuần năm 2023 là bao nhiêu?")
    assert entities.company_codes == ()
    assert entities.ambiguity == ("company_missing",)


def test_year_and_quarter_and_date_periods() -> None:
    assert parse_query_entities("Doanh thu thuần của DBC năm 2023 là bao nhiêu?").periods == (
        "2023",
    )
    assert parse_query_entities(
        "Doanh thu thuần của DBC quý IV năm 2023 là bao nhiêu?"
    ).periods == ("2023-Q4",)
    assert parse_query_entities(
        "Doanh thu thuần của DBC quý I năm 2023 là bao nhiêu?"
    ).periods == ("2023-Q1",)
    assert parse_query_entities(
        "Tra cứu doanh thu thuần của DBC tại ngày 31/12/2023."
    ).periods == ("2023-12-31",)


def test_compare_years_yields_both_periods() -> None:
    entities = parse_query_entities(
        "So sánh tổng tài sản của DBC giữa năm 2022 và năm 2023."
    )
    assert entities.periods == ("2022", "2023")


def test_incomplete_quarter_without_year_is_flagged() -> None:
    entities = parse_query_entities("Doanh thu thuần của DBC quý III là bao nhiêu?")
    assert entities.periods == ()
    assert "period_incomplete" in entities.ambiguity


def test_relative_period_is_flagged_not_resolved() -> None:
    entities = parse_query_entities("Doanh thu thuần năm nay của DBC là bao nhiêu?")
    assert entities.periods == ()
    assert entities.ambiguity == ("period_relative_unresolved",)


def test_missing_period_is_flagged() -> None:
    entities = parse_query_entities("Doanh thu thuần của DBC là bao nhiêu?")
    assert entities.periods == ()
    assert entities.ambiguity == ("period_missing",)


def test_longest_metric_alias_wins_over_a_shorter_substring() -> None:
    # "Doanh thu thuần" (net_revenue) is a strict prefix of "Doanh thu thuần
    # về bán hàng và cung cấp dịch vụ" (also net_revenue in this dictionary,
    # but a different, longer alias) — longest-match must consume the whole
    # phrase as one alias, not stop at the shorter one.
    entities = parse_query_entities(
        "Tra cứu doanh thu thuần về bán hàng và cung cấp dịch vụ của DBC năm 2023."
    )
    assert entities.metrics == ("net_revenue",)


def test_unknown_metric_phrase_is_flagged() -> None:
    entities = parse_query_entities("Tra cứu tổng lợi thế cạnh tranh của DBC năm 2023.")
    assert entities.metrics == ()
    assert entities.ambiguity == ("metric_unknown",)


def test_statement_type_is_only_set_when_named_explicitly() -> None:
    without_statement = parse_query_entities("Tra cứu tổng tài sản của DBC năm 2023.")
    assert without_statement.statement_types == ()
    assert without_statement.ambiguity == ()

    with_statement = parse_query_entities(
        "Tra cứu tổng tài sản trên bảng cân đối kế toán của DBC năm 2023."
    )
    assert with_statement.statement_types == ("balance_sheet",)


def test_spans_point_back_at_the_literal_question_text() -> None:
    question = "Tra cứu tổng tài sản của DBC năm 2023."
    entities = parse_query_entities(question)
    assert entities.spans
    for span in entities.spans:
        assert question[span.start : span.end] == span.surface


def test_to_retrieval_filters_drops_ambiguous_fields() -> None:
    entities = parse_query_entities("Doanh thu thuần năm nay của DBC là bao nhiêu?")
    filters = to_retrieval_filters(entities)
    assert filters.company_codes == ("DBC",)
    assert filters.periods == ()  # period_relative_unresolved must not leak a guess
