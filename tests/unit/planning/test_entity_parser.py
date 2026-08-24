"""Unit tests for the deterministic Day 10 entity parser."""

from __future__ import annotations

from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import (
    ordered_metric_canonicals,
    parse_query_entities,
)


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
    entities = parse_query_entities("Tra cứu tổng tài sản của Ngân hàng TMCP Á Châu năm 2023.")
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
    """ "FPT" is a registered ticker but is also a substring of FTS's display
    name ("CTCP Chứng khoán FPT"); only FTS should be extracted here."""
    entities = parse_query_entities("Tra cứu lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023.")
    assert entities.company_codes == ("FTS",)


def test_missing_company_is_flagged_not_guessed() -> None:
    entities = parse_query_entities("Doanh thu thuần năm 2023 là bao nhiêu?")
    assert entities.company_codes == ()
    assert entities.ambiguity == ("company_missing",)


def test_year_and_quarter_phrasings_keep_their_existing_grammar() -> None:
    """Spec §6.4 normalizes date phrasings to bare fiscal years only — plain
    years and quarters keep the exact forms extracted today: "2023-Q4" stays
    non-bare (only bare years pass the downstream period grammar), which
    documents the still-unsupported quarter grammar."""
    assert parse_query_entities("Doanh thu thuần của DBC năm 2023 là bao nhiêu?").periods == (
        "2023",
    )
    assert parse_query_entities(
        "Doanh thu thuần của DBC quý IV năm 2023 là bao nhiêu?"
    ).periods == ("2023-Q4",)
    assert parse_query_entities("Doanh thu thuần của DBC quý I năm 2023 là bao nhiêu?").periods == (
        "2023-Q1",
    )


def test_date_phrasing_is_normalized_to_the_bare_fiscal_year() -> None:
    """Spec §6.4: a full date ("tại ngày 31/12/2023.") names the fiscal-year
    period, so the parser emits the bare year instead of the ISO form that
    `period_grammar_unsupported` rejects downstream."""
    assert parse_query_entities("Tra cứu doanh thu thuần của DBC tại ngày 31/12/2023.").periods == (
        "2023",
    )


def test_compare_years_yields_both_periods() -> None:
    entities = parse_query_entities("So sánh tổng tài sản của DBC giữa năm 2022 và năm 2023.")
    assert entities.periods == ("2022", "2023")


def test_bare_trailing_year_after_connector_is_captured() -> None:
    """ "giữa năm 2016 và 2017" must not silently drop the second, bare year."""
    entities = parse_query_entities("So sánh lưu chuyển tiền thuần của CTG giữa năm 2022 và 2023.")
    assert entities.periods == ("2022", "2023")


def test_bare_trailing_year_after_den_is_captured() -> None:
    entities = parse_query_entities(
        "Tính tốc độ tăng trưởng doanh thu thuần của NVL từ năm 2022 đến 2023."
    )
    assert entities.periods == ("2022", "2023")


def test_bare_year_inside_ticker_or_note_number_is_not_captured() -> None:
    """A bare year must only resolve when it follows a period connector —
    not any digit sequence that happens to look like a year (e.g. a note
    reference or an amount)."""
    entities = parse_query_entities("Thuyết minh số 2023 của DBC năm 2022 là gì?")
    assert entities.periods == ("2022",)


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
    assert entities.metric_phrases == ("tổng lợi thế cạnh tranh",)
    assert entities.ambiguity == ("metric_unknown",)


def test_banking_metric_not_in_normalization_alias_table_is_still_resolved() -> None:
    """`cho vay khách hàng` (loans to customers) has 0% canonical coverage in
    `normalization/metrics.py` (ADR 0004 §1.6) — the release cannot be touched
    (would change dataset_fingerprint) so this must resolve via a question-side
    lexicon, not `METRIC_ALIASES`."""
    entities = parse_query_entities("Tra cứu cho vay khách hàng của STB tại cuối năm 2024.")
    assert "loans_to_customers" in entities.metrics
    assert "metric_unknown" not in entities.ambiguity


def test_common_abbreviation_lctt_is_resolved() -> None:
    entities = parse_query_entities("So sánh LCTT trực tiếp của MBB giữa năm 2016 và 2017.")
    assert "net_cash_flow" in entities.metrics
    assert "metric_unknown" not in entities.ambiguity


def test_question_side_metric_lexicon_does_not_alter_normalization_alias_table() -> None:
    """Guards ADR 0004 §1.7: extending question-side vocabulary must never grow
    `METRIC_ALIASES`, since that table is baked into the locked release."""
    from financial_report_qa.normalization.metrics import METRIC_ALIASES

    assert "cho vay khách hàng" not in METRIC_ALIASES
    assert "lctt" not in METRIC_ALIASES


def test_ordered_metric_canonicals_preserves_reading_order_for_ratio_phrasing() -> None:
    """`entities.metrics` is sorted (contract invariant, `_canonical_tuple`),
    which loses which metric was named first -- load-bearing for `ratio`,
    where "A trên B" means A/B, not B/A. Day 23 plan Step 2: measured 25/26
    real 2-metric questions are this exact 'X trên/trong Y' ratio shape."""
    entities = parse_query_entities(
        "Lợi nhuận sau thuế trên tổng tài sản cuối năm 2016 của CEO là bao nhiêu phần trăm?"
    )
    assert entities.metrics == ("profit_after_tax", "total_assets")  # sorted, order lost
    assert ordered_metric_canonicals(entities) == ("profit_after_tax", "total_assets")


def test_ordered_metric_canonicals_reverses_when_denominator_named_first() -> None:
    entities = parse_query_entities(
        "Tổng tài sản trên lợi nhuận sau thuế cuối năm 2016 của CEO là bao nhiêu lần?"
    )
    assert ordered_metric_canonicals(entities) == ("total_assets", "profit_after_tax")


def test_statement_type_is_only_set_when_named_explicitly() -> None:
    without_statement = parse_query_entities("Tra cứu tổng tài sản của DBC năm 2023.")
    assert without_statement.statement_types == ()
    assert without_statement.ambiguity == ()

    with_statement = parse_query_entities(
        "Tra cứu tổng tài sản trên bảng cân đối kế toán của DBC năm 2023."
    )
    assert with_statement.statement_types == ("balance_sheet",)


def test_statement_scope_is_unstated_when_not_mentioned() -> None:
    """Day 21 plan §1.6: 62.3% of official ViFinQA questions state neither
    scope keyword -- must remain None, not a guess."""
    entities = parse_query_entities("Tra cứu tổng tài sản của DBC năm 2023.")
    assert entities.statement_scope is None
    assert entities.ambiguity == ()


def test_statement_scope_resolves_separate_from_cong_ty_me() -> None:
    """Real corpus example (questions.jsonl id=1): 'công ty mẹ' is the most
    common scope phrase (36.4% of official questions), meaning `separate`."""
    entities = parse_query_entities(
        "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu?"
    )
    assert entities.statement_scope == "separate"


def test_statement_scope_resolves_separate_from_rieng() -> None:
    entities = parse_query_entities("Tổng tài sản riêng của GEG năm 2022 là bao nhiêu?")
    assert entities.statement_scope == "separate"


def test_statement_scope_resolves_consolidated_from_hop_nhat() -> None:
    entities = parse_query_entities("Tổng tài sản hợp nhất của ACB năm 2023 là bao nhiêu?")
    assert entities.statement_scope == "consolidated"


def test_statement_scope_resolves_consolidated_from_toan_tap_doan() -> None:
    entities = parse_query_entities("Doanh thu toàn tập đoàn của FIT năm 2024 là bao nhiêu?")
    assert entities.statement_scope == "consolidated"


def test_statement_scope_both_keywords_left_unstated_not_guessed() -> None:
    """A question naming both scopes wants a cross-scope comparison the
    compiler does not support yet (Day 21 plan §1.6: 1/1012 official
    questions) -- fall through to `unstated` (visible via `scope_inferred`
    downstream) rather than silently pick one."""
    entities = parse_query_entities("So sánh doanh thu riêng và hợp nhất của MPC năm 2017.")
    assert entities.statement_scope is None


def test_statement_scope_span_points_back_at_literal_text() -> None:
    question = "Tổng tài sản hợp nhất của ACB năm 2023 là bao nhiêu?"
    entities = parse_query_entities(question)
    scope_spans = [span for span in entities.spans if span.field == "statement_scope"]
    assert len(scope_spans) == 1
    assert question[scope_spans[0].start : scope_spans[0].end] == scope_spans[0].surface


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


def test_fiscal_year_phrasing_resolves_the_period() -> None:
    """Day 24 §4.1: `_YEAR_RE` required `năm` immediately before the digits,
    so the extremely common "năm tài chính YYYY" phrasing resolved to no
    period at all. Measured: 61 questions carry a period ambiguity and 60 of
    them have a 4-digit year sitting right there in the text."""
    entities = parse_query_entities(
        "Giá trị thuần phải thu ngắn hạn của HSG là bao nhiêu vào cuối năm tài chính 2017?"
    )
    assert entities.periods == ("2017",)
    assert "period_missing" not in entities.ambiguity


def test_plain_year_phrasing_still_resolves() -> None:
    entities = parse_query_entities("Doanh thu thuần của DBC năm 2023 là bao nhiêu?")
    assert entities.periods == ("2023",)


def test_year_modifier_does_not_swallow_an_unrelated_number() -> None:
    """The modifier slot must stay narrow: `năm <chữ> 2023` may not match an
    arbitrary span, or a note reference would masquerade as a period."""
    entities = parse_query_entities("Thuyết minh số 2023 của DBC năm 2022 là gì?")
    assert entities.periods == ("2022",)


def test_metric_phrases_extraction_single_metric() -> None:
    entities = parse_query_entities("Doanh thu thuần của VNM năm 2023 là bao nhiêu?")
    assert entities.metric_phrases == ("Doanh thu thuần",)
    assert entities.metric_phrase == "Doanh thu thuần"
    assert entities.operation == "lookup"
    assert entities.requested_unit is None


def test_metric_phrases_extraction_unregistered_metric() -> None:
    entities = parse_query_entities("Tra cứu biên lợi nhuận gộp của HPG trong năm 2022?")
    assert entities.metric_phrases == ("biên lợi nhuận gộp",)
    assert entities.metric_phrase == "biên lợi nhuận gộp"
    assert entities.operation == "lookup"
    assert entities.requested_unit is None


def test_metric_phrases_multiple_metrics() -> None:
    entities = parse_query_entities(
        "Tổng tài sản ngắn hạn và nợ phải trả của FPT năm 2021 chênh lệch bao nhiêu?"
    )
    assert entities.metric_phrases == ("Tổng tài sản ngắn hạn", "nợ phải trả")
    assert entities.operation == "difference"
    assert entities.requested_unit is None


def test_operation_inference_ratio() -> None:
    entities = parse_query_entities(
        "Lợi nhuận sau thuế trên tổng tài sản năm 2023 của VNM là bao nhiêu phần trăm?"
    )
    assert entities.operation == "ratio"
    assert entities.requested_unit == "percent"


def test_operation_inference_aggregate_sum() -> None:
    entities = parse_query_entities("Tính tổng cộng doanh thu thuần của VNM từ năm 2020 đến 2023?")
    assert entities.operation == "sum"


def test_operation_inference_aggregate_average() -> None:
    entities = parse_query_entities("Tính trung bình doanh thu thuần của VNM từ năm 2020 đến 2023?")
    assert entities.operation == "average"


def test_requested_unit_percent() -> None:
    entities = parse_query_entities("Tỷ lệ tăng trưởng doanh thu là bao nhiêu %?")
    assert entities.requested_unit == "percent"


def test_requested_unit_billion() -> None:
    entities = parse_query_entities("Doanh thu thuần của VNM năm 2023 là bao nhiêu tỷ đồng?")
    assert entities.requested_unit == "billion_vnd"


def test_slash_date_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities(
        "Vốn chủ sở hữu của FIT là bao nhiêu tỷ đồng vào ngày 31/12/2015?"
    )
    assert entities.periods == ("2015",)


def test_spelled_out_date_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities(
        "Vay và nợ của OGC đến ngày 31 tháng 12 năm 2020 là bao nhiêu tỷ đồng?"
    )
    assert entities.periods == ("2020",)


def test_end_of_year_wording_is_normalized_to_the_bare_fiscal_year() -> None:
    entities = parse_query_entities("Tỷ lệ sở hữu của PLX vào cuối năm 2016 là bao nhiêu?")
    assert entities.periods == ("2016",)


def test_date_normalization_does_not_invent_a_period() -> None:
    """Không có kỳ nào trong câu thì vẫn phải không có kỳ nào."""
    entities = parse_query_entities("Doanh thu thuần của ACB là bao nhiêu?")
    assert entities.periods == ()
