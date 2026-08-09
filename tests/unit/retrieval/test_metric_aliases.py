from financial_report_qa.retrieval.contracts import (
    MetricLabelObservation,
    TableDocument,
    TableMetadata,
)
from financial_report_qa.retrieval.metric_aliases import (
    MetricAliasRule,
    build_metric_alias_lexicon,
    expand_metric_query,
)


def _document(table_character: str, *, canonical: str, raw: str | None) -> TableDocument:
    table_id = "tbl_" + table_character * 64
    return TableDocument(
        table_id=table_id,
        doc_id=f"doc_{table_character}",
        text="metric table",
        metadata=TableMetadata(
            table_id=table_id,
            doc_id=f"doc_{table_character}",
            source_path=f"{table_character}.txt",
            line_start=1,
            line_end=1,
        ),
        metric_labels=(MetricLabelObservation(canonical=canonical, raw=raw),),
    )


def _rule(alias_tokens: tuple[str, ...], canonical_metric: str) -> MetricAliasRule:
    return MetricAliasRule(
        alias_tokens=alias_tokens,
        canonical_metric=canonical_metric,
        canonical_tokens=tuple(canonical_metric.split("_")),
    )


def test_build_metric_alias_lexicon_excludes_ambiguous_aliases() -> None:
    documents = (
        _document("a", canonical="profit_after_tax", raw="Lợi nhuận"),
        _document("b", canonical="operating_profit", raw="Lợi nhuận"),
        _document("c", canonical="net_revenue", raw="Doanh thu thuần"),
    )

    lexicon = build_metric_alias_lexicon(documents)

    assert all(rule.alias_tokens != ("lợi", "nhuận") for rule in lexicon)
    assert any(rule.canonical_metric == "net_revenue" for rule in lexicon)


def test_build_metric_alias_lexicon_is_stable_for_duplicate_observations() -> None:
    document = _document("a", canonical="net_revenue", raw="Doanh thu thuần")
    other_document = _document("b", canonical="total_assets", raw="Tổng tài sản")

    assert build_metric_alias_lexicon((document, document)) == build_metric_alias_lexicon(
        (document,)
    )
    assert build_metric_alias_lexicon((document, other_document)) == build_metric_alias_lexicon(
        (other_document, document)
    )


def test_expansion_uses_whole_token_boundaries() -> None:
    tokens, expansions = expand_metric_query(
        ("profit", "after", "taxation"),
        (_rule(("profit", "after", "tax"), "profit_after_tax"),),
    )

    assert tokens == ("profit", "after", "taxation")
    assert expansions == ()


def test_expansion_prefers_longest_non_overlapping_alias() -> None:
    lexicon = (
        _rule(
            ("lợi", "nhuận", "sau", "thuế", "chưa", "phân", "phối"),
            "retained_earnings",
        ),
        _rule(("lợi", "nhuận", "sau", "thuế"), "profit_after_tax"),
    )

    tokens, expansions = expand_metric_query(
        ("lợi", "nhuận", "sau", "thuế", "chưa", "phân", "phối"), lexicon
    )

    assert "retained" in tokens
    assert "earnings" in tokens
    assert expansions[0].canonical_metric == "retained_earnings"
    assert "profit" not in expansions[0].added_tokens


def test_expansion_does_not_duplicate_existing_tokens() -> None:
    tokens, expansions = expand_metric_query(
        ("net", "revenue"),
        (_rule(("net", "revenue"), "net_revenue"),),
    )

    assert tokens == ("net", "revenue")
    assert expansions == ()


def test_expansion_deduplicates_repeated_canonical_tokens() -> None:
    tokens, expansions = expand_metric_query(
        ("doanh", "thu"),
        (_rule(("doanh", "thu"), "net_net_revenue"),),
    )

    assert tokens == ("doanh", "thu", "net", "revenue")
    assert expansions[0].added_tokens == ("net", "revenue")
