"""Corpus-bound metric alias expansion for the Day 8 BM25 retriever."""

from __future__ import annotations

from dataclasses import dataclass

from financial_report_qa.retrieval.contracts import MetricExpansion, TableDocument
from financial_report_qa.retrieval.index import tokenize_text


@dataclass(frozen=True)
class MetricAliasRule:
    """One unambiguous observed alias and its canonical metric tokens."""

    alias_tokens: tuple[str, ...]
    canonical_metric: str
    canonical_tokens: tuple[str, ...]


def build_metric_alias_lexicon(
    documents: tuple[TableDocument, ...],
) -> tuple[MetricAliasRule, ...]:
    """Derive only unambiguous raw-label aliases from immutable corpus documents."""
    aliases: dict[tuple[str, ...], set[str]] = {}
    for document in documents:
        for observation in document.metric_labels:
            if observation.raw is None:
                continue
            alias_tokens = tokenize_text(observation.raw)
            canonical_tokens = tokenize_text(observation.canonical.replace("_", " "))
            if not alias_tokens or not canonical_tokens:
                continue
            aliases.setdefault(alias_tokens, set()).add(observation.canonical)

    rules: list[MetricAliasRule] = []
    for alias_tokens, canonical_metrics in aliases.items():
        if len(canonical_metrics) != 1:
            continue
        canonical_metric = next(iter(canonical_metrics))
        canonical_tokens = tokenize_text(canonical_metric.replace("_", " "))
        if not set(canonical_tokens).difference(alias_tokens):
            continue
        rules.append(
            MetricAliasRule(
                alias_tokens=alias_tokens,
                canonical_metric=canonical_metric,
                canonical_tokens=canonical_tokens,
            )
        )
    return tuple(
        sorted(
            rules,
            key=lambda rule: (
                -len(rule.alias_tokens),
                rule.alias_tokens,
                rule.canonical_metric,
            ),
        )
    )


def expand_metric_query(
    query_tokens: tuple[str, ...],
    lexicon: tuple[MetricAliasRule, ...],
) -> tuple[tuple[str, ...], tuple[MetricExpansion, ...]]:
    """Expand longest non-overlapping whole-token aliases with canonical tokens."""
    ordered_rules = tuple(
        sorted(
            lexicon,
            key=lambda rule: (
                -len(rule.alias_tokens),
                rule.alias_tokens,
                rule.canonical_metric,
            ),
        )
    )
    expanded_tokens = list(dict.fromkeys(query_tokens))
    seen_tokens = set(expanded_tokens)
    expansions: list[MetricExpansion] = []
    offset = 0
    while offset < len(query_tokens):
        matched_rule = next(
            (
                rule
                for rule in ordered_rules
                if query_tokens[offset : offset + len(rule.alias_tokens)] == rule.alias_tokens
            ),
            None,
        )
        if matched_rule is None:
            offset += 1
            continue
        added_tokens = tuple(
            dict.fromkeys(
                token for token in matched_rule.canonical_tokens if token not in seen_tokens
            )
        )
        if added_tokens:
            expanded_tokens.extend(added_tokens)
            seen_tokens.update(added_tokens)
            expansions.append(
                MetricExpansion(
                    alias_tokens=matched_rule.alias_tokens,
                    canonical_metric=matched_rule.canonical_metric,
                    added_tokens=added_tokens,
                )
            )
        offset += len(matched_rule.alias_tokens)
    return tuple(expanded_tokens), tuple(expansions)
