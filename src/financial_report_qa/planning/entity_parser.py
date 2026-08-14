"""Deterministic rule-based entity extraction for natural-language questions.

Reuses the curated dictionaries already trusted by table normalization
(`financial_report_qa.normalization`) instead of inventing a parallel
vocabulary. Every value returned is either directly evidenced in the question
text or flagged with an `AmbiguityCode` — this module never guesses.
"""

from __future__ import annotations

import datetime
import re
import unicodedata

from financial_report_qa.normalization._shared import normalized_key
from financial_report_qa.normalization.companies import (
    _BARE_TICKER_SCAN_RE,
    COMPANY_REGISTRY,
    company_name_codes_in_text,
    explicit_tickers_in_text,
)
from financial_report_qa.normalization.metrics import (
    BALANCE_SHEET_ALIASES,
    CASH_FLOW_ALIASES,
    INCOME_STATEMENT_ALIASES,
)
from financial_report_qa.normalization.statements import (
    _STATEMENT_FAMILIES,
    normalize_statement_type,
)
from financial_report_qa.planning.entity_contracts import AmbiguityCode, ParsedSpan, QueryEntities
from financial_report_qa.retrieval.index import tokenize_text

_ROMAN_QUARTERS = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}

_DATE_RE = re.compile(
    r"(?:tại\s+)?ngày\s+(\d{1,2})[/\-](\d{1,2})[/\-]((?:19|20)\d{2})"
    r"|\b(\d{1,2})[/\-](\d{1,2})[/\-]((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(
    # "iv" must be tried before "i{1,3}" so "quý IV" is not truncated to "quý I".
    r"quý\s*(iv|i{1,3}|[1-4])(?:\s*(?:năm|/|-)?\s*((?:19|20)\d{2}))?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"năm\s+((?:19|20)\d{2})", re.IGNORECASE)
_RELATIVE_PERIOD_RE = re.compile(
    r"năm\s+(?:nay|hiện\s*hành|trước(?:\s+đó)?)|quý\s+(?:này|trước|vừa\s+rồi)", re.IGNORECASE
)


def _normalize_question(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).split())


def _company_span(
    question: str, code: str, canonical_name: str, aliases: tuple[str, ...]
) -> ParsedSpan | None:
    for candidate in sorted({code, canonical_name, *aliases}, key=len, reverse=True):
        index = question.find(candidate)
        if index >= 0:
            return ParsedSpan(
                field="company", surface=candidate, start=index, end=index + len(candidate)
            )
    return None


def _company_name_spans(question: str, name_codes: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Locate the matched-name substring for each name-evidenced ticker.

    Used only to keep bare-ticker scanning from double-counting a token that
    is itself a substring of another company's full name (e.g. "FPT" inside
    "CTCP Chứng khoán FPT", the display name of ticker FTS).
    """
    spans: list[tuple[int, int]] = []
    for code in name_codes:
        record = COMPANY_REGISTRY.get(code)
        if record is None:
            continue
        for candidate in sorted({record.canonical_name, *record.aliases}, key=len, reverse=True):
            index = question.find(candidate)
            if index >= 0:
                spans.append((index, index + len(candidate)))
                break
    return tuple(spans)


def _bare_tickers_outside(
    question: str, exclude_spans: tuple[tuple[int, int], ...]
) -> tuple[str, ...]:
    codes: set[str] = set()
    for match in _BARE_TICKER_SCAN_RE.finditer(question):
        token = match.group(0)
        if token not in COMPANY_REGISTRY:
            continue
        start, end = match.span()
        if any(start >= s and end <= e for s, e in exclude_spans):
            continue
        codes.add(token)
    return tuple(sorted(codes))


def _parse_company(
    question: str,
) -> tuple[tuple[str, ...], tuple[AmbiguityCode, ...], tuple[ParsedSpan, ...]]:
    explicit = set(explicit_tickers_in_text(question))
    name_codes = set(company_name_codes_in_text(question))
    if explicit and name_codes and explicit != name_codes:
        return (), ("company_conflict",), ()

    name_spans = _company_name_spans(question, tuple(sorted(name_codes)))
    bare_codes = set(_bare_tickers_outside(question, name_spans))
    codes = tuple(sorted(explicit | name_codes | bare_codes))
    if not codes:
        return (), ("company_missing",), ()

    spans: list[ParsedSpan] = []
    for code in codes:
        record = COMPANY_REGISTRY.get(code)
        canonical_name = record.canonical_name if record else code
        aliases = record.aliases if record else ()
        span = _company_span(question, code, canonical_name, aliases)
        if span is not None:
            spans.append(span)
    return codes, (), tuple(spans)


def _quarter_value(quarter_token: str, year_token: str | None) -> str | None:
    if year_token is None:
        return None
    return f"{int(year_token)}-Q{_ROMAN_QUARTERS[quarter_token.lower()]}"


def _parse_period(
    question: str,
) -> tuple[tuple[str, ...], tuple[AmbiguityCode, ...], tuple[ParsedSpan, ...]]:
    values: set[str] = set()
    ambiguity: set[AmbiguityCode] = set()
    spans: list[ParsedSpan] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < c_end and end > c_start for c_start, c_end in consumed)

    for match in _DATE_RE.finditer(question):
        start, end = match.span()
        if overlaps(start, end):
            continue
        day_s, month_s, year_s = (g for g in match.groups() if g is not None)
        try:
            value = datetime.date(int(year_s), int(month_s), int(day_s)).isoformat()
        except ValueError:
            continue
        values.add(value)
        consumed.append((start, end))
        spans.append(ParsedSpan(field="period", surface=match.group(0), start=start, end=end))

    for match in _QUARTER_RE.finditer(question):
        start, end = match.span()
        if overlaps(start, end):
            continue
        quarter_token, year_token = match.groups()
        quarter_value = _quarter_value(quarter_token, year_token)
        consumed.append((start, end))
        if quarter_value is None:
            ambiguity.add("period_incomplete")
            continue
        values.add(quarter_value)
        spans.append(ParsedSpan(field="period", surface=match.group(0), start=start, end=end))

    for match in _YEAR_RE.finditer(question):
        start, end = match.span()
        if overlaps(start, end):
            continue
        values.add(match.group(1))
        consumed.append((start, end))
        spans.append(ParsedSpan(field="period", surface=match.group(0), start=start, end=end))

    if _RELATIVE_PERIOD_RE.search(question):
        ambiguity.add("period_relative_unresolved")

    if not values and not ambiguity:
        ambiguity.add("period_missing")

    return tuple(sorted(values)), tuple(sorted(ambiguity)), tuple(spans)


def _metric_lexicon() -> tuple[tuple[tuple[str, ...], str, str], ...]:
    """Build a longest-match metric lexicon from curated normalization aliases.

    Each entry is `(alias_tokens, canonical_metric, raw_alias)`. Distinct raw
    aliases can only collide here if their normalized keys are equal, which
    `financial_report_qa.normalization.metrics.METRIC_ALIASES` already
    forbids at import time (`validate_aliases` raises otherwise) — so no
    within-lexicon ambiguity is reachable by construction.
    """
    merged = {**INCOME_STATEMENT_ALIASES, **BALANCE_SHEET_ALIASES, **CASH_FLOW_ALIASES}
    entries = []
    for raw_alias, canonical in merged.items():
        tokens = tokenize_text(raw_alias)
        if tokens:
            entries.append((tokens, canonical, raw_alias))
    return tuple(sorted(entries, key=lambda item: (-len(item[0]), item[0], item[1])))


_METRIC_LEXICON = _metric_lexicon()


def _parse_metric(
    question: str,
) -> tuple[tuple[str, ...], tuple[AmbiguityCode, ...], tuple[ParsedSpan, ...]]:
    tokens = tokenize_text(question)
    metrics: set[str] = set()
    matched_raw: set[str] = set()
    offset = 0
    while offset < len(tokens):
        match = next(
            (
                entry
                for entry in _METRIC_LEXICON
                if tokens[offset : offset + len(entry[0])] == entry[0]
            ),
            None,
        )
        if match is None:
            offset += 1
            continue
        alias_tokens, canonical, raw_alias = match
        metrics.add(canonical)
        matched_raw.add(raw_alias)
        offset += len(alias_tokens)

    spans: list[ParsedSpan] = []
    for raw_alias in matched_raw:
        found = re.search(re.escape(raw_alias), question, re.IGNORECASE)
        if found is not None:
            spans.append(
                ParsedSpan(
                    field="metric", surface=found.group(0), start=found.start(), end=found.end()
                )
            )

    ambiguity: tuple[AmbiguityCode, ...] = () if metrics else ("metric_unknown",)
    return tuple(sorted(metrics)), ambiguity, tuple(spans)


def _statement_span(question: str, family: str) -> ParsedSpan | None:
    key = normalized_key(question)
    for alias in _STATEMENT_FAMILIES.get(family, ()):
        if alias in key:
            found = re.search(re.escape(alias), question, re.IGNORECASE)
            if found is not None:
                return ParsedSpan(
                    field="statement_type",
                    surface=found.group(0),
                    start=found.start(),
                    end=found.end(),
                )
    return None


def _parse_statement_type(
    question: str,
) -> tuple[tuple[str, ...], tuple[AmbiguityCode, ...], tuple[ParsedSpan, ...]]:
    decision = normalize_statement_type(question)
    if decision.issue_code is not None:
        return (), ("statement_conflict",), ()
    if decision.value is None:
        return (), (), ()
    span = _statement_span(question, decision.value)
    return (decision.value,), (), ((span,) if span is not None else ())


def parse_query_entities(question: str) -> QueryEntities:
    """Extract company, period, metric, and statement-type entities from a question.

    Never guesses: a field is populated only when its evidence is
    unambiguous, otherwise it is left empty and an `AmbiguityCode` explains
    why, matching the "Company/kỳ mơ hồ → yêu cầu làm rõ hoặc abstain" rule.
    """
    normalized_question = _normalize_question(question)
    company_codes, company_ambiguity, company_spans = _parse_company(normalized_question)
    periods, period_ambiguity, period_spans = _parse_period(normalized_question)
    metrics, metric_ambiguity, metric_spans = _parse_metric(normalized_question)
    statement_result = _parse_statement_type(normalized_question)
    statement_types, statement_ambiguity, statement_spans = statement_result

    ambiguity = tuple(
        sorted(
            set(company_ambiguity)
            | set(period_ambiguity)
            | set(metric_ambiguity)
            | set(statement_ambiguity)
        )
    )
    spans = tuple(
        sorted(
            company_spans + period_spans + metric_spans + statement_spans,
            key=lambda span: (span.start, span.end, span.field),
        )
    )
    return QueryEntities(
        question=normalized_question,
        company_codes=company_codes,
        periods=periods,
        metrics=metrics,
        statement_types=statement_types,
        ambiguity=ambiguity,
        spans=spans,
    )
