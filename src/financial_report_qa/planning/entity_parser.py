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
from financial_report_qa.planning.entity_contracts import (
    AmbiguityCode,
    ParsedSpan,
    QueryEntities,
    StatementScope,
)
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
_BARE_YEAR_AFTER_CONNECTOR_RE = re.compile(
    r"(?:đến|và|với|tới|so\s+với|-|–)\s*((?:19|20)\d{2})\b", re.IGNORECASE
)
_RELATIVE_PERIOD_RE = re.compile(
    r"năm\s+(?:nay|hiện\s*hành|trước(?:\s+đó)?)|quý\s+(?:này|trước|vừa\s+rồi)", re.IGNORECASE
)
# Day 21 plan §1.6/ADR 0010 decision A1: measured 37.7% of official ViFinQA
# questions state a scope, overwhelmingly "công ty mẹ"/"riêng" (36.4%) over
# "hợp nhất" (1.3%).
_SEPARATE_SCOPE_RE = re.compile(r"\briêng\b|\bcông\s+ty\s+mẹ\b", re.IGNORECASE)
_CONSOLIDATED_SCOPE_RE = re.compile(r"\bhợp\s+nhất\b|\btoàn\s+tập\s+đoàn\b", re.IGNORECASE)


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

    if values:
        # A bare year ("... giữa năm 2022 và 2023") only counts as a second
        # period once an unambiguous "năm YYYY"/quarter/date already anchors
        # the sentence to a period list — otherwise a bare digit sequence
        # after a hyphen or "và" is too weak a signal (e.g. a note number).
        for match in _BARE_YEAR_AFTER_CONNECTOR_RE.finditer(question):
            start, end = match.span(1)
            if overlaps(start, end):
                continue
            values.add(match.group(1))
            consumed.append((start, end))
            spans.append(ParsedSpan(field="period", surface=match.group(1), start=start, end=end))

    if _RELATIVE_PERIOD_RE.search(question):
        ambiguity.add("period_relative_unresolved")

    if not values and not ambiguity:
        ambiguity.add("period_missing")

    return tuple(sorted(values)), tuple(sorted(ambiguity)), tuple(spans)


# Question-side-only synonyms with 0% canonical coverage in the locked release
# (ADR 0004 §1.6 / Day 16 §1.6-1.7): `normalization.metrics.METRIC_ALIASES` is
# baked into `cells.row_label_canonical` at build time, so adding entries there
# would change `dataset_fingerprint` and invalidate every pinned baseline. These
# names are therefore NOT in `plan_contracts.CANONICAL_METRICS` — a rule planner
# must locate them via `MetricSelector.raw_text`, never `.canonical` (ADR 0004
# Option C). Keeping them out of `METRIC_ALIASES` also keeps them out of the
# `validate_aliases` collision check that table enforces at import time.
_EXTRA_METRIC_ALIASES: dict[str, str] = {
    "cho vay khách hàng": "loans_to_customers",
    "chứng khoán đầu tư": "investment_securities",
    "chứng khoán kinh doanh": "trading_securities",
    "tiền gửi của khách hàng": "customer_deposits",
    "tiền gửi có kỳ hạn": "term_deposits",
    "dự phòng rủi ro tín dụng": "credit_loss_provision",
    "tỷ lệ sở hữu": "ownership_ratio",
    "lctt": "net_cash_flow",
    # "lnst chưa phân phối" already resolves via METRIC_ALIASES -> retained_earnings.
    "chi phí thuế hiện hành": "current_income_tax_expense",
    "chi phí thuế thu nhập doanh nghiệp": "current_income_tax_expense",
}


def _metric_lexicon() -> tuple[tuple[tuple[str, ...], str, str], ...]:
    """Build a longest-match metric lexicon from curated normalization aliases
    plus the question-side-only extras above.

    Each entry is `(alias_tokens, canonical_metric, raw_alias)`. Distinct raw
    aliases can only collide here if their normalized keys are equal, which
    `financial_report_qa.normalization.metrics.METRIC_ALIASES` already
    forbids at import time (`validate_aliases` raises otherwise) — so no
    within-lexicon ambiguity is reachable by construction. `_EXTRA_METRIC_ALIASES`
    is checked separately for collisions in tests, not at import time.
    """
    merged = {
        **INCOME_STATEMENT_ALIASES,
        **BALANCE_SHEET_ALIASES,
        **CASH_FLOW_ALIASES,
        **_EXTRA_METRIC_ALIASES,
    }
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


def _parse_statement_scope(
    question: str,
) -> tuple[StatementScope | None, tuple[ParsedSpan, ...]]:
    """Never guesses: a question naming both scopes wants a cross-scope
    comparison no compiler operation supports yet (Day 21 plan §1.6, 1/1012
    official questions) -- left unstated rather than picking one silently."""
    separate_match = _SEPARATE_SCOPE_RE.search(question)
    consolidated_match = _CONSOLIDATED_SCOPE_RE.search(question)
    if separate_match is not None and consolidated_match is not None:
        return None, ()
    if separate_match is not None:
        span = ParsedSpan(
            field="statement_scope",
            surface=separate_match.group(0),
            start=separate_match.start(),
            end=separate_match.end(),
        )
        return "separate", (span,)
    if consolidated_match is not None:
        span = ParsedSpan(
            field="statement_scope",
            surface=consolidated_match.group(0),
            start=consolidated_match.start(),
            end=consolidated_match.end(),
        )
        return "consolidated", (span,)
    return None, ()


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
    statement_scope, scope_spans = _parse_statement_scope(normalized_question)

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
            company_spans + period_spans + metric_spans + statement_spans + scope_spans,
            key=lambda span: (span.start, span.end, span.field),
        )
    )
    return QueryEntities(
        question=normalized_question,
        company_codes=company_codes,
        periods=periods,
        metrics=metrics,
        statement_types=statement_types,
        statement_scope=statement_scope,
        ambiguity=ambiguity,
        spans=spans,
    )
