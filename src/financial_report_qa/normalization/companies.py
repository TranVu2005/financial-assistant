"""Conservative company/ticker normalization for Vietnamese issuers.

The document inventory remains the authority for a document's company code.
The registry is used to resolve user/title evidence and to report conflicts; it
must never silently replace inventory provenance.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.normalization._shared import Decision, normalized_key
from financial_report_qa.schemas.documents import DocumentRecord

_REGISTRY_RESOURCE = "company_registry.csv"
_TICKER_RE = re.compile(r"^[A-Z0-9]{2,10}$")
_EXPLICIT_TICKER_PATTERN = re.compile(
    r"(?:mã\s*ck|mã\s*chứng\s*khoán|stock\s*code|ticker)"
    r"\s*[:\-\=]?\s*([a-zA-Z0-9]{2,10})\b",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    """One canonical issuer identity plus explicitly curated historical aliases."""

    ticker: str
    canonical_name: str
    aliases: tuple[str, ...] = ()


def _company_name_key(value: str) -> str:
    """Return an accent-insensitive comparison key for an issuer name."""

    folded = unicodedata.normalize("NFKD", value.casefold()).replace("đ", "d")
    ascii_letters = "".join(char for char in folded if not unicodedata.combining(char))
    key = " ".join(_NON_WORD_RE.sub(" ", ascii_letters).split())

    # Legal-form abbreviations are formatting differences, not issuer identity.
    key = re.sub(r"\bctcp\b", "cong ty co phan", key)
    key = re.sub(r"\bcong ty cp\b", "cong ty co phan", key)
    key = re.sub(r"\btmcp\b", "thuong mai co phan", key)
    return " ".join(key.split())


def _name_variants(value: str) -> tuple[str, ...]:
    """Build conservative exact-match variants without inventing brand aliases."""

    key = _company_name_key(value)
    variants = {key}

    company_prefix = "cong ty co phan "
    bank_prefix = "ngan hang thuong mai co phan "
    company_suffix = " cong ty co phan"

    if key.startswith(company_prefix):
        variants.add(key.removeprefix(company_prefix))
    if key.startswith(bank_prefix):
        bank_name = key.removeprefix(bank_prefix)
        variants.add(f"ngan hang {bank_name}")
    if key.endswith(company_suffix):
        variants.add(key.removesuffix(company_suffix))

    return tuple(sorted(variant for variant in variants if variant))


def _parse_aliases(raw: str) -> tuple[str, ...]:
    return tuple(alias.strip() for alias in raw.split("|") if alias.strip())


def _load_registry() -> dict[str, CompanyRecord]:
    resource = files("financial_report_qa.normalization").joinpath(_REGISTRY_RESOURCE)
    registry: dict[str, CompanyRecord] = {}

    with resource.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["ticker", "canonical_name", "aliases"]:
            raise NormalizationError("invalid company registry header")

        for line_number, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip().upper()
            canonical_name = (row.get("canonical_name") or "").strip()
            aliases = _parse_aliases(row.get("aliases") or "")

            if _TICKER_RE.fullmatch(ticker) is None:
                raise NormalizationError(
                    f"invalid ticker in company registry at line {line_number}: {ticker!r}"
                )
            if not canonical_name:
                raise NormalizationError(f"missing canonical company name at line {line_number}")
            if ticker in registry:
                raise NormalizationError(f"duplicate ticker in company registry: {ticker}")

            registry[ticker] = CompanyRecord(
                ticker=ticker,
                canonical_name=canonical_name,
                aliases=aliases,
            )

    if not registry:
        raise NormalizationError("company registry is empty")
    return registry


def _build_name_indexes(
    registry: Mapping[str, CompanyRecord],
) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    exact_index: dict[str, str] = {}
    contained_names: list[tuple[str, str]] = []

    for record in registry.values():
        explicit_names = (record.canonical_name, *record.aliases)
        for raw_name in explicit_names:
            full_key = _company_name_key(raw_name)
            # Only curated full names are searched inside longer table titles.
            # Stripped legal-name variants could confuse a subsidiary with its
            # parent group (for example HNG and HAG).
            if len(full_key.split()) >= 3:
                contained_names.append((full_key, record.ticker))

            for alias_key in _name_variants(raw_name):
                previous = exact_index.get(alias_key)
                if previous is not None and previous != record.ticker:
                    raise NormalizationError(
                        f"company alias maps to both {previous} and {record.ticker}: {raw_name!r}"
                    )
                exact_index[alias_key] = record.ticker

    # Longest first makes behavior deterministic and favors the most specific
    # legal name when one name is nested inside another.
    contained_names.sort(key=lambda item: (-len(item[0]), item[0], item[1]))
    return exact_index, tuple(contained_names)


_MUTABLE_COMPANY_REGISTRY = _load_registry()
COMPANY_REGISTRY: Mapping[str, CompanyRecord] = MappingProxyType(_MUTABLE_COMPANY_REGISTRY)
_EXACT_NAME_INDEX, _CONTAINED_NAMES = _build_name_indexes(COMPANY_REGISTRY)


def company_name_for_code(company_code: str) -> str | None:
    """Return the registry's canonical display name for a ticker, if known."""

    record = COMPANY_REGISTRY.get(company_code.strip().upper())
    return record.canonical_name if record is not None else None


def _explicit_tickers(value: str) -> set[str]:
    return {
        match.group(1).upper() for match in _EXPLICIT_TICKER_PATTERN.finditer(normalized_key(value))
    }


def _contained_company_codes(name_key: str) -> set[str]:
    """Find every company whose alias appears in `name_key`, greedily
    preferring the longest alias at each text span (`_CONTAINED_NAMES` is
    pre-sorted longest-first) so a subsidiary's name never gets falsely
    claimed out from under a mention of its own parent group's longer legal
    name (e.g. HNG's alias inside HAG's canonical name) -- but a genuinely
    separate mention of a *different* company elsewhere in the same text
    (Day 22 coverage follow-up: multi-company comparison questions) is no
    longer dropped just because some other company's alias elsewhere in the
    text happens to be longer.
    """
    padded = f" {name_key} "
    claimed_spans: list[tuple[int, int]] = []
    codes: set[str] = set()
    for alias_key, ticker in _CONTAINED_NAMES:
        needle = f" {alias_key} "
        start = padded.find(needle)
        while start != -1:
            end = start + len(needle)
            if not any(start < c_end and end > c_start for c_start, c_end in claimed_spans):
                claimed_spans.append((start, end))
                codes.add(ticker)
                break
            start = padded.find(needle, start + 1)

    return codes


def _company_evidence_codes(value: str) -> set[str]:
    stripped = value.strip()
    if not stripped:
        return set()

    ticker = stripped.upper()
    if _TICKER_RE.fullmatch(ticker) is not None:
        return {ticker}

    evidence = _explicit_tickers(stripped)
    name_key = _company_name_key(stripped)
    exact = _EXACT_NAME_INDEX.get(name_key)
    if exact is not None:
        evidence.add(exact)
    else:
        evidence.update(_contained_company_codes(name_key))
    return evidence


_BARE_TICKER_SCAN_RE = re.compile(r"\b[A-Z0-9]{2,10}\b")


def explicit_tickers_in_text(value: str) -> tuple[str, ...]:
    """Return tickers explicitly labeled by a "mã CK"/"ticker" marker in text."""

    return tuple(sorted(_explicit_tickers(value)))


def bare_tickers_in_text(value: str) -> tuple[str, ...]:
    """Return registry tickers written as bare uppercase tokens in text.

    Gated by registry membership so an unrelated all-caps word (``GDP``,
    ``EBITDA``, a Roman-numeral quarter) is never mistaken for a ticker.
    """

    matches = _BARE_TICKER_SCAN_RE.findall(value)
    return tuple(sorted({token for token in matches if token in COMPANY_REGISTRY}))


def company_name_codes_in_text(value: str) -> tuple[str, ...]:
    """Return tickers evidenced only by an issuer name or alias appearing in text."""

    name_key = _company_name_key(value)
    codes = set(_contained_company_codes(name_key))
    exact = _EXACT_NAME_INDEX.get(name_key)
    if exact is not None:
        codes.add(exact)
    return tuple(sorted(codes))


def company_codes_in_text(value: str) -> tuple[str, ...]:
    """Find tickers unambiguously evidenced anywhere inside free text.

    Unlike `_company_evidence_codes`, which treats its entire input as one
    title-like field, this scans a longer string (a natural-language
    question) for embedded evidence: explicit "mã CK: DBC" patterns, bare
    uppercase tickers that are actual registry members (never a bare guess
    such as "GDP" or "Q1"), and full or partial issuer names contained
    anywhere in the text. Returns every distinct ticker found; the caller
    decides whether several distinct tickers mean "compare these companies"
    or "the evidence contradicts itself".
    """

    if not value.strip():
        return ()

    evidence = set(explicit_tickers_in_text(value))
    evidence.update(bare_tickers_in_text(value))
    evidence.update(company_name_codes_in_text(value))
    return tuple(sorted(evidence))


def resolve_company_code(raw_value: str | None) -> Decision[str]:
    """Resolve a ticker or issuer name without fuzzy/substring ticker guessing.

    Unknown values are intentionally returned without an issue because the
    supplied registry is a seed list, not a complete Vietnamese market master.
    Multiple contradictory signals are reported as ``company_conflict``.
    """

    if raw_value is None:
        return Decision(value=None)

    evidence = _company_evidence_codes(raw_value)
    if len(evidence) == 1:
        return Decision(value=next(iter(evidence)))
    if len(evidence) > 1:
        return Decision(value=None, issue_code="company_conflict")
    return Decision(value=None)


def normalize_company(document: DocumentRecord, title_raw: str | None) -> Decision[str]:
    """Validate title evidence while preserving the inventory company code."""

    expected = document.company_code.upper()
    if title_raw is None:
        return Decision(value=expected)

    explicit_tickers = _explicit_tickers(title_raw)
    if len(explicit_tickers) > 1:
        return Decision(value=expected, issue_code="company_conflict")
    if explicit_tickers and expected not in explicit_tickers:
        return Decision(value=expected, issue_code="company_conflict")
    return Decision(value=expected)


def validate_company_codes(company_codes: Iterable[str]) -> tuple[str, ...]:
    """Return normalized codes absent from the seed registry.

    This is an audit helper, not an ingestion gate: unknown codes remain valid
    because the registry intentionally has open-set behavior.
    """

    normalized = {code.strip().upper() for code in company_codes}
    return tuple(sorted(code for code in normalized if code not in COMPANY_REGISTRY))
