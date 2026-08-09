import re
from decimal import Decimal
from typing import Literal

from financial_report_qa.normalization._shared import Decision, normalized_key, validate_aliases

CanonicalUnit = Literal["VND", "VND_thousand", "VND_million", "VND_billion", "percent", "ratio"]

_MULTIPLIERS: dict[CanonicalUnit, Decimal] = {
    "VND": Decimal(1),
    "VND_thousand": Decimal(1000),
    "VND_million": Decimal(1000000),
    "VND_billion": Decimal(1000000000),
    "percent": Decimal("0.01"),
    "ratio": Decimal(1),
}

_MONETARY_UNITS: set[CanonicalUnit] = {
    "VND",
    "VND_thousand",
    "VND_million",
    "VND_billion",
}

_RAW_UNIT_ALIASES: dict[str, CanonicalUnit] = validate_aliases(
    {
        "vnd": "VND",
        "đồng": "VND",
        "dong": "VND",
        "nghìn đồng": "VND_thousand",
        "ngàn đồng": "VND_thousand",
        "thousand vnd": "VND_thousand",
        "nghìn vnd": "VND_thousand",
        "ngàn vnd": "VND_thousand",
        "1.000 vnd": "VND_thousand",
        "1,000 vnd": "VND_thousand",
        "triệu đồng": "VND_million",
        "triệu vnd": "VND_million",
        "million vnd": "VND_million",
        "1.000.000 vnd": "VND_million",
        "1,000,000 vnd": "VND_million",
        "tỷ đồng": "VND_billion",
        "tỷ vnd": "VND_billion",
        "billion vnd": "VND_billion",
        "1.000.000.000 vnd": "VND_billion",
        "1,000,000,000 vnd": "VND_billion",
        "vnđ": "VND",
        "đồng vn": "VND",
        "nghìn vnđ": "VND_thousand",
        "ngàn vnđ": "VND_thousand",
        "triệu vnđ": "VND_million",
        "tỷ vnđ": "VND_billion",
        "%": "percent",
        "phần trăm": "percent",
        "percent": "percent",
        "lần": "ratio",
        "ratio": "ratio",
    }
)


def _strip_prefix(key: str) -> str:
    for prefix in ("đơn vị tính:", "đơn vị:", "đvt:"):
        if key.startswith(prefix):
            return key[len(prefix) :].strip()
    return key


_UNIT_EVIDENCE_RE = re.compile(
    r"(?i)(vnd|vnđ|đồng|dong|nghìn|ngàn|ngan|triệu|trieu|tỷ|%|phần trăm|lần|lầ?n)"
)

_UNIT_CONTEXT_ALIASES: tuple[tuple[str, CanonicalUnit], ...] = tuple(
    sorted(
        (
            (alias, canonical)
            for alias, canonical in _RAW_UNIT_ALIASES.items()
            if alias not in {"percent", "ratio", "lần", "%"}
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _period_like_without_unit(key: str) -> bool:
    return bool(
        re.match(
            r"^(?:năm|ngay|ngày|tháng|thang|quý|quy|q)?\s*\d{2,4}$",
            key,
            re.IGNORECASE,
        )
        or key.startswith("năm ")
        or key.startswith("ngày ")
        or key.startswith("tháng ")
        or key.startswith("quý ")
    )


def _extract_unit(key: str) -> CanonicalUnit | None:
    for alias, canonical in _UNIT_CONTEXT_ALIASES:
        if alias and alias in key:
            return canonical
    if "%" in key or "phần trăm" in key:
        return "percent"
    return None


def _has_unit_evidence(value: str) -> bool:
    """Return True if *value* contains at least one token that looks like a unit."""
    return bool(_UNIT_EVIDENCE_RE.search(value))


def has_unit_evidence(raw: str | None) -> bool:
    if raw is None:
        return False
    key = normalized_key(raw)
    if not key:
        return False
    for prefix in ("đơn vị tính:", "đơn vị:", "đvt:"):
        if key.startswith(prefix):
            return True
    stripped = _strip_prefix(key)
    if _period_like_without_unit(stripped) and not _extract_unit(stripped):
        return False
    return _has_unit_evidence(stripped)


def normalize_unit(raw: str | None) -> Decision[CanonicalUnit]:
    if raw is None:
        return Decision(value=None)

    key = normalized_key(raw)
    if not key:
        return Decision(value=None)

    stripped = _strip_prefix(key)

    if _period_like_without_unit(stripped) and not _extract_unit(stripped):
        return Decision(value=None)

    if stripped in _RAW_UNIT_ALIASES:
        return Decision(value=_RAW_UNIT_ALIASES[stripped])

    # Check for parenthesized unit like "tỷ lệ (%)" or "số tiền (triệu đồng)"
    paren_match = re.search(r"\(([^)]+)\)", stripped)
    if paren_match:
        inner = normalized_key(paren_match.group(1))
        if inner in _RAW_UNIT_ALIASES:
            return Decision(value=_RAW_UNIT_ALIASES[inner])

    extracted = _extract_unit(stripped)
    if extracted is not None:
        return Decision(value=extracted)

    return Decision(value=None, issue_code="unit_unknown")


def strip_unit_context(raw: str | None) -> str:
    """Remove unit tokens from a composite header before period parsing."""
    if raw is None:
        return ""
    key = normalized_key(raw)
    for alias, _ in _UNIT_CONTEXT_ALIASES:
        key = key.replace(alias, " ")
    key = key.replace("%", " ").replace("phần trăm", " ")
    return " ".join(key.split())


def is_monetary_unit(unit: CanonicalUnit | None) -> bool:
    return unit in _MONETARY_UNITS


def resolve_unit(
    cell_hint: str | None, column_raw: str | None, table_raw: str | None
) -> Decision[CanonicalUnit]:
    # Resolve from the most specific evidence outward. A valid cell hint or
    # column unit must not conflict with a broader table default.
    unknown_seen = False
    if cell_hint is not None:
        cell_dec = normalize_unit(cell_hint)
        if cell_dec.value is not None:
            return cell_dec
        unknown_seen = unknown_seen or cell_dec.issue_code == "unit_unknown"

    if column_raw is not None and _has_unit_evidence(column_raw):
        col_dec = normalize_unit(column_raw)
        if col_dec.value is not None:
            return col_dec
        unknown_seen = unknown_seen or col_dec.issue_code == "unit_unknown"

    if table_raw is not None:
        tbl_dec = normalize_unit(table_raw)
        if tbl_dec.value is not None:
            return tbl_dec
        unknown_seen = unknown_seen or tbl_dec.issue_code == "unit_unknown"

    if unknown_seen:
        return Decision(value=None, issue_code="unit_unknown")
    return Decision(value=None)


def unit_multiplier(unit: CanonicalUnit) -> Decimal:
    return _MULTIPLIERS[unit]


def economic_value(value: Decimal, unit: CanonicalUnit) -> Decimal:
    return value * _MULTIPLIERS[unit]


def convert_scale(value: Decimal, source: CanonicalUnit, target: CanonicalUnit) -> Decimal:
    if (source in _MONETARY_UNITS) != (target in _MONETARY_UNITS):
        raise ValueError("incompatible scale conversion")
    if source not in _MONETARY_UNITS and source != target:
        raise ValueError("incompatible scale conversion")

    return (value * _MULTIPLIERS[source]) / _MULTIPLIERS[target]
