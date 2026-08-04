import re
from decimal import Decimal
from typing import Literal

from financial_report_qa.normalization._shared import Decision, normalized_key, validate_aliases

CanonicalUnit = Literal[
    "VND", "VND_thousand", "VND_million", "VND_billion", "percent", "ratio"
]

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
    r"(?i)(vnd|vnđ|đồng|dong|nghìn|ngàn|ngan|triệu|trieu|tỷ|ty|%|phần trăm|lần|lầ?n)"
)


def _has_unit_evidence(value: str) -> bool:
    """Return True if *value* contains at least one token that looks like a unit."""
    return bool(_UNIT_EVIDENCE_RE.search(value))


def normalize_unit(raw: str | None) -> Decision[CanonicalUnit]:
    if raw is None:
        return Decision(value=None)

    key = normalized_key(raw)
    if not key:
        return Decision(value=None)

    stripped = _strip_prefix(key)

    if stripped in _RAW_UNIT_ALIASES:
        return Decision(value=_RAW_UNIT_ALIASES[stripped])

    # Check for parenthesized unit like "tỷ lệ (%)" or "số tiền (triệu đồng)"
    paren_match = re.search(r"\(([^)]+)\)", stripped)
    if paren_match:
        inner = normalized_key(paren_match.group(1))
        if inner in _RAW_UNIT_ALIASES:
            return Decision(value=_RAW_UNIT_ALIASES[inner])


    return Decision(value=None, issue_code="unit_unknown")


def resolve_unit(
    cell_hint: str | None, column_raw: str | None, table_raw: str | None
) -> Decision[CanonicalUnit]:
    decisions: list[Decision[CanonicalUnit]] = []

    if cell_hint is not None:
        cell_dec = normalize_unit(cell_hint)
        decisions.append(cell_dec)

    if column_raw is not None and _has_unit_evidence(column_raw):
        col_dec = normalize_unit(column_raw)
        if col_dec.value is not None or col_dec.issue_code is not None:
            decisions.append(col_dec)

    if table_raw is not None:
        tbl_dec = normalize_unit(table_raw)
        if tbl_dec.value is not None or tbl_dec.issue_code is not None:
            decisions.append(tbl_dec)

    units: set[CanonicalUnit] = set()
    has_unknown = False

    for dec in decisions:
        if dec.value is not None:
            units.add(dec.value)
        elif dec.issue_code == "unit_unknown":
            has_unknown = True

    if len(units) > 1:
        return Decision(value=None, issue_code="unit_conflict")
    if len(units) == 1:
        return Decision(value=next(iter(units)))

    if has_unknown:
        return Decision(value=None, issue_code="unit_unknown")
    return Decision(value=None)


def unit_multiplier(unit: CanonicalUnit) -> Decimal:
    return _MULTIPLIERS[unit]


def economic_value(value: Decimal, unit: CanonicalUnit) -> Decimal:
    return value * _MULTIPLIERS[unit]


def convert_scale(
    value: Decimal, source: CanonicalUnit, target: CanonicalUnit
) -> Decimal:
    if (source in _MONETARY_UNITS) != (target in _MONETARY_UNITS):
        raise ValueError("incompatible scale conversion")
    if source not in _MONETARY_UNITS and source != target:
        raise ValueError("incompatible scale conversion")

    return (value * _MULTIPLIERS[source]) / _MULTIPLIERS[target]
