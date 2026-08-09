import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from financial_report_qa.schemas.normalization import NormalizationIssueCode

MISSING_MARKERS = {"-", "—", "–", "n/a", "na", "null", "none", "", "."}
NumberContext = Literal["unknown", "monetary", "percent"]


def is_missing_number(raw: str) -> bool:
    normalized = unicodedata.normalize("NFKC", raw).strip()
    s = " ".join(normalized.split())
    s_lower = s.casefold()
    return s_lower in MISSING_MARKERS


def is_numeric_candidate(raw: str) -> bool:
    normalized = unicodedata.normalize("NFKC", raw).strip()
    s = " ".join(normalized.split())
    if is_missing_number(raw):
        return False
    if re.fullmatch(r"\d{1,2}\s*[-–—]\s*\d{1,2}", s):
        return False
    if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", s):
        return False
    if (s.endswith("O") or s.endswith("o")) and re.search(r"\d", s):
        s = s[:-1] + "0"
    if not re.search(r"\d", s):
        return False
    return bool(re.fullmatch(r"[0-9., +\-()%]+", s))


@dataclass(frozen=True)
class NumberDecision:
    value: Decimal | None
    unit_hint: Literal["percent"] | None = None
    issue_code: NormalizationIssueCode | None = None


def _valid_grouping(groups: list[str]) -> bool:
    if not groups:
        return False
    if not (1 <= len(groups[0]) <= 3) or not groups[0].isdigit():
        return False
    return all(len(g) == 3 and g.isdigit() for g in groups[1:])


def _resolve_single_separator(
    integer: str, separator: str, *, allow_single_group: bool = False
) -> str | None:
    groups = integer.split(separator)
    if len(groups) > 2:
        return "".join(groups) if _valid_grouping(groups) else None
    left, right = groups
    if len(right) == 3 and left.isdigit() and 1 <= len(left) <= 3:
        # A lone three-digit separator is ambiguous unless an explicit
        # financial-negative wrapper or trailing OCR punctuation supplies
        # evidence that it is thousands grouping.
        return left + right if allow_single_group else None
    if len(right) != 3 and left.isdigit() and right.isdigit():
        return f"{left}.{right}"
    return None


def parse_number(raw: str, *, context: NumberContext = "unknown") -> NumberDecision:
    # 1. Normalize Unicode space variants in working copy; keep raw untouched
    normalized = unicodedata.normalize("NFKC", raw).strip()
    # Normalize non-breaking spaces and whitespace
    s = " ".join(normalized.split())
    s_lower = s.casefold()

    # 2. Check missing markers
    if s_lower in MISSING_MARKERS:
        return NumberDecision(value=None, issue_code="number_missing")

    # OCR cleanup for trailing O/o as 0 or trailing sentence point
    had_trailing_separator = False
    if (s.endswith("O") or s.endswith("o")) and re.search(r"\d", s):
        s = s[:-1] + "0"
    if (s.endswith(".") or s.endswith(",")) and len(s) > 1 and s[-2].isdigit():
        had_trailing_separator = True
        s = s[:-1].strip()

    unit_hint: Literal["percent"] | None = None

    # 3. Peel trailing %
    if s.endswith("%"):
        unit_hint = "percent"
        s = s[:-1].strip()

    # 4. Sign and parentheses handling
    is_negative = False
    if s.startswith("(") or s.endswith(")"):
        if s.startswith("(") and s.endswith(")"):
            is_negative = True
            s = s[1:-1].strip()
        else:
            return NumberDecision(value=None, unit_hint=unit_hint, issue_code="number_invalid")

    if s.startswith("+"):
        s = s[1:].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()

    if not s:
        return NumberDecision(value=None, unit_hint=unit_hint, issue_code="number_missing")

    # 5. Validate character set (digits, spaces, dots, commas)
    if not re.fullmatch(r"[0-9., ]+", s):
        return NumberDecision(value=None, unit_hint=unit_hint, issue_code="number_invalid")

    # 6. Handle spaces as grouping separators
    if " " in s:
        parts = s.split(" ")
        if len(parts) > 1 and _valid_grouping(parts):
            s = "".join(parts)
        else:
            return NumberDecision(value=None, unit_hint=unit_hint, issue_code="number_invalid")

    # 7. Resolve separators (. and ,)
    has_dot = "." in s
    has_comma = "," in s

    canonical_str: str | None = None
    issue_code: NormalizationIssueCode | None = None

    if not has_dot and not has_comma:
        canonical_str = s
    elif has_dot and not has_comma:
        if (unit_hint == "percent" or context == "percent") and re.fullmatch(r"\d+\.\d{1,3}", s):
            canonical_str = s
            resolved = None
        else:
            resolved = _resolve_single_separator(
                s,
                ".",
                allow_single_group=is_negative or had_trailing_separator or context == "monetary",
            )
        if resolved is not None:
            canonical_str = resolved
        elif canonical_str is None:
            groups = s.split(".")
            if len(groups) > 2 and not _valid_grouping(groups):
                issue_code = "number_invalid"
            else:
                issue_code = "number_ambiguous"
    elif has_comma and not has_dot:
        if (unit_hint == "percent" or context == "percent") and re.fullmatch(r"\d+,\d{1,3}", s):
            canonical_str = s.replace(",", ".")
            resolved = None
        else:
            resolved = _resolve_single_separator(
                s,
                ",",
                allow_single_group=is_negative or had_trailing_separator or context == "monetary",
            )
        if resolved is not None:
            canonical_str = resolved
        elif canonical_str is None:
            groups = s.split(",")
            if len(groups) > 2 and not _valid_grouping(groups):
                issue_code = "number_ambiguous"
            else:
                issue_code = "number_ambiguous"
    else:
        # Both dot and comma are present
        dot_idx = s.rfind(".")
        comma_idx = s.rfind(",")
        if comma_idx > dot_idx:
            # Comma is rightmost
            decimal_part = s[comma_idx + 1 :]
            int_part = s[:comma_idx]
            if len(decimal_part) in {1, 2} and _valid_grouping(int_part.split(".")):
                canonical_str = f"{int_part.replace('.', '')}.{decimal_part}"
            else:
                issue_code = "number_ambiguous"
        else:
            # Dot is rightmost
            decimal_part = s[dot_idx + 1 :]
            int_part = s[:dot_idx]
            if len(decimal_part) in {1, 2} and _valid_grouping(int_part.split(",")):
                canonical_str = f"{int_part.replace(',', '')}.{decimal_part}"
            else:
                issue_code = "number_ambiguous"

    if canonical_str is None:
        return NumberDecision(
            value=None,
            unit_hint=unit_hint,
            issue_code=issue_code or "number_invalid",
        )

    try:
        val = Decimal(canonical_str)
        if is_negative:
            val = -val
        return NumberDecision(value=val, unit_hint=unit_hint)
    except InvalidOperation:
        return NumberDecision(value=None, unit_hint=unit_hint, issue_code="number_invalid")
