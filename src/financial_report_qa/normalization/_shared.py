import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from financial_report_qa.core.errors import NormalizationError
from financial_report_qa.schemas.normalization import NormalizationIssue, NormalizationIssueCode

T = TypeVar("T")
RULESET_VERSION = "2026.08.6"


@dataclass(frozen=True)
class Decision(Generic[T]):
    value: T | None
    issue_code: NormalizationIssueCode | None = None


def normalized_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sanitize_selector_text(value: str) -> str:
    """NFKC-normalize and collapse whitespace, without changing case.

    Meant for text that has to survive as a literal value -- a row or
    column label quoted back into a query or a citation -- not just a
    comparison key like `normalized_key` produces. Real corpus column
    headers can concatenate two source lines with an embedded newline
    (`cell_frame.py`'s own docstring: extraction joins a header with the
    row above and its unit row), and label fields forbid control
    characters outright -- such text raises a `ValidationError` instead of
    naming the row it means.
    `.split()`/`" ".join(...)` collapses any whitespace run (space, tab,
    newline) to a single space rather than stripping it, so
    "31/12/2019\nVND" survives as the still-readable "31/12/2019 VND"
    instead of losing the boundary between its two halves.
    """
    return " ".join(unicodedata.normalize("NFKC", value).split())


def validate_aliases(aliases: Mapping[str, T]) -> dict[str, T]:
    validated: dict[str, T] = {}
    for raw, canonical in aliases.items():
        key = normalized_key(raw)
        if key in validated and validated[key] != canonical:
            raise NormalizationError(f"conflicting alias: {raw!r}")
        validated[key] = canonical
    return validated


def _none_first(value: str | None) -> tuple[bool, str]:
    return value is not None, value or ""


def issue_sort_key(issue: NormalizationIssue) -> tuple[object, ...]:
    return (
        issue.doc_id,
        _none_first(issue.table_id),
        _none_first(issue.cell_id),
        issue.field,
        issue.code,
        _none_first(issue.raw_value),
    )
