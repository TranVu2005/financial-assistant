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
