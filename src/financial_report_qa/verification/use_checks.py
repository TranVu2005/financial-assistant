"""Verify B: what the model said it used against what it actually bound.

This is the only check that catches index slippage -- the model describing
the right cell while emitting the wrong position in a numbered list. Nothing
downstream can catch it, because the resulting answer is still a legitimate
number read out of a real table.

It does NOT catch a model that genuinely believes the wrong row answers the
question. That limit is deliberate and stated in §9 of the spec.

Every rule is exact after normalization. No fuzzy threshold: a threshold is a
knob someone has to tune, and being wrong about it fails silently.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from financial_report_qa.execution.program_contracts import BoundValue, UseClaim

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"(19|20)\d{2}")


def _normalize(text: str) -> str:
    """Casefold, collapse whitespace, strip edge punctuation. Accents are kept
    -- stripping them would merge Vietnamese terms that are genuinely
    different."""
    folded = unicodedata.normalize("NFC", text).casefold()
    return _WHITESPACE.sub(" ", folded).strip(" .,:;-–—()[]")


def _year_of(text: str) -> int | None:
    match = _YEAR.search(text)
    return int(match.group(0)) if match else None


def _row_matches(claim: str, bound: BoundValue) -> bool:
    claimed = _normalize(claim)
    if not claimed:
        # Chuỗi rỗng sau chuẩn hoá sẽ tự khớp mọi dòng qua endswith(""); fail closed.
        return False
    if claimed == _normalize(bound.row_label_raw):
        return True
    if claimed == _normalize(bound.row_path):
        return True
    if bound.row_label_canonical is not None and claimed == _normalize(bound.row_label_canonical):
        return True
    return _normalize(bound.row_path).endswith(claimed)


def _column_matches(claim: str, bound: BoundValue) -> bool:
    claimed_year = _year_of(claim)
    if claimed_year is None:
        # Model không nêu năm -> không có gì để bác bỏ; hàng đã kiểm riêng.
        return True
    return claimed_year == bound.period


@dataclass(frozen=True)
class UseCheckResult:
    """Whether every placeholder's claim agrees with its actual binding."""

    matched: bool
    mismatches: tuple[str, ...]


def check_use_bindings(uses: Sequence[UseClaim], bindings: Sequence[BoundValue]) -> UseCheckResult:
    """Compare each `UseClaim` to the cell its placeholder really bound to."""
    claims = {claim.num: claim for claim in uses}
    mismatches: list[str] = []
    for bound in bindings:
        claim = claims.pop(bound.num_index, None)
        if claim is None:
            mismatches.append(f"[NUM_{bound.num_index}] has no use claim")
            continue
        if not _row_matches(claim.row, bound):
            mismatches.append(
                f"[NUM_{bound.num_index}] claims row {claim.row!r} but bound {bound.row_path!r}"
            )
        elif not _column_matches(claim.col, bound):
            mismatches.append(
                f"[NUM_{bound.num_index}] claims column {claim.col!r} "
                f"but bound period {bound.period}"
            )
    for leftover in sorted(claims):
        mismatches.append(f"use claim for [NUM_{leftover}] has no binding")
    return UseCheckResult(matched=not mismatches, mismatches=tuple(mismatches))
