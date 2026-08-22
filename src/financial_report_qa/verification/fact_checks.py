"""plan.md §15: verify each fact independently, before the formula.

`checks.py`'s five checks all operate on the finished `CompiledQuery` --
recompute the whole formula, check the whole answer's unit, check the whole
evidence set against retrieval. None of them re-examines a single cell's own
row, column, or unit against the release. This module is that missing layer:

```text
Question
   |
Fact 1 -> verify row / column / unit
Fact 2 -> verify row / column / unit
   |
Formula -> verify operation      (checks.check_recompute_mismatch)
   |
Answer
```

`verify_fact` re-locates one `GroundedFact` from scratch, independently of
whatever produced it: it rebuilds the cell frame for the fact's own table and
calls `execution.locator.locate` with a position-bound `MetricSelector`
(`table_id` + `row_index`, plan.md §14) at the fact's period. That is a
different code path than the one that produced the fact in the first place
(`fact_grounding.grounded_facts` zips `CompiledQuery.evidence` with
`replay_rows`), so a bug there -- pairing evidence with the wrong replay row,
an off-by-one in `row_index` -- surfaces here as a real mismatch rather than
trivially re-confirming itself.

Two outcomes, both blocking:

- the position no longer resolves to any cell at the fact's period (or, when
  the fact names a column, no cell at that column) -- `fact_not_found`;
- it resolves, but to a different value or unit than the fact recorded --
  `fact_value_mismatch`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.locator import locate
from financial_report_qa.planning.grounding_contracts import GroundedFact
from financial_report_qa.planning.plan_contracts import MetricSelector
from financial_report_qa.verification.contracts import VerificationIssue


def _bounded_raw_text(label: str) -> str:
    """Clamp a raw corpus label to `MetricSelector.raw_text`'s 512-char cap.

    This selector is position-bound (`table_id` + `row_index` both set), so
    `raw_text` is never used as a match key here -- only OCR-merged labels
    over the cap would otherwise crash validation mid-verification.
    """
    cleaned = "".join(ch for ch in label if ord(ch) > 0x1F)
    stripped = cleaned.strip() or "?"
    return stripped[:512]


def verify_fact(fact: GroundedFact, release_dir: Path) -> VerificationIssue | None:
    """Independently re-locate `fact` in the release; `None` if it agrees."""
    frame = build_cell_frame(release_dir, (fact.table_id,))
    selector = MetricSelector(
        raw_text=_bounded_raw_text(fact.row_label),
        table_id=fact.table_id,
        row_index=fact.row_index,
    )
    result = locate(
        frame,
        selector.model_copy(update={"column_text": _bounded_raw_text(fact.column)})
        if fact.column
        else selector,
        fact.period,
        company_code=fact.company_code,
    )
    if result.match is None:
        return VerificationIssue(
            code="fact_not_found",
            message=(
                f"{fact.fact_id}: no cell re-located at table {fact.table_id} "
                f"row {fact.row_index} period {fact.period} ({result.error_code})"
            ),
        )
    if result.match.value != fact.raw_value or result.match.unit != fact.unit:
        return VerificationIssue(
            code="fact_value_mismatch",
            message=(
                f"{fact.fact_id}: re-located value {result.match.value!r} "
                f"{result.match.unit!r} does not match recorded "
                f"{fact.raw_value!r} {fact.unit!r}"
            ),
        )
    return None


def verify_facts(facts: Sequence[GroundedFact], release_dir: Path) -> tuple[VerificationIssue, ...]:
    """One issue per fact that fails re-location; empty when every fact
    behind the answer independently re-confirms its own row/column/unit."""
    return tuple(issue for fact in facts if (issue := verify_fact(fact, release_dir)) is not None)
