"""Bind `[NUM_i]` to real cells, then render the same program two ways.

The arithmetic path (`masked_program.evaluate`) produces the answer. The
pandas path here produces the `pandas_query` the submission carries, because
compliance C5 requires the query to reference a CSV column and C7 requires it
to replay to the same answer -- neither of which a bare `[NUM_0] - [NUM_1]`
can satisfy. Both readings walk the identical guarded AST, so C7 doubles as a
free third consistency check between them.

The lookup shape keeps a semantic clause (`row_label_*`) alongside the
positional ones: the positional clauses make the cell unique, the semantic
clause is what makes the emitted query explain which line the answer came
from. Compliance already strips `row_idx`/`col_idx`/`period` comparisons
before its C4 literal scan, so the positional clauses cannot be mistaken for
a hardcoded answer.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from decimal import Decimal

import pandas as pd

from financial_report_qa.core.errors import ProgramBindingError
from financial_report_qa.execution.masked_program import (
    NAME_PATTERN,
    SCALE_SUFFIX,
    parse_program,
)
from financial_report_qa.execution.pandas_query import _lit
from financial_report_qa.execution.program_contracts import (
    BoundValue,
    CellCandidate,
    ProgramDecision,
    ScaleName,
)

_BINOP_SYMBOL: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


def values_by_position(frame: pd.DataFrame) -> dict[tuple[str, int, int], Decimal]:
    """Index the cell frame by `(table_id, row_idx, col_idx)`.

    Values go through `str()` before `Decimal` so a long significand keeps
    every digit -- the same round-trip hazard `compliance.check_bundle`
    documents for `pd.read_csv`. A cell whose value cannot become a finite
    `Decimal` (NaN, None, infinity, unconvertible junk) is omitted from the
    map rather than bound: `bind_values` then fails closed on the missing
    position with a typed `ProgramBindingError`. Binding NaN silently would
    poison arithmetic downstream, and `Decimal("None")` crashing with an
    uncaught `InvalidOperation` would bypass `run_question`'s ProgramError
    handling -- both violate the global rule that an empty cell never binds.
    """
    values: dict[tuple[str, int, int], Decimal] = {}
    for row in frame.itertuples():
        try:
            value = Decimal(str(row.value))
        except (ArithmeticError, ValueError):
            continue  # unconvertible (e.g. None -> Decimal("None") -> InvalidOperation)
        if not value.is_finite():
            continue  # NaN/Infinity parse fine but must never enter arithmetic
        values[(str(row.table_id), int(row.row_idx), int(row.col_idx))] = value  # type: ignore[arg-type]
    return values


def bind_values(
    decision: ProgramDecision,
    candidates: Sequence[CellCandidate],
    values: Mapping[tuple[str, int, int], Decimal],
) -> tuple[BoundValue, ...]:
    """Resolve every `cells[i]` to a real cell. `[NUM_i]` is `cells[i]`."""
    by_index = {candidate.index: candidate for candidate in candidates}
    bindings: list[BoundValue] = []
    for num_index, candidate_index in enumerate(decision.cells):
        candidate = by_index.get(candidate_index)
        if candidate is None:
            raise ProgramBindingError(
                f"candidate_index_out_of_range: {candidate_index} "
                f"is not one of {len(candidates)} candidates"
            )
        position = (candidate.table_id, candidate.row_idx, candidate.col_idx)
        if position not in values:
            raise ProgramBindingError(
                f"no numeric value at {position} for candidate {candidate_index}"
            )
        bindings.append(
            BoundValue(
                num_index=num_index,
                candidate_index=candidate_index,
                table_id=candidate.table_id,
                row_idx=candidate.row_idx,
                col_idx=candidate.col_idx,
                row_path=candidate.row_path,
                row_label_raw=candidate.row_label_raw,
                row_label_canonical=candidate.row_label_canonical,
                col_path=candidate.col_path,
                period=candidate.period,
                value=values[position],
                unit=candidate.unit,
            )
        )
    return tuple(bindings)


def render_cell_lookup(bound: BoundValue) -> str:
    """Render one bound cell as a unique, self-explaining CSV lookup."""
    if bound.row_label_canonical is not None:
        label_clause = f"(df1.row_label_canonical == {_lit(bound.row_label_canonical)})"
    else:
        label_clause = f"(df1.row_label_raw == {_lit(bound.row_label_raw)})"
    clauses = [
        label_clause,
        f"(df1.table_id == {_lit(bound.table_id)})",
        f"(df1.row_idx == {bound.row_idx})",
        f"(df1.col_idx == {bound.col_idx})",
    ]
    return f'df1[{" & ".join(clauses)}]["value"].iloc[0]'


def render_program_pandas(
    program: str, bindings: Sequence[BoundValue], scale: ScaleName
) -> str:
    """Render the guarded program with every `[NUM_i]` replaced by a lookup."""
    tree = parse_program(program, value_count=len(bindings))
    lookups = [render_cell_lookup(bound) for bound in bindings]
    return _render(tree.body, lookups) + SCALE_SUFFIX[scale]


def _render(node: ast.AST, lookups: Sequence[str]) -> str:
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        assert match is not None  # guarded
        return lookups[int(match.group(1))]
    if isinstance(node, ast.UnaryOp):
        return f"-({_render(node.operand, lookups)})"
    if isinstance(node, ast.Call):
        return f"abs({_render(node.args[0], lookups)})"
    assert isinstance(node, ast.BinOp)  # guarded
    symbol = _BINOP_SYMBOL[type(node.op)]
    return f"({_render(node.left, lookups)} {symbol} {_render(node.right, lookups)})"
