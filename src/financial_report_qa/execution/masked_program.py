"""Masked-PAL program: AST guard plus an arithmetic interpreter.

N4' of the 2026-08-24 spec says a generated program may not contain a
numeric literal. The guard enforces that by rejecting *every* `ast.Constant`
-- there is no harmless one, because the moment a coefficient is allowed the
invariant stops being checkable by a machine. Scaling a result into percent
or into millions is a presentation concern, applied afterwards by
`apply_scale` from a closed enum the model chooses, never written by the
model into the program itself.

Values reach the interpreter only through `values[i]`, bound deterministically
from the release. Like `pandas_query.py`, this module never calls
`eval`/`exec` and denies by default.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from decimal import Decimal, DivisionByZero, InvalidOperation

from financial_report_qa.core.errors import ProgramEvalError, ProgramGuardError
from financial_report_qa.execution.program_contracts import ScaleName

_PLACEHOLDER_PATTERN = re.compile(r"\[NUM_(\d+)\]")
NAME_PATTERN = re.compile(r"^NUM_(\d+)$")

#: Bounded like `pandas_query.py`'s budgets: a legitimate financial formula
#: is short, so anything long is a sign the model wandered off the grammar.
_MAX_PROGRAM_LENGTH = 512
_MAX_NODE_COUNT = 200

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)

_SCALE_FACTORS: dict[str, Decimal] = {
    "none": Decimal(1),
    "percent": Decimal(100),
    "thousand": Decimal(1) / Decimal(1000),
    "million": Decimal(1) / Decimal(1000000),
    "billion": Decimal(1) / Decimal(1000000000),
}

#: Hậu tố pandas tương ứng từng scale. Phải khớp `_SCALE_FACTORS` -- Task 3
#: có test ghim rằng hai đường cho cùng kết quả.
SCALE_SUFFIX: dict[str, str] = {
    "none": "",
    "percent": " * 100",
    "thousand": " / 1000",
    "million": " / 1000000",
    "billion": " / 1000000000",
}


def substitute_placeholders(program: str) -> str:
    """Rewrite `[NUM_0]` to `NUM_0` so the expression parses as Python."""
    return _PLACEHOLDER_PATTERN.sub(r"NUM_\1", program)


def parse_program(program: str, *, value_count: int) -> ast.Expression:
    """Parse and guard one program, or raise `ProgramGuardError`."""
    if len(program) > _MAX_PROGRAM_LENGTH:
        raise ProgramGuardError(
            f"program exceeds max length {_MAX_PROGRAM_LENGTH}: {len(program)} chars"
        )
    source = substitute_placeholders(program)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise ProgramGuardError(f"program is not a single expression: {error}") from error

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_NODE_COUNT:
        raise ProgramGuardError(f"program exceeds max node count {_MAX_NODE_COUNT}: {node_count}")

    _guard(tree.body, value_count=value_count)
    return tree


def _guard(node: ast.AST, *, value_count: int) -> None:
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise ProgramGuardError(f"operator not allowed: {type(node.op).__name__}")
        _guard(node.left, value_count=value_count)
        _guard(node.right, value_count=value_count)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise ProgramGuardError(f"unary operator not allowed: {type(node.op).__name__}")
        _guard(node.operand, value_count=value_count)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "abs":
            raise ProgramGuardError("only abs() may be called")
        if len(node.args) != 1 or node.keywords:
            raise ProgramGuardError("abs() takes exactly one positional argument")
        _guard(node.args[0], value_count=value_count)
        return
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        if match is None:
            raise ProgramGuardError(f"name not allowed: {node.id}")
        index = int(match.group(1))
        if not 0 <= index < value_count:
            raise ProgramGuardError(
                f"[NUM_{index}] is out of range for {value_count} bound value(s)"
            )
        return
    if isinstance(node, ast.Constant):
        raise ProgramGuardError(f"literal not allowed: {node.value!r}")
    raise ProgramGuardError(f"node not allowed: {type(node).__name__}")


def evaluate(tree: ast.Expression, values: Sequence[Decimal]) -> Decimal:
    """Evaluate a guarded expression over already-bound values."""
    return _evaluate(tree.body, values)


def _evaluate(node: ast.AST, values: Sequence[Decimal]) -> Decimal:
    if isinstance(node, ast.Name):
        match = NAME_PATTERN.match(node.id)
        assert match is not None  # guarded
        return values[int(match.group(1))]
    if isinstance(node, ast.UnaryOp):
        return -_evaluate(node.operand, values)
    if isinstance(node, ast.Call):
        return abs(_evaluate(node.args[0], values))
    assert isinstance(node, ast.BinOp)  # guarded
    left = _evaluate(node.left, values)
    right = _evaluate(node.right, values)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    try:
        return left / right
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as error:
        raise ProgramEvalError("division_by_zero") from error


def run_program(program: str, values: Sequence[Decimal]) -> Decimal:
    """Guard, then evaluate, one program over its bound values."""
    tree = parse_program(program, value_count=len(values))
    result = evaluate(tree, values)
    if not result.is_finite():
        raise ProgramEvalError("non_finite_result")
    return result


def apply_scale(value: Decimal, scale: ScaleName) -> Decimal:
    """Scale a raw result for presentation. The model picks the enum, never
    the coefficient -- that is what keeps N4' absolute."""
    return value * _SCALE_FACTORS[scale]
