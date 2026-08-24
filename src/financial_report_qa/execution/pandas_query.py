"""Pandas-query whitelist replayer (ADR 0007 decision F1).

`replay_pandas_query` independently re-executes a packaged `pandas_query`
string — never `eval`/`exec` — through a small AST whitelist interpreter,
deny-by-default on any node or attribute it does not recognize. It is the
replay half of the submission contract: whatever query string ships in
`submission.json` must re-execute, inside this grammar, to the shipped
answer (the masked-PAL path renders its queries via
`program_binding.render_program_pandas`; the compiler-era renderer that used
to live here was removed with the operation-enum path, spec 2026-08-24
§8.1/§8.2).

**String literals are escaped via `json.dumps`, not f-string interpolation**
(ADR 0008 decision A2). Day 19 plan §1.1 found 1,988 real corpus row labels
containing `"` (e.g. `Khấu hao tài sản cố định ("TSCĐ")`) that crashed naive
interpolation with an uncaught `SyntaxError`, and §1.4 found that unescaped
`|`/`&`/`)` in a label could silently change the rendered expression's
operator precedence with no exception raised at all. `_lit()` closes both: a
JSON string literal is a valid Python string literal for every non-control
character.
"""

from __future__ import annotations

import ast
import json
from decimal import Decimal
from typing import Any

import pandas as pd

_ALLOWED_ATTRS = frozenset(
    {
        "company_code",
        "row_label_canonical",
        "row_label_raw",
        "column_label",
        "period",
        "value",
        # Spec 2026-08-21 §5.2: position breaks ties between same-label rows.
        "table_id",
        "row_idx",
        # Masked-PAL lookups (spec 2026-08-24) also pin the column positionally:
        # one row spans many cells, only `col_idx` tells them apart.
        "col_idx",
    }
)
_ALLOWED_METHODS = frozenset({"isin", "sort_values", "mean", "sum"})

# Day 19 plan Sec 1.8/2.D (ADR 0008 decision D3): structural budgets enforced
# before evaluation, not a preemptive timeout (unavailable on win32 -- no
# SIGALRM/setitimer/resource module). Measured: a 1,000-level-deep BinOp chain
# raises Python's own RecursionError at the default limit of 1,000; an
# isin([200_000 literals]) call parses to ~200,000 AST nodes. Real rendered
# queries on gold70 are a few hundred characters and under 15 nodes deep.
_MAX_QUERY_LENGTH = 4096
_MAX_AST_NODES = 2000
_MAX_AST_DEPTH = 50


def _lit(value: str) -> str:
    """Render `value` as a Python string literal via JSON escaping (Day 19 plan
    Sec 1.1/2.A). A JSON string literal is a valid Python string literal for
    every non-control character, so this survives real corpus labels
    containing `"` or `\\` without breaking the surrounding expression's
    syntax -- unlike naive f-string interpolation (`f'"{value}"'`), which both
    crashes on embedded quotes and can silently change operator precedence
    when `value` contains `)` `|` `&`.
    """
    return json.dumps(value, ensure_ascii=False)


def _ast_depth(node: ast.AST) -> int:
    """Iterative (non-recursive) depth walk so a maliciously deep tree cannot
    raise RecursionError while we are trying to reject it."""
    max_depth = 0
    stack: list[tuple[ast.AST, int]] = [(node, 1)]
    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(current):
            stack.append((child, depth + 1))
    return max_depth


def replay_pandas_query(query: str, frame: pd.DataFrame) -> Decimal:
    """Execute a packaged `pandas_query` string through a whitelist AST
    interpreter and return its scalar result. Raises ValueError on anything
    outside the whitelisted grammar or outside the structural budgets in
    ADR 0008 decision D3 (query length, AST node count, AST depth) — deny by
    default, never eval/exec.
    """
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(f"query exceeds max length {_MAX_QUERY_LENGTH}: {len(query)} chars")

    tree = ast.parse(query, mode="eval")

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_AST_NODES:
        raise ValueError(f"query exceeds max AST node count {_MAX_AST_NODES}: {node_count} nodes")

    depth = _ast_depth(tree.body)
    if depth > _MAX_AST_DEPTH:
        raise ValueError(f"query exceeds max AST depth {_MAX_AST_DEPTH}: {depth}")

    result = _eval_node(tree.body, frame)
    if isinstance(result, Decimal):
        return result
    try:
        return Decimal(str(result))
    except ArithmeticError as exc:
        raise ValueError(f"result is not a valid scalar: {result!r}") from exc


def _eval_node(node: ast.AST, frame: pd.DataFrame) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.List):
        return [_eval_node(element, frame) for element in node.elts]

    if isinstance(node, ast.Name):
        if node.id == "df1":
            return frame
        raise ValueError(f"unsupported name: {node.id}")

    if isinstance(node, ast.Attribute):
        # `.loc` is checked before the column whitelist: `df1.loc` is an
        # indexer, not a column, and would otherwise be rejected as one.
        if node.attr == "loc":
            return _eval_node(node.value, frame).loc
        if isinstance(node.value, ast.Name) and node.value.id == "df1":
            if node.attr not in _ALLOWED_ATTRS:
                raise ValueError(f"unsupported column: {node.attr}")
            return frame[node.attr]
        if node.attr == "iloc":
            operand = _eval_node(node.value, frame)
            return operand.iloc
        raise ValueError(f"unsupported attribute: {node.attr}")

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(element, frame) for element in node.elts)

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            raise ValueError("only a single '==' comparison is supported")
        left = _eval_node(node.left, frame)
        right = _eval_node(node.comparators[0], frame)
        return left == right

    if isinstance(node, ast.UnaryOp):
        # Masked-PAL programs allow unary minus, so its rendered query carries
        # one. Everything else (UAdd/Invert/...) stays outside the grammar.
        if not isinstance(node.op, ast.USub):
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return -_eval_node(node.operand, frame)

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, frame)
        right = _eval_node(node.right, frame)
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Div):
            return left / right
        # `SCALE_SUFFIX["percent"]` renders `... * 100`, and the program
        # grammar itself allows multiplication between bound cells.
        if isinstance(node.op, ast.Mult):
            return left * right
        raise ValueError(f"unsupported operator: {type(node.op).__name__}")

    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, frame)
        key = _eval_node(node.slice, frame)
        return value[key]

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "abs":
            return abs(_eval_node(node.args[0], frame))
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("only whitelisted method calls are supported")
        method = node.func.attr
        if method not in _ALLOWED_METHODS:
            raise ValueError(f"unsupported method: {method}")
        operand = _eval_node(node.func.value, frame)
        if method == "isin":
            values = _eval_node(node.args[0], frame)
            return operand.isin(values)
        if method == "mean":
            values = [Decimal(str(v)) for v in operand.tolist()]
            return sum(values, Decimal(0)) / len(values)
        if method == "sum":
            values = [Decimal(str(v)) for v in operand.tolist()]
            return sum(values, Decimal(0))
        if method == "sort_values":
            ascending = True
            for keyword in node.keywords:
                if keyword.arg == "ascending":
                    ascending = bool(_eval_node(keyword.value, frame))
            return operand.sort_values(by="value", ascending=ascending)
        raise ValueError(f"unsupported method: {method}")

    raise ValueError(f"unsupported syntax: {type(node).__name__}")
