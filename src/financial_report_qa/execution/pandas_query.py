"""Day 18 `pandas_query` renderer + whitelist replayer (ADR 0007 decision F1).

`render_pandas_query` turns a `FinancialQueryPlan` into the human-readable
Pandas expression plan.md's submission contract requires. `replay_pandas_query`
independently re-executes that exact string — never `eval`/`exec` — through a
small AST whitelist interpreter, deny-by-default on any node or attribute it
does not recognize. The compiler (task 18.9) is the one place that must call
both and assert they agree; that agreement is the actual Definition-of-Done
evidence, not the string by itself.

**Scope boundary, stated explicitly rather than hidden:** the rendered
expression reads the `value` column as-is, with no unit-conversion syntax —
unit conversion is `operations.py`'s job (ADR 0007 decision E1), not this
module's. The compiler therefore must replay against a frame whose `value`
column has already been converted to the operation's answer unit (normally
just the evidence cells themselves), not the raw multi-unit corpus. Measured
on gold70 (Day 18 plan §1.3), evidence cells for a single compiled answer
share one unit in every observed case (0/82 slots spanned >1 unit), so this
is a documented boundary, not an active correctness gap.

**String literals are escaped via `json.dumps`, not f-string interpolation**
(ADR 0008 decision A2). Day 19 plan §1.1 found 1,988 real corpus row labels
containing `"` (e.g. `Khấu hao tài sản cố định ("TSCĐ")`) that crashed
`compile_plan` with an uncaught `SyntaxError` under naive interpolation, and
§1.4 found that unescaped `|`/`&`/`)` in a company code could silently change
the rendered expression's operator precedence with no exception raised at
all. `_lit()` closes both: a JSON string literal is a valid Python string
literal for every non-control character.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from decimal import Decimal
from typing import Any

import pandas as pd

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

_ALLOWED_ATTRS = frozenset(
    {"company_code", "row_label_canonical", "row_label_raw", "period", "value"}
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


def _metric_column_and_value(selector: MetricSelector) -> tuple[str, str]:
    if selector.canonical is not None:
        return "row_label_canonical", selector.canonical
    assert selector.raw_text is not None
    return "row_label_raw", selector.raw_text


def _cell_expr(*, company: str | None, selector: MetricSelector, period: str) -> str:
    column, value = _metric_column_and_value(selector)
    clauses = []
    if company is not None:
        clauses.append(f"(df1.company_code == {_lit(company)})")
    clauses.append(f"(df1.{column} == {_lit(value)})")
    clauses.append(f"(df1.period == {int(period)})")
    condition = " & ".join(clauses)
    return f'df1[{condition}]["value"].iloc[0]'


def _aggregate_expr(
    *, company: str | None, selector: MetricSelector, periods: Sequence[str], method: str
) -> str:
    column, value = _metric_column_and_value(selector)
    clauses = []
    if company is not None:
        clauses.append(f"(df1.company_code == {_lit(company)})")
    clauses.append(f"(df1.{column} == {_lit(value)})")
    period_list = ", ".join(str(int(period)) for period in periods)
    clauses.append(f"(df1.period.isin([{period_list}]))")
    condition = " & ".join(clauses)
    return f'df1[{condition}]["value"].{method}()'


def _aggregate_expr_over_companies(
    *, companies: Sequence[str], selector: MetricSelector, period: str, method: str
) -> str:
    """Mirrors `_aggregate_expr`, but for the other arity `_validate_aggregate`
    allows: >1 company, exactly 1 period (Day 23 plan Step 2)."""
    column, value = _metric_column_and_value(selector)
    companies_list = ", ".join(_lit(company) for company in companies)
    clauses = [
        f"(df1.company_code.isin([{companies_list}]))",
        f"(df1.{column} == {_lit(value)})",
        f"(df1.period == {int(period)})",
    ]
    condition = " & ".join(clauses)
    return f'df1[{condition}]["value"].{method}()'


def render_pandas_query(plan: FinancialQueryPlan) -> str:
    """Render a readable Pandas expression for one plan, per operation."""
    period = plan.periods[0] if plan.periods else None

    if plan.operation == "lookup":
        assert plan.metric is not None
        return _cell_expr(company=plan.companies[0], selector=plan.metric, period=period or "")

    if plan.operation in ("difference", "growth_rate"):
        assert plan.metric is not None
        start, end = plan.periods[0], plan.periods[-1]
        start_expr = _cell_expr(company=plan.companies[0], selector=plan.metric, period=start)
        end_expr = _cell_expr(company=plan.companies[0], selector=plan.metric, period=end)
        if plan.operation == "difference":
            return f"{end_expr} - {start_expr}"
        return f"({end_expr} - {start_expr}) / abs({start_expr})"

    if plan.operation == "compare":
        assert plan.metric_a is not None and plan.metric_b is not None
        a_expr = _cell_expr(company=plan.companies[0], selector=plan.metric_a, period=period or "")
        b_expr = _cell_expr(company=plan.companies[0], selector=plan.metric_b, period=period or "")
        return f"{a_expr} - {b_expr}"

    if plan.operation == "compare_companies":
        assert plan.metric is not None
        a_expr = _cell_expr(company=plan.companies[0], selector=plan.metric, period=period or "")
        b_expr = _cell_expr(company=plan.companies[1], selector=plan.metric, period=period or "")
        return f"{a_expr} - {b_expr}"

    if plan.operation == "ratio":
        assert plan.numerator_metric is not None and plan.denominator_metric is not None
        numerator = _cell_expr(
            company=plan.companies[0], selector=plan.numerator_metric, period=period or ""
        )
        denominator = _cell_expr(
            company=plan.companies[0], selector=plan.denominator_metric, period=period or ""
        )
        return f"{numerator} / {denominator}"

    if plan.operation in ("average", "sum"):
        assert plan.metric is not None
        method = "mean" if plan.operation == "average" else "sum"
        if len(plan.companies) > 1:
            return _aggregate_expr_over_companies(
                companies=plan.companies,
                selector=plan.metric,
                period=period or "",
                method=method,
            )
        return _aggregate_expr(
            company=plan.companies[0], selector=plan.metric, periods=plan.periods, method=method
        )

    if plan.operation == "rank":
        assert plan.metric is not None and plan.top_k is not None
        column, value = _metric_column_and_value(plan.metric)
        companies_list = ", ".join(_lit(company) for company in plan.companies)
        condition = (
            f"(df1.company_code.isin([{companies_list}])) & "
            f"(df1.{column} == {_lit(value)}) & "
            f"(df1.period == {int(period or '0')})"
        )
        index = plan.top_k - 1
        return f'df1[{condition}].sort_values("value", ascending=False)["value"].iloc[{index}]'

    raise ValueError(f"unsupported operation for pandas_query rendering: {plan.operation}")


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
    """Execute a rendered `pandas_query` string through a whitelist AST
    interpreter and return its scalar result. Raises ValueError on anything
    outside the grammar `render_pandas_query` produces or outside the
    structural budgets in ADR 0008 decision D3 (query length, AST node count,
    AST depth) — deny by default, never eval/exec.
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
        if isinstance(node.value, ast.Name) and node.value.id == "df1":
            if node.attr not in _ALLOWED_ATTRS:
                raise ValueError(f"unsupported column: {node.attr}")
            return frame[node.attr]
        if node.attr == "iloc":
            operand = _eval_node(node.value, frame)
            return operand.iloc
        raise ValueError(f"unsupported attribute: {node.attr}")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            raise ValueError("only a single '==' comparison is supported")
        left = _eval_node(node.left, frame)
        right = _eval_node(node.comparators[0], frame)
        return left == right

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
