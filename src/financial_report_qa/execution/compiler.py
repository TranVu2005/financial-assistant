"""Day 18/19 compile_plan orchestrator: FinancialQueryPlan -> CompiledQuery.

Ties together `cell_frame` (ADR 0007 A1/B1/C2), `locator` (D1), `operations`
(E1), `pandas_query` (F1) and `sandbox` (ADR 0008 B2/C1/D3). This is the one
place all five modules meet, and the one place the F1 replay check actually
runs: every `answered` result is re-derived independently -- through
`sandbox.replay_in_sandbox`, never by calling the `pandas_query` replayer
directly (ADR 0008 decision B2) -- on a small evidence-only frame before it
is returned, and a mismatch is a build-breaking bug
(`ExecutionReplayMismatchError`), never a silently wrong answer. Before any
of that, the plan is self-validated (ADR 0008 decision E1): compile_plan
never trusts that a caller already ran `validate_plan_semantics`.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.core.errors import ExecutionReplayMismatchError
from financial_report_qa.execution import operations
from financial_report_qa.execution.cell_frame import build_cell_frame
from financial_report_qa.execution.contracts import (
    CellMatch,
    CompiledQuery,
    ExecutionIssueCode,
    ReplayRow,
)
from financial_report_qa.execution.locator import LocateResult, locate
from financial_report_qa.execution.pandas_query import render_pandas_query
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.execution.scope_filter import (
    filter_table_ids_by_scope,
    resolve_statement_scope,
)
from financial_report_qa.normalization.units import CanonicalUnit
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector
from financial_report_qa.planning.plan_validator import validate_plan_semantics


class _CompileFailure(Exception):
    def __init__(self, code: ExecutionIssueCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(operation: str, code: ExecutionIssueCode, message: str, query: str) -> CompiledQuery:
    return CompiledQuery(
        operation=operation,  # type: ignore[arg-type]
        status="error",
        answer=None,
        unit=None,
        evidence=(),
        pandas_query=query,
        error_code=code,
        error_message=message,
    )


def _require(result: LocateResult) -> CellMatch:
    if result.match is None:
        assert result.error_code is not None and result.error_message is not None
        raise _CompileFailure(result.error_code, result.error_message)
    return result.match


def _replay_row(
    *, company_code: str, selector: MetricSelector, period: int, value: Decimal
) -> dict[str, object]:
    return {
        "company_code": company_code,
        "row_label_canonical": selector.canonical,
        "row_label_raw": selector.raw_text,
        "column_label": selector.column_text,
        "period": period,
        "value": value,
        # plan.md §14: the replay frame must carry whatever the rendered
        # query reads. A position-bound selector renders `df.loc[(df1.table_id
        # == ...) & (df1.row_idx == ...)]`, so those columns have to exist in
        # `df1` -- otherwise the sandbox replay fails on a column it was
        # itself asked to filter by.
        "table_id": selector.table_id,
        "row_idx": selector.row_index,
    }


def _replay_row_contract(row: dict[str, object]) -> ReplayRow:
    """Project one replay-frame row onto the exported contract.

    The frame column is `row_idx` because that is the name the rendered
    `df.loc[...]` filters on; the contract field is `row_index` to match
    `CellMatch`/`GroundedFact`. Renamed here rather than in either place, so
    neither the query grammar nor the provenance vocabulary has to bend.
    """
    payload = {key: value for key, value in row.items() if key != "row_idx"}
    payload["row_index"] = row.get("row_idx")
    return ReplayRow.model_validate(payload)


def _replay_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["period"] = frame["period"].astype("Int64")
    return frame


def compile_plan(
    plan: FinancialQueryPlan,
    release_dir: Path,
    *,
    execution_settings: ExecutionSettings,
) -> CompiledQuery:
    """Compile one plan against a locked release into a locked answer or a
    typed error. Never returns a guessed value (ADR 0007 decision D1)."""
    # ADR 0008 decision E1: never trust that a caller already validated the
    # plan's arity (Day 19 plan Sec 1.10 -- `top_k` arity was only enforced
    # when a caller happened to call this). Validation must run before
    # `render_pandas_query`, which itself assumes a well-formed plan and
    # raises an uncaught AssertionError on e.g. a `rank` plan missing
    # `top_k`. `candidate_table_ids` is treated as its own known set here:
    # compile_plan operates only within a plan's own bounded (1-12, ADR 0007
    # A1) candidate ids, so the release-existence check belongs to the
    # router that produced the plan, not here.
    issues = validate_plan_semantics(plan, known_table_ids=frozenset(plan.candidate_table_ids))
    if issues:
        message = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        return _error(plan.operation, "plan_rejected", message, "<plan rejected before rendering>")

    query = render_pandas_query(plan)

    if plan.operation not in execution_settings.allow_operations:
        return _error(
            plan.operation,
            "operation_not_allowed",
            f"operation '{plan.operation}' is not in the execution allowlist",
            query,
        )

    effective_scope, scope_inferred = resolve_statement_scope(
        plan_scope=plan.statement_scope,
        default_scope=execution_settings.default_statement_scope,
    )
    candidate_table_ids = plan.candidate_table_ids
    if effective_scope is not None:
        candidate_table_ids = filter_table_ids_by_scope(
            release_dir, plan.candidate_table_ids, effective_scope
        )
        if not candidate_table_ids:
            return _error(
                plan.operation,
                "candidate_table_ids_scope_empty",
                f"no candidate table has statement_scope={effective_scope!r}",
                query,
            )

    frame = build_cell_frame(release_dir, candidate_table_ids)
    if len(frame) > execution_settings.max_rows:
        return _error(
            plan.operation,
            "row_limit_exceeded",
            f"candidate frame has {len(frame)} rows, "
            f"exceeding max_rows={execution_settings.max_rows}",
            query,
        )
    period = int(plan.periods[0]) if plan.periods else None

    try:
        evidence, replay_rows, answer, unit = _dispatch(
            plan,
            frame,
            period,
            resolve_ambiguity_by_priority=execution_settings.resolve_ambiguity_by_priority,
        )
        if plan.expected_unit is not None and unit is not None and plan.expected_unit != unit:
            orig_unit = unit
            answer = operations.convert_cell_value(answer, orig_unit, plan.expected_unit)
            unit = plan.expected_unit

            from financial_report_qa.normalization.units import _MONETARY_UNITS, unit_multiplier

            if orig_unit in _MONETARY_UNITS:
                factor = unit_multiplier(orig_unit) / unit_multiplier(plan.expected_unit)
                updated_rows = []
                for row in replay_rows:
                    new_row = dict(row)
                    new_row["value"] = row["value"] * factor
                    updated_rows.append(new_row)
                replay_rows = updated_rows
            else:
                factor = unit_multiplier(orig_unit) / unit_multiplier(plan.expected_unit)
                query = f"({query}) * {factor}"
    except _CompileFailure as failure:
        return _error(plan.operation, failure.code, failure.message, query)
    except ValueError as exc:
        return _error(plan.operation, "unit_incompatible", str(exc), query)
    except ZeroDivisionError as exc:
        return _error(plan.operation, "division_by_zero", str(exc), query)

    # ADR 0008 decisions B2/C1: replay only through the sandbox, which
    # converts every exception the replayer can raise into a typed result
    # instead of letting it propagate (Day 19 plan Sec 1.3 measured 5
    # distinct uncaught exception types before this fix).
    sandbox_result = replay_in_sandbox(
        query, _replay_frame(replay_rows), timeout_seconds=execution_settings.timeout_seconds
    )
    if sandbox_result.error_code is not None:
        assert sandbox_result.error_message is not None
        return _error(
            plan.operation, sandbox_result.error_code, sandbox_result.error_message, query
        )
    replayed = sandbox_result.value
    if replayed != answer:
        raise ExecutionReplayMismatchError(
            f"pandas_query replay {replayed!r} does not match compiled answer {answer!r} "
            f"for operation '{plan.operation}': {query}"
        )

    return CompiledQuery(
        operation=plan.operation,
        status="answered",
        answer=answer,
        unit=unit,
        evidence=evidence,
        pandas_query=query,
        error_code=None,
        error_message=None,
        scope_inferred=scope_inferred,
        replay_rows=tuple(_replay_row_contract(row) for row in replay_rows),
    )


def _dispatch(
    plan: FinancialQueryPlan,
    frame: pd.DataFrame,
    period: int | None,
    *,
    resolve_ambiguity_by_priority: bool = False,
) -> tuple[tuple[CellMatch, ...], list[dict[str, object]], Decimal, CanonicalUnit]:
    company = plan.companies[0]

    def _cell(selector: MetricSelector, at_period: int, *, company_code: str) -> CellMatch:
        """One place that carries the tie-break flag down to `locate()`.

        `_dispatch` calls `locate()` at 12 branches; adding the keyword to
        each call site directly would push several lines over ruff's
        100-character limit.
        """
        return _require(
            locate(
                frame,
                selector,
                at_period,
                company_code=company_code,
                resolve_ambiguity_by_priority=resolve_ambiguity_by_priority,
            )
        )

    if plan.operation == "lookup":
        assert plan.metric is not None and period is not None
        cell = _cell(plan.metric, period, company_code=company)
        answer, unit = operations.compile_lookup(cell)
        rows = [
            _replay_row(company_code=company, selector=plan.metric, period=period, value=answer)
        ]
        return (cell,), rows, answer, unit

    if plan.operation in ("difference", "growth_rate"):
        assert plan.metric is not None
        start_period, end_period = int(plan.periods[0]), int(plan.periods[-1])
        start = _cell(plan.metric, start_period, company_code=company)
        end = _cell(plan.metric, end_period, company_code=company)
        if plan.operation == "difference":
            answer, unit = operations.compile_difference(end, start)
        else:
            answer, unit = operations.compile_growth_rate(end, start)
        start_converted = _reconvert(start, end.unit)
        rows = [
            _replay_row(
                company_code=company,
                selector=plan.metric,
                period=start_period,
                value=start_converted,
            ),
            _replay_row(
                company_code=company, selector=plan.metric, period=end_period, value=end.value
            ),
        ]
        return (start, end), rows, answer, unit

    if plan.operation == "compare":
        assert plan.metric_a is not None and plan.metric_b is not None and period is not None
        metric_a = _cell(plan.metric_a, period, company_code=company)
        metric_b = _cell(plan.metric_b, period, company_code=company)
        answer, unit = operations.compile_compare(metric_a, metric_b)
        rows = [
            _replay_row(
                company_code=company, selector=plan.metric_a, period=period, value=metric_a.value
            ),
            _replay_row(
                company_code=company,
                selector=plan.metric_b,
                period=period,
                value=_reconvert(metric_b, metric_a.unit),
            ),
        ]
        return (metric_a, metric_b), rows, answer, unit

    if plan.operation == "compare_companies":
        assert plan.metric is not None and period is not None
        company_a, company_b = plan.companies[0], plan.companies[1]
        cell_a = _cell(plan.metric, period, company_code=company_a)
        cell_b = _cell(plan.metric, period, company_code=company_b)
        answer, unit = operations.compile_compare_companies(cell_a, cell_b)
        rows = [
            _replay_row(
                company_code=company_a, selector=plan.metric, period=period, value=cell_a.value
            ),
            _replay_row(
                company_code=company_b,
                selector=plan.metric,
                period=period,
                value=_reconvert(cell_b, cell_a.unit),
            ),
        ]
        return (cell_a, cell_b), rows, answer, unit

    if plan.operation == "ratio":
        assert (
            plan.numerator_metric is not None
            and plan.denominator_metric is not None
            and period is not None
        )
        numerator = _cell(plan.numerator_metric, period, company_code=company)
        denominator = _cell(plan.denominator_metric, period, company_code=company)
        answer, unit = operations.compile_ratio(numerator, denominator)
        rows = [
            _replay_row(
                company_code=company,
                selector=plan.numerator_metric,
                period=period,
                value=numerator.value,
            ),
            _replay_row(
                company_code=company,
                selector=plan.denominator_metric,
                period=period,
                value=_reconvert(denominator, numerator.unit),
            ),
        ]
        return (numerator, denominator), rows, answer, unit

    if plan.operation in ("average", "sum"):
        assert plan.metric is not None
        # `_validate_aggregate` (plan_validator.py) allows exactly one of
        # (companies, periods) to vary -- this must dispatch on which one
        # actually does, not always assume periods (Day 23 plan Step 2
        # regression: a >1-company plan was silently averaged/summed over
        # just companies[0], the rest never located at all).
        if len(plan.companies) > 1:
            assert period is not None
            cells = tuple(_cell(plan.metric, period, company_code=c) for c in plan.companies)
            target_unit = cells[0].unit
            rows = [
                _replay_row(
                    company_code=c,
                    selector=plan.metric,
                    period=period,
                    value=_reconvert(cell, target_unit),
                )
                for c, cell in zip(plan.companies, cells, strict=True)
            ]
        else:
            cells = tuple(
                _cell(plan.metric, int(p), company_code=company) for p in plan.periods
            )
            target_unit = cells[0].unit
            rows = [
                _replay_row(
                    company_code=company,
                    selector=plan.metric,
                    period=int(p),
                    value=_reconvert(cell, target_unit),
                )
                for p, cell in zip(plan.periods, cells, strict=True)
            ]
        answer, unit = (
            operations.compile_average(cells)
            if plan.operation == "average"
            else operations.compile_sum(cells)
        )
        return cells, rows, answer, unit

    if plan.operation == "rank":
        assert plan.metric is not None and plan.top_k is not None and period is not None
        cells = tuple(_cell(plan.metric, period, company_code=c) for c in plan.companies)
        answer, unit = operations.compile_rank(cells, top_k=plan.top_k)
        target_unit = cells[0].unit
        rows = [
            _replay_row(
                company_code=c,
                selector=plan.metric,
                period=period,
                value=_reconvert(cell, target_unit),
            )
            for c, cell in zip(plan.companies, cells, strict=True)
        ]
        return cells, rows, answer, unit

    raise _CompileFailure("operation_not_allowed", f"unsupported operation: {plan.operation}")


def _reconvert(cell: CellMatch, target_unit: str) -> Decimal:
    if cell.unit == target_unit:
        return cell.value
    return operations.convert_cell_value(cell.value, cell.unit, target_unit)
