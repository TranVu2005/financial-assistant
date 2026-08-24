"""Day 19 execution sandbox (ADR 0008 decisions B2/C1/D3).

`replay_in_sandbox` is the sole caller of `replay_pandas_query` (ADR 0008
decision B2 — a test asserts no other module in `execution/` imports it).
`pandas_query.py` already denies structurally out-of-budget queries (length,
AST node count, AST depth) before running, and its interpreter's semantic
whitelist already denies most attacks with a `ValueError`. But Day 19 plan
Sec 1.3 measured that the replayer can still raise `SyntaxError`, `KeyError`,
or `decimal.InvalidOperation` on hostile or malformed input, and none of
those were being caught anywhere -- this module closes that gap by treating
*any* `Exception` from replay as a denial, never a crash.

Timing is a post-hoc measurement, not a preemptive interrupt (ADR 0008
decision D3): Day 19 plan Sec 1.6 measured that `signal.SIGALRM`,
`signal.setitimer`, and the `resource` module are all absent on win32, so
there is no in-process way to abort a running computation from the outside.
A query that runs longer than `timeout_seconds` is reported as
`budget_exceeded` only after it finishes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import pandas as pd

from financial_report_qa.execution.pandas_query import replay_pandas_query

# The sandbox's own denial codes. These used to be borrowed from the
# compiler-era `ExecutionIssueCode` (spec 2026-08-24 §8.2 removed that
# module); they describe replay denials, not compile failures, so they live
# here now.
SandboxErrorCode = Literal["query_rejected", "budget_exceeded"]


@dataclass(frozen=True)
class SandboxResult:
    """Either a resolved scalar, or a typed reason replay was denied."""

    value: Decimal | None
    error_code: SandboxErrorCode | None
    error_message: str | None
    elapsed_seconds: float


def replay_in_sandbox(query: str, frame: pd.DataFrame, *, timeout_seconds: float) -> SandboxResult:
    """Replay `query` against `frame`, converting every failure mode into a
    typed `SandboxResult` instead of letting it propagate. Only `Exception`
    subclasses are caught -- `BaseException` (KeyboardInterrupt, SystemExit)
    propagates unchanged.
    """
    started = time.perf_counter()
    try:
        value = replay_pandas_query(query, frame)
    except Exception as exc:  # noqa: BLE001 -- deny-by-default, see module docstring
        elapsed = time.perf_counter() - started
        return SandboxResult(
            value=None,
            error_code="query_rejected",
            error_message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
        )
    elapsed = time.perf_counter() - started

    if elapsed > timeout_seconds:
        return SandboxResult(
            value=None,
            error_code="budget_exceeded",
            error_message=f"replay took {elapsed:.3f}s, exceeding budget {timeout_seconds:.3f}s",
            elapsed_seconds=elapsed,
        )

    return SandboxResult(value=value, error_code=None, error_message=None, elapsed_seconds=elapsed)
