"""Day 19 plan §3 task 19.8: security tests for the eight malicious-input
classes measured in `docs/plans/day19-sandbox-executor.md`.

Cases live in `data/qa/malicious-plan-cases-v1.jsonl`, each anchored to the
specific measurement that motivated it. Day 19 plan §1.10 measured that only
two free-form fields survive every upstream constraint to reach the
executor: `MetricSelector.raw_text` and `FinancialQueryPlan.companies` --
`plan_construction` cases target those two fields directly.
`attribute_escalation` and `depth_bomb` are AST-level attacks that only make
sense as raw `pandas_query` strings, tested against the sandbox directly.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.data.dataset_builder import CELL_SCHEMA, DOCUMENT_SCHEMA, TABLE_SCHEMA
from financial_report_qa.execution.compiler import compile_plan
from financial_report_qa.execution.sandbox import replay_in_sandbox
from financial_report_qa.planning.plan_contracts import FinancialQueryPlan, MetricSelector

_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "qa" / "malicious-plan-cases-v1.jsonl"
TABLE_ID = "tbl_" + "1" * 64
DOC_ID = "doc_" + "a" * 64

_ALLOW_ALL = ExecutionSettings(
    timeout_seconds=5,
    max_rows=20000,
    allow_operations=("lookup",),
)


def _load_cases() -> list[dict[str, object]]:
    with _CASES_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_payload(case: dict[str, object]) -> str:
    payload = case["payload"]
    assert isinstance(payload, str)
    if payload == "A_REPEAT_50000":
        return "A" * 50_000
    if payload == "1000_NESTED_BINOPS":
        return "1" + " + 1" * 1000
    return payload


def _write_release(tmp_path: Path, *, row_label_raw: str) -> Path:
    release_dir = tmp_path / "release"
    release_dir.mkdir(exist_ok=True)
    document = {
        "doc_id": DOC_ID,
        "repo_id": "repo",
        "revision": "1",
        "relative_path": "PNJ/2018/report.txt",
        "company_code": "PNJ",
        "report_year": 2018,
        "statement_scope": "consolidated",
        "sha256": "0" * 64,
        "file_size_bytes": 10,
        "encoding": "utf-8",
        "inventory_status": "ready",
        "ruleset_version": "1",
        "normalization_fingerprint": "0" * 64,
    }
    table = {
        "table_id": TABLE_ID,
        "doc_id": DOC_ID,
        "source_ordinal": 0,
        "title_raw": "Thuyet minh",
        "statement_type": "notes",
        "unit_raw": "VND",
        "unit_normalized": "vnd",
        "line_start": 1,
        "line_end": 10,
        "row_count": 1,
        "column_count": 2,
        "quality_score": 0.9,
        "csv_path": None,
    }
    cell = {
        "cell_id": "cell_" + "a" * 64,
        "table_id": TABLE_ID,
        "row_idx": 0,
        "col_idx": 1,
        "row_label_raw": row_label_raw,
        "row_label_canonical": None,
        "row_group_context_raw": None,
        "column_label_raw": "Năm 2018",
        "column_label_canonical": None,
        "value_raw": "42",
        "value_numeric": Decimal("42"),
        "period": "2018",
        "unit": "VND",
        "source_line_start": 1,
        "source_line_end": 1,
        "extraction_confidence": 0.9,
    }
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([document], schema=DOCUMENT_SCHEMA), release_dir / "documents.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([table], schema=TABLE_SCHEMA), release_dir / "tables.parquet"
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pylist([cell], schema=CELL_SCHEMA), release_dir / "cells.parquet"
    )
    return release_dir


def _plan_construction_case(case: dict[str, object], payload: str) -> FinancialQueryPlan | None:
    """Return the constructed plan, or None if schema construction rejected it."""
    field = case["target_field"]
    try:
        if field == "raw_text":
            metric = MetricSelector(raw_text=payload)
            return FinancialQueryPlan(
                operation="lookup",
                companies=("PNJ",),
                periods=("2018",),
                candidate_table_ids=(TABLE_ID,),
                metric=metric,
            )
        assert field == "companies"
        return FinancialQueryPlan(
            operation="lookup",
            companies=(payload,),
            periods=("2018",),
            candidate_table_ids=(TABLE_ID,),
            metric=MetricSelector(canonical="cash_and_cash_equivalents"),
        )
    except ValidationError:
        return None


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["case_id"])
def test_malicious_case_never_crashes_uncaught(case: dict[str, object], tmp_path: Path) -> None:
    payload = _resolve_payload(case)
    layer = case["layer"]
    expected = case["expected_outcome"]

    if layer == "plan_construction":
        plan = _plan_construction_case(case, payload)
        if expected == "schema_rejected":
            assert plan is None, f"{case['case_id']} should be rejected at schema construction"
            return
        # safe_render: schema may accept it (no control chars/length violation)
        # -- the guarantee is that compiling it never crashes and never
        # fabricates an answer, not that it is rejected at construction time.
        if plan is None:
            return
        release_dir = _write_release(tmp_path, row_label_raw="Tien mat that")
        result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
        assert result.status == "error"
        assert result.answer is None

    elif layer == "compile_plan":
        plan = _plan_construction_case(case, payload)
        assert plan is not None, f"{case['case_id']} must construct: real corpus label"
        release_dir = _write_release(tmp_path, row_label_raw=payload)
        result = compile_plan(plan, release_dir, execution_settings=_ALLOW_ALL)
        assert result.status == expected
        assert result.answer == Decimal("42")

    elif layer == "sandbox":
        frame = pd.DataFrame(
            {
                "company_code": ["PNJ"],
                "row_label_canonical": ["x"],
                "row_label_raw": ["x"],
                "unit": ["VND"],
                "value": [Decimal("1")],
                "period": pd.array([2018], dtype="Int64"),
            }
        )
        result = replay_in_sandbox(payload, frame, timeout_seconds=5.0)
        assert result.value is None
        assert result.error_code in ("query_rejected", "budget_exceeded")

    else:
        raise AssertionError(f"unknown layer: {layer}")
