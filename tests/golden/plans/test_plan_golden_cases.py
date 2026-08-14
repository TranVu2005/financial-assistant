"""Day 15 golden cases: valid plans for all 8 operations, invalid cases with codes.

Every valid fixture is sourced from real release facts (see manifest.json for
provenance — a gold70 question_id where one exists, otherwise a note that no
gold70 question has that shape). Every invalid fixture violates exactly one
rule, tagged in the manifest with the layer that must catch it: pydantic
construction ("schema") or `validate_plan_semantics` ("semantic").
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from financial_report_qa.planning.plan_contracts import FinancialQueryPlan
from financial_report_qa.planning.plan_validator import validate_plan_semantics

_ROOT = Path(__file__).parents[3]
_PLANS_DIR = Path(__file__).parent
_MANIFEST = json.loads((_PLANS_DIR / "manifest.json").read_text(encoding="utf-8"))
_RELEASE_TABLES = _ROOT / "data/processed/release_v2_422df141c935/tables.parquet"


def _known_table_ids() -> frozenset[str]:
    return frozenset(
        pq.read_table(_RELEASE_TABLES, columns=["table_id"])  # type: ignore[no-untyped-call]
        .column("table_id")
        .to_pylist()
    )


@pytest.mark.parametrize("name", sorted(_MANIFEST["valid"]))
def test_valid_plan_parses_and_has_no_semantic_issues(name: str) -> None:
    payload = json.loads((_PLANS_DIR / "valid" / name).read_text(encoding="utf-8"))
    plan = FinancialQueryPlan.model_validate(payload)
    issues = validate_plan_semantics(plan, known_table_ids=_known_table_ids())
    assert issues == (), f"{name}: unexpected issues {issues}"


@pytest.mark.parametrize("name", sorted(_MANIFEST["invalid"]))
def test_invalid_plan_is_rejected_with_expected_reason(name: str) -> None:
    expectation = _MANIFEST["invalid"][name]
    payload = json.loads((_PLANS_DIR / "invalid" / name).read_text(encoding="utf-8"))

    if expectation["layer"] == "schema":
        with pytest.raises(ValidationError, match=re.escape(expectation["match"])):
            FinancialQueryPlan.model_validate(payload)
        return

    plan = FinancialQueryPlan.model_validate(payload)
    issues = validate_plan_semantics(plan, known_table_ids=_known_table_ids())
    codes = tuple(issue.code for issue in issues)
    assert expectation["code"] in codes, f"{name}: got {codes}, expected {expectation['code']}"
