"""Regression coverage for persisted Day 9--13 evaluation schemas."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from financial_report_qa.planning.entity_evaluation import (
    EntityEvaluationReport,
    HeldOutEntityReport,
)
from financial_report_qa.retrieval.cli import _DenseBuildObservation
from financial_report_qa.retrieval.dense_evaluation import (
    Day9ComparisonReport,
    DenseEvaluationRun,
)
from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReport
from financial_report_qa.retrieval.expansion_evaluation import ExpansionGridReport
from financial_report_qa.retrieval.fusion_evaluation import FusionGridReport
from financial_report_qa.retrieval.graph_evaluation import GraphCoverageReport


def _artifact_root() -> Path | None:
    checkout = Path(__file__).resolve().parents[3]
    candidates = [checkout / "artifacts" / "evaluations"]
    if checkout.parent.name == ".worktrees":
        candidates.append(checkout.parent.parent / "artifacts" / "evaluations")
    return next(
        (
            candidate
            for candidate in candidates
            if all((candidate / f"day{day}").is_dir() for day in (9, 10, 11, 13))
        ),
        None,
    )


def _legacy_model(path: Path) -> type[BaseModel]:
    name = path.name
    if name.endswith("-build.json") or name in {"bge-build.json", "e5-build.json"}:
        return _DenseBuildObservation
    if name in {"bge-report.json", "e5-report.json", "dense-bge-m3.json", "dense-e5-small.json"}:
        return DenseEvaluationRun
    if name.startswith("retrieval-day8-"):
        return RetrievalEvaluationReport
    if name.startswith("retrieval-day9-dense-"):
        return Day9ComparisonReport
    if name.startswith("retrieval-day10-fusion-"):
        return FusionGridReport
    if name.startswith("retrieval-day11-graph-"):
        return GraphCoverageReport
    if name.startswith("retrieval-day12-expansion-"):
        return ExpansionGridReport
    if name.startswith("entity-cases-"):
        return EntityEvaluationReport
    if name.startswith("entity-held-out-"):
        return HeldOutEntityReport
    raise AssertionError(f"No legacy model registered for {path}")


def test_all_existing_day9_to_day13_json_artifacts_parse_with_legacy_models() -> None:
    """Adding V2 fields must not invalidate any persisted strict legacy report."""
    root = _artifact_root()
    if root is None:
        pytest.skip("historical evaluation artifacts are not available in this checkout")
    paths = tuple(
        sorted(
            path
            for day in (9, 10, 11, 13)
            for path in (root / f"day{day}").rglob("*.json")
        )
    )
    assert paths, "expected historical evaluation artifacts"

    parsed = []
    for path in paths:
        parsed.append(_legacy_model(path).model_validate_json(path.read_bytes()))

    assert len(parsed) == len(paths)
