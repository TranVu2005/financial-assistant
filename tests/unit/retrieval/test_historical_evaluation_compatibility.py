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

_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures/retrieval/legacy_reports"
_LEGACY_MODELS: dict[str, type[BaseModel]] = {
    "day9-dense-build.json": _DenseBuildObservation,
    "day9-dense-run.json": DenseEvaluationRun,
    "day8-bm25-report.json": RetrievalEvaluationReport,
    "day9-comparison-report.json": Day9ComparisonReport,
    "day10-entity-report.json": EntityEvaluationReport,
    "day10-held-out-report.json": HeldOutEntityReport,
    "day10-fusion-report.json": FusionGridReport,
    "day11-graph-report.json": GraphCoverageReport,
    "day12-expansion-report.json": ExpansionGridReport,
}


@pytest.mark.parametrize(("fixture_name", "model"), sorted(_LEGACY_MODELS.items()))
def test_frozen_legacy_report_fixture_parses_without_v2_fields(
    fixture_name: str, model: type[BaseModel]
) -> None:
    """Every registered legacy model has required, committed compatibility evidence."""
    fixture = _FIXTURE_ROOT / fixture_name

    assert fixture.is_file(), f"missing committed legacy fixture: {fixture_name}"
    parsed = model.model_validate_json(fixture.read_bytes())

    assert parsed is not None
