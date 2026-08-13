from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReportV2
from financial_report_qa.retrieval.system_evaluation import RetrievalSystemReportV2

_REPO_ROOT = Path(__file__).parents[3]
_V2_ROOT = _REPO_ROOT / "artifacts/evaluations/day13/v2"
_EXPECTED_F2_AT_R = {
    "bm25-v3": 0.49134585652442808,
    "dense-bge-m3": 0.2241741393527108,
    "dense-e5-small": 0.18460884353741497,
    "fusion-bge": 0.49412878787878795,
    "fusion-e5": 0.49412878787878795,
    "graph-expansion": 0.49412878787878795,
}


def test_every_claimed_system_has_a_complete_source_bound_v2_artifact() -> None:
    paths = sorted(_V2_ROOT.glob("retrieval-v2-*-422df141c935.json"))
    reports = [RetrievalSystemReportV2.model_validate_json(path.read_bytes()) for path in paths]

    assert {report.system_name for report in reports} == set(_EXPECTED_F2_AT_R)
    for report in reports:
        assert report.question_count == len(report.per_question) == 70
        assert set(report.by_intent) == {"compare", "growth", "lookup"}
        assert set(report.by_gold_cardinality) == {
            "one_table",
            "two_tables",
            "three_or_more",
        }
        assert set(report.by_period_cardinality) == {"one_period", "multiple_periods"}
        assert set(report.by_statement_filter) == {"filtered", "unfiltered"}
        assert set(report.by_report_era) == {"2015_2019", "2020_2023", "2024_2025"}
        source_path = _REPO_ROOT / report.source_path
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report.source_sha256
        assert report.macro.f2_at_r == pytest.approx(_EXPECTED_F2_AT_R[report.system_name])


def test_bm25_diagnostic_v2_artifact_is_complete_and_has_safe_ranks() -> None:
    report = RetrievalEvaluationReportV2.model_validate_json(
        (_V2_ROOT / "bm25-diagnostic/retrieval-v2-422df141c935.json").read_bytes()
    )

    assert report.question_count == len(report.per_question) == 70
    assert report.diagnostic_k == 100
    assert set(report.by_gold_cardinality) == {
        "one_table",
        "two_tables",
        "three_or_more",
    }
    assert all(
        rank is None or 11 <= rank <= 100
        for item in report.per_question
        for rank in item.gold_rank_beyond_10.values()
    )
