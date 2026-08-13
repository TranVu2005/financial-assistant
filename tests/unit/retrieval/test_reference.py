from __future__ import annotations

import json
from pathlib import Path

import pytest

from financial_report_qa.retrieval.dense_evaluation import _validate_bm25_reference
from financial_report_qa.retrieval.evaluation import RetrievalEvaluationReport
from financial_report_qa.retrieval.reference import (
    CURRENT_BM25_REFERENCE,
    load_bm25_reference_report,
    resolve_gold_reference,
)

_REPO_ROOT = Path(__file__).parents[3]
_CURRENT_REPORT = (
    _REPO_ROOT
    / "artifacts/evaluations/day13/bm25/retrieval-day8-422df141c935.json"
)
_GOLD30_REPORT = (
    _REPO_ROOT
    / "artifacts/evaluations/day13/gold30/bm25/retrieval-day8-422df141c935.json"
)
_CURRENT_GOLD = _REPO_ROOT / "data/qa/retrieval-gold-v1.jsonl"


def test_gold30_reference_report_remains_replayable() -> None:
    resolved = load_bm25_reference_report(_GOLD30_REPORT)

    assert resolved.descriptor.version == "gold30"
    assert resolved.report.question_count == 30
    _validate_bm25_reference(resolved.report)


def test_gold30_reference_can_be_selected_byte_exactly_from_current_gold70() -> None:
    resolved = resolve_gold_reference(_CURRENT_GOLD, version="gold30")

    assert resolved.descriptor.version == "gold30"
    assert len(resolved.selected_question_ids) == 30
    assert resolved.selected_jsonl_sha256 == resolved.descriptor.gold_sha256


def test_current_reference_rejects_a_truncated_report(tmp_path: Path) -> None:
    payload = json.loads(_CURRENT_REPORT.read_text(encoding="utf-8"))
    payload["per_question"] = []
    truncated = tmp_path / "truncated.json"
    truncated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="question_count must equal per_question"):
        _validate_bm25_reference(RetrievalEvaluationReport.model_validate(payload))
    with pytest.raises(ValueError, match="artifact SHA-256"):
        load_bm25_reference_report(truncated)


def test_gold70_is_the_current_reference() -> None:
    resolved = load_bm25_reference_report(_CURRENT_REPORT)

    assert resolved.descriptor == CURRENT_BM25_REFERENCE
    assert resolved.report.question_count == 70


def test_reference_rejects_a_tampered_intent_breakdown() -> None:
    report = RetrievalEvaluationReport.model_validate_json(_CURRENT_REPORT.read_bytes())
    tampered = report.model_copy(update={"by_intent": {}})

    with pytest.raises(ValueError, match="by_intent"):
        _validate_bm25_reference(tampered)
