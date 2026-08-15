"""Day 22 integration test: export + validate the submission bundle against
the REAL locked release, REAL BM25 v4 index, and a handful of REAL ViFinQA
questions -- not a synthetic fixture. Exercises exactly the wiring the CLI
uses (index loading, dataset_fingerprint match, real parquet reads), without
asserting answer correctness (no gold labels exist for this question set,
Day 22 plan §3): the point is that the pipeline runs end-to-end on real data
without crashing and, if anything is answered, the packaged ZIP validates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from financial_report_qa.core.config import ExecutionSettings
from financial_report_qa.retrieval.index import load_bm25_index
from financial_report_qa.retrieval.release import resolve_retrieval_release
from financial_report_qa.retrieval.service import RetrievalService
from financial_report_qa.submission.contracts import RawQuestion
from financial_report_qa.submission.exporter import export_submission, write_submission_zip
from financial_report_qa.submission.validator import validate_submission_zip

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RELEASE_LOCK = _REPO_ROOT / "data" / "qa" / "week1_pilot_422df141c935" / "dataset-pilot-v1.json"
_BM25_INDEX_DIR = (
    _REPO_ROOT
    / "data"
    / "indexes"
    / "bm25-v4"
    / "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
)
_QUESTIONS_PATH = _REPO_ROOT / "data" / "raw" / "ViFinQA" / "questions" / "questions.jsonl"

_ALLOW_ALL = ExecutionSettings(
    timeout_seconds=5,
    max_rows=200000,
    allow_operations=(
        "lookup",
        "compare",
        "compare_companies",
        "difference",
        "growth_rate",
        "ratio",
        "average",
        "sum",
        "rank",
    ),
)


def _skip_unless_real_corpus_available() -> None:
    if not (_RELEASE_LOCK.exists() and _BM25_INDEX_DIR.exists() and _QUESTIONS_PATH.exists()):
        pytest.skip("real locked release / bm25-v4 index / ViFinQA questions not present locally")


def test_export_and_validate_a_handful_of_real_questions(tmp_path: Path) -> None:
    _skip_unless_real_corpus_available()

    release = resolve_retrieval_release(_RELEASE_LOCK, repo_root=_REPO_ROOT)
    index = load_bm25_index(_BM25_INDEX_DIR)
    assert index.manifest.dataset_fingerprint == release.dataset_fingerprint
    service = RetrievalService(index)

    with _QUESTIONS_PATH.open(encoding="utf-8") as handle:
        import json

        questions = tuple(
            RawQuestion.model_validate(json.loads(line))
            for _, line in zip(range(3), handle, strict=False)
        )
    assert len(questions) == 3

    report, items, csv_rows = export_submission(
        questions,
        service,
        release.release_dir,
        execution_settings=_ALLOW_ALL,
        dataset_fingerprint=release.dataset_fingerprint,
        k=10,
    )
    assert report.question_count == 3

    zip_path = tmp_path / "submission.zip"
    write_submission_zip(items, csv_rows, zip_path)
    expected_ids = [outcome.id for outcome in report.outcomes if outcome.status == "answered"]
    validation = validate_submission_zip(zip_path, expected_ids)
    assert validation.valid is True, validation.issues
