"""Loader for manually reviewed, provenance-bound retrieval gold questions."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from financial_report_qa.core.errors import RetrievalGoldError
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.release import EXPECTED_FINGERPRINT

REQUIRED_GOLD_QUESTION_COUNT = 30


def load_reviewed_gold(
    path: Path,
    *,
    expected_count: int | None = None,
    expected_fingerprint: str = EXPECTED_FINGERPRINT,
) -> tuple[GoldRetrievalQuestion, ...]:
    """Load validated JSONL gold records without silently repairing annotations."""
    if not path.is_file():
        raise RetrievalGoldError(f"Reviewed retrieval gold file not found: {path}")
    records: list[GoldRetrievalQuestion] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = GoldRetrievalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RetrievalGoldError(f"Invalid reviewed gold record at line {line_number}") from exc
        if record.question_id in seen_ids:
            raise RetrievalGoldError(f"duplicate reviewed gold question_id: {record.question_id}")
        if record.dataset_fingerprint != expected_fingerprint:
            raise RetrievalGoldError(f"Reviewed gold fingerprint mismatch for {record.question_id}")
        evidence_ids = tuple(sorted(evidence.table_id for evidence in record.gold_evidence))
        if tuple(sorted(record.gold_table_ids)) != evidence_ids:
            raise RetrievalGoldError(
                f"Gold table IDs and verified evidence differ for {record.question_id}"
            )
        seen_ids.add(record.question_id)
        records.append(record)
    result = tuple(sorted(records, key=lambda record: record.question_id))
    if expected_count is not None and len(result) != expected_count:
        raise RetrievalGoldError(
            f"Expected {expected_count} reviewed gold questions, found {len(result)}"
        )
    return result
