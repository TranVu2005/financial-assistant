"""Loader for manually reviewed, provenance-bound retrieval gold questions."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import ValidationError

from financial_report_qa.core.errors import RetrievalGoldError
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion, RetrievalFilters
from financial_report_qa.retrieval.release import EXPECTED_FINGERPRINT, ResolvedRetrievalRelease

REQUIRED_GOLD_QUESTION_COUNT = 70


def stable_question_id(
    question: str,
    filters: RetrievalFilters,
    gold_table_ids: tuple[str, ...],
    dataset_fingerprint: str,
) -> str:
    """Derive an ID from immutable semantic fields, never BM25 predictions."""
    normalized_question = " ".join(unicodedata.normalize("NFKC", question).split())
    payload = json.dumps(
        {
            "contract_version": "retrieval-gold-v1",
            "dataset_fingerprint": dataset_fingerprint,
            "filters": filters.model_dump(mode="json"),
            "gold_table_ids": list(gold_table_ids),
            "question": normalized_question,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "retq_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    previous_id: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise RetrievalGoldError(f"Reviewed retrieval gold contains blank line {line_number}")
        try:
            record = GoldRetrievalQuestion.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise RetrievalGoldError(f"Invalid reviewed gold record at line {line_number}") from exc
        if record.question_id in seen_ids:
            raise RetrievalGoldError(f"duplicate reviewed gold question_id: {record.question_id}")
        if previous_id is not None and record.question_id < previous_id:
            raise RetrievalGoldError(
                "Reviewed retrieval gold records must be sorted by question_id"
            )
        if record.dataset_fingerprint != expected_fingerprint:
            raise RetrievalGoldError(f"Reviewed gold fingerprint mismatch for {record.question_id}")
        evidence_ids = tuple(sorted(evidence.table_id for evidence in record.gold_evidence))
        if tuple(sorted(record.gold_table_ids)) != evidence_ids:
            raise RetrievalGoldError(
                f"Gold table IDs and verified evidence differ for {record.question_id}"
            )
        seen_ids.add(record.question_id)
        previous_id = record.question_id
        records.append(record)
    result = tuple(records)
    if expected_count is not None and len(result) != expected_count:
        raise RetrievalGoldError(
            f"Expected {expected_count} reviewed gold questions, found {len(result)}"
        )
    return result


def load_gold_questions(
    path: Path,
    release: ResolvedRetrievalRelease,
    *,
    require_count: int = REQUIRED_GOLD_QUESTION_COUNT,
) -> tuple[GoldRetrievalQuestion, ...]:
    """Load gold only when every label matches the locked release provenance."""
    questions = load_reviewed_gold(
        path, expected_count=require_count, expected_fingerprint=release.dataset_fingerprint
    )
    tables = {
        str(row["table_id"]): row
        for row in pq.read_table(release.release_dir / "tables.parquet").to_pylist()  # type: ignore[no-untyped-call]
    }
    documents = {
        str(row["doc_id"]): row
        for row in pq.read_table(release.release_dir / "documents.parquet").to_pylist()  # type: ignore[no-untyped-call]
    }
    cell_periods: dict[str, set[str]] = {}
    for row in pq.read_table(  # type: ignore[no-untyped-call]
        release.release_dir / "cells.parquet", columns=["table_id", "period"]
    ).to_pylist():
        period = row.get("period")
        if period is not None:
            cell_periods.setdefault(str(row["table_id"]), set()).add(str(period))
    seen_questions: set[str] = set()
    for question in questions:
        if question.question_id != stable_question_id(
            question.question,
            question.filters,
            question.gold_table_ids,
            question.dataset_fingerprint,
        ):
            raise RetrievalGoldError(f"question_id is not stable for {question.question_id}")
        if question.question in seen_questions:
            raise RetrievalGoldError("duplicate reviewed gold question text")
        seen_questions.add(question.question)
        for evidence in question.gold_evidence:
            table = tables.get(evidence.table_id)
            if table is None:
                raise RetrievalGoldError(
                    f"gold table is absent from locked release: {evidence.table_id}"
                )
            document = documents.get(str(table["doc_id"]))
            if document is None:
                raise RetrievalGoldError(f"gold table has no locked document: {evidence.table_id}")
            if (
                evidence.relative_path != document.get("relative_path")
                or evidence.line_start != table.get("line_start")
                or evidence.line_end != table.get("line_end")
            ):
                raise RetrievalGoldError(f"gold evidence provenance mismatch: {evidence.table_id}")
            if (
                question.filters.company_codes
                and document.get("company_code") not in question.filters.company_codes
            ):
                raise RetrievalGoldError(f"gold table violates company filter: {evidence.table_id}")
            table_periods = cell_periods.get(evidence.table_id, set()).union(
                {str(document.get("report_year"))}
                if document.get("report_year") is not None
                else set()
            )
            if question.filters.periods and not table_periods.intersection(
                question.filters.periods
            ):
                raise RetrievalGoldError(f"gold table violates period filter: {evidence.table_id}")
            if (
                question.filters.statement_types
                and table.get("statement_type") not in question.filters.statement_types
            ):
                raise RetrievalGoldError(
                    f"gold table violates statement filter: {evidence.table_id}"
                )
    return questions
