import hashlib
import json
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from financial_report_qa.core.errors import RetrievalGoldError
from financial_report_qa.evaluation.week1_release import ReleaseLock
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import (
    load_gold_questions,
    load_reviewed_gold,
    stable_question_id,
)
from financial_report_qa.retrieval.release import (
    EXPECTED_FINGERPRINT,
    ResolvedRetrievalRelease,
)


def _record(question_id: str = "retq_001") -> dict[str, object]:
    table_id = "tbl_" + "a" * 64
    return {
        "question_id": question_id,
        "question": "Doanh thu năm 2024 là bao nhiêu?",
        "intent": "lookup",
        "filters": {"company_codes": ["ACB"], "periods": ["2024"]},
        "gold_table_ids": [table_id],
        "gold_evidence": [
            {
                "table_id": table_id,
                "relative_path": "data/sources/acb.txt",
                "line_start": 1,
                "line_end": 3,
                "verified": True,
            }
        ],
        "reviewed_by": "reviewer",
        "reviewed_at": "2026-08-08T00:00:00+00:00",
        "dataset_fingerprint": EXPECTED_FINGERPRINT,
    }


def test_load_reviewed_gold_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    record = _record("retq_" + "1" * 64)
    path.write_text("\n".join(json.dumps(record) for _ in range(2)), encoding="utf-8")

    with pytest.raises(RetrievalGoldError, match="duplicate"):
        load_reviewed_gold(path)


def test_load_reviewed_gold_requires_expected_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    record = _record("retq_" + "1" * 64)
    record["dataset_fingerprint"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RetrievalGoldError, match="fingerprint"):
        load_reviewed_gold(path)


def test_stable_question_id_uses_exact_canonical_payload() -> None:
    filters = RetrievalFilters(company_codes=("VCB",), periods=("2023",))
    question = "  Doanh\u00a0thu  VCB  "
    gold_table_ids = ("tbl_" + "a" * 64,)
    fingerprint = "b" * 64
    normalized_question = " ".join(unicodedata.normalize("NFKC", question).split())
    payload = {
        "contract_version": "retrieval-gold-v1",
        "dataset_fingerprint": fingerprint,
        "filters": filters.model_dump(mode="json"),
        "gold_table_ids": list(gold_table_ids),
        "question": normalized_question,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert stable_question_id(question, filters, gold_table_ids, fingerprint) == (
        "retq_" + hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    )


def test_stable_question_id_treats_nfkc_and_whitespace_as_equivalent() -> None:
    filters = RetrievalFilters(company_codes=("VCB",))
    gold_table_ids = ("tbl_" + "a" * 64,)
    fingerprint = "b" * 64

    normalized_id = stable_question_id(
        "Doanh\u00a0thu \uff36\uff23\uff22", filters, gold_table_ids, fingerprint
    )
    plain_id = stable_question_id("Doanh thu VCB", filters, gold_table_ids, fingerprint)

    assert normalized_id == plain_id


def test_load_reviewed_gold_rejects_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    first = _record("retq_" + "1" * 64)
    second = _record("retq_" + "2" * 64)
    path.write_text(
        f"{json.dumps(first)}\n\n{json.dumps(second)}\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalGoldError, match="blank line"):
        load_reviewed_gold(path)


def test_load_reviewed_gold_rejects_noncanonical_record_order(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    first = _record("retq_" + "2" * 64)
    second = _record("retq_" + "1" * 64)
    second["question"] = "Cau hoi khac"
    path.write_text(
        f"{json.dumps(first)}\n{json.dumps(second)}\n",
        encoding="utf-8",
    )

    with pytest.raises(RetrievalGoldError, match="sorted"):
        load_reviewed_gold(path)


def test_gold_period_filter_accepts_cell_period_not_only_report_year(tmp_path: Path) -> None:
    table_id = "tbl_" + "a" * 64
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "table_id": [table_id],
                "doc_id": ["doc_a"],
                "statement_type": ["income_statement"],
                "line_start": [1],
                "line_end": [3],
            }
        ),
        release_dir / "tables.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table(
            {
                "doc_id": ["doc_a"],
                "company_code": ["ACB"],
                "report_year": [2024],
                "relative_path": ["ACB/report.txt"],
            }
        ),
        release_dir / "documents.parquet",
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.table({"table_id": [table_id], "period": ["2023"]}),
        release_dir / "cells.parquet",
    )
    lock = ReleaseLock(
        alias="dataset-pilot-v1",
        sampling_version="week1-pilot-v1",
        dataset_fingerprint=EXPECTED_FINGERPRINT,
        source_manifest_sha256="0" * 64,
        release_path="release",
        gate_result_path="gate.json",
        evaluation_inputs_sha256="1" * 64,
    )
    release = ResolvedRetrievalRelease(
        lock=lock,
        dataset_fingerprint=EXPECTED_FINGERPRINT,
        release_dir=release_dir,
        gate_result_path=tmp_path / "gate.json",
        lock_path=tmp_path / "lock.json",
        manifest={},
        lock_sha256="2" * 64,
    )
    filters = RetrievalFilters(company_codes=("ACB",), periods=("2023",))
    record = _record()
    record["filters"] = filters.model_dump(mode="json")
    record["question_id"] = stable_question_id(
        str(record["question"]), filters, (table_id,), EXPECTED_FINGERPRINT
    )
    record["gold_evidence"] = [
        {
            "table_id": table_id,
            "relative_path": "ACB/report.txt",
            "line_start": 1,
            "line_end": 3,
            "verified": True,
        }
    ]
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    questions = load_gold_questions(gold_path, release, require_count=1)

    assert questions[0].filters.periods == ("2023",)
