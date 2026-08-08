import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from financial_report_qa.core.errors import RetrievalGoldError
from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import load_reviewed_gold, stable_question_id
from financial_report_qa.retrieval.release import EXPECTED_FINGERPRINT


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
