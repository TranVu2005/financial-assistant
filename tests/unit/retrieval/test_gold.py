import json
from pathlib import Path

import pytest

from financial_report_qa.core.errors import RetrievalGoldError
from financial_report_qa.retrieval.gold import load_reviewed_gold
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
