"""Đổi phiếu đã gán nhãn tay thành `retrieval-gold-v2.jsonl` hợp lệ.

Đọc `answers.jsonl` của `scripts/sample_gold_v2.py`, kiểm tra từng nhãn ngược
về release, rồi ghi ra đúng schema `GoldRetrievalQuestion`.

Kiểm tra, fail rõ chứ không tự sửa:

  * `table_id` phải tồn tại trong `tables.parquet`.
  * Tài liệu chứa bảng đó phải khớp công ty **và** năm mà entity parser tách
    ra từ câu hỏi. Đây là cái lưới bắt lỗi chép nhầm `table_id` giữa hai phiếu
    -- một bảng của công ty khác lọt vào gold sẽ làm hỏng số đo âm thầm.
  * Câu để rỗng `gold_table_ids` bị loại khỏi tập gold, có thống kê ở cuối.

`question_id` sinh bằng `stable_question_id` -- hàm này băm cả
`gold_table_ids`, nên ID chỉ tính được **sau** khi gán nhãn xong. Trong phiếu
ta dùng `id` chính thức của ViFinQA làm khoá tạm.

Chạy:

    uv run python scripts/build_gold_v2.py \\
        --release-dir data/processed/release_v2_422df141c935 \\
        --answers data/qa/gold-v2-review/answers.jsonl \\
        --output data/qa/retrieval-gold-v2.jsonl \\
        --reviewed-by vu-tran
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from financial_report_qa.planning.entity_contracts import to_retrieval_filters
from financial_report_qa.planning.entity_parser import parse_query_entities
from financial_report_qa.retrieval.contracts import GoldRetrievalQuestion
from financial_report_qa.retrieval.gold import stable_question_id

_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"

_GROWTH = re.compile(r"tăng trưởng|thay đổi bao nhiêu|biến động|tăng bao nhiêu|giảm bao nhiêu")
_COMPARE = re.compile(r"so sánh|chênh lệch|cao nhất|thấp nhất|lớn nhất|nhỏ nhất|năm nào|đối chiếu")


def _intent(question: str, override: str | None) -> str:
    """`lookup`/`compare`/`growth`. Người gán nhãn ghi đè được qua `intent`."""
    if override:
        return override
    lowered = question.casefold()
    if _GROWTH.search(lowered):
        return "growth"
    if _COMPARE.search(lowered):
        return "compare"
    return "lookup"


def _table_provenance(
    connection: duckdb.DuckDBPyConnection, release_dir: Path, table_ids: list[str]
) -> dict[str, dict[str, Any]]:
    if not table_ids:
        return {}
    rows = connection.execute(
        """
        SELECT t.table_id, t.line_start, t.line_end,
               d.relative_path, d.company_code, d.report_year
        FROM read_parquet(?) AS t
        JOIN read_parquet(?) AS d USING (doc_id)
        WHERE t.table_id IN (SELECT UNNEST(?))
        """,
        [
            str(release_dir / "tables.parquet"),
            str(release_dir / "documents.parquet"),
            table_ids,
        ],
    ).fetchall()
    return {
        str(row[0]): {
            "line_start": int(row[1]),
            "line_end": int(row[2]),
            "relative_path": str(row[3]),
            "company_code": str(row[4]),
            "report_year": int(row[5]),
        }
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    arguments = parser.parse_args()

    answers = [
        json.loads(line)
        for line in arguments.answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions = false")
    connection.execute("SET autoload_known_extensions = false")
    provenance = _table_provenance(
        connection,
        arguments.release_dir,
        sorted({str(t) for row in answers for t in row.get("gold_table_ids", [])}),
    )
    connection.close()

    reviewed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    records: list[str] = []
    skipped: list[tuple[int, str]] = []
    problems: list[str] = []

    for row in answers:
        question_id_raw = int(row["id"])
        question = str(row["question"])
        table_ids = [str(t).strip() for t in row.get("gold_table_ids", []) if str(t).strip()]
        if not table_ids:
            skipped.append((question_id_raw, str(row.get("note", "")).strip() or "chưa gán nhãn"))
            continue

        entities = parse_query_entities(question)
        filters = to_retrieval_filters(entities)
        evidence: list[dict[str, Any]] = []
        rejected = False

        for table_id in sorted(set(table_ids)):
            record = provenance.get(table_id)
            if record is None:
                problems.append(
                    f"câu {question_id_raw}: table_id không có trong release: {table_id}"
                )
                rejected = True
                continue
            if filters.company_codes and record["company_code"] not in filters.company_codes:
                problems.append(
                    f"câu {question_id_raw}: {table_id} thuộc {record['company_code']}, "
                    f"câu hỏi nói {list(filters.company_codes)}"
                )
                rejected = True
            years = {int(p[:4]) for p in filters.periods if p[:4].isdigit()}
            if years and record["report_year"] not in years:
                problems.append(
                    f"câu {question_id_raw}: {table_id} thuộc năm {record['report_year']}, "
                    f"câu hỏi nói {sorted(years)}"
                )
                rejected = True
            evidence.append(
                {
                    "table_id": table_id,
                    "relative_path": record["relative_path"],
                    "line_start": record["line_start"],
                    "line_end": record["line_end"],
                    "verified": True,
                }
            )

        if rejected:
            continue

        canonical = tuple(sorted(set(table_ids)))
        payload = {
            "question_id": stable_question_id(question, filters, canonical, _FINGERPRINT),
            "question": question,
            "intent": _intent(question, row.get("intent")),
            "filters": filters.model_dump(mode="json"),
            "gold_table_ids": list(canonical),
            "reviewed_by": arguments.reviewed_by,
            "reviewed_at": reviewed_at,
            "gold_evidence": evidence,
            "dataset_fingerprint": _FINGERPRINT,
        }
        GoldRetrievalQuestion.model_validate(payload)
        records.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    if problems:
        print(f"❌ {len(problems)} nhãn bị từ chối:")
        for problem in problems:
            print(f"   {problem}")
        print("\nSửa answers.jsonl rồi chạy lại. Không ghi file gold.")
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(sorted(records)) + "\n", encoding="utf-8")

    print(f"đã gán nhãn : {len(records)}")
    print(f"bỏ qua      : {len(skipped)}")
    for question_id_raw, note in skipped:
        print(f"   câu {question_id_raw}: {note}")
    print(f"\nghi ra: {arguments.output}")
    print(f"dùng với: retrieval sweep-k --gold {arguments.output} --gold-count {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
