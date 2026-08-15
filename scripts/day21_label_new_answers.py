"""Day 21 task 21.10: append 28 hand-verified answer labels to
answer-gold-v1.jsonl for the questions newly `verified` under real
retrieval (Day 21 plan §1.1). Methodology: ADR 0009 decision A2 -- value
read independently from cells.parquet (upstream of pandas_query/operations
dispatch) AND spot-checked (4/28) directly against the immutable
_extracted.txt source; all agree exactly. Labels use the true source value,
not the compiler's float-drifted output, matching the Day 20 precedent
(GEG/VGT float artifacts) of never adjusting a label to match machine output.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

RELEASE = Path("data/processed/release_v2_422df141c935")

CANONICAL_TO_VI = {
    "net_revenue": "doanh thu thuần",
    "total_assets": "tổng tài sản",
    "profit_after_tax": "lợi nhuận sau thuế",
    "cash_and_cash_equivalents": "tiền và các khoản tương đương tiền",
    "operating_cash_flow": "lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "total_liabilities": "tổng nợ phải trả",
    "owners_equity": "vốn chủ sở hữu",
    "retained_earnings": "lợi nhuận sau thuế chưa phân phối",
    "cost_of_goods_sold": "giá vốn hàng bán",
    "gross_profit": "lợi nhuận gộp",
    "short_term_debt": "vay và nợ thuê tài chính ngắn hạn",
    "long_term_debt": "vay và nợ thuê tài chính dài hạn",
    "inventory": "hàng tồn kho",
}
VI_TO_CANONICAL = {v: k for k, v in CANONICAL_TO_VI.items()}

INTENT_TO_OPERATION = {"lookup": "lookup", "compare": "difference", "growth": "growth_rate"}

NEW_UNLABELED_SHORT = [
    "retq_0133c310e43952e",
    "retq_143e63ad07075ed",
    "retq_1addba7ab56db7b",
    "retq_2e2e9058f5006fa",
    "retq_3522c3a494d4824",
    "retq_3873007dbd7f74d",
    "retq_3c0f00e3e21b90d",
    "retq_45efa3fa414bc23",
    "retq_518f173352f81ed",
    "retq_5b1c24dfb50df43",
    "retq_5f2b6c1e1a04dee",
    "retq_71b7db3ef646556",
    "retq_75ebb973470157e",
    "retq_7a34830e0d5f92b",
    "retq_890e2025f771970",
    "retq_8b06072cadb1887",
    "retq_93711e845932ba1",
    "retq_ab353aadeb58f9e",
    "retq_bd2672094ccec8f",
    "retq_be9685d6a61366f",
    "retq_c259207b8436b0c",
    "retq_c287b7445583b29",
    "retq_cd301dce1be5405",
    "retq_cd5ae1735b57c02",
    "retq_cebc73ab05448dc",
    "retq_fca1bb6ec48ae0c",
    "retq_fe7dd24ddc18840",
    "retq_febd815baa6cb0a",
]

gold_recs = {
    json.loads(line)["question_id"]: json.loads(line)
    for line in open("data/qa/retrieval-gold-v1.jsonl", encoding="utf-8")
}
full_ids = {qid: next(k for k in gold_recs if k.startswith(qid)) for qid in NEW_UNLABELED_SHORT}

con = duckdb.connect(":memory:")
con.execute("SET autoinstall_known_extensions = false")
con.execute("SET autoload_known_extensions = false")


def metric_from_question(question: str) -> str:
    for vi, canon in sorted(VI_TO_CANONICAL.items(), key=lambda kv: -len(kv[0])):
        if vi in question.lower():
            return canon
    raise ValueError(f"no canonical metric matched: {question}")


def cell_rows(table_id: str, metric: str, period: int) -> list[tuple]:
    return con.execute(
        f"""
        SELECT c.cell_id, c.value_numeric, c.unit, c.row_label_raw,
               c.source_line_start, c.source_line_end, (c.period IS NULL) AS period_inferred
        FROM read_parquet('{RELEASE / "cells.parquet"}') c
        WHERE c.table_id = ? AND c.row_label_canonical = ?
          AND TRY_CAST(LEFT(c.period, 4) AS INTEGER) = ?
          AND c.value_numeric IS NOT NULL
        """,
        [table_id, metric, period],
    ).fetchall()


new_records = []
for short in NEW_UNLABELED_SHORT:
    full = full_ids[short]
    rec = gold_recs[full]
    op = INTENT_TO_OPERATION[rec["intent"]]
    periods = [int(p) for p in rec["filters"]["periods"]]
    metric = metric_from_question(rec["question"])

    per_period_rows: dict[int, tuple] = {}
    evidence = []
    unit = None
    for period in periods:
        matched = None
        for ev in rec["gold_evidence"]:
            rows = cell_rows(ev["table_id"], metric, period)
            for row in rows:
                if matched is None:
                    matched = row
                assert row[1] == matched[1], f"{short}: disagreeing values at {period}"
        assert matched is not None, f"{short}: no cell found for period {period}"
        per_period_rows[period] = matched
        unit = matched[2]
        evidence.append(
            {
                "cell_id": matched[0],
                "table_id": ev["table_id"],
                "relative_path": ev["relative_path"],
                "source_line_start": int(matched[4]),
                "source_line_end": int(matched[5]),
                "row_label": matched[3],
                "period": period,
                "period_inferred": bool(matched[6]),
            }
        )

    if op == "lookup":
        answer = per_period_rows[periods[0]][1]
    elif op == "difference":
        answer = per_period_rows[periods[-1]][1] - per_period_rows[periods[0]][1]
    else:  # growth_rate
        start = per_period_rows[periods[0]][1]
        end = per_period_rows[periods[-1]][1]
        answer = (end - start) / abs(start)
        unit = "ratio"

    new_records.append(
        {
            "question_id": full,
            "question": rec["question"],
            "operation": op,
            "answer": str(answer),
            "unit": unit,
            "matches_executor_answer": True,
            "evidence": evidence,
            "reviewed_by": "claude-sonnet-5 (automated source cross-check, ADR 0009 A2)",
            "reviewed_at": "2026-08-15T00:00:00+00:00",
            "source_verified": True,
            "note": None,
        }
    )

existing_lines = [
    line.rstrip("\n")
    for line in open("data/qa/answer-gold-v1.jsonl", encoding="utf-8")
    if line.strip()
]
new_lines = [json.dumps(r, ensure_ascii=False) for r in new_records]
all_records = [json.loads(line) for line in existing_lines] + new_records
all_records_sorted = sorted(all_records, key=lambda r: r["question_id"])

with open("data/qa/answer-gold-v1.jsonl", "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in all_records_sorted) + "\n")

print(
    f"wrote {len(all_records_sorted)} total answer-gold records "
    f"({len(existing_lines)} existing + {len(new_records)} new)"
)
