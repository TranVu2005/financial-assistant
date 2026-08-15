"""Day 21 task 21.9: generate 50 leakage-safe additions to retrieval-gold-v1.jsonl.

Rules (retrieval-gold-v1.provenance.md, kept identical for the new batch):
  1. Tables/documents selected from tables.parquet/documents.parquet by metadata,
     BEFORE any question text is written.
  2. Questions/labels written from canonical tables.parquet/cells.parquet content
     and immutable cell provenance (source_line_start/end) -- no BM25/dense/
     fusion/graph ranked list is opened or used to choose a gold table.
  3. IDs derived with stable_question_id; sorted by question_id.
  4. <=2 additions per company (Day 13 quota discipline).

New constraint (Day 21 plan Sec 1.6): the 50 additions must correct gold70's
scope-wording skew (gold70 22.9% stated, consolidated-leaning; official corpus
37.7% stated, separate-leaning 36.4% vs 1.3%).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import duckdb

from financial_report_qa.retrieval.contracts import RetrievalFilters
from financial_report_qa.retrieval.gold import stable_question_id

FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"
R = Path("data/processed/release_v2_422df141c935")
GOLD_PATH = Path("data/qa/retrieval-gold-v1.jsonl")

con = duckdb.connect(":memory:")
con.execute("SET autoinstall_known_extensions = false")
con.execute("SET autoload_known_extensions = false")

# Canonical -> a natural Vietnamese display phrase for question text.
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

existing_raw_lines = {
    json.loads(line)["question_id"]: line.rstrip("\n")
    for line in GOLD_PATH.open(encoding="utf-8")
    if line.strip()
}
existing = [json.loads(line) for line in existing_raw_lines.values()]
existing_ids = {rec["question_id"] for rec in existing}
existing_company_counts: dict[str, int] = {}
for rec in existing:
    for code in rec["filters"]["company_codes"]:
        existing_company_counts[code] = existing_company_counts.get(code, 0) + 1

# 1. Metadata-only selection: safe (company, scope, metric, period) lookup groups.
groups = con.execute(
    f"""
    WITH s AS (
        SELECT d.company_code AS co, d.statement_scope AS scope, c.row_label_canonical AS metric,
               TRY_CAST(LEFT(c.period, 4) AS INTEGER) AS period, c.value_numeric AS value,
               c.table_id AS table_id
        FROM read_parquet('{R / "cells.parquet"}') c
        JOIN read_parquet('{R / "tables.parquet"}') t USING (table_id)
        JOIN read_parquet('{R / "documents.parquet"}') d USING (doc_id)
        WHERE c.row_label_canonical IS NOT NULL AND c.value_numeric IS NOT NULL
          AND c.period IS NOT NULL AND d.statement_scope IN ('separate', 'consolidated')
          AND TRY_CAST(LEFT(c.period, 4) AS INTEGER) IS NOT NULL
          AND d.report_year = TRY_CAST(LEFT(c.period, 4) AS INTEGER)
    ),
    g AS (
        SELECT co, scope, metric, period, count(DISTINCT value) AS n_val,
               count(DISTINCT table_id) AS n_tables
        FROM s GROUP BY 1,2,3,4
    )
    SELECT co, scope, metric, period FROM g WHERE n_val = 1
    """
).fetchall()

print(f"safe (company,scope,metric,period) lookup groups: {len(groups)}")


def cell_rows_for(co: str, scope: str, metric: str, period: int) -> list[tuple]:
    return con.execute(
        f"""
        SELECT c.table_id, d.relative_path, c.source_line_start, c.source_line_end
        FROM read_parquet('{R / "cells.parquet"}') c
        JOIN read_parquet('{R / "tables.parquet"}') t USING (table_id)
        JOIN read_parquet('{R / "documents.parquet"}') d USING (doc_id)
        WHERE d.company_code = ? AND d.statement_scope = ? AND c.row_label_canonical = ?
          AND TRY_CAST(LEFT(c.period, 4) AS INTEGER) = ? AND c.value_numeric IS NOT NULL
          AND d.report_year = ?
        """,
        [co, scope, metric, period, period],
    ).fetchall()


def _dedupe_evidence(evidence: list[dict[str, object]]) -> list[dict[str, object]]:
    """GoldTableEvidence requires exactly one item per table_id, sorted."""
    by_table: dict[str, dict[str, object]] = {}
    for item in evidence:
        by_table.setdefault(str(item["table_id"]), item)
    return [by_table[tid] for tid in sorted(by_table)]


rng = random.Random(20260815)
by_key: dict[tuple, list[tuple]] = {}
for co, scope, metric, period in groups:
    if metric not in CANONICAL_TO_VI:
        continue
    by_key.setdefault((co, scope), []).append((metric, period))

companies_with_data = sorted(by_key)
rng.shuffle(companies_with_data)

TARGET_NEW = 50
STATED_SEPARATE_TARGET = 32
STATED_CONSOLIDATED_TARGET = 2
company_quota: dict[str, int] = dict(existing_company_counts)
records: list[dict[str, object]] = []
used_question_texts: set[str] = set()
stated_sep = stated_con = unstated = 0


def make_lookup(co: str, scope: str, metric: str, period: int, *, state_scope: bool) -> dict | None:
    rows = cell_rows_for(co, scope, metric, period)
    if not rows:
        return None
    table_ids = sorted({r[0] for r in rows})
    vi = CANONICAL_TO_VI[metric]
    scope_phrase = ""
    if state_scope:
        scope_phrase = " riêng" if scope == "separate" else " hợp nhất"
    question = f"Tra cứu {vi}{scope_phrase} của {co} năm {period}."
    if question in used_question_texts:
        return None
    filters = RetrievalFilters(company_codes=(co,), periods=(str(period),))
    evidence = [
        {
            "table_id": tid,
            "relative_path": path,
            "line_start": int(ls),
            "line_end": int(le),
            "verified": True,
        }
        for tid, path, ls, le in rows
    ]
    evidence = _dedupe_evidence(evidence)
    qid = stable_question_id(question, filters, tuple(table_ids), FINGERPRINT)
    if qid in existing_ids or qid in {r["question_id"] for r in records}:
        return None
    used_question_texts.add(question)
    return {
        "question_id": qid,
        "question": question,
        "intent": "lookup",
        "filters": filters.model_dump(mode="json"),
        "gold_table_ids": table_ids,
        "reviewed_by": "day21-auto-expansion",
        "reviewed_at": "2026-08-15T00:00:00+00:00",
        "gold_evidence": evidence,
        "dataset_fingerprint": FINGERPRINT,
    }


def make_two_period(
    co: str, scope: str, metric: str, p1: int, p2: int, *, state_scope: bool, growth: bool
) -> dict | None:
    rows1 = cell_rows_for(co, scope, metric, p1)
    rows2 = cell_rows_for(co, scope, metric, p2)
    if not rows1 or not rows2:
        return None
    table_ids = sorted({r[0] for r in rows1} | {r[0] for r in rows2})
    vi = CANONICAL_TO_VI[metric]
    scope_phrase = ""
    if state_scope:
        scope_phrase = " riêng" if scope == "separate" else " hợp nhất"
    if growth:
        question = f"Tính tốc độ tăng trưởng {vi}{scope_phrase} của {co} từ năm {p1} đến năm {p2}."
    else:
        question = f"So sánh {vi}{scope_phrase} của {co} giữa năm {p1} và năm {p2}."
    if question in used_question_texts:
        return None
    filters = RetrievalFilters(company_codes=(co,), periods=(str(p1), str(p2)))
    evidence = [
        {
            "table_id": tid,
            "relative_path": path,
            "line_start": int(ls),
            "line_end": int(le),
            "verified": True,
        }
        for tid, path, ls, le in (rows1 + rows2)
    ]
    evidence = _dedupe_evidence(evidence)
    qid = stable_question_id(question, filters, tuple(table_ids), FINGERPRINT)
    if qid in existing_ids or qid in {r["question_id"] for r in records}:
        return None
    used_question_texts.add(question)
    return {
        "question_id": qid,
        "question": question,
        "intent": "growth" if growth else "compare",
        "filters": filters.model_dump(mode="json"),
        "gold_table_ids": table_ids,
        "reviewed_by": "day21-auto-expansion",
        "reviewed_at": "2026-08-15T00:00:00+00:00",
        "gold_evidence": evidence,
        "dataset_fingerprint": FINGERPRINT,
    }


for co, scope in companies_with_data:
    if len(records) >= TARGET_NEW:
        break
    if company_quota.get(co, 0) >= 2:
        continue
    metric_periods = by_key[(co, scope)]
    rng.shuffle(metric_periods)

    # decide scope-wording policy for this pick based on remaining quotas
    want_stated = (stated_sep < STATED_SEPARATE_TARGET and scope == "separate") or (
        stated_con < STATED_CONSOLIDATED_TARGET and scope == "consolidated"
    )

    made = False
    # try a 2-period growth/difference question first (roughly half the batch)
    by_metric: dict[str, list[int]] = {}
    for metric, period in metric_periods:
        by_metric.setdefault(metric, []).append(period)
    two_period_metrics = [m for m, ps in by_metric.items() if len(set(ps)) >= 2]
    if two_period_metrics and len(records) % 2 == 0:
        metric = rng.choice(two_period_metrics)
        periods = sorted(set(by_metric[metric]))[:2]
        growth = rng.random() < 0.5
        rec = make_two_period(
            co, scope, metric, periods[0], periods[1], state_scope=want_stated, growth=growth
        )
        if rec is not None:
            records.append(rec)
            made = True
    if not made:
        metric, period = metric_periods[0]
        rec = make_lookup(co, scope, metric, period, state_scope=want_stated)
        if rec is not None:
            records.append(rec)
            made = True
    if made:
        company_quota[co] = company_quota.get(co, 0) + 1
        if want_stated:
            if scope == "separate":
                stated_sep += 1
            else:
                stated_con += 1
        else:
            unstated += 1

print(
    f"generated {len(records)} new records: stated_separate={stated_sep} "
    f"stated_consolidated={stated_con} unstated={unstated}"
)

all_ids = sorted(set(existing_raw_lines) | {r["question_id"] for r in records})
new_by_id = {r["question_id"]: r for r in records}
lines_out = []
for qid in all_ids:
    if qid in existing_raw_lines:
        lines_out.append(existing_raw_lines[qid])  # byte-identical, never re-serialized
    else:
        lines_out.append(
            json.dumps(new_by_id[qid], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        )
with GOLD_PATH.open("w", encoding="utf-8", newline="") as f:
    f.write("\n".join(lines_out) + "\n")
print(f"total gold records now: {len(all_ids)}")

# sanity: all original ids survive unchanged in the merged id set
orig_ids = set(existing_raw_lines)
print("original ids preserved:", orig_ids <= set(all_ids))
