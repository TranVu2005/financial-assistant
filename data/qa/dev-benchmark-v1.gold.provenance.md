# Dev benchmark v1 — gold annotation provenance

`dev-benchmark-v1.jsonl` was a *selection* scaffold: 144 stratified questions with every
`gold_table`/`gold_rows`/`gold_columns`/`gold_values` field `null` and `needs_annotation: true`
(see `dev-benchmark-v1.provenance.md`). Without gold labels the benchmark can only compare an
outcome *status* against a recorded baseline — it cannot tell a right answer from a wrong one.

`dev-benchmark-v1.gold.jsonl` holds the annotations, keyed by `vifinqa_id`. **All 144 questions
now have a record**, in one of two states:

| status | count | meaning |
|---|---:|---|
| `annotated` | 98 | gold row, column, raw value and answer read from the source report |
| `needs_review` | 46 | deliberately *not* answered — the reason is recorded per question |
| **total** | **144** | |

Every `annotated` record carries the source path, line number, row label, column header, the raw
value string exactly as printed, its unit, and the answer converted to the unit the question asks
for. Nothing is a placeholder and nothing is a guess: where the source did not settle the
question, the record says so instead of inventing a number.

## Method (follows ADR 0009 decision A2)

Labels are read from the source report text, **independently of the pipeline being scored**.
Nothing in `scripts/gold_annotation/` imports `financial_report_qa`; if it did, the gold set would
be measuring the pipeline against itself.

1. `srcread.py` — standalone parser for `data/raw/ViFinQA/financial_statements/**/*_extracted.txt`.
   Splits the `<table><tr><td>` blocks into rows, keeps each row's line number, page marker and the
   table's header row, parses Vietnamese number formatting (`.` thousands, `,` decimals, `(…)`
   negative), and caches parsed rows per file. `grep()` does full-text search for metrics that live
   in prose rather than in a table.
2. `propose.py` — resolves the issuer from `code_stock.csv` (longest issuer *name* wins over any
   bare ticker: "CTCP Chứng khoán FPT" is **FTS**, not FPT; "CTCP"/"Công ty CP"/"TMCP" are expanded
   so spelled-out names match), the year(s), and the statement scope, then ranks candidate rows by
   content-token overlap.
3. `review.py` — the bulk review sheet. Its `normalize_row` handles the three OCR layouts that
   otherwise hide balance-sheet rows: cells aligned 1:1 with the header, a row missing its label
   cell, and extra leading cells where the line numbering ("IV.", "10") sits ahead of the metric
   name. Rows whose table actually has a column for the asked period are ranked above notes that
   merely repeat the label.
4. `pair.py` — targeted lookups of one metric across several years or companies, which is what the
   difference / growth / argmax questions need.
5. `record.py` — appends a decision **only after re-reading it from source**: the named row must
   exist at the named line (matched against the first cell, the numbering-shifted label, or any
   cell of the row) and the raw value string must appear in that row. A transcription error fails
   loudly instead of silently becoming a wrong label. It caught two real mistakes during this pass.

## Record shape

```json
{"vifinqa_id": 6, "status": "annotated", "operation": "lookup",
 "company": "VSC", "scope": "separate", "period": 2017,
 "gold_table": {"relative_path": "VSC/2017/…_extracted.txt", "line": 224},
 "gold_rows": ["Lưu chuyển tiền thuần từ hoạt động kinh doanh"],
 "gold_columns": ["2017 VND"],
 "gold_values": [{"raw": "145.731.366.146", "numeric": 145731366146.0, "source_unit": "VND"}],
 "answer": 145.731366146, "answer_unit": "VND_billion", "notes": "…"}
```

Multi-step questions add `supporting_evidence` — every intermediate figure with its own path, line
and raw string — so a difference, an average or an argmax can be re-checked without redoing the
search. `kind` records the question shape: `single_lookup` (60 annotated), `argmax` (14),
`two_period` (12), `two_company` (6), `multi_period` (4), `ratio`, `multi_hop`.

`answer_unit` uses the unit the question asks for, including `VND_hundred_billion` for
"trăm tỷ đồng" (1e11), `VND_trillion` for "nghìn tỷ đồng" (1e12), `year` for "năm nào", `count`,
`shares` and `percentage_point`.

## Scope when the question does not state one

62% of ViFinQA questions name no statement scope, and consolidated and separate reports usually
carry different values for the same metric. Four cases, all recorded explicitly rather than
guessed:

- only one scope contains the metric → that scope, no ambiguity (id 25: only the consolidated
  report has a "Giá vốn hàng hóa" row; the separate one has "Giá vốn hàng bán", a different metric);
- both scopes carry the same value → scope does not affect the answer (ids 52, 276);
- both carry *different* values → consolidated is recorded as the primary answer and the separate
  reading is kept in `also_acceptable` (20 records). A scorer should accept either;
- the wording cannot separate them and the values differ by orders of magnitude → `needs_review`.

## Why 46 are `needs_review`

- **23 multi-hop** questions that need a whole matrix of lookups before any arithmetic — e.g. id 443
  (6 issuers × 5 metrics × 2 years, then a median, a filter and a ranking). Each record names the
  companies, the metrics and the operation, so the remaining work is enumerated rather than
  rediscovered.
- **9 single lookups** where the metric is not in the report under any recognisable name (id 228:
  the separate VGC 2025 report has no raw-material inventory row at all), or exists only inside a
  related-party note whose scope readings differ 2.5× (id 122).
- **6 two-period and 3 ratio** questions whose denominator is a *derived* concept the reports never
  print — "tổng nợ phải trả tài chính" (id 670), "tổng nợ tài chính" (id 645) — so the answer
  depends on a definition the question does not give.
- **2 argmax** questions with no unique answer: id 907's three candidate years are all exactly 407.

## What this gold set already showed

- Question 175 ("Tổng cộng dự phòng phải trả cuối năm 2020 của VGT"): the source answer is
  32.587.523.656 VND (note 27, row "Số dư cuối năm", column "Tổng cộng VND"). Both numbers the
  pipeline produced for it during the V2 benchmark runs — 136.932 and 235.664 — are wrong, and both
  came from rows naming a company rather than a metric.
- Questions 4, 6, 27, 132, 145, 167, 340, 730 confirm the executor's arithmetic matched the source,
  i.e. the `recompute_mismatch` verifier defect (fixed separately) had been rejecting correct
  answers.
- Question 4 shows why issuer resolution matters: "CTCP Chứng khoán FPT" is FTS, and the answer
  (444,918 triệu đồng) sits in an FTS report, not an FPT one.

## Continuing

```bash
python scripts/gold_annotation/review.py --limit=8
python scripts/gold_annotation/pair.py < lookups.json
python scripts/gold_annotation/record.py < batch.json
```

`record.py` merges by `vifinqa_id`, so re-running it updates a record in place — that is how a
`needs_review` record becomes `annotated` once its open question is settled.
