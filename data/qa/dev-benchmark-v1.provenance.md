# Dev benchmark v1 provenance

`plan.md` §19 asks for a small (100-150 question) benchmark to run "trước khi sửa toàn bộ 1.012 câu" -- a dev-loop regression check, not the scored gold set. It should deliberately cover the pipeline's known failure categories so a code change's effect on each one is visible in a run that finishes in seconds, not the full 1,012-question export.

## Source

`dev-benchmark-v1.jsonl` (144 questions) was selected from a real, already-executed full run of the official 1,012-question ViFinQA set:

```text
artifacts/evaluations/day22/submission_llm/submission-export-422df141c935.json
```

(dataset fingerprint `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`, rule planner + LLM planner fallback, the same architecture `submission export` runs today). This report was not generated for this task -- it already existed on disk from prior pipeline development and was reused as-is rather than re-running the full 1,012-question export, which is expensive and was not needed to answer "which 144 questions should the dev benchmark contain."

## Selection method

[scripts/build_dev_benchmark_v1.py](../../scripts/build_dev_benchmark_v1.py) buckets the report's 1,012 outcomes by `plan.md` §19's named categories (`status == "answered"` counts as a category of its own -- a baseline control group, since a benchmark containing only failures cannot detect a change that breaks a previously-working question) and samples a fixed quota per category with `random.Random(19)` (seeded on the plan.md section number, for reproducibility), then sorts the result by the official ViFinQA `id`. Quotas were chosen so every one of the 7 categories `plan.md` §19 explicitly lists is represented, capped so no single common category (`metric_not_found`, `llm_plan_invalid`) dominates the file:

| Category | Quota | Achieved | Population in source report |
|---|---:|---:|---:|
| `answered` (control group) | 25 | 25 | 32 |
| `metric_not_found` | 20 | 20 | 338 |
| `llm_plan_invalid` | 20 | 20 | 458 |
| `cell_ambiguous` | 20 | 20 | 60 |
| `period_unresolved` | 20 | 20 | 56 |
| `no_candidate_tables` | 15 | 15 | 43 |
| `unit_missing` | 12 | 12 | 12 (all of them) |
| `llm_invalid_json` | 12 | 12 | 12 (all of them) |
| **Total** | | **144** | |

Re-running the script against the same report file is byte-for-byte reproducible (fixed seed, deterministic bucketing/sort).

## What this file is not

Each record carries `gold_table`/`gold_rows`/`gold_columns`/`gold_values`/`operation` fields per `plan.md` §19's own annotation schema, but they are **all `null`, with `needs_annotation: true`** -- this script performs selection only, not annotation. Populating real gold values requires reading each question's source financial statement by hand (`data/raw/ViFinQA/financial_statements/...`), the same leakage-safe process `answer-gold-v1.provenance.md` and `retrieval-gold-v1.provenance.md` document for the scored gold sets. That full annotation pass is out of scope here and tracked as a follow-up, not silently skipped: **this file is a stratified selection scaffold, not yet a usable regression oracle.**

Until annotated, this benchmark is useful for exactly one thing: run `submission export` restricted to these 144 `vifinqa_id`s after a change and compare each question's `status`/`code` against its `source_status`/`source_code` baseline recorded in this file -- did a question that used to fail with `metric_not_found` now fail with something else (progress), or did a question that used to be `answered` regress? That comparison does not require gold values, only the baseline outcome already captured per record.

## Regenerating

```bash
python scripts/build_dev_benchmark_v1.py \
  --report artifacts/evaluations/day22/submission_llm/submission-export-422df141c935.json \
  --out data/qa/dev-benchmark-v1.jsonl
```
