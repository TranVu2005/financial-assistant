---
name: vifinqa-bm25-retrieval
description: Use when implementing, extending, reviewing, debugging, or verifying the ViFinQA Day 8 BM25 table-retrieval baseline, including release locks, table indexing, metadata filters, lexical ranking, reviewed gold questions, score traces, Recall@10, F2@10, or deterministic retrieval artifacts.
---

# ViFinQA BM25 Retrieval

## Overview

Build an auditable lexical baseline from the immutable Week-1 release. Preserve identity from release lock to report; never claim Day-8 metrics from missing gold data or synthetic fixtures.

**REQUIRED SUB-SKILLS:** Use `superpowers:using-git-worktrees`, `superpowers:test-driven-development`, `superpowers:requesting-code-review`, and `superpowers:verification-before-completion`. For the full plan, use `superpowers:subagent-driven-development` or `superpowers:executing-plans`.

## Source of truth

Read completely before editing:

- `docs/superpowers/specs/2026-08-08-day-8-bm25-retrieval-design.md`
- `docs/superpowers/plans/2026-08-08-day-8-bm25-retrieval.md`

Tasks 1–7 cover coding, gold contracts, and this skill. Tasks 8–11 are verification/evidence and require their real inputs.

## Non-negotiable contracts

| Area | Required behavior |
|---|---|
| Corpus | Resolve `dataset-pilot-v1`; require a passing Week-1 gate and equal lock/gate/release fingerprints. Never choose an arbitrary `release_v2_*`. |
| Gold | Require exactly 30 non-empty-gold questions bound to the fingerprint. Each gold ID needs persisted reviewer, timestamp, source path/span, and `verified=true`. Missing, partial, incompatible, or prediction-derived gold fails closed. |
| Scope | Consume explicit expert filters. Automatic entity parsing belongs to Day 10. |
| Documents | One stable document per `table_id`: title, statement, observed metric aliases, company code, periods, units. Exclude raw numbers. |
| Filters | OR within a field; AND across fields. `periods=("2022", "2023")` means either period. Empty eligible set returns no result without fallback. Trace each non-empty field's matched and remaining counts. |
| Ranking | BM25 only; rank eligible rows by `(-score, table_id)`. Avoid `bm25s.retrieve(weight_mask=...)`; 0.3.10 can return excluded zero-score rows. No vocabulary tokens means no candidates. |
| Metrics | At `k=10`: `P=TP/10`, `R=TP/|gold|`, `F2=5PR/(4P+R)`. Macro-average all 30 questions; zero hits score zero. |
| Evidence | Build/evaluate twice into independent paths and compare hashes. Fixtures prove behavior, not real-corpus completion. |

## Workflow

1. Preserve unrelated dirty changes and isolate code work.
2. Write and run the specified failing test before each production behavior.
3. Implement only the active task; run its focused pytest, Ruff, mypy, and `git diff --check` gates.
4. Review identity, provenance, filters, ties, arithmetic, and hashes before committing only listed paths.
5. Report unavailable lock/gold inputs; never restore or replace user-owned data silently.

Full gate:

```powershell
uv run --frozen --no-sync pytest -q tests/unit/retrieval tests/integration/retrieval/test_day8_cli.py
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

## Red flags

- Parsing company/year during Day 8.
- Assuming gold exists or labeling BM25 output as gold.
- Using `TP/retrieved` for P@10.
- Requiring all requested periods in every table.
- Ranking zero-token queries or reporting one-run/fixture metrics as Day 8.

## Handoff

`Use $vifinqa-bm25-retrieval for Task <N>; start with its failing test and return exact identity and verification evidence.`
