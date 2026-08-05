---
name: vifinqa-txt-ingestion
description: Use when implementing, extending, reviewing, or debugging the provenance-preserving ViFinQA TXT ingestion pipeline, including UTF-8 source reading, HTML or structured-text table detection, table extraction, continuation handling, and regression tests.
---

# ViFinQA TXT Ingestion

## Overview

Build deterministic ingestion from an immutable TXT snapshot. Preserve exact source provenance; reject uncertain tables rather than inventing structure.

**REQUIRED SUB-SKILLS:** Use `superpowers:using-git-worktrees`, `superpowers:test-driven-development`, `superpowers:requesting-code-review`, and `superpowers:verification-before-completion`.

## Delivery workflow

1. Inspect `DocumentRecord`, `TableRecord`, `CellRecord`, errors, and ingestion tests. Preserve unrelated dirty changes; use an isolated worktree.
2. Add one failing unit or golden test for each behavior before production code. Verify it fails for the missing behavior.
3. Define or extend contracts in `src/financial_report_qa/ingestion/`. Every output table and cell must retain raw text, source line/span, and stable ID.
4. Implement in this order: reader -> detector -> extractor -> continuation merge -> public API -> smoke script.
5. Run focused tests during development; run the full verification gate before integration. Do not claim corpus coverage if the pinned local corpus revision is absent.

## Non-negotiable contracts

| Area | Required behavior |
|---|---|
| Source reader | Require a ready `DocumentRecord`; resolve only safe relative paths; stream-check size and SHA-256 before strict UTF-8/UTF-8-SIG decode; preserve LF, CRLF, CR, raw lines, and offsets. |
| Detection | Prefer HTML. Exclude an entire HTML region from text fallback even when HTML is malformed or nested. Record an auditable rejection reason. |
| Text fallback | Accept only strong tabular evidence. Normalize Unicode (NFKC/casefold) before header scoring. Reject bullets, numbered lists, narrative text, and ambiguous layouts. Calculate density from populated cells. |
| Extraction | Parse `th`, `td`, entities, `<br>`, `rowspan`, and `colspan`; cap span expansion. Preserve placements/provenance. Infer title/unit only from nearby lines or header bands, never body cells. |
| Continuation | Merge only adjacent, compatible tables with matching schema/header and no evidence of a new table. Preserve original cell IDs and spans. |

## Tests to add or preserve

- Reader: snapshot mismatch, unsafe path, unsupported encoding, BOM, each line ending, page marker.
- HTML: nested/malformed rejection, entities, breaks, spans, large span cap.
- Fallback: Unicode header, ragged rows, empty-cell density, bullets/numeric lists/narrative rejection.
- End-to-end: golden fixtures for structured fallback and cross-page continuation.

## Verification gate

Run from the repository root:

```powershell
uv run --frozen --no-sync pytest -q
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
git diff --check
```

Run `scripts/smoke_ingestion.py --root <vifinqa-root>` only against an explicitly named immutable corpus revision. Report missing corpus access as a limitation, not a passing smoke result.

## Handoff prompt

`Use $vifinqa-txt-ingestion to implement <specific ingestion behavior>; begin with a failing regression test and report provenance, rejection, and verification evidence.`

## Common mistakes

- Parsing rejected nested HTML again through the text fallback.
- Computing confidence before Unicode normalization or from blank cells.
- Inferring units from body rows or merging similarly shaped but independent tables.
- Treating tests written after implementation or partial-suite output as verification.

