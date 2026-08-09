# BM25 Length-Normalization Remediation Design

## Objective

Reduce retrieval fragmentation caused by BM25 over-favoring short note fragments over
longer primary financial-statement tables, without changing reviewed gold labels,
filters, query IDs, or the BM25-only retrieval model.

## Evidence

The immutable 30-question evaluation on the v2 corpus showed macro Recall@10 `0.8333333`,
F2@10 `0.4034392`, and five zero-hit questions. A read-only sweep over the same documents,
filters, gold, tokenizer, and corpus-bound metric expansion found that Lucene BM25 with
`k1=1.5`, `b=0.25`, and `delta=0.5` yields Recall@10 `0.8833333`, F2@10 `0.4312169`,
and three zero-hit questions. The remaining HDB misses target a fragment whose review label
is semantically narrower than other equivalent source fragments; they are not fixed by the
length-normalization change.

## Design

Set the pinned BM25 length-normalization parameter from `b=0.75` to `b=0.25`. The tokenizer,
Lucene method, `k1=1.5`, `delta=0.5`, filter-first selection, corpus-bound alias expansion,
and stable `(-score, table_id)` ranking remain unchanged.

This is a corpus-wide BM25 parameter change, not a query-ID rule, metadata boost, scope
heuristic, or gold edit. It lowers the document-length penalty applied to primary statements
with many legitimate metrics, while preserving lexical BM25 ranking.

Persist the changed retrieval contract as `bm25-index-v3`, `builder_version="v3"`, and
`query_expansion_version="v1"`. The loader rejects v2 artifacts so every result is backed by
a v3 rebuild. The manifest continues to record `b`, document hashes, and artifact hashes.

## Tests and verification

Add a unit regression with a long document and a short note that share the same query terms;
under the pinned v3 parameter, the long document must rank first when it contains the
additional relevant title term. Update index persistence tests to expect v3 and reject v2.

Run focused retrieval tests, build two independent v3 real-corpus indexes, evaluate both with
the immutable lock and reviewed gold, compare index/report hashes, and record measured metrics.
Acceptance is the existing provisional floor: Recall@10 >= `0.8833333333333333`, F2@10 >=
`0.41798941798941797`, and zero-hit <= `3`.

## Non-goals

- No change to `data/qa/retrieval-gold-v1.jsonl`.
- No automatic filter extraction, dense retrieval, rank fusion, or structural boosts.
- No claim that the residual HDB gold-fragment mismatch is solved.
