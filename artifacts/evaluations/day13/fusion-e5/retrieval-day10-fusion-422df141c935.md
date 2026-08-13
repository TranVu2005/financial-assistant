# Day 10 Fusion Grid Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 70
- BM25 v3 reference: Precision@10=0.150000 Recall@10=0.880952 F2@10=0.422455
- Default system: **bm25-v3**
- Best weights: bm25=1.0 dense=0.0 (rrf_k=60, depth=50)
- Decision reason: best grid point bm25=1.0/dense=0.0 reaches BM25 v3 on both F2 and Recall, but uses no dense weight (dense=0.0), so it carries no real dense contribution -- it is BM25 v3 itself; BM25 v3 stays default

## Full pre-registered grid

| bm25 | dense | Precision@10 | Recall@10 | F2@10 | ΔF2 vs BM25 | ΔRecall vs BM25 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 0.150000 | 0.880952 | 0.422455 | +0.000000 | +0.000000 |
| 1.0 | 1.0 | 0.125714 | 0.758333 | 0.353464 | -0.068991 | -0.122619 |
| 2.0 | 1.0 | 0.131429 | 0.789286 | 0.370059 | -0.052396 | -0.091667 |
| 3.0 | 1.0 | 0.142857 | 0.839286 | 0.400671 | -0.021784 | -0.041667 |
| 4.0 | 1.0 | 0.148571 | 0.867857 | 0.416544 | -0.005911 | -0.013095 |
| 0.0 | 1.0 | 0.102857 | 0.615476 | 0.285437 | -0.137018 | -0.265476 |

## Failure counts by weight point

| bm25 | dense | full | partial | zero | no_eligible |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 59 | 4 | 7 | 0 |
| 1.0 | 1.0 | 49 | 7 | 14 | 0 |
| 2.0 | 1.0 | 51 | 8 | 11 | 0 |
| 3.0 | 1.0 | 55 | 7 | 8 | 0 |
| 4.0 | 1.0 | 58 | 5 | 7 | 0 |
| 0.0 | 1.0 | 41 | 4 | 25 | 0 |
