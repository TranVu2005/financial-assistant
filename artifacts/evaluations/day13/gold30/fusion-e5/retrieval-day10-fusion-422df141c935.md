# Day 10 Fusion Grid Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 30
- BM25 v3 reference: Precision@10=0.146667 Recall@10=0.883333 F2@10=0.431217
- Default system: **bm25-v3**
- Best weights: bm25=1.0 dense=0.0 (rrf_k=60, depth=50)
- Decision reason: best grid point bm25=1.0/dense=0.0 reaches BM25 v3 on both F2 and Recall, but uses no dense weight (dense=0.0), so it carries no real dense contribution -- it is BM25 v3 itself; BM25 v3 stays default

## Full pre-registered grid

| bm25 | dense | Precision@10 | Recall@10 | F2@10 | ΔF2 vs BM25 | ΔRecall vs BM25 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 0.146667 | 0.883333 | 0.431217 | +0.000000 | +0.000000 |
| 1.0 | 1.0 | 0.076667 | 0.533333 | 0.236772 | -0.194444 | -0.350000 |
| 2.0 | 1.0 | 0.093333 | 0.616667 | 0.283069 | -0.148148 | -0.266667 |
| 3.0 | 1.0 | 0.120000 | 0.733333 | 0.354497 | -0.076720 | -0.150000 |
| 4.0 | 1.0 | 0.133333 | 0.800000 | 0.391534 | -0.039683 | -0.083333 |
| 0.0 | 1.0 | 0.033333 | 0.283333 | 0.111111 | -0.320106 | -0.600000 |

## Failure counts by weight point

| bm25 | dense | full | partial | zero | no_eligible |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 26 | 1 | 3 | 0 |
| 1.0 | 1.0 | 14 | 4 | 12 | 0 |
| 2.0 | 1.0 | 16 | 5 | 9 | 0 |
| 3.0 | 1.0 | 20 | 4 | 6 | 0 |
| 4.0 | 1.0 | 23 | 2 | 5 | 0 |
| 0.0 | 1.0 | 8 | 1 | 21 | 0 |
