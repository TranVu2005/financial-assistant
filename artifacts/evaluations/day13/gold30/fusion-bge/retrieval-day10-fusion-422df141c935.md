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
| 1.0 | 1.0 | 0.073333 | 0.516667 | 0.227513 | -0.203704 | -0.366667 |
| 2.0 | 1.0 | 0.080000 | 0.566667 | 0.248677 | -0.182540 | -0.316667 |
| 3.0 | 1.0 | 0.113333 | 0.733333 | 0.341270 | -0.089947 | -0.150000 |
| 4.0 | 1.0 | 0.113333 | 0.700000 | 0.335979 | -0.095238 | -0.183333 |
| 0.0 | 1.0 | 0.033333 | 0.266667 | 0.108466 | -0.322751 | -0.616667 |

## Failure counts by weight point

| bm25 | dense | full | partial | zero | no_eligible |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 26 | 1 | 3 | 0 |
| 1.0 | 1.0 | 14 | 3 | 13 | 0 |
| 2.0 | 1.0 | 16 | 2 | 12 | 0 |
| 3.0 | 1.0 | 19 | 6 | 5 | 0 |
| 4.0 | 1.0 | 18 | 6 | 6 | 0 |
| 0.0 | 1.0 | 8 | 0 | 22 | 0 |
