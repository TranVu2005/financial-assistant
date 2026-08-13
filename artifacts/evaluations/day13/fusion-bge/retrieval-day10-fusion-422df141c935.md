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
| 1.0 | 1.0 | 0.124286 | 0.741667 | 0.347640 | -0.074814 | -0.139286 |
| 2.0 | 1.0 | 0.124286 | 0.753571 | 0.350217 | -0.072237 | -0.127381 |
| 3.0 | 1.0 | 0.138571 | 0.825000 | 0.389900 | -0.032555 | -0.055952 |
| 4.0 | 1.0 | 0.138571 | 0.810714 | 0.387632 | -0.034822 | -0.070238 |
| 0.0 | 1.0 | 0.102857 | 0.585714 | 0.279371 | -0.143083 | -0.295238 |

## Failure counts by weight point

| bm25 | dense | full | partial | zero | no_eligible |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.0 | 59 | 4 | 7 | 0 |
| 1.0 | 1.0 | 49 | 5 | 16 | 0 |
| 2.0 | 1.0 | 50 | 5 | 15 | 0 |
| 3.0 | 1.0 | 53 | 9 | 8 | 0 |
| 4.0 | 1.0 | 52 | 9 | 9 | 0 |
| 0.0 | 1.0 | 41 | 0 | 29 | 0 |
