# Day 12 Graph Expansion Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 30
- Decision: Day 12 records the best pre-registered point only; Day 14 decides whether graph expansion is retained.
- Caveat: Only 4/30 questions have headroom; they involve only two distinct missing tables, both already present in BM25 top-50. This evaluation does not establish a default system.

## Full pre-registered grid

| Relations | Alpha | Expand non-seeds | Precision@10 | Recall@10 | F2@10 | Latency p95 (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0 | False | 0.146667 | 0.883333 | 0.431217 | 2.002722 |
| same_document | 0.25 | False | 0.026667 | 0.200000 | 0.084656 | 0.287889 |
| same_document | 0.25 | True | 0.026667 | 0.200000 | 0.084656 | 0.275637 |
| same_document | 0.5 | False | 0.016667 | 0.133333 | 0.054233 | 0.313707 |
| same_document | 0.5 | True | 0.013333 | 0.116667 | 0.044974 | 0.328219 |
| same_document | 1 | False | 0.016667 | 0.133333 | 0.054233 | 0.253539 |
| same_document | 1 | True | 0.010000 | 0.100000 | 0.035714 | 0.255730 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.25 | False | 0.063333 | 0.400000 | 0.189153 | 1.943645 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.25 | True | 0.020000 | 0.166667 | 0.066138 | 1.735761 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.5 | False | 0.063333 | 0.400000 | 0.189153 | 2.058957 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.5 | True | 0.020000 | 0.166667 | 0.066138 | 1.714368 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 1 | False | 0.063333 | 0.400000 | 0.189153 | 1.056603 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 1 | True | 0.020000 | 0.166667 | 0.066138 | 1.091883 |
