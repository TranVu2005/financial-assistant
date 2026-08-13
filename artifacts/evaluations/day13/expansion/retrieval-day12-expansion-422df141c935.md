# Day 12 Graph Expansion Evaluation

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Questions: 70
- Decision: Day 12 records the best pre-registered point only; Day 14 decides whether graph expansion is retained.
- Caveat: The 70-question Day 13 gold set supplies the current expansion evidence; Day 14 uses the separately reviewed failure export for root-cause decisions. This evaluation does not establish a default system.

## Full pre-registered grid

| Relations | Alpha | Expand non-seeds | Precision@10 | Recall@10 | F2@10 | Latency p95 (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0 | False | 0.150000 | 0.880952 | 0.422455 | 0.822052 |
| same_document | 0.25 | False | 0.094286 | 0.554762 | 0.260478 | 0.111829 |
| same_document | 0.25 | True | 0.094286 | 0.554762 | 0.260478 | 0.111143 |
| same_document | 0.5 | False | 0.088571 | 0.511905 | 0.242337 | 0.117252 |
| same_document | 0.5 | True | 0.087143 | 0.504762 | 0.238369 | 0.109752 |
| same_document | 1 | False | 0.085714 | 0.494048 | 0.234488 | 0.112104 |
| same_document | 1 | True | 0.082857 | 0.479762 | 0.226551 | 0.109670 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.25 | False | 0.111429 | 0.636905 | 0.307153 | 0.369870 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.25 | True | 0.092857 | 0.536905 | 0.254432 | 0.858348 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.5 | False | 0.111429 | 0.636905 | 0.307153 | 0.342467 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 0.5 | True | 0.092857 | 0.536905 | 0.254432 | 0.839635 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 1 | False | 0.112857 | 0.651190 | 0.312255 | 0.385618 |
| adjacent_period, explained_by_note, same_document, same_statement_type, shared_metric | 1 | True | 0.094286 | 0.551190 | 0.259534 | 0.369240 |
