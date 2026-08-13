# Day 11 Graph Coverage

- Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`
- Documents (nodes): 146011
- Nodes with no edge in any relation: 0

## Coverage by relation

| relation | buckets | membership | nodes w/ edges | isolated | directed edges | p50 | p95 | max | weight min | weight max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adjacent_period | 1109 | 185493 | 146011 | 0 | 61967388 | 374 | 841 | 1565 | 1.000000 | 1.000000 |
| explained_by_note | 1961 | 30891 | 5735 | 140276 | 24847 | 0 | 0 | 20 | 1.000000 | 1.000000 |
| same_document | 1963 | 146011 | 146008 | 3 | 12008152 | 75 | 126 | 244 | 0.000130 | 0.333333 |
| same_statement_type | 372 | 30891 | 30885 | 115126 | 3730008 | 0 | 156 | 282 | 1.000000 | 1.000000 |
| shared_metric | 3807 | 62167 | 25057 | 120954 | 1211850 | 0 | 56 | 243 | 0.032258 | 1.000000 |

## Excluded relations

| relation | measured pair count | reason |
| --- | ---: | --- |
| same_company | 117156769 | every reviewed gold question already hard-filters company_codes before ranking (retrieval/filtering.py), so same_company edges would only connect tables already inside the eligible pool -- no new information |
| same_period | 18683741 | every reviewed gold question already hard-filters periods before ranking (retrieval/filtering.py), so same_period edges would only connect tables already inside the eligible pool -- no new information |
