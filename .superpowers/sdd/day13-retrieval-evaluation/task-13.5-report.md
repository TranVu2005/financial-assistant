# Task 13.5 Report — Re-baseline after 70-question gold

Status: COMPLETE
Branch/worktree: `codex/day13-retrieval-evaluation` at `D:/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation`
Dataset fingerprint: `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`

## Mandatory ordering and inputs

The mandated sequence was preserved:

1. BM25 evaluated on gold70 before changing the reference lock.
2. Tests were changed first and observed RED against the old production lock; the production lock was then updated and focused tests observed GREEN.
3. The exact dense query caches were path-validated and removed; dense BGE-M3 and E5 were evaluated in WSL2 `financial-dense-gpu` with `--encoder-device cuda`.
4. Fusion BGE/E5, graph, then expansion were evaluated.
5. The pre-existing gold30 Day13 reports were copied to `artifacts/evaluations/day13/gold30/` before current reports were overwritten.

Immutable/read-only input identity:

| Input | SHA-256 |
| --- | --- |
| `D:/GitHub/financial-assistant/data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json` | `24E9CA4005C9B9ECFB74A6CCE273D9F174124885F55F5031C9617709C4E41AB2` |
| `data/qa/retrieval-gold-v1.jsonl` (70 lines) | `0AAEEC29325596BF8E56FA91FE330D57C6B731E42842AB3096D04D9CAE43678F` |
| BM25 v3 manifest | `D7A1433209CE2F8B3FBC24559DE57EC1AE4F73CE7E38F6B6D32E599223B26C68` |
| Dense corpus manifest | `43589A1CD11F6A9851544C0F3BF9AB1388081EAE395E3EF9881BA6628ABFB1D6` |
| BGE index manifest | `535D4EC0F5FBC8D6101EA4E8EF51D62393CB787DC5606E850EEA782CD26357AA` |
| E5 index manifest | `61D58F487FD55DC788F642B8B2ED1FDD8E43E45A5B01964F4FA9F9FEA3E4E39E` |
| Graph manifest | `0D18A1FC94EE56CC76904B8AFFD0D93E43F28360AFC08A6263E4A9B876A292DF` |

## Results

All retrieval systems used 70 questions. Legacy metrics retain the fixed Precision@10/F2@10 contract. Extended metrics were derived from the persisted top-10 predictions with the Task 13.3 formulas and stored in `metrics-v2-422df141c935.json`.

| System | TP | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 v3 | 105 | 0.150000000000 | 0.583333333333 | 0.725000000000 | 0.880952380952 | 0.422454529597 | 0.621938775510 | 0.422619047619 | 0.491345856524 |
| dense BGE-M3 | 69 | 0.098571428571 | 0.246428571429 | 0.411904761905 | 0.552380952381 | 0.265920587349 | 0.298503401361 | 0.196428571429 | 0.224174139353 |
| dense E5-small | 71 | 0.101428571429 | 0.220238095238 | 0.420238095238 | 0.601190476190 | 0.280334744620 | 0.274875283447 | 0.159523809524 | 0.184608843537 |
| fusion BGE best | 105 | 0.150000000000 | 0.588095238095 | 0.725000000000 | 0.880952380952 | 0.422454529597 | 0.615748299320 | 0.427380952381 | 0.494128787879 |
| fusion E5 best | 105 | 0.150000000000 | 0.588095238095 | 0.725000000000 | 0.880952380952 | 0.422454529597 | 0.615748299320 | 0.427380952381 | 0.494128787879 |
| graph expansion best | 105 | 0.150000000000 | 0.588095238095 | 0.725000000000 | 0.880952380952 | 0.422454529597 | 0.615748299320 | 0.427380952381 | 0.494128787879 |

Fusion BGE and E5 both selected `bm25=1.0, dense=0.0, rrf_k=60, depth=50`; default remains `bm25-v3`. Expansion selected the alpha-zero anchor: `alpha=0.0`, `fan_out=25`, `seed_depth=50`, `rrf_k=60`, `expand_non_seeds=false`, with all five registered relations. Graph coverage remained identity-stable at 146,011 documents; all nodes have at least one edge across the retained relations.

## CUDA/cache evidence

WSL2 preflight:

- Environment: `/home/vutran/.local/share/micromamba/envs/financial-dense-gpu`
- PyTorch: `2.6.0+cu124`
- `torch.cuda.is_available() == True`
- GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU`
- Worktree source import: `/mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/src/financial_report_qa/__init__.py`

Only these validated rebuildable original-checkout caches were removed:

- `D:/GitHub/financial-assistant/data/indexes/dense-query-cache/day9-a-bge`
- `D:/GitHub/financial-assistant/data/indexes/dense-query-cache/day9-a-e5`

New worktree-local caches contain exactly 70 vectors each. Dense evidence:

| Encoder | encoder_spec_sha256 | Cold misses | Warm hits | cold macro == warm macro |
| --- | --- | ---: | ---: | --- |
| BGE-M3 | `795294d329d055d3c1d5eecc11afff466da7ccadcdb4cd60efc2a5785ae0414b` | 70 | 70 | true |
| multilingual-E5-small | `1fa278c2f4d5941ddf96e07177d3ae6b8bfecb16ffc5a139ea08c423ed344fbe` | 70 | 70 | true |

Build-observation SHA-256: BGE `068BF8DC7D13A3D08ACEA7336CBDFB5914770A4DCC64F82E9FB4DB198CD00A87`; E5 `41E88C706CC51E547E4EB03C3C6D6F8EDB6EFDF7432C1B24CD110313E85AB93B`.

## Commands

BM25 (Windows dependency environment, worktree source):

```powershell
$env:PYTHONPATH='D:\GitHub\financial-assistant\.worktrees\day13-retrieval-evaluation\src'
uv run --project D:\GitHub\financial-assistant --frozen --no-sync financial-report-qa retrieval evaluate --release-lock D:\GitHub\financial-assistant\data\qa\week1_pilot_422df141c935\dataset-pilot-v1.json --index-dir D:\GitHub\financial-assistant\data\indexes\bm25-v3\422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a --gold-path D:\GitHub\financial-assistant\.worktrees\day13-retrieval-evaluation\data\qa\retrieval-gold-v1.jsonl --output-dir D:\GitHub\financial-assistant\.worktrees\day13-retrieval-evaluation\artifacts\evaluations\day13\bm25
```

Dense and fusion were run from `/mnt/d/GitHub/financial-assistant` with:

```bash
PYTHONPATH=/mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/src /home/vutran/.local/share/micromamba/envs/financial-dense-gpu/bin/financial-report-qa retrieval evaluate-dense ... --encoder bge-m3 --encoder-device cuda ... --output-path /mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/artifacts/evaluations/day13/dense-bge-m3.json
PYTHONPATH=/mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/src /home/vutran/.local/share/micromamba/envs/financial-dense-gpu/bin/financial-report-qa retrieval evaluate-dense ... --encoder multilingual-e5-small --encoder-device cuda ... --output-path /mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/artifacts/evaluations/day13/dense-e5-small.json
PYTHONPATH=/mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/src /home/vutran/.local/share/micromamba/envs/financial-dense-gpu/bin/financial-report-qa retrieval evaluate-fusion ... --encoder bge-m3 --encoder-device cuda ... --output-dir /mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/artifacts/evaluations/day13/fusion-bge
PYTHONPATH=/mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/src /home/vutran/.local/share/micromamba/envs/financial-dense-gpu/bin/financial-report-qa retrieval evaluate-fusion ... --encoder multilingual-e5-small --encoder-device cuda ... --output-dir /mnt/d/GitHub/financial-assistant/.worktrees/day13-retrieval-evaluation/artifacts/evaluations/day13/fusion-e5
```

Graph and expansion used the same WSL environment/worktree source:

```bash
financial-report-qa retrieval evaluate-graph --release-lock ... --graph-dir .../graph-day11-a/422df141c935... --output-dir .../artifacts/evaluations/day13/graph
financial-report-qa retrieval evaluate-expansion --release-lock ... --index-dir .../bm25-v3/422df141c935... --graph-dir .../graph-day11-a/422df141c935... --gold-path .../data/qa/retrieval-gold-v1.jsonl --bm25-report .../day13/bm25/retrieval-day8-422df141c935.json --output-dir .../day13/expansion
```

Expansion completed with exit 0 after 172.2 seconds. No validator was patched, mocked, or bypassed in the real command.

## Deterministic artifacts

| Artifact | SHA-256 |
| --- | --- |
| BM25 JSON | `71EA90DEDE2E3B6BB873F527CCBC8A6B3A80285D7D3D15D55439B566E2BDB639` |
| BM25 Markdown | `FFC247C9AA77E7C21D1B10A3AF9DCF8B6BBD157902FF97191CA325468EFAF70F` |
| dense BGE JSON | `1F1E1C9466BC7AEA50A1B7B2CBF71A17C0E3CF555AE8229536AC05D497127937` |
| dense E5 JSON | `DA49E6C634B2C4E2B3C003C4A6C9C974764C77319AADB660F169DA49E5C978F6` |
| fusion BGE JSON | `DD6D67168A4BA625E6FDD05F84AB1A4F19A5EEF1F69F746AF8760F36FB94F602` |
| fusion E5 JSON | `2716F4A890DBC58B58514F7CD41B1FBAA6A0FB22C79028BC6526C82138C016CE` |
| graph JSON | `017F90BAA0B032D46A81D7173DB70E0E523BABEC98B9410BDA2FD0757036D3B0` |
| expansion JSON | `C6377699D8E5732D8FEE6F76F79738A90B660011F82E8E9E25AC6D8B4FBAD3B8` |
| V2 metrics summary | `00D0FE092756F75EEC05FE904AE1720D584842BFA28974A81BDEBC7A1351EBFF` |
| failure JSON (Task 13.4 input retained) | `0530751D4275074385EA4A63A2DA0FE5CC249C1B4D3B243CCABC750103296CAC` |

The `gold30/` archive contains the 12 original files with their original hashes; for example BM25 JSON `2653B2713FD34E032F2492CFFCF530D1EB80285452BCD59C22097559A6EF1EED` and expansion JSON `3474116425B908F6CF1427A6070286E103CFE7CD30F5B87BB47CE04B4795D3A2`.

## Verification and self-review

TDD evidence:

- RED: focused dense/fusion/expansion suite failed 10 tests because production still required the gold30 metrics/count; caveat test also detected stale `4/30`.
- GREEN: 18 focused unit tests passed after updating the lock and stale evidence text.
- Full-suite RED found two integration fixtures still emitting the gold30 lock.
- Focused integration GREEN: 2 passed after updating only those fixture values.

Fresh gates:

- Full pytest: `753 passed, 4 skipped` (one existing pytest config warning).
- Changed paths Ruff: clean.
- Changed paths mypy: clean (8 files).
- Full repo Ruff: known baseline `102 errors`, zero new changed-path errors.
- Full repo mypy: known baseline `33 errors in 5 files (checked 157 source files)`, zero new changed-path errors.
- `git diff --check`: clean.

Self-review found no source/release/index mutation. Production behavior change is limited to the BM25 gold70 reference lock and replacement of the stale expansion evidence statement. Historical strict report models remain unchanged. Real expansion completed end-to-end with the reference gate active, satisfying Task 13.5 DoD.
