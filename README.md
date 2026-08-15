# Financial Report QA

Hệ thống hỏi–đáp báo cáo tài chính tiếng Việt theo hướng **data-first** và **auditable**. Mỗi kết
quả số phải truy ngược được qua phép tính, ô dữ liệu, bảng, trang và tài liệu nguồn.

> Trạng thái: đang xây dựng nền tảng dữ liệu và môi trường phát triển. Chưa phải bản production.

## Nguyên tắc kiến trúc

Dự án dùng modular monolith: toàn bộ application code thuộc package `financial_report_qa`, còn
dữ liệu, cấu hình, script vận hành, test và tài liệu nằm ngoài `src/`.

```text
financial-assistant/
├── .github/                    # CI và dependency automation
├── configs/                    # cấu hình có version, không chứa secret
├── data/                       # raw, interim, processed, indexes, manifests, QA
├── docker/                     # image Linux tái lập
├── docs/                       # kiến trúc, development và ADR
├── scripts/                    # wrapper mỏng cho terminal
├── src/financial_report_qa/    # application package
├── tests/                      # unit, integration, golden, security
├── Makefile
├── pyproject.toml
└── uv.lock
```

Xem [kiến trúc chi tiết](docs/architecture.md) và
[ADR modular monolith](docs/decisions/0001-modular-monolith.md).

## Yêu cầu

- Windows 11 với WSL2.
- Distro Linux trong WSL, khuyến nghị Ubuntu.
- Git, GNU Make và [uv](https://docs.astral.sh/uv/).
- Python 3.11 x64; `uv` cài và quản lý runtime theo `.python-version`.
- RTX 3050 6 GB chỉ cần cho model local ở giai đoạn sau; downloader và test nền tảng chạy CPU.

WSL2/Linux là môi trường phát triển chuẩn. Không cần duy trì một bộ lệnh PowerShell song song.

## Thiết lập

Trong WSL:

```bash
cd /mnt/d/GitHub/financial-assistant
make setup
cp .env.example .env
make check
make build
```

`make setup` đồng bộ dependency development từ `uv.lock` và cài pre-commit hooks.

## Các lệnh chính

```bash
make lint           # Ruff
make format         # định dạng code
make typecheck      # mypy strict
make test           # pytest
make check          # toàn bộ quality gate
make build          # wheel + source distribution
make download-data  # chỉ dry-run dataset
```

CLI của sản phẩm:

```bash
uv run --frozen --no-sync financial-report-qa --help
uv run --frozen --no-sync financial-report-qa download-data --help
```

## Tải dataset

Kiểm tra chính xác số file và dung lượng trước; lệnh này không tải dữ liệu:

```bash
make download-data
```

Sau khi kiểm tra dung lượng, tải hoặc tiếp tục snapshot đầy đủ:

```bash
uv run --frozen --no-sync financial-report-qa download-data \
  --reserve-gb 100 \
  --download
```

Mặc định dữ liệu được ghi vào:

```text
data/raw/ocr_annual_financials/
```

Downloader giữ nguyên đường dẫn Unicode, ghim commit của Hugging Face sau dry-run, kiểm tra dung
lượng và dùng metadata cục bộ để tiếp tục khi tải gián đoạn. Việc tải corpus không tự tạo CSV hay
Parquet; tách và chuẩn hóa bảng là bước ingestion riêng.

Xem [hướng dẫn tải dữ liệu](docs/data-download.md).

## Vòng đời dữ liệu

```text
data/raw/          # nguồn bất biến, không commit
data/quarantine/   # file không đọc được hoặc không đạt kiểm tra
data/interim/      # artifact trích xuất có thể tái tạo
data/processed/    # documents/tables/cells dạng canonical Parquet
data/indexes/      # index retrieval có thể tái tạo
data/manifests/    # inventory và fingerprint nhỏ
data/qa/           # câu hỏi/annotation được phép lưu
```

Không commit raw reports, OCR output sinh lại được, Parquet, index, model, artifact đánh giá hoặc
secret.

## Cấu trúc package

```text
src/financial_report_qa/
├── core/           # settings, errors, logging
├── data/           # dataset acquisition và inventory
├── schemas/        # Pydantic contracts
├── ingestion/      # TXT/HTML table extraction
├── normalization/  # công ty, kỳ, metric, số và đơn vị
├── retrieval/      # lexical, dense, fusion, graph
├── planning/       # constrained financial query plan
├── execution/      # deterministic computation và verification
├── evaluation/     # metrics và error analysis
└── cli.py
```

Các module tương lai đã có ranh giới trách nhiệm nhưng chỉ được thêm implementation khi đến
milestone tương ứng.

## Quality và CI

Mọi pull request phải qua cùng quality gate với local:

1. Ruff lint.
2. Ruff format check.
3. mypy strict.
4. pytest.
5. Build wheel và source distribution.

GitHub Actions chạy gate trên Linux. Dependabot theo dõi dependency `uv`, Docker và GitHub
Actions theo lịch hàng tuần.

## Tài liệu

- [Kiến trúc](docs/architecture.md)
- [Môi trường phát triển](docs/development.md)
- [Tải dataset](docs/data-download.md)
- [Kế hoạch sản phẩm 30 ngày](plan.md)

## Week 1 Quality Gate (`dataset-pilot-v1`)

Tuần 1 kết thúc với một bộ annotation cố định trên 60 tài liệu pilot và một cổng chất lượng
tái lập được. Toàn bộ retrieval work tiêu thụ **release lock**, không phải thư mục `data/processed/` tùy ý.

### Lệnh vận hành chuẩn

```powershell
# 1. Chọn 60 tài liệu pilot và khởi tạo thư mục annotation
uv run --frozen --no-sync financial-report-qa week1-gate prepare `
  --manifest-path data/manifests/documents.jsonl `
  --release-path  data/processed/release_v2_37a61be7aebd `
  --annotation-root data/qa/week1_pilot_37a61be7aebd

# 2. Sinh worksheet gợi ý (không tự approve)
uv run --frozen --no-sync financial-report-qa week1-gate prepare-review `
  --manifest-path data/manifests/documents.jsonl `
  --release-path  data/processed/release_v2_37a61be7aebd `
  --corpus-dir    data/raw/financial_statements `
  --annotation-dir data/qa/week1_pilot_37a61be7aebd `
  --output-path   data/interim/week1_gate_review/37a61be7aebd/table-review.csv

# 3. Sau khi review thủ công, finalize
uv run --frozen --no-sync financial-report-qa week1-gate finalize-tables `
  --manifest-path data/manifests/documents.jsonl `
  --release-path  data/processed/release_v2_37a61be7aebd `
  --annotation-dir data/qa/week1_pilot_37a61be7aebd `
  --review-path   data/interim/week1_gate_review/37a61be7aebd/table-review.csv

# 4. Lấy mẫu 30 cell để audit thủ công
uv run --frozen --no-sync financial-report-qa week1-gate sample-cells `
  --manifest-path data/manifests/documents.jsonl `
  --release-path  data/processed/release_v2_37a61be7aebd `
  --corpus-dir    data/raw/financial_statements `
  --annotation-dir data/qa/week1_pilot_37a61be7aebd

# 5. Evaluate (sau khi đã đánh verified=true trong cell-audit.csv)
uv run --frozen --no-sync financial-report-qa week1-gate evaluate `
  --manifest-path data/manifests/documents.jsonl `
  --release-path  data/processed/release_v2_37a61be7aebd `
  --corpus-dir    data/raw/financial_statements `
  --annotation-dir data/qa/week1_pilot_37a61be7aebd `
  --report-root   data/interim/week1_gate/37a61be7aebd

# 6. Publish release lock (chỉ sau khi gate passed)
uv run --frozen --no-sync financial-report-qa week1-gate lock-release `
  --release-path    data/processed/release_v2_37a61be7aebd `
  --gate-result-path data/interim/week1_gate/37a61be7aebd/gate-result.json `
  --output-path     data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json
```

### Kết quả cổng Tuần 1 (2026-08-07)

| Check | Kết quả |
|---|---|
| `pilot_document_count` | 60/60 ✅ |
| `statement_type_coverage` | 70/30 (>100%) ✅ |
| `overall_table_usability` | 256/277 = 92.4% ≥ 85% ✅ |
| `accepted_cell_provenance` | 30 866/30 866 ✅ |
| `manual_cell_audit` | 30/30 ✅ |
| `eligible_strata_usability` | n/a (0 strata ≥10 annotations) ✅ |

- **Release fingerprint:** `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`
- **Manifest SHA-256:** `924d165211c63bbfc718b790f217ec356f80236e21fa0d8aa2acb497e186a5cf`
- **Release lock:** `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`
- **Replay determinism:** SHA-256 identiques sur 3 fichiers rapport (gate-result.json, gate-report.md, pareto-errors.csv)

## Day 8 BM25 Retrieval Baseline

Hệ thống truy xuất bảng dựa trên từ khóa (lexical) dùng mô hình BM25 kết hợp với các bộ lọc siêu dữ liệu (metadata) được định cấu hình chặt chẽ theo thiết kế Day 8.

### Quy tắc khóa dữ liệu (Release-Lock Rule)
Tất cả các câu lệnh tạo chỉ mục và đánh giá đều phải nhận và xác thực dựa trên tệp khóa dữ liệu tuần 1 bất biến:
- Thư mục release: `data/qa/week1_pilot_37a61be7aebd/`
- Vân tay dữ liệu (fingerprint): `37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f`
- Tệp khóa phát hành: `data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json`

### Ba lệnh truy xuất chính
1. **Tạo chỉ mục BM25 (build-index):**
   ```bash
   uv run --frozen --no-sync financial-report-qa retrieval build-index \
     --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
     --output-root data/indexes/bm25
   ```

2. **Xác thực bộ câu hỏi gold (validate-gold):**
   ```bash
   uv run --frozen --no-sync financial-report-qa retrieval validate-gold \
     --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
     --gold-path data/qa/retrieval-gold-v1.jsonl
   ```

3. **Chạy đánh giá và xuất báo cáo (evaluate):**
   ```bash
   uv run --frozen --no-sync financial-report-qa retrieval evaluate \
     --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
     --index-dir data/indexes/bm25/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f \
     --gold-path data/qa/retrieval-gold-v1.jsonl \
     --output-dir artifacts/evaluations
   ```

### Evidence BM25 v2 (2026-08-09)

Clean committed source (`4ee2fa6`) imports retrieval without the normalization runtime and passes
45 focused retrieval tests, Ruff, and mypy. The locked release contains 146,011 tables and the
reviewed gold contains 30 questions (gold SHA-256
`13888830E7DDE393BF3ED0E4561C02340912A6F36AB2B32503EF2FB2CFAC63F5`). Two independent v2
indexes and JSON/Markdown reports are byte-identical. Observed clean-source macro Recall@10 is
`0.8333333` and F2@10 is `0.4034392` (five ranking misses remain), below the provisional floors;
gold and query-specific rules were not changed. Full working-tree pytest is `556 passed, 1 skipped`;
full Ruff (84 errors) and mypy (33 errors) retain pre-existing non-retrieval failures.

## Day 9 Dense Retrieval

Runbook cho BGE-M3/FAISS GPU, cleanup fail-closed và theo dõi log: [docs/runbooks/day9-faiss-gpu.md](docs/runbooks/day9-faiss-gpu.md).
Native Windows vẫn CPU-only với `faiss-cpu`; FAISS GPU yêu cầu WSL2/Linux hoặc tự build FAISS từ
source với CUDA.

### GPU-accelerated encoding (2026-08-10)

`SentenceTransformerDenseEncoder` mặc định `device="cpu"` để giữ reproducibility theo Global
Constraints. CPU benchmark thật trên mẫu 400 tài liệu đo được `multilingual-e5-small` 36.5
docs/s (ETA ~1.1h/lần build) và `bge-m3` chỉ 4.9 docs/s (ETA ~8.2h/lần build) — với 2 lần build
mỗi encoder theo yêu cầu Task 7, tổng ~18.7h CPU thuần. Để giữ trong ngân sách 5–10h/ngày, đã mở
rộng có kiểm soát: `DenseEncoderSpec.device` nhận thêm `"cuda"`, CLI `build-dense-index` và
`evaluate-dense` có cờ `--encoder-device {cpu,cuda}` (mặc định vẫn `cpu`, không đổi hành vi cũ),
và khi `device="cuda"` thì `SentenceTransformerDenseEncoder` bật
`torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True)` cùng biến môi trường
`CUBLAS_WORKSPACE_CONFIG=:4096:8` để giữ A/B replay bit-identical trên cùng GPU.

Môi trường build: conda env `financial-dense-gpu` (tách biệt hoàn toàn khỏi `financial-faiss-gpu`
đã verified ở trên) với `torch==2.6.0+cu124` — cần `torch>=2.6` vì `transformers` mới chặn
`torch.load` cho checkpoint không phải safetensors (CVE-2025-32434) và snapshot BGE-M3 pinned chỉ
có `pytorch_model.bin`. FAISS trong env này là `faiss-cpu==1.15.0` (không cần FAISS GPU vì
`index.add()` không phải điểm nghẽn; điểm nghẽn là encode).

**Determinism thực nghiệm** (mẫu 200 tài liệu, 2 tiến trình Python độc lập):

| Encoder | docs/s (GPU) | SHA-256 vectors (2 lần chạy) |
|---|---:|---|
| multilingual-e5-small | 169 | `fe06042879872573bb2b68349961495ee755860b9bf4345cd1c516df0da6f4fd` (khớp) |
| bge-m3 | 51–63 | `7492b7e908f6eed2f65e9b579c136b53193056d2cd24b80c82f9e83459d205e7` (khớp) |

**Build thật trên corpus đầy đủ (146,011 tài liệu, A và B độc lập):**

```bash
uv run --frozen --no-sync financial-report-qa retrieval build-dense-corpus --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --output-root data/indexes/dense-day9-a
uv run --frozen --no-sync financial-report-qa retrieval build-dense-corpus --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --output-root data/indexes/dense-day9-b
financial-report-qa retrieval build-dense-index --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --corpus-dir data/indexes/dense-day9-a/<fp>/corpus --encoder bge-m3 --output-root data/indexes/dense-day9-a/<fp>/encoders --observation-path artifacts/evaluations/day9/bge-build.json --faiss-device cpu --encoder-device cuda --local-files-only
```

(lặp lại cho corpus B và cho `multilingual-e5-small`, cùng cờ `--encoder-device cuda`)

| Encoder | Revision | Dim | Build A/B time | Index size | index.faiss SHA-256 (A=B) |
|---|---|---:|---:|---:|---|
| bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | 1024 | 2557s / 2600s (~43 phút) | 598,061,101 bytes | `089a5aed2e70890c5c73c7a3dcc2de0e366fe297197481e520f05380e7ad4f26` |
| multilingual-e5-small | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | 384 | 331s / 343s (~5.6 phút) | 224,272,941 bytes | `3da3c40cbd11c11b1b6786f794afaf995f7cbfe49e0b10e52ade33038e20b346` |

`manifest.json` A/B cũng byte-identical: bge-m3
`c82ccaaab3bc36ffd1a9a17af52510812e56be7c964ab60c9d2ea3a5e5f1ab8f`, multilingual-e5-small
`010184c2cb8d431cb5c365bbafefad0b4ee3d862dc7aa384645798d6942e5c96`.

**Evaluate cold/warm (30 câu gold, 2 lần mỗi encoder):** mọi report có đúng 30 cold miss + 30 warm
hit, metric cold==warm tuyệt đối, không score vô hạn/NaN,
`deterministic_projection(A) == deterministic_projection(B)` cho cả hai encoder.

| Encoder | Cold p50/p95 | Warm p50/p95 | Precision@10 | Recall@10 | F2@10 |
|---|---:|---:|---:|---:|---:|
| bm25-v3 (reference) | — | — | 0.146667 | 0.883333 | 0.431217 |
| bge-m3 | 0.114/0.212s | 0.071/0.091s | 0.033333 | 0.250000 | 0.105820 |
| multilingual-e5-small | 0.126/0.150s | 0.073/0.098s | 0.036667 | 0.316667 | 0.123016 |

**Cả hai dense encoder đều thua rõ rệt BM25 v3 trên tập gold 30 câu hiện tại** (F2@10 dense chỉ
bằng ~24–29% của BM25). Trong 3 câu BM25 v3 "zero gold hits", `multilingual-e5-small` phục hồi
đúng 1 câu (`retq_7ea882a1f84e4570db63f91b3174bd7a21ec251f763a8d7d68b5b0d9fb2aca0c`) thành full
hit; `bge-m3` không phục hồi câu nào. Không sửa gold/câu hỏi/filter để có kết quả này.

Lệnh so sánh và xác minh:

```bash
uv run --frozen --no-sync financial-report-qa retrieval evaluate-dense --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --corpus-dir data/indexes/dense-day9-a/<fp>/corpus --index-dir data/indexes/dense-day9-a/<fp>/encoders/bge-m3-795294d329d0 --encoder bge-m3 --encoder-device cuda --gold-path data/qa/retrieval-gold-v1.jsonl --cache-dir data/indexes/dense-query-cache/day9-a-bge --observation-path artifacts/evaluations/day9/bge-build.json --output-path artifacts/evaluations/day9/bge-report.json
uv run --frozen --no-sync financial-report-qa retrieval compare-day9 --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json --bm25-report artifacts/evaluations/day9/bm25-v3/retrieval-day8-37a61be7aebd.json --bge-report artifacts/evaluations/day9/bge-report.json --e5-report artifacts/evaluations/day9/e5-report.json --output-dir artifacts/evaluations/day9
```

**Verification:** `pytest -q tests/unit/retrieval tests/integration/retrieval` → `84 passed, 3
skipped`; full working-tree `pytest -q` → `596 passed, 4 skipped`; full `ruff check .` → 102
errors, 0 trong file Day 9 (`dense_contracts.py`, `dense_encoder.py`, `cli.py`, test files sửa
đổi đều sạch riêng lẻ); full `mypy` → 33 errors, 0 trong file Day 9; `git diff --check` sạch. Số
lỗi ruff/mypy toàn repo là nợ kỹ thuật có sẵn ở `notebooks/`, `evaluation/week1_*`,
`normalization/service.py` — không liên quan đến retrieval/Day 9.

## Day 10 Fusion và Entity Parser

Entity parser xác định thực thể (company/period/metric/statement_type) từ câu hỏi bằng rule +
dictionary tái dùng nguyên vẹn (`normalization.companies`, `normalization.metrics`,
`normalization.statements`); không đoán khi thiếu bằng chứng — field mơ hồ bị để rỗng kèm
`AmbiguityCode` thay vì chọn bừa. Fusion hợp nhất BM25 và dense bằng weighted Reciprocal Rank
Fusion trên một lưới trọng số công bố trước trong `fusion_contracts.PRE_REGISTERED_WEIGHT_GRID`.

### Thước đo entity parser (hai bộ, không trộn)

- **Generated cases** (`entity_cases.py`): sinh từ giá trị thật trong release đã khóa
  (company_code, period, statement_type từ parquet; metric từ dictionary chuẩn hóa), nhãn đúng
  theo xây dựng — không cần gán tay. 14 template × 100 công ty = **1.400 case**, phủ đủ 6 mã
  ambiguity thực sự đạt được (`company_missing`, `company_conflict`, `period_relative_unresolved`,
  `period_incomplete`, `metric_unknown`, `period_missing`).
- **Held-out gold** (`retrieval-gold-v1`, 30 câu người viết): chạy đúng một lần, cuối cùng, chỉ
  chấm `company_codes`/`periods` (hai field duy nhất gán đủ 30/30 câu — `statement_types` chỉ có ở
  5/30 nên không dùng làm thước đo công bằng).

### Ba lệnh chính

```bash
uv run --frozen --no-sync financial-report-qa planning generate-entity-cases \
  --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --output-path data/qa/entity-cases-v1.jsonl

uv run --frozen --no-sync financial-report-qa planning evaluate-entities \
  --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --case-path data/qa/entity-cases-v1.jsonl \
  --gold-path data/qa/retrieval-gold-v1.jsonl \
  --output-dir artifacts/evaluations/day10

uv run --frozen --no-sync financial-report-qa retrieval evaluate-fusion \
  --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --index-dir data/indexes/bm25-day9-reference/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f \
  --corpus-dir data/indexes/dense-day9-a/<fp>/corpus \
  --dense-index-dir data/indexes/dense-day9-a/<fp>/encoders/multilingual-e5-small-1fa278c2f4d5 \
  --encoder multilingual-e5-small --gold-path data/qa/retrieval-gold-v1.jsonl \
  --cache-dir data/indexes/dense-query-cache/day9-a-e5 \
  --bm25-report artifacts/evaluations/day9/bm25-v3/retrieval-day8-37a61be7aebd.json \
  --output-dir artifacts/evaluations/day10
```

### Evidence Day 10 (2026-08-10)

**Entity parser:** case-set SHA-256 `c57f014d0fc51e8ed5b4dad371903e110bd6e6ec694c829ca422d2c3191247e1`
(hai lần sinh độc lập byte-identical). Trên 1.400 generated case: **exact-match rate = 1.000000**
(0 failure); precision/recall/F1 = 1.0 ở cả 5 trường (company_codes, periods, metrics,
statement_types, ambiguity). Trên 30 câu **held-out gold** (chạy đúng một lần):
**exact-match rate (company+period) = 1.000000**, company và period precision/recall/F1 đều 1.0,
0 failure. Không sửa gold hay chỉnh rule sau khi thấy kết quả held-out.

**Fusion:** đánh giá toàn bộ lưới 6 điểm trọng số trên 30 câu gold, dùng `bm25-v3` (F2@10
`0.431217`, Recall@10 `0.883333`) làm tham chiếu khóa cứng.

| bm25 | dense | Precision@10 | Recall@10 | F2@10 | ΔF2 vs BM25 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.0 | 0.146667 | 0.883333 | 0.431217 | +0.000000 |
| 1.0 | 1.0 | 0.096667 | 0.666667 | 0.297619 | −0.133598 |
| 2.0 | 1.0 | 0.116667 | 0.766667 | 0.353175 | −0.078042 |
| 3.0 | 1.0 | 0.136667 | 0.850000 | 0.406085 | −0.025132 |
| 4.0 | 1.0 | 0.143333 | 0.883333 | 0.424603 | −0.006614 |
| 0.0 | 1.0 | 0.036667 | 0.316667 | 0.123016 | −0.308201 |

**Mọi trọng số dense > 0 đều làm giảm F2 so với BM25 v3**, khớp phát hiện Day 9 rằng dense yếu hơn
BM25 rõ rệt trên tập gold hiện tại. Điểm tốt nhất theo quy tắc quyết định công bố trước là
`bm25=1/dense=0` — về bản chất là BM25 nguyên trạng, không có đóng góp thực từ dense — nên
`default_system = "fusion"` chỉ đúng theo nghĩa kỹ thuật (đạt đúng ngưỡng `≥ BM25 v3` ở cả F2 và
Recall vì đó chính là BM25). Đây là kết quả trung thực, không phải đóng góp thật của dense; ablation
Day 14 (`BM25; dense; fusion; fusion + graph`) sẽ dùng lại bảng này.

`deterministic_projection` khớp tuyệt đối giữa hai lần build GPU độc lập Day 9
(`dense-day9-a` và `dense-day9-b`) khi chạy lại fusion trên cả hai. Do `.venv` chính của máy này
không có CUDA (`torch.cuda.is_available() == False`), lệnh `evaluate-fusion` phía trên cần môi
trường có CUDA hoặc dense index build bằng `--encoder-device cpu` để tự chạy; bằng chứng ở trên
được tạo bằng cách nạp lại đúng các vector embedding GPU đã cache và xác minh bit-identical từ Day
9 (`data/indexes/dense-query-cache/day9-a-e5`, `day9-b-e5` — cả hai đều có sẵn đủ 30/30 câu gold),
qua một script không nằm trong `src/` chỉ để tái dùng cache có sẵn, không mã hoá câu hỏi mới. Vì
vậy trạng thái cache "cold" thật (buộc `encode_query`) chưa được xác minh lại bằng GPU thật ở đây;
hành vi cold-cache/warm-cache được xác minh riêng bằng unit test dùng fake encoder
(`tests/unit/retrieval/test_fusion.py`, `test_fusion_evaluation.py`), và cold==warm bit-identical
trên GPU thật đã được Day 9 xác minh cho đúng cơ chế cache mà fusion tái sử dụng nguyên vẹn.

**Verification:** `pytest -q tests/unit/planning tests/unit/retrieval tests/integration/retrieval
tests/integration/planning` → `133 passed, 3 skipped`; full working-tree `pytest -q` → `645
passed, 4 skipped` (tăng đúng 49 test so với baseline Day 9); full `ruff check .` → 102 errors, 0
trong file Day 10; full `mypy src tests` → 33 errors, 0 trong file Day 10; `git diff --check` sạch
trên toàn bộ file Day 10. Số lỗi ruff/mypy có sẵn không đổi so với Day 9, xác nhận Day 10 không
thêm nợ kỹ thuật mới.

## Day 11 Graph

GTR-lite table-relation graph (`retrieval/graph.py`, `graph_service.py`, `graph_evaluation.py`).
Cạnh không được materialize eagerly: `build_graph` dựng **bucketed adjacency**
(`relation -> key -> positions`), cùng scheme "vị trí trong dãy đã sort" mà `dense_index.py` dùng
để gắn hàng FAISS với `table_id`. `TableGraphService.neighbors()` suy ra `GraphEdge` khi được gọi;
không có fan-out cap ở Ngày 11 — đó là việc của Ngày 12.

### Năm quan hệ

| Quan hệ | Đối xứng | Weight |
|---|---|---|
| `same_document` | ✅ | `1/(1+line_gap)` |
| `shared_metric` | ✅ | Jaccard trên tập canonical metric |
| `adjacent_period` | ✅ | hằng số `1.0` |
| `same_statement_type` | ✅ | hằng số `1.0` |
| `explained_by_note` | ❌ (statement → notes cùng tài liệu) | hằng số `1.0` |

**`same_company` và `same_period` bị loại khỏi bộ quan hệ** — đo được 117.156.769 và 18.686.209
cặp vô hướng trên corpus khóa 146.011 bảng, không materialize được; quan trọng hơn, cả 30 câu gold
đều đã hard-filter `company_codes`/`periods` trước khi xếp hạng (`retrieval/filtering.py`) nên hai
quan hệ này chỉ nối tới bảng đã nằm trong pool eligible — không thêm thông tin. Lý do và số đo lưu
trong `GraphManifest.excluded_relations`, không bỏ lặng lẽ.

### Hai lệnh chính

```bash
uv run --frozen --no-sync financial-report-qa retrieval build-graph \
  --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --output-root data/indexes/graph-day11-a

uv run --frozen --no-sync financial-report-qa retrieval evaluate-graph \
  --release-lock data/qa/week1_pilot_37a61be7aebd/dataset-pilot-v1.json \
  --graph-dir data/indexes/graph-day11-a/37a61be7aebde1fbcfe3aca42e6ba4ff37ae87bdd1a9ba6696506bcd188e7d1f \
  --output-dir artifacts/evaluations/day11
```

### Evidence Day 11 (2026-08-13)

Build thật hai lần độc lập (A/B) trên corpus khóa 146.011 bảng: `buckets.jsonl` và
`manifest.json` byte-identical tuyệt đối.

| relation | buckets | membership | nodes w/ edges | isolated | directed edges | weight min | weight max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `adjacent_period` | 1.108 | 185.514 | 146.011 | 0 | 61.976.574 | 1.000000 | 1.000000 |
| `explained_by_note` | 1.263 | 4.301 | 680 | 145.331 | 3.148 | 1.000000 | 1.000000 |
| `same_document` | 1.963 | 146.011 | 146.008 | 3 | 12.008.152 | 0.000130 | 0.333333 |
| `same_statement_type` | 313 | 4.301 | 4.274 | 141.737 | 138.906 | 1.000000 | 1.000000 |
| `shared_metric` | 3.807 | 62.096 | 25.011 | 121.000 | 1.209.760 | 0.032258 | 1.000000 |

**0 node cô lập trên toàn bộ 5 quan hệ** (`nodes_with_no_edge_in_any_relation = 0`).

**Lỗi phát hiện khi chạy trên dữ liệu thật, không lộ ra ở fixture nhỏ:** 20.558 bảng (14,1%) có
nhiều `periods` cùng chung một năm (vd. `"2024"` và `"2024-12-31"`) và 1.449 bảng có
`metric_labels` trùng `canonical` khác `raw` — bucket ban đầu cộng trùng vị trí bảng nguồn (không
tạo self-loop, nhưng làm sai `membership_counts`/degree). Sửa bằng cách khử trùng năm/canonical
trước khi thêm vào bucket trong `build_graph`; thêm test hồi quy
`test_a_table_does_not_duplicate_its_own_position_within_one_bucket`. Sau khi sửa, hai build A/B
vẫn byte-identical và số liệu coverage khớp đúng số đo thủ công trên `documents.jsonl`.

**Verification:** `pytest -q tests/unit/retrieval tests/integration/retrieval` → `137 passed, 3
skipped`; full working-tree `pytest -q` → `695 passed, 4 skipped` (tăng đúng 50 test so với Day
10); full `ruff check .` sạch trên toàn bộ file Day 11; full `mypy src tests` → 33 errors có sẵn, 0
trong file Day 11 (không đổi so với Day 9/10); `git diff --check` sạch trên toàn bộ file Day 11.

## Day 12 Graph Expansion and Rerank

Day 12 evaluates a fixed 13-point graph rerank grid. Seeds are BM25 top-50; each retained graph
edge contributes rank-only RRF evidence, with `fan_out=25` per `(seed, relation)`. Every expanded
node is rechecked by the same hard metadata filters, then contradiction tiers are applied before
`(contradiction_count, -score, table_id)` ordering. `alpha=0` is the locked BM25 control and must
reproduce its ranking exactly.

The report intentionally has no `default_system`: only 4/30 questions have headroom, involving two
distinct missing tables, both already in BM25 top-50. The paired `expand_non_seeds` conditions
separate one-hop expansion from reranking; Day 14 retains or rejects graph support.

```bash
uv run --frozen --no-sync financial-report-qa retrieval evaluate-expansion \
  --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json \
  --index-dir data/indexes/bm25-v3/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a \
  --graph-dir data/indexes/graph-day11-a/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a \
  --gold-path data/qa/retrieval-gold-v1.jsonl \
  --bm25-report artifacts/evaluations/day13/bm25/retrieval-day8-422df141c935.json \
  --output-dir artifacts/evaluations/day12
```

### Re-run trên release `422df141c935` (2026-08-14)

Sau khi release được rebuild để sửa `row_group_context_raw` và giới hạn header 3 dòng (xem mục
sửa lỗi review ở trên), toàn bộ index (BM25/dense/graph) và 30 câu gold đã được rebuild/re-stamp
lại trên fingerprint mới. Grid Day 12 chạy lại cho kết quả **giống hệt về mặt định tính**: điểm
`alpha=0` tái lập đúng BM25 (F2=0.431217, Recall=0.883333, khớp số với baseline BM25 v3), không
điểm nào trên grid vượt qua với `dense`/graph weight > 0 → `default_system` vẫn là BM25 v3, đúng
kết luận Day 12 ban đầu.

## Day 13 Retrieval Evaluation

Đánh giá hiện hành dùng 70 câu gold đã review, khóa vào dataset fingerprint
`422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`. SHA-256 của
`data/qa/retrieval-gold-v1.jsonl` là
`0AAEEC29325596BF8E56FA91FE330D57C6B731E42842AB3096D04D9CAE43678F`; 30 record gốc được giữ
nguyên byte và 40 record mới được chọn/gán nhãn từ source trước khi mở ranked output.

### Phân bố gold70

| Chiều phân bố | Số câu |
|---|---:|
| Intent | lookup 24; compare 23; growth 23 |
| Số bảng gold | 1 bảng: 36; 2 bảng: 22; 3 bảng: 8; 4 bảng: 4 |
| Số period filter | 1 period: 37; nhiều period: 33 |
| Era theo period mới nhất | 2015–2019: 21; 2020–2023: 41; 2024–2025: 8 |
| Statement filter | balance sheet 10; cash flow 9; income statement 14; notes 12; không filter 25 |

Tập này phủ 40 công ty, 10 năm filter từ 2016 đến 2025 và có 34 câu multi-table. Chi tiết
provenance và quota nằm tại [`data/qa/retrieval-gold-v1.provenance.md`](data/qa/retrieval-gold-v1.provenance.md).

### Kết quả mở rộng trên 70 câu

Tất cả metric dưới đây được đọc từ các full report trong
[`artifacts/evaluations/day13/v2/`](artifacts/evaluations/day13/v2/). Mỗi system report lưu đủ
70 kết quả theo câu và bốn breakdown ngoài intent; summary macro-only cũ đã được thay thế.
Metric chỉ dùng top 10. Chỉ artifact BM25 `bm25-diagnostic/` chạy
truy hồi riêng đến hạng 100 để failure analysis, với điều kiện top-10 prefix phải khớp
metric trace; dense/fusion/expansion report không tuyên bố lưu cutoff chẩn đoán.

| System | TP | P@10 | R@3 | R@5 | R@10 | F2@10 | MRR | P@R | F2@R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 v3 | 105 | 0.150000 | 0.583333 | 0.725000 | 0.880952 | 0.422455 | 0.621939 | 0.422619 | 0.491346 |
| dense BGE-M3 | 69 | 0.098571 | 0.246429 | 0.411905 | 0.552381 | 0.265921 | 0.298503 | 0.196429 | 0.224174 |
| dense E5-small | 71 | 0.101429 | 0.220238 | 0.420238 | 0.601190 | 0.280335 | 0.274875 | 0.159524 | 0.184609 |
| fusion BGE | 105 | 0.150000 | 0.588095 | 0.725000 | 0.880952 | 0.422455 | 0.615748 | 0.427381 | 0.494129 |
| fusion E5 | 105 | 0.150000 | 0.588095 | 0.725000 | 0.880952 | 0.422455 | 0.615748 | 0.427381 | 0.494129 |
| graph expansion | 105 | 0.150000 | 0.588095 | 0.725000 | 0.880952 | 0.422455 | 0.615748 | 0.427381 | 0.494129 |

Hai fusion cùng chọn `bm25=1.0, dense=0.0`; graph expansion chọn anchor `alpha=0.0`. Vì dense và
graph không đóng góp tại các điểm thắng, default trung thực vẫn là BM25 v3. Báo cáo graph độc lập
[`retrieval-day11-graph-422df141c935.json`](artifacts/evaluations/day13/graph/retrieval-day11-graph-422df141c935.json)
xác nhận đủ 146.011 document và 0 node cô lập trên toàn bộ quan hệ giữ lại.

### Failure analysis và cổng Day 14

[`failures-422df141c935.json`](artifacts/evaluations/day13/failures-422df141c935.json) ghi nhận
11/70 câu lỗi top-10: 7 `zero_gold_hits`, 4 `partial_gold_hits`, 0 `no_eligible_documents`, 0
`no_index_tokens`. Root cause gán tay có evidence: 6 `missing_alias`, 4 `ranking_only`, 1
`gold_label_error`; các nhóm `filter_too_narrow`, `filter_too_wide`, `ocr_corruption` và `unknown`
đều bằng 0. Gold bị thiếu đều xuất hiện ở hạng chẩn đoán 11–44. Nhãn gắn tay được
version-control độc lập tại
[`data/qa/retrieval-failure-annotations-v1.jsonl`](data/qa/retrieval-failure-annotations-v1.jsonl);
không nhãn nào được sinh từ ranked list.

### CLI tái lập V2/failure từ clean clone

Với release lock và BM25 index đã có, các lệnh sau không cần network/GPU:

```bash
uv run --frozen --no-sync financial-report-qa retrieval evaluate-v2 \
  --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json \
  --index-dir data/indexes/bm25-v3/422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a \
  --gold-path data/qa/retrieval-gold-v1.jsonl --gold-version gold70 \
  --diagnostic-k 100 --output-dir artifacts/evaluations/day13/v2/bm25-diagnostic

uv run --frozen --no-sync financial-report-qa retrieval derive-v2 \
  --release-lock data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json \
  --gold-path data/qa/retrieval-gold-v1.jsonl --gold-version gold70 \
  --source-report artifacts/evaluations/day13/dense-bge-m3.json \
  --source-kind dense --system-name dense-bge-m3 \
  --output-dir artifacts/evaluations/day13/v2

uv run --frozen --no-sync financial-report-qa retrieval export-failures \
  --evaluation-report artifacts/evaluations/day13/v2/bm25-diagnostic/retrieval-v2-422df141c935.json \
  --annotations data/qa/retrieval-failure-annotations-v1.jsonl \
  --output-dir artifacts/evaluations/day13
```

Lặp `derive-v2` với `source-kind=legacy|dense|fusion|expansion` và report tương ứng để
tái sinh sạch 6 system artifacts. Khóa reference có hai descriptor bất biến: `gold30`
(lịch sử) và `gold70` (hiện hành), gắn với fingerprint, SHA gold, digest question ID,
question count, macro và SHA artifact; dùng `--gold-version gold30` để replay subset cố định.

Day 13 hoàn tất phạm vi **đánh giá**, nhưng chưa đạt cổng Day 14 theo ADR 0002: kết quả tốt nhất
quan sát được là F2@R `0.494129` và Recall@10 `0.880952`, thấp hơn hai ngưỡng tương ứng `0.80` và
`0.90`. Ưu tiên tiếp theo là sửa gold label đã xác nhận và cải thiện normalization/alias; không
đổi công thức metric hoặc tuyên bố gate pass.

## Day 14 Week 2 Gate Review

Chẩn đoán: cả 15 bảng gold bị trượt khỏi top-10 (11 câu lỗi ở Day 13) đều nằm trong top-100,
hạng 11–44 — không có bảng nào ngoài tầm với. Đây là bài toán **xếp hạng**, không phải thu hồi,
nên đổi embedding model không giải quyết được gì (khớp với việc dense thua BM25 ở mọi cấu hình
suốt Day 9/10/13).

Nguyên nhân gốc: [`documents.py`](src/financial_report_qa/retrieval/documents.py) trước đây lọc
`FILTER (WHERE c.row_label_canonical IS NOT NULL)` khi gom nhãn dòng vào document BM25 — khi
normalization không canonical hoá được nhãn thì **cả `raw` cũng bị vứt**. Đo trên release khóa:
chỉ 5,93% nhãn được canonical hoá, khiến 120.920/146.011 bảng (82,8%) không có nhãn dòng nào
trong document, trong đó 109.499 bảng có sẵn nhãn raw bị bỏ đi.

### Sửa lỗi và rebuild (`documents.py`, BM25 v4)

Thêm dòng `unconfirmed labels:` vào document text, giữ nhãn `row_label_raw` khi canonical hoá thất
bại, tách riêng khỏi dòng `metrics`/`metric aliases` (đã canonical hoá) để không lẫn tín hiệu đã
xác thực với tín hiệu chưa xác thực. Sửa một nhãn gold sai kèm theo (`gold_label_error` từ Day 13):
câu hỏi MML 2017 dùng nhầm bảng "Công ty liên kết sở hữu gián tiếp" thay vì bảng "Các công ty con
sở hữu trực tiếp" — đã thay đúng bảng và nới filter `statement_types` (bảng đúng không được phân
loại `notes` dù nằm trong phần thuyết minh, một lỗ hổng phân loại riêng ngoài phạm vi sửa lần này).
Rebuild BM25 v4/dense/graph trên release `422df141c935…` không đổi (chỉ nội dung document đổi),
gold 70 câu.

| | F2@10 | F2@R | Precision@R | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BM25 v3 (cũ) | 0.422455 | 0.491346 | 0.422619 | 0.880952 | 0.621939 |
| **BM25 v4 (mới)** | **0.437709** | **0.483581** | 0.410714 | **0.914286** | 0.622778 |
| dense bge-m3 | 0.288204 | 0.268108 | 0.238095 | 0.630952 | 0.359836 |
| dense e5-small | 0.286606 | 0.270792 | 0.234524 | 0.609524 | 0.321247 |
| fusion bge / e5, graph expansion | = BM25 v4 (control thắng ở mọi cấu hình lưới) | | | | |

**Recall@10 đạt cổng** (`0.914286 ≥ 0.90`, TP 105→109/122). **F2@R chưa đạt** (`0.483581 < 0.80`)
và giảm nhẹ so với v3 (`-0.007765`): thêm từ vựng giúp nhiều câu khó tìm đúng bảng hơn (Recall@3
`+0.063`, Recall@5 `+0.089`) nhưng làm nhiễu thứ hạng của các câu vốn đã dễ, giảm Precision@R —
đánh đổi thật, không phải hồi quy.

Khoảng cách era 2015–2019 (nghi ngờ ở Day 13 nhưng chưa xác định nguyên nhân — tỷ lệ canonical hoá
đo được phẳng ~5.5–6.4% ở mọi năm) **biến mất sau khi sửa**: Precision@R của 2015–2019 so với
2024–2025 đổi từ tỷ lệ `2.35×` xuống `0.95×`. Xác nhận đây là hệ quả của lỗi vứt nhãn dòng, không
phải chất lượng OCR/normalization của báo cáo cũ.

### Quyết định graph expansion (ADR 0003)

Lưới 13 điểm chạy lại trên corpus đã sửa: **mọi điểm `alpha > 0` đều tệ hơn control `alpha=0`** ở
cả F2 và Recall@10, chênh 0.13–0.20 tuyệt đối — dứt khoát hơn nhiều so với Day 12 (4/70 câu còn
headroom) và Day 13 (chênh 0.003 giữa best và control). Cơ chế: với ngân sách top-10 cố định, mọi
bảng lân cận graph chèn thêm đều đẩy một bảng BM25-xếp-hạng-cao ra ngoài top-10, và BM25 v4 đã
mạnh hơn nhiều sau khi sửa nhãn. Quyết định:
[ADR 0003](docs/decisions/0003-graph-expansion-decision.md) — bỏ graph expansion khỏi hệ thống
mặc định, giữ code/index/báo cáo làm phụ lục có thể bật lại nếu gold sau này bổ sung câu cần bảng
`notes` (hiện 0/70 câu, quan hệ `explained_by_note` chưa từng được đánh giá thật).

### Cổng Day 14

Recall@10 đạt (`0.914286 ≥ 0.90`); F2@R chưa đạt (`0.483581 < 0.80`). Theo nhánh xử lý đã chốt
trước ở [docs/plans/day14-week2-gate-review.md](docs/plans/day14-week2-gate-review.md): sang
Tuần 3 với nợ kỹ thuật ghi ở Ngày 27 (đánh giá reranker nhẹ trên top-50 trước khi tối ưu latency);
planner/compiler Tuần 3 phải chịu được top-10 còn nhiễu.

## Day 15 FinancialQueryPlan

**Phạm vi:** `data/qa/retrieval-gold-v1.jsonl` (gold70) là gold **đánh giá truy hồi** — 3 intent
`lookup`/`compare`/`growth`, dùng để đo Recall@10/F2@R. Nó **không phải** tập câu hỏi nộp bài và
không map 1-1 sang 8 operation thực thi (`lookup`, `compare`, `difference`, `growth_rate`, `ratio`,
`average`, `sum`, `rank`) của `FinancialQueryPlan`. `data/official/test_questions.json` — tập câu
hỏi nộp bài thật — chưa tồn tại trong repo tại thời điểm viết Ngày 15; 8 operation đang bị đóng
băng dựa trên [plan.md § 2.3](plan.md) mà chưa thấy phân bố câu hỏi thật (rủi ro ghi ở
[plan.md § 14](plan.md)). Các câu hỏi gold70 dạng liệt kê thuyết minh (ví dụ *"Tra cứu bốn bảng
thuyết minh về danh sách công ty con..."*) nằm ngoài phạm vi `FinancialQueryPlan` vì chúng không có
đáp án số.

### Chẩn đoán: Ngày 14 chưa đủ cho compiler

Ngày 14 chỉ sửa nhánh **truy hồi** (giữ `row_label_raw` trong text document BM25), không đổi
`cells.row_label_canonical` — thứ compiler dùng để định vị hàng. Đo trên 70 câu gold (đối chiếu
entity parser Day 10 với canonical label thật của bảng gold):

| Kết quả | Số câu | Tỷ lệ |
|---|---:|---:|
| Mọi metric giải được tới canonical | 42 | 60,0 % |
| Parser không rút được metric nào (phần lớn câu `notes` liệt kê) | 20 | 28,6 % |
| Có metric nhưng không khớp bảng gold | 6 | 8,6 % |
| Khớp một phần | 2 | 2,9 % |

Bảng gold chỉ 56/91 (61,5 %) có ≥ 1 canonical row label. Nhưng mật độ dữ kiện `(company, period,
metric)` sau khi đã có canonical rất cao: một khi metric đã tồn tại, `growth_rate` (cần 2 period)
đạt 93,2 %, `ratio` (cần 2 metric) đạt 99,3 %, `rank`/`compare` (cần 2 company) đạt 95,7 % — nút
thắt hoàn toàn nằm ở bước định vị metric đầu tiên, không phải arity nhiều thực thể.

### ADR 0004: `metric_selector` hai nhánh

[ADR 0004](docs/decisions/0004-metric-locator-strategy.md) chọn `MetricSelector` với đúng một
trong hai khoá: `{"canonical": "net_revenue"}` (56 giá trị trong
[`normalization/metrics.py`](src/financial_report_qa/normalization/metrics.py)) hoặc
`{"raw_text": "Cho vay khách hàng"}` (chuỗi sao chép nguyên văn từ bảng ứng viên, không viết tắt).
Validator Ngày 15 chỉ kiểm tra hình dạng (`canonical` thuộc 56 giá trị, đúng một nhánh có mặt);
việc `raw_text` có thật sự khớp một hàng trong `candidate_table_ids` cụ thể hay không là việc của
compiler (Day 18), không lặp lại ở đây.

### Schema và semantic validator

`planning/plan_contracts.py` (structural: `extra="forbid"`, `MetricSelector`, ngữ pháp `period`
canonical `"YYYY"` — từ chối `"YYYY-MM-DD"`, `candidate_table_ids` 1–12 phần tử không trùng) và
`planning/plan_validator.py` (bảng arity theo operation, `PlanErrorCode` theo tiền lệ
`AmbiguityCode`/`NormalizationIssueCode`). Tám operation được định nghĩa quyết định (không có
trong `plan.md` gốc, chốt lại ở đây để không mơ hồ khi thực thi):

| Operation | companies | periods | metric field | Field riêng |
|---|:---:|:---:|---|---|
| `lookup` | 1 | 1 | `metric` | — |
| `compare` | 1 | 1 | `metric_a`, `metric_b` (hiệu số) | — |
| `difference` | 1 | 2 (tăng dần) | `metric` | — |
| `growth_rate` | 1 | 2 (tăng dần) | `metric` | `expected_unit` phải là `percent` |
| `ratio` | 1 | 1 | `numerator_metric`, `denominator_metric` | `expected_unit` phải là `ratio`/`percent` |
| `average`/`sum` | đúng một trong hai chiều biến thiên | | `metric` | — |
| `rank` | ≥ 2 | 1 | `metric` | `top_k` bắt buộc, `1 ≤ top_k < len(companies)` |

`compare` = hai metric cùng một company/period (không phải hai company hay hai period — đã có
`difference`/`growth_rate` riêng cho việc đó). `average`/`sum` từ chối cả hai trường hợp suy biến:
không chiều nào biến thiên (company=1, period=1 — không có gì để gộp) và cả hai chiều cùng biến
thiên (mơ hồ tích chéo).

Field không thuộc operation phải **vắng mặt** (`None`), không phải để trống ngầm — ép bằng
`_check_forbidden_extras`, liệt kê field **được phép** thay vì field bị cấm, để field mới mặc định
bị cấm ở operation chưa khai báo thay vì mặc định lọt qua.

### Property test bắt lỗi thật

`tests/unit/planning/test_plan_property.py` so khớp `validate_plan_semantics` với một hàm đặc tả
arity viết độc lập (không dùng lại code validator). Lần chạy đầu tiên **thất bại thật**: `compare`,
`difference`, `growth_rate`, `ratio`, `average`, `sum`, `rank` thiếu kiểm tra cấm cho
`numerator_metric`/`denominator_metric`/`metric_a`/`metric_b` — chỉ `lookup` (viết đầu tiên, làm
mẫu copy) kiểm đủ cả 5 field cấm. Đây đúng loại lỗi property test được thiết kế để bắt: thiếu sót
lặp lại có hệ thống khi copy-paste qua nhiều nhánh tương tự, mà unit test theo từng operation
riêng lẻ không tự động phát hiện chéo được. Sửa bằng thiết kế khai báo `_check_forbidden_extras`
ở trên, sau đó property test (300 example ngẫu nhiên) và toàn bộ 93 test `planning`/`golden/plans`
đều xanh.

### JSON examples

8 ví dụ hợp lệ trong [`tests/golden/plans/valid/`](tests/golden/plans/valid/): 2/8 (`lookup`,
`growth_rate`) lấy trực tiếp từ câu hỏi gold70 thật (`question_id` ghi trong
[`manifest.json`](tests/golden/plans/manifest.json)); 6/8 còn lại dựng từ dữ kiện thật đo được
trong release vì gold70 (3 intent lookup/compare/growth) không có câu hỏi khớp hình dạng
`compare`/`difference`/`ratio`/`average`/`sum`/`rank` — ghi rõ trong manifest, không bịa số liệu.
12 case invalid trong [`tests/golden/plans/invalid/`](tests/golden/plans/invalid/), mỗi case đúng
một vi phạm, phân theo tầng bắt lỗi (`schema` = pydantic construction, `semantic` =
`validate_plan_semantics`).

`pytest -q`: 830 passed, 4 skipped. `ruff check .`: sạch. `mypy`: 0 lỗi mới (168 file).

## Day 16 Deterministic Parsing và Rule Planner

**Chẩn đoán:** đo trực tiếp `entity_parser.py` trên gold70 và 1.400 entity case (không đoán) phát
hiện 6 vấn đề. [docs/plans/day16-deterministic-planning.md](docs/plans/day16-deterministic-planning.md)
là bản đo đầy đủ; [ADR 0005](docs/decisions/0005-operation-coverage-gaps.md) chốt 3 quyết định
thiết kế. Đã khắc phục 3/6 phát hiện, hoãn có chủ đích 2/6 (không phải nợ ẩn), 1/6 là ràng buộc
kiến trúc đúng như thiết kế:

| Phát hiện | Xử lý |
|---|---|
| Lỗi năm trần trụi ("giữa năm 2016 **và 2017**") bỏ mất 20 % câu gold70 | **Sửa** — bắt năm trần trụi sau từ nối kỳ, chỉ khi đã có kỳ neo trước đó (0 dương tính giả/1.400 case) |
| `intent` gold70 ≠ `operation` Ngày 15 (10/23 câu `compare` thực chất là `difference`) | **Ghi văn bản** — bảng ánh xạ trong ADR 0005 |
| Lỗ hổng operation: so sánh chéo công ty `(≥2 company, 1 period, 1 metric)`, 100/1.400 case | **Sửa (ADR 0005 quyết định A1)** — thêm operation `compare_companies` |
| "Biến động đầu năm–cuối năm" là hai **hàng**, không phải hai kỳ | **Hoãn có chủ đích (B2)** — abstain có mã, chờ compiler Ngày 18 |
| Từ điển thiếu họ ngân hàng (`cho vay khách hàng`, `chứng khoán đầu tư`... 0 % canonical) | **Sửa một phần** — lexicon riêng phía câu hỏi, cứu 10/20 câu `metric_unknown` |
| `cells.row_label_canonical` đóng băng trong release khóa | **Không sửa** — đúng ràng buộc thiết kế, xác nhận `dataset_fingerprint` không đổi |

### Rule planner: `QueryEntities → FinancialQueryPlan | abstain`

`planning/rule_planner.py::build_plan` thuần, không tự gọi truy hồi (`candidate_table_ids` được
tiêm vào). Luật xác định: `(≥2 company, 1 period)` → `compare_companies`; `(1 company, 1 period)` →
`lookup`; `(1 company, 2 period)` → `growth_rate` nếu câu chứa "tăng trưởng"/"biến động"/"tốc độ",
ngược lại `difference`. Metric không thuộc 56 giá trị canonical (từ vựng ngân hàng mới thêm) dùng
nhánh `raw_text` của `MetricSelector` (ADR 0004 Option C), không ép vào `canonical`. Mọi plan trả về
đã qua `validate_plan_semantics`; không semantic issue nào lọt ra ngoài — vi phạm nội bộ tự động
rơi về abstain thay vì trả plan sai.

5 mã abstain (`PlanAbstainCode`), theo đúng khuôn code-và-field của `AmbiguityCode`/`PlanErrorCode`:
`entity_ambiguous`, `period_grammar_unsupported`, `multi_metric_unsupported`, `operation_unknown`,
`metric_role_unassignable` (dự phòng, chưa có đường dẫn nào kích hoạt ở Ngày 16).

### Bộ case và harness đánh giá

`planning/plan_cases.py` gán nhãn `expected_operation`/`expected_abstain_code` cho toàn bộ 1.400
entity case đã ghim (Ngày 10) bằng một hàm thuần theo `template_id` — **không** chạy rule planner
rồi copy kết quả, giữ đúng kỷ luật "bug trong planner không thể tự làm hỏng đáp án của chính nó".
Phạm vi cố ý thu hẹp: gán nhãn 14 template sẵn có, không thêm template mới cho từ vựng operation ở
`day16-deterministic-planning.md § 1.8` (`gấp mấy lần`, `biên`...) vì từ vựng đó xuất hiện **0 lần**
trong cả gold70 lẫn entity case — thêm template cho nó sẽ là suy đoán, ghi rõ trong docstring của
module để không bị hiểu nhầm là bao phủ đầy đủ.

`planning evaluate-plans` (mới, cùng khuôn `evaluate-entities`) cho hai báo cáo độc lập, không trộn:

| Báo cáo | Nguồn | Kết quả thật |
|---|---|---:|
| `plan-cases-*.md` — số chính, được lặp | 1.400 plan case | operation accuracy **1,000000**, abstain recall **1,000000**, **false-plan rate 0,000000** (chỉ tiêu cứng DoD) |
| `plan-held-out-*.md` — **chỉ mô tả**, chạy 1 lần | gold70 (70 câu) | plannable rate **72,9 %** (51/70): 20 `lookup`, 18 `growth_rate`, 13 `difference`; abstain: 10 `entity_ambiguous`, 9 `multi_metric_unsupported` |

Báo cáo held-out gold70 **không phải** điểm chính xác — gold70 chỉ có 3 intent
(`lookup`/`compare`/`growth`) không map 1-1 sang 9 `PlanOperation`, nên không có đáp án đúng từng
câu để so khớp (đúng phát hiện #2). Nó chỉ mô tả khả năng lập plan, ghi rõ trong docstring và cả
tiêu đề Markdown của báo cáo. So với ước tính sơ bộ 43/70 (61,4 %) trước khi sửa lỗi năm và mở
từ điển, con số đo thật 51/70 cao hơn — phần lớn nhờ 14 câu được cứu khỏi bug năm trần trụi.

`pytest -q`: 871 passed, 4 skipped. `ruff check .`: sạch. `mypy`: 0 lỗi (89 file).

## Day 17 LLM Planner

**Chẩn đoán trước khi viết code:** đo lại 19 câu gold70 bị rule planner Ngày 16 abstain cho thấy
**không câu nào thật sự cần LLM** — 7 câu khớp đúng arity `compare` (lỗ hổng định tuyến xác định,
không phải lỗ hổng ngôn ngữ), 8 câu liệt kê/bộ phận không có đáp án số, 2 câu thiếu từ điển, 2 câu
vượt arity schema. Vì vậy DoD Ngày 17 **không** đặt là "tăng plannable rate trên gold70" — headroom
đo được bằng 0. Lý do thật để vẫn xây: rule planner chỉ phát ra 4/9 operation trên 1.400 plan case
(`compare`/`ratio`/`average`/`sum`/`rank` chưa từng được phát ra), và
`data/official/test_questions.json` vẫn chưa tồn tại — LLM là bảo hiểm phủ sóng cho ngữ pháp chưa
từng thấy, không phải cải thiện một con số đã đo. Xem
[docs/plans/day17-llm-planner.md](docs/plans/day17-llm-planner.md) và
[ADR 0006](docs/decisions/0006-llm-planner-role.md) (quyết định A1 router / B1 không có row label
trong prompt / C1 chấm trên plan case / D một lượt repair duy nhất cho cả 3 loại lỗi).

### Kiến trúc: router → LLM planner → client

```
plan_router.route_plan          # ADR 0006 A1: luật chạy trước, LLM chỉ chạy khi luật abstain
  ├─ rule_planner.build_plan     # Ngày 16, không đổi
  └─ llm_planner.build_plan      # question -> FinancialQueryPlan | abstain, tối đa 1 lần repair
       ├─ llm_prompt.py          # schema rút gọn viết tay + 9 few-shot, không dump model_json_schema()
       ├─ llm_contracts.py       # LLMPlanOutput = FinancialQueryPlan trừ candidate_table_ids
       └─ llm_client.py          # httpx OpenAI-compatible, bounded retry, không retry 4xx
```

`llm_planner.build_plan` không bao giờ trả một plan chưa qua `validate_plan_semantics` — cùng bất
biến rule planner đã giữ từ Ngày 16 (ADR 0005 §Hệ quả). Ba loại lỗi (JSON hỏng, `LLMPlanOutput` sai
schema, `validate_plan_semantics` từ chối) đi chung đúng một lượt repair rồi abstain có mã
(`llm_invalid_json` / `llm_plan_invalid` / `llm_unavailable`) — không có lượt thứ hai.

### Ba ràng buộc đã đo trước khi thiết kế prompt

| Ràng buộc | Số đo | Hệ quả thiết kế |
|---|---:|---|
| `FinancialQueryPlan.model_json_schema()` | 963 token (bge-m3) = 23,5 % cửa sổ 4.096 | Prompt dùng bảng arity viết tay (`_OPERATION_GUIDE`), không dump schema |
| Row label 6 bảng ứng viên, p90 | ~1.300 token | Loại khỏi prompt (quyết định B1); mọi field metric chấp nhận `raw_text` dự phòng |
| 12 `candidate_table_ids` vs `max_output_tokens=160` | 192–400 token > 160 | LLM không sinh table id; caller tiêm sau khi parse (`LLMPlanOutput` không có field này) |

Prompt hệ thống thật đo được **3.188 chars ≈ 1.594 token ước lượng** (tỷ lệ 2,0 chars/token bi quan
có chủ đích so với 2,16–4,79 đo bằng bge-m3), nằm trong ngân sách nửa cửa sổ 2.048 token.

### `configs/*.yaml` — từ code chết thành có kiểu

Khối `llm:` trong `configs/base.yaml`/`configs/local_rtx3050.yaml` trước Ngày 17 có **0 tham chiếu
Python**. `core/config.py::LLMSettings` (`extra="forbid"`) và `load_llm_settings()` nối dây thật,
lớp sau đè lớp trước — bắt được ngay một khóa lạ thay vì âm thầm bỏ qua. Đã sửa luôn
`execution.allow_operations` thiếu `compare_companies` (sót lại từ Ngày 16/ADR 0005).

### Đo được: an toàn khi 0 mô hình sống, chưa đo được: độ chính xác LLM thật

**Không có mô hình llama.cpp trong môi trường này** (`models/` rỗng, `127.0.0.1:8080`
`ConnectTimeout`) — nợ đã biết, ghi rõ trong kế hoạch Ngày 17 trước khi bắt đầu, không phải phát
hiện sau. `planning evaluate-llm-plans` (CLI mới) chạy được **hoàn toàn offline** qua
`ReplayCacheClient`: cache rỗng → mọi câu hỏi abstain `llm_unavailable` thay vì crash hay bịa plan.

| Báo cáo | Cases | Kết quả (cache rỗng, không mô hình) |
|---|---:|---|
| `llm-plan-cases-*.md` — LLM planner đứng một mình | 600 (`expected_operation`) | operation accuracy 0,0; invalid-JSON rate 0,0 (mọi case là `llm_unavailable`, không phải JSON hỏng) |
| `router-abstain-*.md` — router đầy đủ | 800 (`expected_abstain_code`) | **abstain recall 1,000000; false-plan rate 0,000000** |

Con số quan trọng nhất là dòng thứ hai: **dù LLM hoàn toàn không khả dụng, router không bao giờ trả
bừa một plan** — đúng chỉ tiêu cứng ADR 0006 đặt ra. `operation_accuracy`/`invalid_json_rate` của
LLM planner trên mô hình thật để lại cho lần chạy `--live` khi có server, ghi cache lại để tái lập
(`sha256(model identity + prompt)`, cùng tiền lệ cache truy vấn dense Ngày 9) — không đặt ngưỡng
cứng vì chưa có số đo thật nào để đặt ngưỡng dựa trên đó.

`pytest -q`: 916 passed, 4 skipped (chỉ symlink/ACL Windows). `ruff check .`: sạch. `mypy`: 0 lỗi
(186 file, `src` + `tests`).

## Day 18 Deterministic Compiler

**Chẩn đoán trước khi viết code:** phần số học **không** phải chỗ khó — chỗ khó là **locator**: đi
từ `FinancialQueryPlan` xuống một ô số trong `cells.parquet`. Đo trên 51 plan mà rule planner Ngày 16
sinh ra cho gold70 (82 khe `plan × kỳ × selector`): chỉ **24/51 plan (47,1 %)** giải được trọn vẹn.
Nguyên nhân gốc là kỳ, không phải metric — chỉ **15,4 %** ô có `period`, và **62,5 % bảng (91.266)
không có `period` trên bất kỳ ô nào** vì dùng bố cục `Số đầu năm`/`Số cuối năm` với năm nằm ở cấp tài
liệu (`documents.report_year`). Thêm quy tắc suy diễn kỳ nâng lên **30/51 (58,8 %)** — số này về sau
khớp chính xác với kết quả CLI thật. Xem
[docs/plans/day18-deterministic-compiler.md](docs/plans/day18-deterministic-compiler.md) và
[ADR 0007](docs/decisions/0007-deterministic-compiler-contract.md) (quyết định A1 đọc thẳng
`cells.parquet` / B1 hình chiếu dạng dài / C2 suy diễn kỳ / D1 không bao giờ đoán / E1 tái dùng
`convert_scale` / F1 replay `pandas_query` là điều kiện DoD).

### Kiến trúc: cell_frame → locator → operations → pandas_query, gộp bởi compiler

```
execution.compiler.compile_plan(plan, release_dir, execution_settings)
  ├─ cell_frame.build_cell_frame     # ADR 0007 A1/B1/C2: hình chiếu dạng dài, kỳ đã chuẩn hoá/suy diễn
  ├─ locator.locate                  # ADR 0007 D1: 4 nhánh — metric_not_found / period_unresolved /
  │                                  #   khớp duy nhất / cell_ambiguous, không bao giờ đoán
  ├─ operations.compile_*            # ADR 0007 E1: 9 hàm, một cho mỗi operation, dùng lại
  │                                  #   normalization/units.py::convert_scale nguyên trạng
  └─ pandas_query.render/replay      # ADR 0007 F1: sinh chuỗi Pandas dễ đọc + replay qua AST
                                     #   whitelist (không eval/exec) trước khi trả `answered`
```

`build_cell_frame` đọc thẳng `cells.parquet` (không qua `data/table_frame.py`'s placement-join —
đó là để dựng lại grid vị trí, không cần cho tra cứu giá trị) và áp hai bộ lọc bắt buộc:
`col_idx > 0` (loại ô nhãn) và `value_numeric IS NOT NULL` (loại ô placeholder như `-`). `locate`
không bao giờ lấy dòng đầu khi có xung đột: 33.321/35.766 nhóm duplicate-row toàn corpus (93,2 %) có
giá trị khác nhau, và nhóm cùng giá trị vẫn phải cùng đơn vị (100 VND ≠ 100 VND_million).

### `pandas_query` phải replay được, không phải chuỗi trang trí

`tables.csv_path` **NULL cho cả 146.011 bảng** trước Ngày 18 — hình chiếu mà `pandas_query` trỏ tới
chưa từng tồn tại. `compile_plan` giờ luôn kiểm chứng: mọi kết quả `answered` phải replay đúng qua
`replay_pandas_query` (một AST interpreter whitelist tự viết — không `eval`/`exec`, đúng tinh thần
Ngày 19) trên một khung dữ liệu chỉ gồm các ô evidence, và một lệch số là `ExecutionReplayMismatchError`
(hỏng build), không phải một đáp án sai âm thầm.

### Kết quả đo trên gold70 (release đã khoá, CLI `execution compile-plans`)

| Số đo | Giá trị |
|---|---:|
| Plan giải được tới ô số | **30/51 (58,8 %)** — khớp đúng trần lý thuyết đo trước khi code |
| `metric_not_found` | 11 |
| `period_unresolved` | 8 |
| `cell_ambiguous` | 2 |
| `unit_incompatible` / `division_by_zero` trên gold70 | 0 (bao phủ riêng bởi golden test) |

29 plan không giải được đều nằm ngoài tầm compiler — thiếu `row_label_canonical` (chỉ phủ 0,9 % ô
toàn corpus) hoặc bảng không có cả `period` tường minh lẫn dấu hiệu suy diễn được. Compiler trả error
code có kiểu cho cả hai, không nới `locate` để "cứu" thêm khe bằng cách đoán.

`pytest -q`: 987 passed, 4 skipped (chỉ symlink/ACL Windows). `ruff check .` / `ruff format --check .`:
sạch. `mypy`: 0 lỗi (103 file `src`).

## Day 19 Sandbox Executor

**Chẩn đoán trước khi viết code:** Ngày 19 không phải bọc thêm một lớp bảo mật quanh thứ đang chạy
tốt — nó phải đóng một biên tin cậy đang rò rỉ **lỗi correctness**, không chỉ lỗi bảo mật giả định.
`render_pandas_query` nội suy chuỗi tự do bằng f-string; **1.988 nhãn dòng có thật** trong corpus
(1.790 bảng, 9.944 ô số) chứa dấu `"` theo đúng quy ước viết tắt tiếng Việt
(`Khấu hao tài sản cố định ("TSCĐ")`). Đưa một nhãn như thế vào `MetricSelector.raw_text` — đúng thứ
ADR 0004 phương án C bảo planner phải làm — khiến `compile_plan` ném `SyntaxError` không ai bắt, tái
hiện 3/3 lần trên dữ liệu MBB/PNJ thật. Xem
[docs/plans/day19-sandbox-executor.md](docs/plans/day19-sandbox-executor.md) và
[ADR 0008](docs/decisions/0008-execution-sandbox-contract.md) (quyết định A2 escape đúng bằng
`json.dumps` / B2 `sandbox.py` là cổng duy nhất tới replay / C1 bắt `Exception` rộng đổi thành mã lỗi
có kiểu / D3 chặn theo cấu trúc, đo thời gian hậu nghiệm / E1 `compile_plan` tự validate / F1 hardening
DuckDB / G1 bốn mã lỗi mới).

### Kiến trúc: sandbox là cổng duy nhất tới replay, compiler tự gác biên

```
execution.compiler.compile_plan(plan, release_dir, execution_settings)
  ├─ plan_validator.validate_plan_semantics   # ADR 0008 E1: tự validate, không tin caller — chạy
  │                                            #   TRƯỚC render_pandas_query (nó giả định plan hợp lệ)
  ├─ cell_frame.build_cell_frame              # kiểm max_rows ngay sau khi build (row_limit_exceeded)
  ├─ locator / operations                     # không đổi so với Ngày 18
  └─ sandbox.replay_in_sandbox                # ADR 0008 B2/C1/D3: gọi pandas_query.replay_pandas_query
                                               #   bên trong, bắt MỌI Exception, đổi thành query_rejected;
                                               #   đo thời gian, budget_exceeded nếu vượt timeout_seconds
```

`pandas_query.py` tự thêm ba ngân sách cấu trúc trước khi diễn giải AST (độ dài chuỗi ≤ 4.096, số node
≤ 2.000, độ sâu ≤ 50) — chặn *trước khi chạy*, vì trên Windows (`win32`) **không tồn tại**
`signal.SIGALRM`/`signal.setitimer`/module `resource`, nên không thể ngắt một phép tính đang chạy từ
bên ngoài. `sandbox.py` chỉ đo thời gian *sau khi* replay xong và trả `budget_exceeded` nếu vượt
`timeout_seconds` — đây là phát hiện hậu nghiệm, không phải preemptive timeout, ghi rõ trong ADR để
không ai hiểu nhầm.

### Một hiệu chỉnh phát hiện giữa chừng bởi chính TDD

Kế hoạch ban đầu định tắt cả `enable_external_access` lẫn autoinstall/autoload extension cho mọi
connection DuckDB. Viết test TDD cho việc này (nhiệm vụ 19.7) lộ ra ngay: `enable_external_access=false`
chặn **toàn bộ** filesystem, kể cả `read_parquet` cục bộ mà `cell_frame` phụ thuộc — 8/8 test
`cell_frame` đỏ với `PermissionException`. Sửa lại: chỉ tắt autoinstall/autoload là đủ để chặn mạng
(thiếu extension `httpfs` thì không đọc được `http(s)://`), không cần và không được tắt
`enable_external_access`. Đây là ví dụ cụ thể cho lý do TDD bắt buộc ở dự án này — một quyết định
"hardening" tưởng vô hại suýt phá chức năng chính nếu không có test đỏ bắt được trước khi merge.

### Kết quả sau khi cài đặt xong

| Số đo | Giá trị |
|---|---:|
| Nhãn corpus thật có `"` compile ra `answered` đúng | có (trước đây `SyntaxError`) |
| gold70 resolved rate sau toàn bộ hardening | **30/51 (58,8 %)** — không đổi so với Ngày 18 |
| Phân rã lỗi gold70 | `metric_not_found` 11, `period_unresolved` 8, `cell_ambiguous` 2 — không đổi |
| Test bảo mật (`tests/security/`, 8 lớp payload) | 8/8 xanh |
| `ExecutionIssueCode` | 10 mã (6 cũ + `plan_rejected`, `query_rejected`, `budget_exceeded`, `row_limit_exceeded`) |
| `max_rows` cấu hình | 20.000 (hạ từ 100.000 — trần tuyệt đối toàn corpus đo được chỉ 4.002) |

`pytest -q`: 1.020 passed, 4 skipped (chỉ symlink/ACL Windows). `ruff check .` / `ruff format --check .`:
sạch trên 225 file. `mypy`: 0 lỗi (104 file `src`).

## Day 20 Answer Verifier và Citation

**Chẩn đoán trước khi viết code:** phát hiện lớn nhất không nằm trong bốn gạch đầu dòng của plan.md —
**bộ QA không có một đáp án số nào được gán nhãn** (`GoldRetrievalQuestion` không có trường "answer"),
nên cổng Ngày 21 ("answer accuracy ≥ 0,85") hiện không tính được. May là gán nhãn khả thi: 33/33 ô
evidence trên gold70 truy ngược được 100% tới đường dẫn tài liệu + số dòng nguồn. Đo thêm hai chốt
chặn: `expected_unit` NULL 30/30 plan **và** mâu thuẫn không thể thoả mãn (`_validate_growth_rate`
chỉ cho `percent`, `compile_growth_rate` hardcode trả `ratio`) — vô hình vì rule planner bỏ trống
trường này, nhưng prompt LLM Ngày 17 đang dạy model đặt `"percent"`; và 20,7% ô số toàn corpus không
có đơn vị, NULL biến thành chuỗi bịa `'nan'` lọt vào `CellMatch.unit`, quy oan thành
`unit_incompatible`. Xem [docs/plans/day20-answer-verifier-citation.md](docs/plans/day20-answer-verifier-citation.md)
và [ADR 0009](docs/decisions/0009-answer-package-contract.md) (quyết định A2 gán nhãn tay từ dòng
nguồn / B1 `ratio`↔`percent` là cùng đơn vị / C1 mã lỗi `unit_missing` riêng / D1 hai phép so dung
sai khác nhau / E1 `AnswerPackage` tự chứa / F1 template trước LLM tuỳ chọn / G1 package
`verification/` mới).

### Kiến trúc: builder gộp checks + templates thành AnswerPackage tự kiểm được

```
verification.builder.build_answer_package(plan, compiled, retrieved_table_ids, citation_lookup)
  ├─ templates.render_answer/render_sentence   # F1: template tiếng Việt, không cần LLM
  ├─ checks.check_recompute_mismatch           # tính lại từ evidence qua operations.py, so tuyệt đối
  ├─ checks.check_unit_not_presentable         # B1: bảng tương đương ratio<->percent
  ├─ checks.check_evidence_outside_retrieval   # E1: evidence phải ⊆ retrieved_table_ids
  ├─ checks.check_display_roundtrip_mismatch   # D1: dung sai = độ chính xác hiển thị đã khai báo
  └─ checks.check_period_inferred_warning      # cảnh báo, không chặn — 6/30 đáp án dựa kỳ suy diễn
```

`numeric_guard.py` là hàng rào riêng cho nhánh LLM diễn đạt tuỳ chọn: mọi token số trong văn bản sinh
ra phải nằm trong danh sách trắng {đáp án đã khoá, `plan.periods`/`top_k`, giá trị ô evidence} — token
lạ thì **từ chối** bản diễn đạt và rơi về template, không phải cảnh báo rồi vẫn dùng.

### Một bug thật bị bắt bởi chính vòng chạy end-to-end

Sau khi cài xong, chạy CLI `verify-answers` thật trên gold70 lần đầu cho ra **17/30 đáp án bị
`rejected` oan** với mã `display_roundtrip_mismatch`. Nguyên nhân: regex kiểm tra trong `checks.py`
chỉ đọc được **nhóm dấu phẩy đầu tiên** của số định dạng hàng nghìn — `"84,420,878 VND"` bị đọc thành
`84420` thay vì `84420878`. Viết test đỏ tái hiện đúng chuỗi thật lấy từ báo cáo gold70, sửa regex,
chạy lại: 0/30 bị ảnh hưởng. Đây đúng lý do dự án này bắt buộc TDD và chạy end-to-end thật trước khi
báo cáo xong việc, không chỉ tin vào test tổng hợp.

### Kết quả sau khi cài đặt xong

| Số đo | Giá trị |
|---|---:|
| `data/qa/answer-gold-v1.jsonl` | 30 nhãn gán tay từ dòng nguồn — lần đầu tồn tại |
| Đối chiếu tự động 51 ô evidence với dòng nguồn | 27/30 khớp tuyệt đối, 3/30 lệch ~2×10⁻³ VND (artifact float ingestion, đã ghi rõ) |
| gold70: answered → verified | **30/30** đều `verified`, 0 `rejected` |
| Accuracy so với answer-gold-v1 | **0,9** (27/30) |
| gold70 resolved rate | **30/51 (58,8 %)** — không đổi so với Ngày 19 |
| `ExecutionIssueCode` | 11 mã (10 cũ + `unit_missing`) |

`pytest -q`: 1.083 passed, 4 skipped (chỉ symlink/ACL Windows). `ruff check .` / `ruff format --check .`:
sạch trên 240 file. `mypy`: 0 lỗi (112 file `src`).

## Day 21 E2E và Review Cổng Tuần 3

**Chẩn đoán trước khi viết code:** "E2E" của Ngày 20 nạp thẳng `gold_table_ids` vào cả
`candidate_table_ids` lẫn `retrieved_table_ids` — retriever bị bỏ qua hoàn toàn. Thay bằng ranking
BM25 v4 thật (Ngày 14): answered 30 → 9/70, accuracy 0,900 → 0,667. Nguyên nhân không phải retrieval
kém — 6/6 ca đã soi có bảng gold nằm trong top-10 — mà là `statement_scope` (báo cáo riêng/hợp nhất)
chưa được `FinancialQueryPlan` mô hình hoá, dù trường đó đã có sẵn trong release khoá. Xem
[docs/plans/day21-e2e-week3-gate.md](docs/plans/day21-e2e-week3-gate.md) và
[ADR 0010](docs/decisions/0010-statement-scope-contract.md) (A1 `statement_scope` là trường plan /
B1 suy diễn có mặc định + `scope_inferred` chặn / C1 khử nhập nhằng đơn vị NULL / D1 nợ nhập nhằng
loại bảng, chưa sửa / E1 harness `pipeline/` retrieval trong vòng lặp / F1 mở rộng QA sửa lệch phân
bố / G1 báo cáo bảng biên 3 chính sách, không một con số).

### Kiến trúc: pipeline/ chạy retrieval thật trong vòng lặp, không còn tắt qua gold table

```
pipeline.evaluation.run_e2e_pipeline(questions, rankings, release_dir, execution_settings)
  với mỗi câu hỏi:
    retrieved = rankings.get(question_id, ())        # BM25 v4 đã lưu, KHÔNG dùng gold_table_ids
    plan = build_plan(entities, candidate_table_ids=retrieved)   # abstain được GHI LẠI, không nuốt
    compiled = compile_plan(plan, ...)                # scope_filter lọc theo statement_scope
    package = build_answer_package(...)                # scope_inferred chặn nếu suy diễn
  -> PipelineQuestionResult(stage=None nếu verified, else stage∈{retrieval,planning,normalization,execution,verification})
```

Quy tắc quy tầng (ADR 0010 E1): `cell_ambiguous` **không** tự động là lỗi retrieval — chỉ khi gold
⊄ retrieved. Đo được 22/24 ca `cell_ambiguous` dưới retrieval thật có gold nằm trong top-10 → quy về
`planning` (plan thiếu chiều scope), không phải `retrieval`. `execution/scope_filter.py` lọc
`candidate_table_ids` theo `plan.statement_scope` (nếu plan khai) hoặc
`ExecutionSettings.default_statement_scope` (nếu không, và đánh dấu `scope_inferred=True`).

### Một bug locator thật, đo được bán kính 859 ca

`locator.py` dùng `drop_duplicates(subset=["value","unit"])`, đếm `(X, None)` và `(X, "VND")` là hai
cặp phân biệt → báo `cell_ambiguous` ở nơi giá trị **giống hệt nhau** (ca OCB:
`2582236224358.0 None` vs `… VND`). Đo toàn corpus: 868 nhóm một-giá-trị bị báo nhập nhằng oan,
**859/868 (99,0 %)** do một ô thiếu đơn vị, chỉ 9/868 do hai đơn vị thật khác nhau. Sửa: khi tập ứng
viên rút về đúng một giá trị và đơn vị chỉ khác ở NULL-hay-không, lấy đơn vị đã biết — vẫn giữ
`cell_ambiguous` khi có ≥2 đơn vị thật khác nhau.

### Không chính sách scope nào "thắng" — cổng phải báo cáo bảng biên, không một số

Đo thật cả ba chính sách trên retrieval thật, 120 câu, đủ nhãn đáp án:

| Chính sách | Answered | Scored | Correct | Sai tự tin | Accuracy |
|---|---:|---:|---:|---:|---:|
| `none` (mặc định) | 39/120 | 39 | 33 | 6 (15,4 %) | **0,846** |
| `default_consolidated` | 71/120 | 53 | 40 | 13 (24,5 %) | 0,755 |
| `abstain_when_unstated` | 28/120 | 28 | 25 | 3 (10,7 %) | 0,893 |

`default_consolidated` gần gấp đôi answered nhưng **accuracy tệ đi** và tăng gấp đôi tỷ lệ sai tự
tin — đúng cái bẫy đo được ở kế hoạch § 1.5. Hệ thống giữ nguyên `none` làm mặc định sản xuất;
`scope_inferred` (mã lỗi verification mới, **chặn**, khác `period_inferred_warning`) đảm bảo một
câu trả lời suy diễn scope không bao giờ được trình bày như chắc chắn.

### Mở rộng bộ QA 70 → 120, sửa lệch phân bố scope

`retrieval-gold-v1.jsonl` thêm 50 câu (giữ nguyên 70 bản ghi gốc **byte-for-byte**, cùng 5 quy tắc
chống rò rỉ Ngày 13: chọn bảng theo metadata trước khi viết câu hỏi, không mở ranked list, evidence
từ `cells.parquet`, `stable_question_id`, sắp theo id). Tỷ lệ nêu scope: 22,9 % → 33,3 % (mục tiêu đo
được là 37,7 % của đề chính thức; không đạt hẳn vì 70 bản ghi gốc không được sửa lời). Retrieval thật
trên 120 câu: Recall@10 **0,9458**, F2@R **0,5085** — cả hai cao hơn mốc 70 câu (0,9143 / 0,4836).
`answer-gold-v1.jsonl` 30 → 58 nhãn: 28 câu mới `verified` được gán tay theo đúng phương pháp ADR
0009 A2 (đọc `cells.parquet` độc lập với executor, đối chiếu 4/28 trực tiếp với dòng nguồn
`_extracted.txt`) — 28/28 khớp ở lớp 1, 4/4 khớp ở lớp 2.

### Kết quả sau khi cài đặt xong

| Số đo | Giá trị |
|---|---:|
| `retrieval-gold-v1.jsonl` | 70 → **120 câu** |
| `answer-gold-v1.jsonl` | 30 → **58 nhãn** |
| Recall@10 / F2@R (120 câu, BM25 v4 thật) | 0,9458 / 0,5085 |
| Answer accuracy (chính sách `none`, retrieval thật) | **0,846** (chưa đạt 0,85, sát cổng) |
| Invalid plan (`plan_rejected`) | **0/120 = 0 %** — đạt (< 5 %) |
| Answered có nguồn đầy đủ | **100 %** — đạt, đúng cấu trúc `AnswerPackage` |
| Lỗi theo tầng | retrieval 6 (5 %), planning 60 (50 %), normalization 15 (12,5 %), verification 0 |
| Bug locator sửa (nhập nhằng oan do đơn vị NULL) | 859/868 ca |

Retrieval **không phải** lỗi chủ đạo (5 % số lỗi) — đúng nhánh xử lý plan.md: không fine-tune planner
bằng retrieval. Nợ kỹ thuật thật chuyển sang ngày sau: `multi_metric_unsupported`/`entity_ambiguous`
(19/120, giới hạn rule planner một-chỉ-tiêu) và nhập nhằng do loại bảng ngoài scope (ADR 0010 D1,
`statement_type` NULL 78,8 % nên chưa có bộ lọc rẻ).

`pytest -q`: 1.141 passed, 4 skipped (chỉ symlink/ACL Windows). `ruff check .` / `ruff format --check .`:
sạch trên 250 file. `mypy`: 0 lỗi (117 file `src`).

## Ngày 22 (thực hiện sớm) — Submission Export

Người dùng cung cấp bộ câu hỏi thi thật (`data/raw/ViFinQA/questions/questions.jsonl`, 1.012 câu,
không nhãn) và yêu cầu ưu tiên trước giao diện Streamlit — làm trước Ngày 22/23 gốc trong `plan.md`,
trùng phạm vi Ngày 24 (Task G). Kế hoạch: [docs/plans/day22-submission-export.md](docs/plans/day22-submission-export.md).

### Kiến trúc: retrieval trực tiếp là mảnh ghép còn thiếu, không phải submission

Mọi harness Ngày 8-21 đo BM25 trên câu hỏi **đã có nhãn** (`GoldRetrievalQuestion.filters`) — chưa
từng có đường "câu hỏi thô → candidate table_ids". `to_retrieval_filters` (Ngày 10) đã làm đúng việc
chiếu `QueryEntities` sang `RetrievalFilters`, chỉ chưa ai nối nó với `RetrievalService.retrieve`.
`retrieval/live_query.py` là dây nối 4 dòng đó — không có logic truy hồi mới, chỉ ghép ba hàm đã kiểm
chứng. `execution/contracts.py` thêm `CompiledQuery.replay_rows` để exporter đọc đúng DataFrame
compiler đã dùng (biến `df1` trong `pandas_query`) thay vì dựng lại logic `_dispatch`.

### Một bug thật chỉ lộ ra trên toàn bộ corpus, không phải trên gold70/gold120

5,2 % bảng thật (7.643/146.011, đo trực tiếp trên `tables.parquet`) có `title_raw = NULL`.
`Citation.table_title` (Ngày 20) bắt buộc chuỗi non-empty — sập ngay câu hỏi thật đầu tiên chạm một
bảng loại này. gold70/gold120 (mẫu nhỏ, có chủ đích) chưa từng chạm ngẫu nhiên trường hợp này. Sửa
bằng TDD: nới `table_title: NonEmptyString | None`.

### Kết quả chạy thật trên toàn bộ 1.012 câu

| Số đo | Giá trị |
|---|---:|
| Câu trả lời được (`submission.json`) | **32/1.012 = 3,16 %** |
| Lỗi theo tầng | retrieval (không candidate): 43, planning (abstain): 623, execution: 314 |
| Validator offline (`submission validate`) | `valid=True items=32` — mọi `pandas_query` replay khớp `answer` qua sandbox thật |
| ZIP | `submissions/submission_422df141c935.zip`, SHA-256 lưu ngoài ZIP |

Hạ tầng export/validate/đóng gói ZIP đã đúng và đã kiểm chứng trên dữ liệu thật (không phải fixture
tổng hợp). Nhưng **đây chưa phải bản nộp Dashboard sẵn sàng**: contract §2.4 quy tắc 4 đòi hỏi phủ
đúng và đủ 100 % câu hỏi thi, còn coverage thật hiện tại chỉ 3,16 % — phần lớn do rule planner Ngày
16 (bảo thủ có chủ đích, "false-plan rate = 0" quan trọng hơn coverage) abstain trên câu hỏi tự
nhiên đa dạng hơn nhiều so với gold70/120 (được sinh có kiểm soát). Cải thiện coverage — LLM planner
(Ngày 17) đã có nhưng chưa bật trong luồng submission, mở rộng ontology, giảm rule planner abstain —
là việc của các ngày sau, không phải nợ ẩn của module này.

`pytest -q`: 1.186 passed, 4 skipped. `ruff check .` / `ruff format --check .`: sạch trên 263 file.
`mypy`: 0 lỗi (123 file `src`). Security: 8 test riêng cho ZIP (ZIP Slip, absolute/drive path,
backslash, symlink metadata, duplicate entry, entry ngoài `data/`, JSON root thừa).

### Theo dõi cùng ngày: nối LLM planner (Ngày 17) vào submission — đo thật, kết quả âm tính

Yêu cầu cải thiện coverage bằng LLM fallback. Máy không có server đúng spec dự án (llama.cpp), nhưng
có sẵn **Ollama đang chạy** (`127.0.0.1:11434/v1`, model `qwen2.5:3b` đã pull). Nối
`submission/exporter.py` gọi `planning.plan_router.route_plan` (rule planner luôn chạy trước, LLM chỉ
chạy khi rule abstain, không đổi hành vi khi rule thành công — đúng ADR 0006 A1) thay vì gọi thẳng
`rule_planner.build_plan`; CLI `submission export` thêm `--llm-config` tuỳ chọn (bỏ qua giữ nguyên
hành vi cũ). TDD bằng `httpx.MockTransport`, `pytest -q`: 1.191 passed.

**Đo thật hai lần** (mẫu 30 câu, rồi toàn bộ 1.012 câu, `qwen2.5:3b` qua Ollama): **0 câu trả lời
thêm cả hai lần** — SHA-256 ZIP giống hệt bản rule-only, vẫn 32/1.012. LLM được gọi cho toàn bộ 623
câu rule abstain: 458 (73,5 %) sinh plan JSON hợp lệ cú pháp nhưng sai schema
(`llm_plan_invalid`), 146 (23,4 %) tự bịa tên chỉ tiêu canonical không có thật (`metric_not_found`),
12 JSON không hợp lệ, 7 lỗi khác. Nguyên nhân: ADR 0006 quyết định B1 cố tình không đưa 56 tên chỉ
tiêu chuẩn vào prompt (tiết kiệm token, xem [docs/plans/day22-submission-export.md](docs/plans/day22-submission-export.md))
— giả định hợp lý với model đúng spec (`qwen3-4b-instruct`) nhưng `qwen2.5:3b` không tuân thủ đủ tốt
để rơi về `raw_text` khi không chắc, thay vào đó tự bịa. Hạ tầng LLM-fallback đúng và an toàn (không
làm mất 32 câu rule đã có), nhưng cải thiện coverage thật cần model mạnh hơn hoặc đưa vocabulary chỉ
tiêu vào prompt — việc của bước sau, không phải bug ở module submission này.

### Theo dõi tiếp: mở rộng company alias từ `code_stock.csv` — sửa được 1 bug thật, coverage thô giảm nhưng đúng đắn hơn

Người dùng chỉ ra `data/raw/ViFinQA/code_stock.csv` là nguồn duy nhất cho tên 100 công ty. Đối
chiếu trực tiếp: khớp 100/100 mã với `company_registry.csv` nội bộ, chỉ lệch tên STB
(`code_stock.csv` ghi nhầm "Sài Gòn Tài Lộc" — giữ làm alias đúng theo nguồn sinh câu hỏi, không
"sửa lại cho đúng"). Đào 34 câu `company_missing` thật: nguyên nhân là khớp tên công ty chỉ nhận
cụm ≥3 từ giống *hệt* tên canonical đầy đủ, không nhận tên ngắn câu hỏi thật hay dùng ("Đô thị Kinh
Bắc" thay vì "Tổng Công ty Phát triển Đô thị Kinh Bắc - CTCP"). Thêm 21 alias ≥3 từ an toàn (giữ
nguyên ngưỡng 3-từ chống nhận nhầm), TDD 9 câu thật + 2 test hồi quy.

Trong lúc đó phát hiện một **bug thật** trong `companies.py::_contained_company_codes`: hàm chỉ giữ
lại công ty có alias dài nhất *toàn câu*, âm thầm bỏ công ty thứ hai khi câu hỏi so sánh ≥2 công ty
bằng tên (không phải mã). Sửa bằng longest-match theo từng vị trí văn bản, không đè lên nhau — vẫn
giữ nguyên tắc chống nhầm công ty con/mẹ (vd. HNG lồng trong tên HAG).

Chạy lại thật trên 1.012 câu: **answered 32 → 28** — nhìn thô là giảm, nhưng lý do là **sửa đúng**:
4 câu "mất" (id 734, 745, 777, 964) trước đó âm thầm trả lời bằng `pandas_query` chỉ tra **một**
công ty cho câu hỏi hỏi **chênh lệch giữa 2 công ty** — trả lời sai câu hỏi thật một cách tự tin,
đúng do bug vừa sửa. Sau khi sửa, cả 4 câu đúng đắn abstain (`metric_not_found`, không tìm được chỉ
tiêu khi tra đúng cả 2 công ty) thay vì tiếp tục trả lời sai. Riêng 34 câu mục tiêu: 10/34 tiến qua
được bước nhận diện công ty (chuyển sang `metric_not_found`/`multi_metric_unsupported` — bị chặn ở
bước sau); 24/34 còn lại là câu hỏi nhiều bước/điều kiện (vd. "biên lợi nhuận gộp của năm ngay sau
năm đầu tiên ghi nhận CFO âm") vượt quá 9 operation hiện có — nợ kỹ thuật khác, không phải lỗi alias.

`pytest -q`: 1.202 passed, 4 skipped. `ruff check .` / `ruff format --check .`: sạch. `mypy`: 0 lỗi.
ZIP mới `submissions/submission_422df141c935.zip`, SHA-256 `aa98c918d...`, validator
`valid=True items=28`.
