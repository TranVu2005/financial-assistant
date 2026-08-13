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
