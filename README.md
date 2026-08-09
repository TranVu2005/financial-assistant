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
