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
uv run financial-report-qa --help
uv run financial-report-qa download-data --help
```

## Tải dataset

Kiểm tra chính xác số file và dung lượng trước; lệnh này không tải dữ liệu:

```bash
make download-data
```

Sau khi kiểm tra dung lượng, tải hoặc tiếp tục snapshot đầy đủ:

```bash
uv run financial-report-qa download-data \
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
