# Môi trường phát triển

## Môi trường chuẩn

- Windows 11 làm host.
- WSL2/Linux làm shell và runtime phát triển chuẩn.
- Python 3.11 và dependency do `uv` quản lý.
- GitHub Actions chạy Linux và dùng cùng lockfile/lệnh kiểm tra với local.

## Thiết lập từ checkout mới

```bash
cd /mnt/d/GitHub/financial-assistant
make setup
cp .env.example .env
make check
make build
```

Nếu WSL chưa có `make`, cài bằng package manager của distro, ví dụ Ubuntu:

```bash
sudo apt-get update
sudo apt-get install --yes make
```

## Lệnh thường dùng

| Lệnh | Công dụng |
|---|---|
| `make setup` | Đồng bộ dependency dev và cài pre-commit hook |
| `make lint` | Chạy Ruff lint |
| `make format` | Định dạng Python bằng Ruff |
| `make format-check` | Kiểm tra format mà không sửa file |
| `make typecheck` | Chạy mypy strict trên `src` và `tests` |
| `make test` | Chạy toàn bộ pytest |
| `make check` | Chạy lint, format check, mypy và pytest |
| `make build` | Tạo wheel và source distribution |
| `make download-data` | Dry-run dataset; không tải dữ liệu |

Tải thật chỉ khi thêm cờ rõ ràng:

```bash
uv run --frozen --no-sync financial-report-qa download-data --reserve-gb 100 --download
```

## Quy trình thay đổi code

1. Tạo branch từ nhánh ổn định.
2. Viết test thất bại cho hành vi mới hoặc lỗi cần sửa.
3. Cài phần nhỏ nhất để test qua.
4. Chạy `make check`.
5. Cập nhật tài liệu khi interface hoặc vận hành thay đổi.
6. Mở pull request; chỉ merge khi CI qua và review hoàn tất.

## Configuration và secret

`configs/` chứa cấu hình công khai, có version. `.env.example` chỉ chứa tên biến và giá trị local
không nhạy cảm. API key/token thật nằm trong `.env` hoặc secret store của CI và không được commit.

Các setting nền tảng hiện tại:

```text
APP_ENV=local|test|staging
DATA_ROOT=data
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL
```

## Docker

Docker dùng để kiểm tra khả năng build/chạy độc lập với Python global của máy phát triển:

```bash
docker build -f docker/Dockerfile -t financial-report-qa:local .
docker run --rm financial-report-qa:local --help
```

Không mount dataset hoặc chạy model trong image nền tảng này.
