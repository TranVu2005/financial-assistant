# Lớp xuất bảng chuẩn hoá — CSV / metadata / text đồng bộ hoá

- **Ngày:** 2026-08-23
- **Trạng thái:** Đề xuất, chờ duyệt

---

## 1. Bối cảnh

Pipeline ingestion/normalization hiện tại (`src/financial_report_qa/ingestion/`,
`src/financial_report_qa/normalization/`) đã xử lý đầy đủ:

- Loại bỏ thẻ HTML + chuẩn hoá whitespace/entity (`html.parser.HTMLParser` với
  `convert_charrefs=True` trong `ingestion/table_extractor.py`).
- Mở rộng ô gộp rowspan/colspan thành lưới 2 chiều đầy đủ (`_materialize_table`).
- Căn chỉnh số cột đồng đều mọi hàng (grid-based, không có hàng thiếu cột).
- Lan truyền tiêu đề nhóm (section banner) làm ngữ cảnh phân cấp cho từng dòng
  con qua `row_group_context_raw` (`_row_group_context_map`).
- Chuẩn hoá số, đơn vị, kỳ, công ty, loại báo cáo (`normalization/*.py`).

Toàn bộ 146.011 bảng đã được build thành release Parquet bất biến tại
`data/processed/release_v2_<fingerprint>/{documents,tables,cells,placements}.parquet`.

**Khoảng trống hiện tại** không nằm ở ingestion/normalization mà ở **lớp xuất
dữ liệu**: chưa có nơi nào sinh ra (a) CSV với header đa cấp được "làm phẳng"
thành đường dẫn gạch dưới, (b) nhãn dòng có tiền tố ngữ cảnh nhóm hiển thị
trực tiếp trong ô, (c) tên file theo tên tài liệu nguồn, (d) metadata sidecar
6 trường (Company/Year/Report Type/Statement/TableID/Unit), (e) bản sao văn
bản gốc với khối bảng được thay bằng liên kết đồng bộ tới CSV tương ứng.

`data/interim/table_csv_export/` hiện có (`data/table_frame.py::export_table_csvs`)
chỉ xuất **lưới thô** (`value_raw`, không header, tên file theo hash `table_id`)
— phục vụ mục đích khác (audit thủ công), không đáp ứng yêu cầu này. Thiết kế
này **không sửa** `export_table_csvs` hiện có, mà bổ sung một lớp xuất mới.

### Giới hạn phạm vi (quan trọng)

Retrieval hiện tại build văn bản index trực tiếp từ Parquet
(`retrieval/row_documents.py`, `retrieval/documents.py`), **độc lập** với CSV
export. Bước này **không tự động cải thiện điểm retrieval** — nó tạo ra đúng
sản phẩm CSV/metadata/text theo yêu cầu. Việc cho retrieval tiêu thụ
`manifest.jsonl` mới (Statement/Report Type/Unit) làm feature bổ sung là một
đề xuất kế tiếp, ngoài phạm vi bản thiết kế này.

---

## 2. Kiến trúc

Gói mới: `src/financial_report_qa/export/`

```
export/
  __init__.py
  csv_export.py      # flatten header, build normalized table, write CSVs + manifest
  synced_text.py      # rewrite source TXT with table blocks replaced by CSV links
  cli.py               # `financial-report-qa export-tables` subcommand
```

Đầu vào: một release Parquet đã build (`documents.parquet`, `tables.parquet`,
`cells.parquet`, `placements.parquet`) + `snapshot_root` (để đọc lại TXT gốc
cho bước synced-text). Không đọc lại HTML/OCR thô, không chạy lại
detection/extraction.

### 2.1 `csv_export.py`

```python
def flatten_header(levels: list[str]) -> str:
    """Nối các cấp header đã khử trùng lặp liên tiếp bằng '_'; khoảng trắng
    trong mỗi cấp cũng thành '_'. VD: ["Tổng cộng", "31/12/2022"]
    -> "Tổng_cộng_31/12/2022"."""

def build_normalized_table(
    cells: list[CellRow], placements: list[PlacementRow], header_rows: int
) -> NormalizedTable:
    """Dựng bảng đã chuẩn hoá cho một table_id:
    - Dòng 0..N cột: header đã flatten (dùng column_label_raw theo thứ tự cấp
      gốc, dedup theo (col_idx, giá trị)).
    - Cột đầu mỗi dòng dữ liệu: "{row_group_context_raw} > {row_label_raw}"
      nếu có group context, ngược lại chỉ "{row_label_raw}".
    - Ô giá trị: value_numeric (dạng chuỗi số Decimal, '.' thập phân) nếu có,
      ngược lại value_raw.strip(); ô trống -> "".
    - Dedup: một (row_idx, col_idx) chỉ xuất hiện đúng 1 lần trong output dù
      cell gốc trải rowspan/colspan nhiều vị trí lưới (đã đảm bảo bởi
      `placements`, không cần xử lý thêm — mỗi placement trỏ về đúng 1 cell_id,
      không có logic ghi đè)."""

@dataclass(frozen=True)
class TableExportMetadata:
    table_id: str
    company: str          # documents.company_code
    year: int             # documents.report_year
    report_type: str       # documents.statement_scope: consolidated/separate/aggregated/other
    statement: str | None  # tables.statement_type: balance_sheet/income_statement/...
    unit: str | None       # tables.unit_normalized (fallback unit_raw)
    csv_path: str          # relative path, POSIX

def export_normalized_csvs(
    release_dir: Path, output_dir: Path
) -> CsvExportManifest:
    """Với mỗi document (nhóm theo doc_id), sắp bảng theo line_start tăng dần,
    đánh số N=1.. ; tên file: f"{doc_base_name}__table_{N}.csv".
    doc_base_name = tên thư mục cha ngay trên tên file trong relative_path
    (VD relative_path=".../AAA_financial_statements_2015_consolidated/..._extracted.txt"
    -> "AAA_financial_statements_2015_consolidated").
    Ghi CSV UTF-8-sig (Excel/pandas mở đúng dấu tiếng Việt).
    Ghi kèm data/interim/<output_dir>/manifest.jsonl: 1 dòng JSON/bảng gồm
    toàn bộ TableExportMetadata ở trên."""
```

Xử lý dedup tên file: nếu 2 bảng cùng `line_start` (không xảy ra theo ràng
buộc `TableRecord`, các bảng trong 1 doc có `line_start` khác nhau) — không
cần xử lý thêm; nếu phát sinh trùng, raise lỗi thay vì âm thầm ghi đè (an
toàn hơn im lặng mất dữ liệu).

### 2.2 `synced_text.py`

```python
def build_synced_text(
    snapshot_root: Path, document: DocumentRecord,
    tables: list[TableExportEntry],  # (line_start, line_end, table_id, csv_relpath)
) -> str:
    """Đọc lại TXT gốc qua `ingestion.txt_reader.read_document` (đã verify
    byte-level sẵn có). Với mỗi bảng, thay nguyên văn các dòng
    [line_start, line_end] (1-based, đúng theo TableRecord đã lưu — đây là
    span đã gộp qua trang nếu bảng bị merge continued_across_page) bằng một
    dòng duy nhất:
        [TABLE: {table_id} -> {csv_relpath}]
    Giữ nguyên tất cả text/page-marker khác. Ghép lại theo thứ tự dòng gốc,
    thay thế theo line_end giảm dần để không lệch offset."""

def export_synced_text(
    release_dir: Path, snapshot_root: Path, csv_manifest: CsvExportManifest,
    output_dir: Path,  # mặc định data/interim/synced_text/
) -> SyncedTextManifest:
    """Với mỗi document có >=1 bảng đã xuất CSV, ghi
    output_dir / relative_path (mirror cấu trúc thư mục snapshot gốc)."""
```

`data/interim/synced_text/` là artifact có thể build lại bất kỳ lúc nào từ
release + snapshot, không sửa `data/raw` (append-only, giữ nguyên).

### 2.3 CLI

Thêm subcommand theo đúng mẫu `submission`/`retrieval` hiện có trong
`src/financial_report_qa/cli.py`:

```
financial-report-qa export-tables \
  --release-dir data/processed/release_v2_<fingerprint> \
  --snapshot-root data/raw/ViFinQA/financial_statements \
  --csv-output-dir data/interim/normalized_table_csv_export \
  --text-output-dir data/interim/synced_text
```

`export/cli.py::main` gọi tuần tự `export_normalized_csvs` rồi
`export_synced_text`, in ra table_count/document_count, trả exit code 0/1.

---

## 3. Cấu trúc dữ liệu đầu ra

### 3.1 CSV (`{doc_base_name}__table_{N}.csv`)

```
Chỉ_tiêu,Thuyết_minh,Số_cuối_năm_31/12/2022,Số_đầu_năm_31/12/2021
Tiền và các khoản tương đương tiền,V.1,100000,80000
TÀI SẢN NGẮN HẠN KHÁC > Chi phí trả trước ngắn hạn,V.8,15000,12000
```

(Ví dụ minh hoạ — cột đầu chỉ mang tiền tố nhóm khi `row_group_context_raw`
khác None.)

### 3.2 `manifest.jsonl` (1 dòng/bảng)

```json
{"table_id": "tbl_...", "company": "VCB", "year": 2023, "report_type": "consolidated", "statement": "balance_sheet", "unit": "million_vnd", "csv_path": "VCB_financial_statements_2023_consolidated__table_17.csv"}
```

### 3.3 Text đồng bộ hoá (trích đoạn)

```
=== PAGE 4 ===
Bảng cân đối kế toán hợp nhất
[TABLE: tbl_ab12...ef -> data/interim/normalized_table_csv_export/VCB_financial_statements_2023_consolidated__table_17.csv]
Thuyết minh V.1 nêu chi tiết...
```

---

## 4. Xử lý lỗi & bất biến

- Không có `table_id` nào trong release bị bỏ sót khỏi `manifest.jsonl`
  (đối chiếu `table_count` trong `release manifest.json`, giống cách
  `write_table_documents` đối chiếu hiện tại).
- `company`/`year`/`report_type` luôn có giá trị (bắt buộc trên
  `DocumentRecord`); `statement`/`unit` có thể `None` — giữ `null` trong JSON,
  không giả định giá trị.
- Nếu `snapshot_root` thiếu file nguồn cho một document đã có trong release
  → lỗi cứng (`DatasetBuildError`-style), không bỏ qua âm thầm (nhất quán với
  `dataset_builder.py` hiện tại).
- CSV ghi qua file tạm cùng thư mục rồi `replace()` (nhất quán với
  `row_documents.py::write_table_documents`), tránh file dở dang khi crash.

## 5. Kiểm thử

- `tests/unit/export/test_csv_export.py`:
  - `flatten_header`: 1 cấp, nhiều cấp trùng lặp cần dedup, cấp có khoảng
    trắng/dấu `/`.
  - `build_normalized_table`: bảng có rowspan (nhãn dòng lặp qua nhiều hàng
    lưới), colspan (header lặp qua nhiều cột), group context 2 cấp lồng
    nhau, bảng không có group context nào (tiền tố rỗng).
  - Tên file: 1 document nhiều bảng → số thứ tự đúng theo `line_start`; ký tự
    đặc biệt trong `doc_base_name` (nếu có) không phá vỡ path an toàn.
- `tests/unit/export/test_synced_text.py`:
  - Thay đúng span line_start/line_end bằng 1 dòng link, giữ nguyên phần còn
    lại byte-for-byte (so với `read_document` gốc trừ đúng đoạn thay).
  - Nhiều bảng trong 1 document, thay đúng thứ tự (giảm dần line_end) không
    lệch offset.
- `tests/integration/export/test_export_tables_cli.py`: chạy CLI trên 1
  release nhỏ dựng từ fixture (theo mẫu `tests/integration` hiện có cho
  `dataset_builder`), xác nhận CSV + manifest + synced text nhất quán với
  nhau (mọi `csv_path` trong manifest tồn tại trên đĩa, mọi link trong synced
  text trỏ đúng file đó).

## 6. Ngoài phạm vi (không làm trong plan này)

- Không sửa `ingestion/`, `normalization/` — dữ liệu nguồn giữ nguyên.
- Không sửa `retrieval/row_documents.py` hay index BM25/dense hiện có.
- Không xoá/sửa `data/interim/table_csv_export/` (raw-grid export) hiện có.
- Không tự động tích hợp `manifest.jsonl` mới vào retrieval — đây là việc
  tiếp theo, cần spec riêng nếu muốn theo đuổi.
