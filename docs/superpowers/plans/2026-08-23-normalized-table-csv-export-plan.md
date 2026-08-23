# Kế hoạch thực thi: Lớp xuất bảng chuẩn hoá — CSV / metadata / text đồng bộ

- **Spec (thẩm quyền cao nhất):** `docs/superpowers/specs/2026-08-23-normalized-table-csv-export-design.md`
- **Nhánh:** `feat/normalized-table-csv-export`
- **Ngày:** 2026-08-23

Mọi yêu cầu dưới đây bổ sung chi tiết thi hành cho spec; khi mâu thuẫn, spec
thắng. Các điểm spec chưa chốt đã được điều phối viên quyết và ghi thành quy
chuẚc trong chính kế hoạch này (mục "Quyết định đã chốt").

---

## Bối cảnh kỹ thuật (sự kiện mã nguồn đã kiểm chứng)

- Release Parquet bất biến: `data/processed/release_v2_<fp>/{documents,tables,cells,placements}.parquet`
  + `manifest.json` (có `table_count`). Schema pyarrow:
  `src/financial_report_qa/data/dataset_builder.py` (`DOCUMENT_SCHEMA`,
  `TABLE_SCHEMA`, `CELL_SCHEMA`, `PLACEMENT_SCHEMA`) — import được.
  - documents: doc_id, relative_path, company_code, report_year,
    statement_scope, sha256, file_size_bytes, encoding, inventory_status, …
  - tables: table_id, doc_id, source_ordinal, statement_type, unit_raw,
    unit_normalized, line_start, line_end, row_count, column_count, …
  - cells: cell_id, table_id, row_idx, col_idx, value_raw, value_numeric
    (decimal128(38,10)), row_label_raw, row_group_context_raw,
    column_label_raw, …
  - placements: table_id, row_idx, col_idx, cell_id.
- Mô hình ô sinh bởi `_materialize_table` (`ingestion/table_extractor.py`):
  - Ô **header** (hàng lưới `< header_rows`): `column_label_raw = None`,
    `row_label_raw = None`, `row_group_context_raw = None`, văn bản header
    nằm trong `value_raw`. Cờ `is_header` gốc của HTML **không** được lưu
    vào release; `header_rows` cũng **không** lưu trong `tables.parquet`.
  - Ô **dữ liệu**: `column_label_raw` = toàn bộ đường dẫn cấp cột nối `\n`
    (đã dedup theo giá trị); `row_label_raw` = text của ô cột metric cùng
    hàng; `row_group_context_raw` = đường dẫn nhóm đã nối sẵn bằng `" > "`
    (deepest last), `None` nếu không có nhóm.
- Đọc TXT gốc đã verify byte-level: `ingestion/txt_reader.py::read_document(root, document)`
  → `DecodedDocument` có `.lines` (`SourceLine(number, text, line_ending)`);
  dựng lại nguyên văn = `"".join(l.text + l.line_ending for l in lines)`.
  `read_document` yêu cầu `inventory_status == "ready"` và chấp nhận fixture
  mock khi `sha256 == "a"*64`; `encoding` phải là `"utf-8"`/`"utf-8-sig"`
  (fixture phải set đủ `file_size_bytes`, `sha256`, `encoding`).
- Ghi file nguyên tử (mẫu bắt buộc theo đúng `retrieval/documents.py::write_table_documents`):
  `tempfile.mkstemp(dir=<thư mục đích>, prefix=f".{tên}.", suffix=".tmp")`
  → ghi → flush/fsync → `os.replace()` → `unlink(missing_ok=True)` trong except.
- Đối chiếu count với release manifest (mẫu `write_table_documents`): đọc
  `release_dir/manifest.json`, so `table_count` (phải là `int`) với số bảng
  đã xuất; lệch → raise.
- Đọc parquet bằng DuckDB in-memory (mẫu `retrieval/row_documents.py`,
  `data/table_frame.py`). Manifest JSONL dùng `orjson` với
  `OPT_SORT_KEYS | OPT_APPEND_NEWLINE` (UTF-8 không escape — đúng yêu cầu
  dấu tiếng Việt).
- CLI dispatcher: `src/financial_report_qa/cli.py` đăng ký subparser
  `add_parser(<tên>, add_help=False, help=...)` rồi lazy-import module con
  trong nhánh dispatch. Test mẫu: `tests/unit/test_cli.py`.
- Fixture release nhỏ có sẵn: fixture `release_dir` trong
  `tests/unit/conftest.py` (dựng parquet bằng `pa.Table.from_pylist` +
  schema trên). Integration test mới làm theo đúng cách đó.
- Tooling: `.venv/Scripts/python.exe -m pytest`, ruff (line-length 100),
  mypy **strict** trên cả src và tests. Lệnh chạy:
  `.venv/Scripts/python.exe -m ruff check src tests && .venv/Scripts/python.exe -m mypy`
  và `.venv/Scripts/python.exe -m pytest ...`.

## Quyết định đã chốt (điền vào chỗ trống của spec)

1. **`header_rows` không có trong release** → export tự suy ra bằng hàm mới
   `detect_header_row_count(cells, placements)`: quét các hàng lưới liên tiếp
   tính từ 0; một hàng được tính là header khi và chỉ khi hàng đó có ≥1 ô và
   **mọi** ô đều có `column_label_raw is None` **và** `row_label_raw is None`
   **và** không phải tất cả `value_raw` đều rỗng (toàn blank → dừng, không
   tính). Dừng ở hàng đầu tiên không thoả. (Đã kiểm chứng phân bố hợp lý
   trên 300 bảng thật: chủ yếu 1–2 dòng header.)
2. **Lỗi domain:** thêm `ExportError(FinancialReportQAError)` vào
   `core/errors.py` (chỉ thêm class mới, không sửa class cũ). Mọi lỗi nghiệp
   vụ của lớp export (count lệch, tên file không an toàn, trùng line_start,
   span overlap, release thiếu manifest…) đều raise `ExportError`.
3. **manifest.jsonl nằm ngay trong `--csv-output-dir`** (`output_dir/manifest.jsonl`).
4. **`csv_path` trong metadata = TÊN FILE** (POSIX, không thư mục) — khớp ví dụ
   §3.2 của spec. Link trong synced text = `(output_dir / tên_file).as_posix()`
   theo đúng đối số người dùng truyền (khớp ví dụ §3.3).
5. **Định dạng số:** `value_numeric` (Decimal từ decimal128(38,10)) được xuất
   dạng `format(value.normalize(), "f")` → `"100000"`, `"-12.5"`; không có
   chữ số 0 thừa, dấu `.` thập phân.
6. **Cột đầu mỗi dòng dữ liệu** luôn nhận nhãn tổ hợp
   `{row_group_context_raw} > {row_label_raw}` (bỏ phần `None`); các cột còn
   lại phát giá trị ô tại đúng vị trí lưới; vị trí lưới thiếu ô → chuỗi rỗng.
7. **Dòng banner nhóm** (chỉ có text ở cột metric) vẫn xuất như một dòng dữ
   liệu bình thường — không loại bỏ.

## Ràng buộc toàn cục (Global Constraints)

1. KHÔNG sửa `ingestion/`, `normalization/`, `retrieval/`, hay nội dung hiện
   có của `data/table_frame.py` (`export_table_csvs` giữ nguyên). Chỉ THÊM
   `src/financial_report_qa/export/`, thêm class lỗi mới, wire `cli.py`, và
   thêm tests.
2. `data/raw/**` read-only; không xoá/sửa `data/interim/table_csv_export/`.
3. CSV encode UTF-8-sig; mọi file ghi qua temp sibling + `os.replace`.
4. `manifest.jsonl`: mỗi dòng là JSON của đúng 7 trường
   `TableExportMetadata` (`table_id`, `company`, `year`, `report_type`,
   `statement`, `unit`, `csv_path`); `statement`/`unit` có thể `null`;
   UTF-8 thật (không `\uXXXX` escape).
5. Không bỏ sót `table_id`: đối chiếu với `manifest.json` của release.
6. Thiếu file nguồn khi build synced text → fail cứng (không skip im lặng).
7. Mọi đường dẫn ghi vào metadata/link là POSIX relative hoặc tên file an toàn;
   `doc_base_name` phải là một segment POSIX an toàn (không rỗng, không
   `/`, `\`, không `.`/`..`) — vi phạm → `ExportError`.
8. Type hints đầy đủ, mypy strict sạch, ruff sạch, pytest output pristine.
9. Commit theo conventional commits (`feat(export): …`, `test(export): …`).

---

## Task 1 — `export/csv_export.py` + unit test

**Files tạo/sửa:**
- `src/financial_report_qa/export/__init__.py` (export public API)
- `src/financial_report_qa/export/csv_export.py`
- `src/financial_report_qa/core/errors.py` (thêm `ExportError`)
- `tests/unit/export/test_csv_export.py`

**Yêu cầu:**

1. `flatten_header(levels: list[str]) -> str`:
   `"__".join?` — KHÔNG. Công thức: bỏ level rỗng/whitespace-only; mỗi level
   còn lại nén whitespace run thành `_` (gợi ý: `"_".join(level.split())`);
   nối các level bằng `_`. Ví dụ spec: `["Tổng cộng", "31/12/2022"]` →
   `"Tổng_cộng_31/12/2022"`; `[]` → `""`.
2. Dataclass local (frozen) phản ánh hàng parquet tối thiểu:
   - `CellRow`: cell_id, table_id, row_idx, col_idx, value_raw: str,
     value_numeric: Decimal | None, row_label_raw: str | None,
     row_group_context_raw: str | None, column_label_raw: str | None
   - `PlacementRow`: table_id, row_idx, col_idx, cell_id
3. `detect_header_row_count(cells, placements) -> int` theo Quyết định 1.
4. `NormalizedTable` frozen dataclass: `headers: tuple[str, ...]`,
   `rows: tuple[tuple[str, ...], ...]`.
5. `build_normalized_table(cells, placements, header_rows) -> NormalizedTable`:
   - `column_count` = max(col_idx)+1; `row_count` = max(row_idx)+1 (0 nếu rỗng).
   - Header cột c: lấy `value_raw` của các ô được đặt tại (r, c) với
     r < header_rows theo thứ tự r tăng dần, gộp các giá trị liền kề trùng
     nhau (dedup consecutive), rồi `flatten_header`.
   - Dòng dữ liệu r ∈ [header_rows, row_count): ô đầu tiên (min col_idx) cung
     cấp `row_label_raw` + `row_group_context_raw` cho nhãn tổ hợp (Quyết định
     6); cột 0 = nhãn tổ hợp; cột c>0 = giá trị ô tại (r,c): nếu
     `value_numeric` khác None → chuỗi theo Quyết định 5; ngược lại
     `value_raw.strip()`; rỗng → `""`; thiếu placement → `""`.
6. `TableExportMetadata` frozen dataclass đúng 7 trường như spec §2.1
   (`csv_path` = tên file, xem Quyết định 4).
7. `CsvExportManifest` frozen dataclass: `output_dir: Path`,
   `manifest_path: Path`, `table_count: int`,
   `entries: tuple[TableExportMetadata, ...]`.
8. `export_normalized_csvs(release_dir: Path, output_dir: Path) -> CsvExportManifest`:
   - Một truy vấn DuckDB join documents×tables×placements×cells, ORDER BY
     (relative_path, line_start, source_ordinal, row_idx, col_idx); group theo
     doc trong Python (mẫu `iter_table_frames`/`row_documents.build_row_documents`).
   - Mỗi doc: sắp bảng theo (line_start, source_ordinal), N = 1..;
     `doc_base_name` = `PurePosixPath(relative_path).parent.name`, kiểm tra an
     toàn theo GC-7; trùng (line_start, source_ordinal) trong 1 doc hoặc trùng
     tên file → `ExportError`.
   - `unit = unit_normalized or unit_raw` (giữ None khi cả hai None);
     `company = company_code`; `year = report_year`;
     `report_type = statement_scope`; `statement = statement_type`.
   - Ghi CSV: `csv.writer` với `lineterminator="\n"`, mở
     `open(tmp, "w", encoding="utf-8-sig", newline="")`; dòng đầu = headers;
     sau đó các rows.
   - Ghi `output_dir/manifest.jsonl` bằng orjson
     `OPT_SORT_KEYS | OPT_APPEND_NEWLINE` (mẫu `write_table_documents`).
   - Đối chiếu `table_count` với `release_dir/manifest.json` (GC-5).
   - Trả `CsvExportManifest`.

**Unit tests (`tests/unit/export/test_csv_export.py`)** — phủ ít nhất:
flatten_header (1 cấp / nhiều cấp trùng kề cần dedup / khoảng trắng & dấu `/`
/ list rỗng); detect_header_row_count (0 dòng khi hàng 0 có ô mang nhãn; dừng
ở hàng toàn blank; nhiều dòng header rowspan); build_normalized_table (rowspan
nhãn dòng lặp qua nhiều hàng lưới, colspan header lặp qua nhiều cột, group
context 2 cấp lồng nhau từ dữ liệu `" > "` có sẵn, bảng không có group context,
bảng header_rows=0, format số theo Quyết định 5, ô numeric None fallback
value_raw.strip(), ô thiếu placement → ""); export_normalized_csvs trên
release fixture nhỏ dựng bằng `pa.Table.from_pylist` + schema (theo mẫu
`tests/unit/conftest.py::release_dir`): tên file đánh số theo line_start,
manifest.jsonl đúng 7 trường & thứ tự xác định, count lệch manifest → ExportError,
doc_base_name không an toàn → ExportError, trùng line_start trong 1 doc → ExportError.

---

## Task 2 — `export/synced_text.py` + unit test

**Files tạo:** `src/financial_report_qa/export/synced_text.py`,
`tests/unit/export/test_synced_text.py`.

**Yêu cầu:**

1. `TableExportEntry` frozen dataclass: `line_start: int`, `line_end: int`,
   `table_id: str`, `csv_relpath: str`.
2. `build_synced_text(snapshot_root: Path, document: DocumentRecord,
   tables: list[TableExportEntry]) -> str`:
   - Đọc qua `read_document(snapshot_root, document)`; dựng mảng dòng gốc.
   - Validate: span 1-based, `1 <= line_start <= line_end <= số dòng`;
     các span không chồng lấn nhau; vi phạm → `ExportError`.
   - Sắp `tables` theo `line_end` GIẢM dần; thay từng đoạn
     `[line_start..line_end]` bằng đúng một dòng
     `"[TABLE: {table_id} -> {csv_relpath}]"` với `line_ending="\n"`.
   - Trả về `"".join(text + ending)` của mảng sau thay thế — ngoài các đoạn
     thay thế phải byte-for-byte giống văn bản gốc tái tạo từ `lines`.
3. `SyncedTextManifest` frozen dataclass: `output_dir: Path`,
   `document_count: int`, `table_count: int`.
4. `export_synced_text(release_dir: Path, snapshot_root: Path,
   csv_manifest: CsvExportManifest, output_dir: Path | None = None)
   -> SyncedTextManifest` (default `output_dir = Path("data/interim/synced_text")`):
   - Đọc `csv_manifest.manifest_path` → map `table_id → csv_path` (tên file);
     link = `(csv_manifest.output_dir / tên).as_posix()`.
   - Truy release parquet lấy (documents.*) cho mọi doc có ≥1 bảng trong map;
     dựng lại `DocumentRecord.model_validate(...)` — CHỈ select đúng các
     trường model có (`extra="forbid"`, parquet có thừa `ruleset_version`,
     `normalization_fingerprint` phải loại ra).
   - Với mỗi doc (thứ tự `relative_path` tăng dần): build text, ghi mirror
     `output_dir / relative_path` (mkdir parents), atomic write (GC-3), nội
     dung encode UTF-8.
   - Lỗi đọc nguồn để propagate nguyên trạng (GC-6).

**Unit tests** — phủ ít nhất: thay đúng span giữ nguyên phần còn lại
byte-for-byte (TXT fixture utf-8 + utf-8-sig BOM, có page marker + prose +
nhiều khối bảng; DocumentRecord mock `sha256="a"*64`, `doc_id=stable_document_id(sha)`,
`file_size_bytes=len(payload)`, `encoding="utf-8"`); nhiều bảng 1 document thay
đúng thứ tự giảm dần line_end không lệch offset; span overlap/out-of-range →
ExportError; thiếu file nguồn → lỗi cứng; end-to-end
`export_synced_text` trên release fixture nhỏ: link trong text tồn tại trên
đĩa và trỏ đúng CSV do Task 1 sinh ra (dùng chung helper dựng fixture nếu cần,
không import từ tests/unit/data).

---

## Task 3 — `export/cli.py` + wire subcommand

**Files tạo/sửa:** `src/financial_report_qa/export/cli.py`,
`src/financial_report_qa/cli.py`, `tests/unit/test_cli.py` (bổ sung case).

**Yêu cầu:**

1. `financial_report_qa/export/cli.py::main(argv=None) -> int`:
   argparse các flag: `--release-dir` (bắt buộc), `--snapshot-root`
   (bắt buộc), `--csv-output-dir` (bắt buộc), `--text-output-dir`
   (tuỳ chọn, default `data/interim/synced_text`) — đều là `Path`.
   Chạy tuần tự `export_normalized_csvs` → `export_synced_text`; in ra stdout
   tổng kết (table_count, document_count); bắt `FinancialReportQAError` →
   `error: …` ra stderr, return 1; thành công return 0.
2. `cli.py`: đăng ký subcommand `export-tables` (add_help=False, help mô tả)
   + nhánh lazy import gọi `financial_report_qa.export.cli:main` — đúng mẫu
   các subcommand hiện có.
3. Test `tests/unit/test_cli.py` thêm case: dispatcher forward argv tới
   export main (monkeypatch module con, mẫu case retrieval hiện có).

## Task 4 — Integration test CLI end-to-end

**Files tạo:** `tests/integration/export/test_export_tables_cli.py`.

**Yêu cầu:** dựng 1 release nhỏ trực tiếp bằng pyarrow (`DOCUMENT_SCHEMA`,
`TABLE_SCHEMA`, `CELL_SCHEMA`, `PLACEMENT_SCHEMA` từ `dataset_builder`) gồm
2 documents × vài bảng (có rowspan/colspan/group context, 1 bảng có
statement/unit null) + `manifest.json` (đúng `table_count`) + snapshot TXT
root chứa 2 file nguồn (sha256/size/encoding khớp DocumentRecord; có thể dùng
`sha256="a"*64` như reader cho phép). Gọi `main([...])` của export CLI qua
subprocess-free import (mẫu `tests/integration/evaluation/test_week1_cli.py`),
assert: exit 0; mọi `csv_path` trong manifest.jsonl tồn tại trên đĩa; nội dung
CSV khớp lưới kỳ vọng (header flatten + nhãn tổ hợp + số); synced text mirror
richt đường dẫn, link `[TABLE: …]` trỏ tới file tồn tại, phần prose/page
marker giữ nguyên; chạy lần 2 với snapshot thiếu file → exit 1 + stderr có
thông báo lỗi.

---

## Tiến độ

| Task | Trạng thái |
|------|-----------|
| 1 | pending |
| 2 | pending |
| 3 | pending |
| 4 | pending |
