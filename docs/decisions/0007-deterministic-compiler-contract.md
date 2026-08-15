# ADR 0007: Hợp đồng compiler tất định

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A1 (compiler đọc thẳng `cells.parquet` qua DuckDB); B1 (hình chiếu DataFrame dạng
  dài, 8 cột); C2 (giải kỳ: `period` tường minh trước, suy diễn từ `report_year` sau); D1 (không bao
  giờ đoán khi nhập nhằng — trả error code có kiểu); E1 (quy đổi đơn vị bằng `normalization/units.py`
  hiện có, không viết lại); F1 (replay `pandas_query` là điều kiện DoD, không phải tuỳ chọn)

## Bối cảnh

[Kế hoạch Ngày 18](../plans/day18-deterministic-compiler.md) đo trên release đã khoá
(`data/processed/release_v2_422df141c935`) và phát hiện: phần số học không phải chỗ khó — chỗ khó là
locator đi từ `FinancialQueryPlan` xuống một ô số. Chỉ 15,4 % ô có `period`; 62,5 % bảng (91.266)
không có `period` trên bất kỳ ô nào vì dùng bố cục `Số đầu năm`/`Số cuối năm` với năm nằm ở cấp tài
liệu; 37,7 % giá trị `period` là ngày ISO chứ không phải `YYYY`; 33.321 nhóm duplicate row có giá trị
xung đột; `tables.csv_path` NULL cho cả 146.011 bảng. ADR này chốt 6 quyết định thiết kế bắt buộc
trước khi viết code compiler.

## Quyết định A: compiler lấy dữ liệu ô từ đâu?

**Đã chọn: A1 — đọc thẳng `cells.parquet` bằng DuckDB, giới hạn trong `plan.candidate_table_ids`.**

Retrieval service (Ngày 8–14) chỉ trả về `table_id`, chưa từng mở ô số. Không có tầng trung gian nào
khác đã cấp API đọc ô theo bảng. `candidate_table_ids` đã bị schema Ngày 15 giới hạn 1–12 bảng, nên
truy vấn luôn có phạm vi nhỏ, không cần tầng cache riêng cho Ngày 18.

## Quyết định B: hình chiếu DataFrame có hình dạng gì?

**Đã chọn: B1 — dạng dài, 8 cột cố định:**
`table_id | row_idx | col_idx | row_label | column_label | period | unit | value`.

`pandas_query` sinh ra đọc được, ví dụ:

```python
df1[(df1.row_label == "TỔNG CỘNG TÀI SẢN") & (df1.period == 2023)]["value"].iloc[0]
```

**Sai khác có chủ ý so với ví dụ ở `plan.md` § 2.4**
(`df1[(df1.company == 'VNM') & (df1.year == 2023)]['net_revenue'].iloc[0]`, hình dạng rộng — mỗi
metric một cột). Corpus báo cáo tài chính Việt Nam không có hình dạng đó: metric nằm ở **dòng**, tên
dòng là tiếng Việt tự do không cố định, ép sang cột riêng cho mỗi metric sẽ phải bịa danh sách cột.
Hợp đồng nộp bài (`plan.md` § 2.4, quy tắc 3 và 7) chỉ ràng buộc *replay `pandas_query` phải khớp
`answer`*, không ràng buộc hình dạng cột cụ thể — nên B1 hợp lệ với hợp đồng dù khác ví dụ minh hoạ.

## Quyết định C: giải kỳ (period) bằng cách nào?

**Đã chọn: C2 — ưu tiên `period` tường minh của ô (chuẩn hoá 4 chữ số đầu để bắt cả dạng ISO), sau đó
suy diễn từ `column_label` chứa "số cuối năm" / "số đầu năm" / "năm trước" cộng
`documents.report_year`.**

Đo: quy tắc suy diễn nâng tỷ lệ plan gold70 giải được từ 24/51 (47,1 %) lên 30/51 (58,8 %), và nâng
số ô `số + kỳ` toàn corpus từ 822.679 lên 1.287.719 (+56,5 %). Kỳ suy diễn được đánh dấu
`period_inferred=True` trong `CellMatch` để Ngày 20 hạ mức tin cậy khi cần.

**Nợ đã biết:** quy tắc đối chiếu offset (`Số cuối năm` → `report_year + 0`, `Số đầu năm` →
`report_year − 1`) mới có **n = 10** ô làm chứng (5 mỗi nhánh), tuy nhất quán 100 % không có phản ví
dụ. Đây là bằng chứng yếu, không phải quy tắc đã chứng minh chặt — ghi rõ để không bị hiểu nhầm khi
đọc lại ADR này.

## Quyết định D: khi nhập nhằng thì làm gì?

**Đã chọn: D1 — không bao giờ đoán.**

| Số ô khớp | Hành vi |
| --- | --- |
| 0 | error `metric_not_found` hoặc `period_unresolved` |
| 1 | trả giá trị |
| n, mọi `value_numeric` bằng nhau | trả giá trị, evidence giữ toàn bộ `cell_id` liên quan |
| n, có giá trị khác nhau | error `cell_ambiguous`, liệt kê ứng viên trong thông báo lỗi |

Đo: 33.321/35.766 nhóm duplicate-row toàn corpus (93,2 %) có giá trị xung đột — lấy dòng đầu là sinh
đáp án sai âm thầm. Trên gold70, nhánh "n ô cùng giá trị" cứu 12/82 khe; nhánh xung đột giữ 2 khe ở
dạng lỗi thay vì số sai.

## Quyết định E: quy đổi đơn vị bằng gì?

**Đã chọn: E1 — tái dùng nguyên `normalization/units.py::convert_scale`/`economic_value`/
`unit_multiplier`, không viết lại bảng hệ số.**

`convert_scale` đã raise `ValueError` khi trộn nhóm tiền tệ với `percent`/`ratio` — compiler bắt
exception này và đổi thành error `unit_incompatible` có kiểu, không thêm logic quy đổi mới.

## Quyết định F: `pandas_query` có phải bằng chứng thật không?

**Đã chọn: F1 — replay `pandas_query` qua whitelist replayer trên chính hình chiếu B1 là điều kiện
Definition of Done, không phải một trường trang trí.**

`tables.csv_path` hiện NULL cho cả 146.011 bảng — hình chiếu mà `pandas_query` trỏ tới chưa tồn tại ở
bất kỳ đâu trong codebase trước Ngày 18. Nếu không chốt và kiểm chứng nó, `pandas_query` là chuỗi
không thể phủ định: viết gì cũng "đúng" vì không có gì để replay lại. Mọi kết quả compile phải chạy
qua replayer (chỉ boolean mask, `[]`, `.iloc`, số học scalar — không `eval`/`exec`, đúng tinh thần
whitelist của Ngày 19) và khớp kết quả compiler tự tính; lệch một trường hợp là hỏng build.

## Số đo hỗ trợ quyết định

| Số đo | Giá trị | Nguồn |
| --- | ---: | --- |
| Plan gold70 giải được tới ô số, chỉ dùng `period` tường minh | 24/51 (47,1 %) | § 1.3 kế hoạch Ngày 18 |
| Plan gold70 giải được tới ô số, cộng suy diễn kỳ (C2) | 30/51 (58,8 %) | § 1.3 |
| Bảng không có `period` trên bất kỳ ô nào | 91.266/146.011 (62,5 %) | § 1.2 |
| Giá trị `period` dạng ISO (`YYYY-MM-DD`), không phải `YYYY` | 359.199/952.363 (37,7 %) | § 1.1 |
| Nhóm duplicate row `(table, row_label_raw, period, col_idx)` có giá trị xung đột | 33.321/35.766 (93,2 %) | § 1.4 |
| Ô có `row_label_canonical` | 55.891/6.199.661 (0,9 %) | § 1.1 |
| Bảng dùng ≥2 đơn vị | 501/47.040 (1,06 %) | § 1.4 |
| Ô âm dùng dấu ngoặc `(…)` | 326.782/327.743 (99,7 %) | § 1.5 |
| Ô có `value_numeric = 0` | 4.649 trên 1.257 bảng | § 1.5 |
| `tables.csv_path` khác NULL | 0/146.011 | § 1.6 |

## Hệ quả

- `execution/cell_frame.py` là nơi duy nhất chuyển `cells.parquet` sang hình chiếu B1; mọi module sau
  đó (`locator.py`, `operations.py`, `pandas_query.py`) chỉ thao tác trên hình chiếu này, không tự
  truy vấn DuckDB.
- `CellMatch` phải mang `period_inferred: bool` để Ngày 20 phân biệt kỳ tường minh với kỳ suy diễn.
- Khối `execution:` trong `configs/*.yaml` được nối dây (nhiệm vụ 18.2), giống việc Ngày 17 đã nối
  dây khối `llm:`.
- Không đổi `dataset_fingerprint`; không đụng `data/processed/` hay `normalization/`.
- **Không mở rộng D1 để "cứu" thêm khe** — 16 khe `metric_label_absent` (thiếu `row_label_canonical`)
  và 13 khe `period_unresolved` là nợ của normalization/retrieval, không phải của compiler. Compiler
  trả error code có kiểu cho cả hai, không nới locator để đoán.
