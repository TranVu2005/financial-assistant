# ADR 0004: Chiến lược định vị metric trong FinancialQueryPlan

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-14
- **Quyết định:** Option C — `metric_selector` hai nhánh (`canonical` | `raw_text`)

## Bối cảnh

`FinancialQueryPlan` cần một trường xác định *hàng nào* trong bảng ứng viên chứa giá trị cần
lấy. Ứng viên tự nhiên là 56 metric canonical đã có trong
[`normalization/metrics.py`](../../src/financial_report_qa/normalization/metrics.py). Nhưng Day 14
chỉ sửa nhánh **truy hồi** (giữ `row_label_raw` trong text document BM25); nó không đổi
`cells.row_label_canonical`, thứ mà một compiler chỉ-canonical sẽ cần để định vị hàng.

## Số đo (release khóa `422df141c935…`)

### Mức độ phủ của canonical hoá

| Đại lượng | Giá trị |
| --- | ---: |
| Cell có `row_label_canonical` / có `row_label_raw` (toàn corpus) | 5,93 % |
| Bảng có ≥ 1 canonical row label (toàn corpus, 146.011 bảng) | 17,2 % |
| Bảng gold (91 bảng, từ 70 câu gold70) có ≥ 1 canonical row label | 61,5 % |
| Cell trong bảng gold có canonical / có raw | 16,98 % |

### Đo trên 70 câu hỏi gold70 (đối chiếu parser Day 10 với canonical label thật của bảng gold)

| Kết quả | Số câu | Tỷ lệ |
| --- | ---: | ---: |
| Mọi metric giải được tới canonical | 42 | 60,0 % |
| Parser không rút được metric nào (phần lớn câu `notes` liệt kê, không có đáp án số) | 20 | 28,6 % |
| Có metric nhưng không cái nào khớp bảng gold | 6 | 8,6 % |
| Khớp một phần | 2 | 2,9 % |

### Mật độ dữ kiện `(company, period, metric)` có `value_numeric`, sau khi chuẩn hoá `period` về
năm trần (`"YYYY-MM-DD"` → `"YYYY"`)

- 18.898 bộ ba distinct, phủ 100 công ty, 16 period, 55/56 metric canonical.
- **Một khi một metric đã tồn tại canonical cho một `(company, period)`, các operation nhiều thực
  thể hầu như luôn thực thi được:**

| Operation cần | Điều kiện | Tỷ lệ đạt (có điều kiện) |
| --- | --- | ---: |
| `growth_rate` | `(company, metric)` có ≥ 2 period distinct | 2.661/2.856 = **93,2 %** |
| `ratio` | `(company, period)` có ≥ 2 metric distinct | 1.083/1.091 = **99,3 %** |
| `rank` / `compare` | `(metric, period)` có ≥ 2 company distinct | 607/634 = **95,7 %** |

**Kết luận quan trọng:** nút thắt không nằm ở arity nhiều-thực-thể của operation. Nút thắt nằm
hoàn toàn ở bước đầu tiên — tìm được **một** metric canonical cho một `(company, period)`. Đây là
đúng vấn đề mà Day 14 đã chẩn đoán và sửa cho document BM25 (giữ `row_label_raw`), nhưng chưa sửa
cho đường dữ liệu cell-level mà compiler sẽ đọc.

## Các phương án đã xét

### A. Chỉ canonical (`metrics: list[str]` giới hạn 56 giá trị)

Đơn giản nhất, khớp thẳng với `cells.row_label_canonical`. Nhưng theo số đo trên, loại thẳng
**40 %** câu hỏi thực (28,6 % không rút được metric + 8,6 % rút được nhưng không khớp bảng gold)
ngay từ bước lập kế hoạch, đẩy toàn bộ nợ sang Day 16 (mở dictionary) mà không có cơ chế fallback
tường minh trong lúc chờ.

### B. Canonical, fallback khớp `row_label_raw` đã chuẩn hoá tự động

LLM/validator tự dò `row_label_raw` gần đúng khi không có canonical. Vấn đề: khi nhiều hàng cùng
"gần khớp" (ví dụ nhiều dòng con trong bảng thuyết minh chứa cùng một từ khoá), không có quy tắc
tất định để chọn đúng một hàng — vi phạm nguyên tắc *"không đoán khi thiếu bằng chứng"* đã áp dụng
xuyên suốt Day 10 (`entity_parser`) và Day 8–9 (`filtering.py`).

### C. `metric_selector` hai nhánh — **chọn**

```json
{"canonical": "net_revenue"}
```
hoặc
```json
{"raw_text": "Cho vay khách hàng"}
```

Đúng một trong hai khoá phải có mặt (bắt bằng validator, không phải `Optional` ngầm). Nhánh
`raw_text` giữ nguyên chuỗi xuất hiện trong nguồn (sau NFKC-normalize, không viết tắt/đoán) — LLM
Ngày 17 phải sao chép nguyên văn từ bảng ứng viên nó thấy, không được bịa cụm từ mới.

## Phạm vi ở Day 15 so với Day 18

Validator Ngày 15 chỉ kiểm tra **hình dạng** của `metric_selector` (đúng một nhánh, `canonical`
thuộc 56 giá trị, `raw_text` không rỗng sau chuẩn hoá) — không mở `cells.parquet` để xác minh
`raw_text` thật sự xuất hiện trong một `candidate_table_id` cụ thể. Việc đó thuộc compiler (Day
18), nơi đã có `TableFrame` đầy đủ trong bộ nhớ cho từng bảng; lặp lại việc đó ở validator vừa tốn
kém vừa trùng lặp trách nhiệm. Nếu `raw_text` không khớp hàng nào lúc thực thi, đó là lỗi
`ExecutionError` (không có bằng chứng), không phải lỗi validate-time.

`candidate_table_ids` **vẫn** được validator Ngày 15 kiểm tra tồn tại trong `tables.parquet` —
đây là existence check rẻ (một bảng, không phải cell), khác về bản chất với việc dò nội dung
`raw_text` bên trong bảng.

## Hệ quả

- `PlanErrorCode` cần thêm ít nhất: `metric_selector_ambiguous` (cả hai khoá cùng có mặt),
  `metric_selector_empty` (không khoá nào), `metric_canonical_unknown` (ngoài 56 giá trị),
  `metric_raw_text_empty`.
- `numerator_metric`/`denominator_metric` của `ratio` dùng cùng kiểu `metric_selector`, không phải
  `str` thuần.
- Đây là nợ kỹ thuật tạm thời, không phải giải pháp lâu dài: Ngày 16 (*"mở rộng dictionary
  alias"*) nên ưu tiên các nhãn xuất hiện nhiều lần trong `raw_text` được LLM chọn ở Tuần 3, đo
  được qua log thực tế — biến nợ ẩn thành backlog có số liệu.
