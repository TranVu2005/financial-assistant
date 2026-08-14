# ADR 0005: Lấp lỗ hổng độ phủ operation của FinancialQueryPlan

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A1 (thêm `compare_companies`) — đã triển khai; B2 (abstain có mã lỗi cho
  "biến động đầu năm–cuối năm") — giữ nguyên, hoãn; C2 (chiếu kỳ dạng ngày/quý về `YYYY`) —
  chưa cần, hoãn tới khi có nhu cầu thật

## Bối cảnh

[Kế hoạch Ngày 16](../plans/day16-deterministic-planning.md) đo bằng dữ liệu thật (gold70,
1.400 entity case, release khóa `422df141c935…`) và phát hiện 6 vấn đề trong đường ống
`entity_parser → FinancialQueryPlan`. ADR này chốt cách xử lý 3 vấn đề cần một quyết định thiết
kế (không chỉ sửa lỗi thuần túy): lỗ hổng operation so sánh chéo công ty, "biến động đầu
năm–cuối năm", và ngữ pháp kỳ dạng ngày/quý.

## Quyết định A: so sánh chéo công ty `(≥2 company, 1 period, 1 metric)`

**Đã chọn: A1 — thêm operation `compare_companies`.**

Template `two_companies` (100/1.400 entity case, ví dụ *"So sánh Các khoản giảm trừ doanh thu
giữa GEG và GEX năm 2017"*) không khớp operation nào trong 8 operation của Ngày 15:
- `rank` bắt buộc `top_k` và mang nghĩa xếp hạng, không phải đặt cạnh nhau.
- `average`/`sum` gộp thành một con số, sai ngữ nghĩa "so sánh".

Đã thêm `compare_companies` vào `PlanOperation`
([`plan_contracts.py:22-31`](../../src/financial_report_qa/planning/plan_contracts.py)), cùng
`_validate_compare_companies`
([`plan_validator.py`](../../src/financial_report_qa/planning/plan_validator.py)) với arity
`(≥2 company, đúng 1 period, đúng 1 metric, không top_k, không metric_pair, không
numerator/denominator)`. Xác nhận độc lập qua property test
(`tests/unit/planning/test_plan_property.py::test_validator_agrees_with_independent_arity_spec`)
và golden case mới (`tests/golden/plans/valid/compare_companies.json`, tái dùng sự kiện thật
GEX/PNJ `net_revenue` 2021 đã có ở `sum.json`).

Đã loại A2 (nới nhánh thứ hai vào `compare`) vì sẽ làm mờ ngữ nghĩa `compare` hiện tại (hai
metric, một công ty) — hai khái niệm khác nhau không nên chung một operation. Đã loại A3 (bắt
abstain) vì đây là hình dạng có tần suất cao (100/1.400 case) và biểu diễn được không tốn thêm
field nào, không có lý do gì để từ chối.

## Quyết định B: "biến động đầu năm–cuối năm" là hai hàng, không phải hai kỳ

**Đã chọn: B2 — abstain có mã lỗi, hoãn tới khi có compiler.**

12/23 câu `intent=growth` của gold70 chỉ có một kỳ vì giá trị đầu kỳ/cuối kỳ nằm ở hai **hàng**
khác nhau trong cùng một bảng (`Số dư đầu năm`: 6.468 bảng; `Số dư cuối năm`: 6.820 bảng), không
phải hai giá trị `cells.period` khác nhau. `FinancialQueryPlan` không có cơ chế "hai hàng cùng
bảng, cùng kỳ".

Không chọn B1 (thêm operation `period_change` với hai hàng) vì đó là field mới không hàm ý được
gì thêm cho compiler Ngày 18 — compiler vẫn chưa có cách định vị "đầu kỳ" vs "cuối kỳ" một cách
đáng tin cậy (nhãn hàng biến thiên: `Số dư đầu năm` / `Số đầu năm` / `Số cuối năm` / `Tại ngày
đầu năm`, không có một mẫu chuẩn). Mở schema cho một cơ chế chưa ai thực thi được là nợ kỹ thuật
sớm, không phải sửa lỗi.

**Hệ quả:** entity parser / rule planner Ngày 16 phải trả về abstain với mã riêng (ví dụ
`period_change_unsupported`) cho câu dạng này thay vì ép vào `growth_rate`. Ghi vào backlog Ngày
18 (compiler): nếu có nhu cầu thật, thêm cơ chế định vị "cặp hàng cùng ngữ nghĩa thời gian" lúc
đó, có compiler để kiểm chứng ngay.

## Quyết định C: kỳ dạng ngày/quý bị schema từ chối

**Đã chọn: C2 (dự phòng) — chiếu về `YYYY`, không sửa ngay.**

`FinancialQueryPlan.periods` ràng buộc `^\d{4}$`. Parser hợp lệ trả về `2025-12-31` (template
`date_lookup`, 100 case) và `2018-Q4` (template `quarter_lookup`, 100 case) — hai template này bị
schema từ chối thẳng nếu không xử lý.

**Không sửa trong đợt này** vì gold70 — bộ đo held-out duy nhất hiện có — có 91/91 giá trị kỳ đều
là `YYYY`; vấn đề chưa có bằng chứng xảy ra ở dữ liệu thật, chỉ ở entity case tổng hợp. Ghi nhận
là nợ đã biết (không phải nợ ẩn): khi rule planner Ngày 16 gặp một kỳ không phải `YYYY`, nó phải
abstain có mã (`period_grammar_unsupported`) thay vì cố chiếu về `YYYY` một cách ngầm định — chiếu
ngầm định (ví dụ luôn lấy năm của ngày) có thể sai nếu ngày rơi vào quý cuối năm tài chính khác
lịch dương, một giả định chưa kiểm chứng được trên dữ liệu hiện có.

## Số đo hỗ trợ quyết định

| Hình dạng | Nguồn | Số lượng | Operation trước ADR | Operation sau ADR |
| --- | --- | ---: | --- | --- |
| `(≥2 company, 1 period, 1 metric)` | entity-cases `two_companies` | 100/1.400 | không có | `compare_companies` |
| `(1 company, 1 period, 1 metric)`, biến động trong năm | gold70 `growth` | 12/23 | ép sai vào `growth_rate` | abstain (`period_change_unsupported`, chưa cài) |
| kỳ dạng `YYYY-MM-DD` / `YYYY-Qn` | entity-cases `date_lookup`, `quarter_lookup` | 200/1.400 | bị schema từ chối | vẫn từ chối, có kế hoạch abstain rõ ràng |

## Hệ quả

- `tests/golden/plans/manifest.json` và `tests/unit/planning/test_plan_property.py` cập nhật cho
  `compare_companies`; 44 test đơn vị + 3 property test + 21 golden case đều xanh.
- Rule planner Ngày 16 (nhiệm vụ 16.5–16.6, chưa triển khai) phải implement 2 mã abstain mới:
  `period_change_unsupported` (Quyết định B) và `period_grammar_unsupported` (Quyết định C) —
  không được cố gắng "đoán" cho hai trường hợp này.
- Không đổi `dataset_fingerprint`; không đụng `normalization/metrics.py` hay bất kỳ artifact
  release nào.
