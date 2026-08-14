# ADR 0006: Vai trò của LLM planner

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A1 (router: luật trước, LLM chỉ chạy khi luật abstain); B1 (prompt không chứa
  row label bảng ứng viên); C1 (chấm điểm trên 1.400 plan case, mô tả — không chấm — trên 19 câu
  gold70 bị abstain); D (ba loại lỗi JSON/schema/semantics đều dùng chung đúng một lượt repair)

## Bối cảnh

[Kế hoạch Ngày 17](../plans/day17-llm-planner.md) đo bằng dữ liệu thật (rule planner Ngày 16,
gold70, 1.400 plan case, tokenizer `bge-m3` offline) và phát hiện: (1) 19 câu gold70 bị rule
planner abstain không câu nào thực sự cần LLM — 7 câu khớp đúng arity `compare` mà luật chưa định
tuyến, 8 câu liệt kê/bộ phận không có đáp án số, 2 câu thiếu từ điển, 2 câu vượt arity schema; (2)
rule planner chỉ phát ra 4/9 operation trên 1.400 case; (3) schema đầy đủ ăn 23,5 % cửa sổ 4.096
token; (4) 12 `candidate_table_ids` tốn nhiều token hơn `max_output_tokens=160` cho phép. ADR này
chốt 4 quyết định thiết kế bắt buộc trước khi viết code.

## Quyết định A: LLM thay thế hay dự phòng?

**Đã chọn: A1 — router, luật chạy trước, LLM chỉ chạy khi luật abstain.**

Rule planner Ngày 16 đã đo được operation accuracy 1,000 và false-plan rate 0,000 trên 1.400 plan
case. Cho LLM chạy trước (A2) chỉ có thể làm tệ đi một đường đã hoàn hảo, không thể làm tốt hơn.
Chạy song song rồi đối chiếu (A3) tốn gấp đôi chi phí suy luận và khi hai bên lệch vẫn cần một luật
phân xử — quay lại đúng A1. Với A1, router **không bao giờ** để LLM ghi đè một plan mà luật đã tạo
ra thành công; LLM chỉ được gọi trên nhánh luật trả về `abstain_codes`.

## Quyết định B: prompt có chứa row label của bảng ứng viên không?

**Đã chọn: B1 — không. Prompt chỉ có câu hỏi + danh sách canonical metric + few-shot.**

Đo trên 91 bảng gold: trung bình 524 chars/bảng, p90 1.058 chars/bảng row label. Với 6 bảng ứng
viên, B2 (đưa row label vào, đúng tinh thần ADR 0004 Option C) tốn thêm 656–1.300 token — trong khi
schema rút gọn + enum 56 metric đã chiếm 34 % cửa sổ 4.096. Ngoài ra `rule_planner._metric_selector`
hiện tại đã lấy `raw_text` từ **câu hỏi** (`span.surface`), không phải từ bảng — B1 nhất quán với
hành vi luật đang chạy, còn B2 sẽ tạo hai hành vi khác nhau giữa hai nhánh của cùng một router.

**Nợ đã biết, không phải nợ ẩn:** đây là lệch thật với câu chữ ADR 0004 ("nhãn nguồn chép nguyên
văn từ một bảng mà planner đã nhìn thấy"). Chưa có bằng chứng nào cho thấy `raw_text` từ bảng tốt
hơn từ câu hỏi ở giai đoạn này — Ngày 18 (compiler) là nơi bằng chứng đó xuất hiện, vì compiler mới
thực sự tra `raw_text` vào bảng và biết nó khớp hay không. Xem lại quyết định B khi đó.

## Quyết định C: chấm điểm trên tập nào?

**Đã chọn: C1 — 1.400 plan case (đáp án từ template, Ngày 16) là bộ chấm điểm chính; gold70 chỉ mô
tả, không chấm.**

Sinh một "QA dev" mới (C2) nghĩa là tự viết đề rồi tự chấm — đúng thiên lệch mà Ngày 16 finding #2
đã cảnh báo (`intent` gold70 không map 1:1 vào `PlanOperation`). 1.400 plan case đã có đáp án suy
từ `template_id`, độc lập với planner. Phép đo tách làm hai, không trộn:

1. **LLM planner đứng một mình** trên 600 case `expected_operation` (bỏ qua bước luật) →
   `operation_accuracy`, `invalid_json_rate`, `repair_success_rate`.
2. **Router (A1)** trên 800 case `expected_abstain_code` → `false_plan_rate` phải giữ 0,0 — đây là
   phép thử khó thật sự: LLM có bịa plan cho câu mà luật đã đúng khi từ chối không.

gold70 chỉ được **mô tả** (plannable rate, phân bố operation) đúng như báo cáo held-out Ngày 16,
không bao giờ gắn một con số "accuracy" vào 19 câu không có đáp án operation đáng tin.

## Quyết định D: cái gì được coi là "repair được"?

**Đã chọn: cả ba loại lỗi (JSON sai cú pháp, `FinancialQueryPlan` ném `ValidationError`,
`validate_plan_semantics` trả về issue) đều đi vào chung đúng một lượt repair; sau đó abstain.**

Không tách repair theo loại lỗi vì plan.md chỉ cho phép "một lần repair JSON tối đa" — chia nhỏ
thành nhiều loại sẽ ngầm cho phép nhiều hơn một lượt gọi lại mô hình. Repair prompt mang theo danh
sách `PlanValidationIssue` đã định kiểu (`plan_validator.py` đã ghi chú "typed for LLM-repair
prompts (Day 17)" từ Ngày 15) khi lỗi là bước 3; với bước 1–2, repair prompt mang theo thông báo lỗi
parse/validate thô. Sau đúng một lượt, mọi kết quả hỏng đều thành abstain có mã
(`llm_invalid_json`, `llm_plan_invalid`), không có lượt thứ hai.

## Số đo hỗ trợ quyết định

| Số đo | Giá trị | Nguồn |
| --- | ---: | --- |
| Câu gold70 bị rule planner abstain thực sự cần LLM (không có đáp án số hoặc luật đã xử lý được) | 0/19 | § 1.1 kế hoạch Ngày 17 |
| Operation rule planner từng phát ra trên 1.400 case | 4/9 | § 1.2 kế hoạch Ngày 17 |
| `model_json_schema()` token (bge-m3, cận dưới lạc quan) | 963 / 4.096 = 23,5 % | § 1.4 |
| Row label 6 bảng ứng viên, mức p90 | ~1.300 token | § 1.4 |
| 12 `candidate_table_ids` (token) vs `max_output_tokens` | 192–400 vs 160 | § 1.5 |
| Plan JSON đủ, bỏ `candidate_table_ids` | 59 token | § 1.5 |

## Hệ quả

- `planning/rule_planner.py::PlanAbstainCode` mở rộng thêm `llm_unavailable`, `llm_invalid_json`,
  `llm_plan_invalid` cho nhánh LLM (nhiệm vụ 17.6).
- `planning/plan_router.py` (nhiệm vụ 17.7) là điểm vào duy nhất kết hợp hai planner; không nơi nào
  khác trong codebase được gọi `llm_planner.build_plan` trực tiếp mà bỏ qua luật.
- Mọi test phải chạy được khi không có server llama.cpp (`models/` hiện rỗng,
  `127.0.0.1:8080` `ConnectTimeout` khi đo ADR này) — dùng `httpx.MockTransport`.
- Không đổi `dataset_fingerprint`; không đụng `data/processed/` hay `normalization/`.
- **Không tinh chỉnh prompt theo 19 câu gold70** — chỉ 19 câu, tinh chỉnh theo đó là overfit chứ
  không phải cải thiện (rủi ro R5, kế hoạch Ngày 17).
