# Kế hoạch Ngày 17 — LLM planner

> Trạng thái: **đã triển khai § 3 (17.1–17.8); § 4 DoD đạt một phần** (xem
> [plan.md § Ngày 17](../../plan.md), [README.md § Day 17](../../README.md) và
> [ADR 0006](../decisions/0006-llm-planner-role.md)). Viết ngày 2026-08-15 sau khi Ngày 16 hoàn tất
> (`ff1ebb0`). Mọi con số ở mục 1 đo trực tiếp bằng chính `rule_planner.py`/`entity_parser.py` hiện
> tại trên release khóa `422df141c935…`, gold70 (`data/qa/retrieval-gold-v1.jsonl`) và bộ 1.400 plan
> case (`data/qa/plan-cases-v1.jsonl`, sha `ef11139ff448…`). Số token đo bằng tokenizer
> `BAAI/bge-m3` có sẵn offline — đây là **cận dưới lạc quan**, vì sentencepiece XLM-R mã hóa tiếng
> Việt tốt hơn BPE của Qwen; ngân sách thật sẽ chật hơn con số ghi ở đây, không rộng hơn.
>
> **Nợ đã biết sau khi triển khai:** không có mô hình llama.cpp sống trong môi trường này
> (`models/` rỗng, `127.0.0.1:8080` `ConnectTimeout` — đúng như § 1.8 đã dự đoán), nên
> `operation_accuracy`/`invalid_json_rate` của LLM planner trên mô hình thật **chưa đo được**; DoD
> mục 2 (đo trên 600/800 plan case) chỉ chạy được ở chế độ offline replay-cache rỗng, tức toàn bộ
> case abstain `llm_unavailable`. Điều đã xác nhận được và quan trọng hơn cho an toàn hệ thống: ngay
> cả khi LLM hoàn toàn không khả dụng, router A1 vẫn giữ `false_plan_rate = 0,0` — không bao giờ
> trả bừa một plan.

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Rule planner (Ngày 16) | `planning/rule_planner.py` | ✅ operation accuracy 1,0; false-plan rate 0,0 |
| Schema plan (Ngày 15) | `planning/plan_contracts.py` — 9 operation | ✅ đóng băng |
| Semantic validator | `planning/plan_validator.py` — 10 error code | ✅ `PlanValidationIssue` đã ghi rõ "typed for LLM-repair prompts (Day 17)" |
| Bộ plan case | `data/qa/plan-cases-v1.jsonl` — 1.400 case, đáp án từ template | ✅ 600 plan-expected + 800 abstain-expected |
| gold70 | `data/qa/retrieval-gold-v1.jsonl` | ⚠️ chỉ mô tả được, không có đáp án operation |
| `httpx` | `pyproject.toml` `>=0.28,<1` | ✅ **không cần thêm dependency nào cho Ngày 17** |
| Cấu hình `llm:` | `configs/base.yaml`, `configs/local_rtx3050.yaml` | ❌ **code chết** — xem § 1.3 |
| Mô hình GGUF | `models/` | ❌ **thư mục rỗng** |
| llama.cpp server | `http://127.0.0.1:8080/v1` | ❌ `ConnectTimeout` |
| `data/official/test_questions.json` | — | ❌ **vẫn chưa tồn tại** |

Ngày 17 **không** thực thi số (Ngày 18) và **không** dựng sandbox (Ngày 19). Phạm vi đúng ba thứ:
một client HTTP có kỷ luật, một planner LLM luôn đi qua validator của Ngày 15, và phép đo trung
thực về việc nó có làm chất lượng **tệ đi** hay không.

---

## 1. Chốt chặn phải đo trước khi viết code

### 1.1. Trên gold70, LLM planner có **0 câu** thật sự cần đến LLM

19/70 câu bị rule planner abstain. Phân rã theo hình dạng `(company, period, metric)` mà
`entity_parser` rút được:

| Mã abstain | `(c, p, m)` | Số câu | Bản chất thật |
| --- | --- | ---: | --- |
| `entity_ambiguous` | `(1, 1, 0)` | 7 | Câu **liệt kê thuyết minh** — "danh sách công ty con", "cơ cấu tài sản/nợ theo nhóm kỳ hạn". Không có metric vô hướng nào để tính. |
| `entity_ambiguous` | `(1, 2, 0)` | 3 | 2 câu **thiếu từ điển** (lưu chuyển tiền từ hoạt động tài chính / đầu tư), 1 câu báo cáo bộ phận. |
| `multi_metric_unsupported` | `(1, 1, 2)` | 7 | **Khớp đúng arity của `compare`** (`metric_a` + `metric_b`, 1 công ty, 1 kỳ) — luật xử lý được, chỉ là `_infer_operation` chưa định tuyến. |
| `multi_metric_unsupported` | `(1, 1, 3)` | 1 | 3 metric — ngoài schema Ngày 15. |
| `multi_metric_unsupported` | `(1, 2, 2)` | 1 | 2 metric × 2 kỳ — ngoài schema Ngày 15. |

Gộp lại: **7 câu là lỗ hổng định tuyến xác định**, **2 câu là lỗ hổng từ điển**, **8 câu không có
đáp án số** (liệt kê/bộ phận/riêng-vs-hợp-nhất), **2 câu vượt arity của schema**. Không một câu nào
thuộc dạng "luật không hiểu nổi câu tiếng Việt này" — tức là dạng duy nhất mà LLM giải được.

Hệ quả trực tiếp cho kế hoạch: **không được lấy "tăng plannable rate trên gold70" làm DoD của Ngày
17.** Đo đúng thì headroom bằng 0. Nếu vẫn ép LLM chạy trên 8 câu liệt kê, kết quả duy nhất có thể
xảy ra là nó bịa ra một metric — đúng thứ `false_plan_rate` của Ngày 16 sinh ra để chặn.

### 1.2. Lý do thật để vẫn xây LLM planner: rule planner chỉ chạm **4/9 operation**

Chạy `build_plan` trên toàn bộ 1.400 plan case, thống kê nhãn nó phát ra:

| Kết quả | Số case |
| --- | ---: |
| `lookup` | 300 |
| `difference` | 100 |
| `growth_rate` | 100 |
| `compare_companies` | 100 |
| abstain `entity_ambiguous` | 600 |
| abstain `period_grammar_unsupported` | 200 |

`compare`, `ratio`, `average`, `sum`, `rank` **không bao giờ được phát ra với bất kỳ đầu vào nào**.
Bốn trong số đó cần từ vựng ("gấp mấy lần", "biên", "bình quân", "cao nhất") mà Ngày 16 § 1.8 đã đo
được **0 lần xuất hiện** trong dữ liệu hiện có — nên viết luật cho chúng là suy đoán, còn để LLM
đọc là hợp lý. Cộng với việc `data/official/test_questions.json` vẫn chưa tồn tại, giá trị của Ngày
17 là **bảo hiểm phủ sóng cho ngữ pháp chưa từng thấy**, không phải cải thiện con số đã đo.

### 1.3. Cấu hình `llm:` trong YAML là code chết

`grep` toàn bộ `*.py`: **0 tham chiếu** tới `base_url`, `max_output_tokens`, `json_schema_constrained`
hay `allow_operations`. `core/config.py::Settings` chỉ có `app_env`, `data_root`, `log_level`.

Kèm theo một chỗ lệch: `configs/local_rtx3050.yaml` → `execution.allow_operations` liệt kê **8**
operation, thiếu `compare_companies` do Ngày 16 thêm vào. Config không ai đọc thì không ai phát
hiện lệch — phải hoặc nối dây thật, hoặc xóa.

### 1.4. Ngân sách token: schema đầy đủ ăn 23,5 % cửa sổ 4.096

| Thành phần | chars | tokens (bge-m3, lạc quan) | % của 4.096 |
| --- | ---: | ---: | ---: |
| `FinancialQueryPlan.model_json_schema()` | 2.081 | **963** | 23,5 % |
| Enum 56 canonical metric | 1.165 | **436** | 10,6 % |
| Row label của 6 bảng ứng viên (trung bình 524 chars/bảng) | 3.148 | ~656 | 16,0 % |
| Row label của 6 bảng ứng viên (p90 1.058 chars/bảng) | 6.348 | ~1.300 | 31,7 % |

Cộng ba dòng đầu ở mức p90 đã là **2.699 token = 66 %** cửa sổ, chưa tính một few-shot nào. Vậy
"prompt chỉ chứa schema rút gọn" trong plan.md không phải sở thích trình bày mà là **ràng buộc đã
đo**: không được nhét `model_json_schema()` vào prompt, phải viết tay một bảng
`operation → trường bắt buộc` gọn.

(Phân bố row label đo trên 91 bảng gold: trung vị 14 dòng / 493 chars, tối đa 35 dòng / 1.347 chars.)

### 1.5. LLM **không được** sinh `candidate_table_ids`

| Đo | Giá trị |
| --- | --- |
| Độ dài một `table_id` | 68 chars (`tbl_` + 64 hex) = 16 token (bge-m3) / ~34 token (BPE) |
| 12 id tối đa theo schema | 864 chars ≈ 192–400 token |
| `max_output_tokens` cấu hình | **160** |
| Plan JSON đủ, **có** 1 table id | 205 chars / 90 token |
| Plan JSON đủ, **bỏ** `candidate_table_ids` | **110 chars / 59 token** |

Ba lý do độc lập cùng chỉ về một hướng: (1) 12 id không lọt `max_output_tokens=160`; (2) chuỗi 64
hex là thứ LLM bịa tệ nhất; (3) `validate_plan_semantics` sẽ chặn bằng `candidate_table_ids_unknown`
nên mọi token sinh ra đều lãng phí. → **caller tiêm `candidate_table_ids`**, đúng hợp đồng mà
`build_plan` đã dùng.

### 1.6. `raw_text` hiện lấy từ **câu hỏi**, không phải từ bảng — lệch với ADR 0004

ADR 0004 Option C mô tả `raw_text` là "nhãn nguồn chép nguyên văn từ một bảng mà planner đã nhìn
thấy". Nhưng `rule_planner._metric_selector` lấy `span.surface` — bề mặt trong **câu hỏi**. Đây là
lệch có thật giữa ADR và code, và nó quyết định trực tiếp mục 1.4: nếu LLM cũng không cần nhìn bảng
thì tiết kiệm được 656–1.300 token; nếu tôn trọng ADR 0004 thì phải trả khoản đó.

Phải chốt, không được trôi vào một trong hai hướng bằng quán tính.

### 1.7. Không tồn tại "QA dev" như plan.md giả định

Đầu ra Ngày 17 viết "plan accuracy và invalid JSON rate trên **QA dev**". Repo chỉ có: 1.400 plan
case (có đáp án, nhãn suy từ `template_id`) và gold70 (không có đáp án operation, Ngày 16 đã kết
luận `intent` ≠ `operation`). **Không có** tập dev nào khác. Phải định nghĩa lại mục tiêu đo bằng
thứ đang có, hoặc thừa nhận đang tự sinh đề rồi tự chấm.

### 1.8. Không có mô hình, không có server

`models/` rỗng; `http://127.0.0.1:8080/v1/models` và `/health` đều `ConnectTimeout`. Nên:

- **Mọi test phải chạy được khi không có server** — `httpx.MockTransport` (httpx đã là dependency).
- Lần chạy thật là một artifact **tùy chọn, tách rời**, không bao giờ là cổng CI.
- `temperature 0` **không** đảm bảo tái lập (kv-cache, batching, phiên bản build llama.cpp khác nhau
  vẫn lệch). Cần cache phản hồi khóa theo `sha256(prompt) + danh tính mô hình`, đúng tiền lệ cache
  truy vấn dense của Ngày 9.

---

## 2. Ba quyết định phải chốt trước khi viết code (→ ADR 0006)

### Quyết định A — LLM thay thế hay dự phòng?

| | Phương án | Đánh giá |
| --- | --- | --- |
| **A1** | **Router: luật chạy trước, LLM chỉ chạy khi luật abstain** | ✅ **Khuyến nghị.** Đường xác định đang có operation accuracy 1,0 — cho LLM chạy trước chỉ có thể làm tệ đi. Ngoài ra A1 tách `false_plan_rate` thành hai phần đo riêng được. |
| A2 | LLM trước, luật là fallback | Vứt bỏ một đường đã đo là hoàn hảo để lấy một đường chưa đo được lần nào. |
| A3 | Chạy cả hai rồi đối chiếu | Gấp đôi chi phí, và khi hai bên lệch vẫn phải có luật phân xử — tức là quay về A1. |

### Quyết định B — Prompt có chứa row label của bảng ứng viên không?

| | Phương án | Đánh giá |
| --- | --- | --- |
| **B1** | **Không: LLM chỉ thấy câu hỏi + danh sách canonical metric** | ✅ **Khuyến nghị cho Ngày 17.** Tiết kiệm 656–1.300 token (§ 1.4), và **nhất quán với `_metric_selector` hiện tại** (§ 1.6). |
| B2 | Có: tôn trọng ADR 0004 Option C | Đúng tinh thần ADR nhưng đắt, và chưa có bằng chứng nào cho thấy `raw_text` từ bảng tốt hơn từ câu hỏi — Ngày 18 (compiler) mới là chỗ bằng chứng đó xuất hiện. |

Chốt B1 và **ghi vào ADR 0006 như nợ đã biết**, xem lại khi compiler Ngày 18 cho biết `raw_text`
thật sự cần gì.

### Quyết định C — Chấm điểm trên tập nào?

| | Phương án | Đánh giá |
| --- | --- | --- |
| **C1** | **Dùng chính 1.400 plan case + mô tả trên 19 câu abstain của gold70** | ✅ **Khuyến nghị.** Đáp án đã có sẵn và suy từ `template_id`, không copy từ planner. |
| C2 | Sinh tập "QA dev" mới | Lại tự viết đề rồi tự chấm — đúng thiên lệch Ngày 16 đã cảnh báo, mà không thêm tín hiệu mới. |

Với C1, phép đo tách làm hai, **không được trộn**:

1. **LLM planner đứng một mình** trên **600 case plan-expected** → `operation_accuracy`,
   `invalid_json_rate`, `repair_success_rate`. Đây là con số "plan accuracy" plan.md đòi.
2. **Router (A1)** trên **800 case abstain-expected** → `false_plan_rate` **phải giữ 0,0**. Đây là
   phép thử thật sự khó: 800 câu mà luật đã từ chối đúng, LLM có bịa ra plan cho câu nào không.

### Quyết định D — Cái gì được coi là "repair được"?

plan.md: "một lần repair JSON tối đa; sau đó abstain". Ba loại hỏng khác nhau:

1. JSON sai cú pháp;
2. JSON đúng nhưng `FinancialQueryPlan` ném `ValidationError`;
3. Plan dựng được nhưng `validate_plan_semantics` trả về issue.

Khuyến nghị: **cả ba** đều đi vào đúng **một** lượt repair, mang theo danh sách
`PlanValidationIssue` đã định kiểu (`plan_validator.py:61` đã ghi rõ nó sinh ra cho việc này); sau
lượt đó → abstain. Không có lượt thứ hai.

---

## 3. Nhiệm vụ

| # | Nhiệm vụ | Đầu ra | Ghi chú |
| --- | --- | --- | --- |
| **17.1** | ADR 0006 chốt A1/B1/C1/D, kèm bảng số liệu § 1 | `docs/decisions/0006-llm-planner-role.md` | Không sinh code |
| **17.2** | Settings LLM có kiểu + nối dây `configs/*.yaml`; sửa lệch `allow_operations` | `core/config.py` | Sửa § 1.3 |
| **17.3** | Client OpenAI-compatible trên `httpx`: timeout, retry có giới hạn, **không retry 4xx**, lỗi định kiểu | `llm/client.py`, `core/errors.py` | Test bằng `httpx.MockTransport`, 0 network call |
| **17.4** | Prompt: schema rút gọn viết tay + 6–10 few-shot, render tất định | `planning/llm_prompt.py` | Kèm **test ngân sách token** chặn trần đã đo ở § 1.4 |
| **17.5** | Model LLM-facing rút gọn (không có `candidate_table_ids`) + payload `response_format` json_schema | `planning/llm_contracts.py` | Test: round-trip sang `FinancialQueryPlan` sau khi tiêm table id |
| **17.6** | LLM planner: parse → repair 1 lần → validate → abstain; thêm `llm_unavailable`, `llm_invalid_json`, `llm_plan_invalid` vào `PlanAbstainCode` | `planning/llm_planner.py` | Không bao giờ trả plan chưa qua `validate_plan_semantics` |
| **17.7** | Router A1, có ghi lại nhánh nào sinh ra plan | `planning/plan_router.py` | |
| **17.8** | Đo: CLI `evaluate-llm-plans`, chế độ replay-cache (offline) và live | `planning/llm_evaluation.py`, `planning/cli.py` | Report ghi danh tính mô hình |

### Thứ tự thực hiện

```
17.1 ─┬─> 17.2 ──> 17.3 ─┐
      ├─> 17.4 ──────────┼─> 17.6 ──> 17.7 ──> 17.8
      └─> 17.5 ──────────┘
```

---

## 4. Định nghĩa hoàn thành

- [ ] Toàn bộ test chạy **không cần server**; `pytest` không phát ra network call nào.
- [ ] `false_plan_rate` của router trên 800 case abstain-expected = **0,0** — giữ nguyên KPI cứng
      của Ngày 16.
- [ ] Router **không bao giờ làm tệ đi**: `exact_match_rate` trên 1.400 case ≥ mức 1,000 của rule
      planner.
- [ ] `operation_accuracy` và `invalid_json_rate` của LLM planner đứng một mình được **báo cáo**
      (không đặt ngưỡng cứng — chưa có mô hình nào để đo trước, đặt ngưỡng bây giờ là bịa).
- [ ] Sau đúng 1 lượt repair, mọi đầu ra hỏng đều thành abstain có mã, không bao giờ thành plan.
- [ ] Report ghi danh tính mô hình (tên, lượng tử hóa, host, temperature, seed) để tái lập được.
- [ ] `data/processed/` và `normalization/` không đổi một byte — `dataset_fingerprint` giữ nguyên.

## 5. Rủi ro

| # | Rủi ro | Giảm thiểu |
| --- | --- | --- |
| R1 | Không có mô hình → không ra được số live trong hôm nay | Cổng nghiệm thu đặt ở đường offline; lần chạy live là artifact tùy chọn, tách rời |
| R2 | `temperature 0` không tái lập được | Cache phản hồi khóa theo `sha256(prompt)` + danh tính mô hình (tiền lệ Ngày 9) |
| R3 | Prompt tràn cửa sổ 4.096 ở p90 | Test ngân sách token trong 17.4; chốt B1 để bỏ hẳn row label |
| R4 | Câu hỏi độc hại chèn lệnh vào prompt | Ngày 19 dựng sandbox; Ngày 17 đã chặn sẵn bằng việc LLM không được sinh `table_id` và mọi plan phải qua validator |
| R5 | **Đầu tư quá tay vào một đường đo được là 0 headroom** (§ 1.1) | Giới hạn Ngày 17 ở router + phép đo. **Không tinh chỉnh prompt theo gold70** — chỉ có 19 câu, tinh chỉnh là overfit chứ không phải cải thiện |
| R6 | 7 câu `(1,1,2)` bị đẩy sang LLM trong khi luật xử lý được | Ghi vào ADR 0006 là việc của Ngày 18+, **không** lén sửa `_infer_operation` trong Ngày 17 |
