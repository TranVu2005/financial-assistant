# Kế hoạch Ngày 15 — FinancialQueryPlan

> Trạng thái: **đã hoàn tất** (xem [plan.md § Ngày 15](../../plan.md) và
> [README.md § Day 15](../../README.md) để biết kết quả thực tế). Viết ngày 2026-08-14 sau khi
> Ngày 14 hoàn tất (`8516ea9`), như tài liệu thiết kế trước khi triển khai. Quyết định `compare`
> (hai metric cùng company/period) và `average`/`sum` (đúng một chiều biến thiên) được chốt lúc
> triển khai, cụ thể hơn phác thảo ở đây — coi `plan.md`/README/ADR 0004 là nguồn sự thật.

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Release khóa | `data/processed/release_v2_422df141c935/` | ✅ |
| Retrieval mặc định | BM25 v4, Recall@10 `0,9143`, F2@R `0,4836` | ✅ đạt cổng Recall |
| Entity parser (Day 10) | `planning/entity_parser.py`, `entity_contracts.py` | ✅ exact-match 1.0 |
| Từ vựng metric | `normalization/metrics.py` — **56 canonical** | ✅ |
| Từ vựng unit | `VND`, `VND_thousand`, `VND_million`, `VND_billion`, `percent`, `ratio` | ✅ |
| Lớp dựng DataFrame | `data/table_frame.py::TableFrame` (`grid` + `values`) | ✅ |
| Nhánh lỗi | `PlanningError` / `PlanningInputError` / `PlanningArtifactError` | ✅ đã có |
| `data/official/test_questions.json` | — | ❌ **chưa tồn tại** |

Ngày 15 **không** đụng tới LLM (đó là Ngày 17). Phạm vi đúng ba thứ: schema, error codes,
semantic validator — cộng phần đo lường bắt buộc ở mục 1 trước khi đóng băng schema.

---

## 1. Chốt chặn phải đo trước khi đóng băng schema

### 1.1. Chỉ 60 % câu hỏi có metric giải được tới hàng canonical

`FinancialQueryPlan.metrics` là `list[str]` canonical, và compiler định vị hàng trong bảng qua
`cells.row_label_canonical`. Đo trên 70 câu gold hiện có (parse bằng chính entity parser Day 10,
đối chiếu với canonical label thật của bảng gold):

| Kết quả | Số câu | Tỷ lệ |
| --- | ---: | ---: |
| Mọi metric giải được tới canonical | 42 | 60,0 % |
| **Parser không rút được metric nào** | **20** | **28,6 %** |
| Có metric nhưng không cái nào khớp bảng gold | 6 | 8,6 % |
| Khớp một phần | 2 | 2,9 % |

Nền dữ liệu bên dưới:

| Đại lượng | Toàn corpus | Riêng 91 bảng gold |
| --- | ---: | ---: |
| Bảng có ≥ 1 canonical row label | 25.091 / 146.011 (17,2 %) | 56 / 91 (**61,5 %**) |
| Cell có `row_label_canonical` / có `row_label_raw` | 5,93 % | 16,98 % |

> **Ngày 14 chỉ sửa nhánh truy hồi, không sửa nhánh thực thi.** Task 14.2 giữ `row_label_raw` vào
> *text* của document BM25 nên retrieval tìm được bảng; nhưng `metric_labels` (canonical) không
> đổi, và đó mới là thứ compiler dùng để định vị hàng. Cùng một lỗ hổng normalization giờ chặn
> Tuần 3 ở một tầng khác. **Không được giả định Ngày 14 đã xử lý xong vấn đề này.**

### 1.2. Gần 30 % câu gold không có đáp án số

20 câu mà parser không rút được metric phần lớn là câu dạng `notes`, ví dụ:

- *"Tra cứu đồng thời bốn bảng thuyết minh về danh sách công ty con của HPG năm 2017."*
- *"Tra cứu ba bảng thuyết minh phân tích thời hạn và rủi ro của tài sản, nợ tài chính MSB 2018."*
- *"So sánh kết quả kinh doanh riêng và hợp nhất của IJC năm 2017."*

Đây **không phải lỗi parser** — chúng thật sự không có một con số làm đáp án. Nhưng
[plan.md § 2.3](../../plan.md) quy định `SubmissionItem.answer: float` và *"`abstained`, `error`
hoặc `answer=None` là lỗi chặn phát hành"*.

Cần phân biệt rạch ròi, và Ngày 15 phải ghi rõ:

- `data/qa/retrieval-gold-v1.jsonl` là gold **đánh giá truy hồi**, cố ý gồm cả câu dạng liệt kê.
  Nó **không phải** tập câu hỏi nộp bài.
- Tập nộp bài thật là `data/official/test_questions.json` — **chưa tồn tại trong repo**.

### 1.3. Rủi ro: đóng băng 8 operation khi chưa thấy câu hỏi thật

Ngày 15 chốt vĩnh viễn `Literal[...]` 8 operation, nhưng chưa ai đọc
`data/official/test_questions.json`. Nếu tập thật có nhiều câu ngoài 8 phép này (ví dụ tỷ trọng
cơ cấu, CAGR nhiều kỳ, xếp hạng có điều kiện), schema sẽ phải mở lại giữa Tuần 3.

Đây là rủi ro lịch trình có thật, không phải giả định — xem Task 15.7.

### 1.4. Ngữ pháp `period` chưa nhất quán trong chính dữ liệu

`cells.period` tồn tại **hai dạng song song**: năm trần (`"2024"`, 66.394 cell) và ngày đầy đủ
(`"2024-12-31"`, 25.830 cell). Schema phải chốt một dạng canonical và validator phải từ chối dạng
còn lại, nếu không compiler sẽ lặng lẽ trượt khớp ở các bảng bảng cân đối.

---

## 2. Các task

### Task 15.1 — Đo phủ trước, thiết kế sau

- [ ] Viết script đo (một lần, không commit vào `src/`) trả lời: với 56 metric canonical × 8
      operation, bao nhiêu phần trăm bảng trong release đủ dữ liệu để thực thi từng operation?
- [ ] Tách riêng: (a) bảng có canonical label, (b) bảng chỉ có raw label, (c) bảng không nhãn.
- [ ] Đo tỷ lệ cell có `value_numeric` theo nhóm (toàn corpus hiện **42,3 %**).
- [ ] **DoD:** một bảng số đưa vào `docs/plans/day15-financial-query-plan.md` mục này, đủ để
      quyết định 15.2 mà không đoán.

### Task 15.2 — ADR 0004: chiến lược định vị metric

Quyết định lớn nhất của Ngày 15. Ba phương án:

| | Phương án | Phủ | Nhược |
| --- | --- | --- | --- |
| **A** | Chỉ canonical | ~60 % câu | Bỏ 40 % câu ngay từ schema; nợ lại toàn bộ cho normalization |
| **B** | Canonical, fallback khớp `row_label_raw` đã chuẩn hoá | Cao hơn nhiều | Cần quy tắc khử nhập nhằng tất định khi nhiều hàng cùng khớp |
| **C** | `metric_selector` hai nhánh: `canonical` \| `raw_pattern`, validator ép mỗi selector phải giải được trên ≥ 1 `candidate_table_id` | Cao nhất, tường minh nhất | Schema phức tạp hơn; LLM Ngày 17 phải học chọn nhánh |

**Khuyến nghị: C.** Lý do: nó biến "không định vị được metric" từ lỗi runtime im lặng của
compiler thành lỗi validation *trước* thực thi — đúng mục tiêu Ngày 15 (*"mọi plan không hợp lệ bị
từ chối trước execution"*). Nhánh `raw_pattern` cũng là nơi duy nhất ghi nhận được rằng nhãn đã
tồn tại trong nguồn nhưng chưa canonical hoá, thay vì giả vờ nó không tồn tại.

- [ ] Viết `docs/decisions/0004-metric-locator-strategy.md` kèm số đo từ 15.1.
- [ ] **DoD:** quyết định có số đo hậu thuẫn; nêu rõ phương án bị loại và vì sao.

### Task 15.3 — JSON examples cho 8 operation (viết TRƯỚC schema)

[plan.md § Task D](../../plan.md) yêu cầu *"viết gold plans và invalid cases trước"*. Làm đúng thứ
tự đó: ví dụ JSON là đặc tả, không phải minh hoạ sinh sau.

- [ ] `tests/golden/plans/valid/<operation>.json` — ít nhất một ví dụ mỗi operation, lấy từ câu
      hỏi thật trong gold70 (ghi `question_id` nguồn), không bịa.
- [ ] `tests/golden/plans/invalid/*.json` — mỗi file một vi phạm **đơn lẻ**, kèm error code kỳ
      vọng. Tối thiểu 12 case: thiếu field bắt buộc theo operation, thừa field bị cấm, sai arity,
      metric ngoài 56 canonical, unit ngoài 6 giá trị, period sai ngữ pháp, `top_k` khi không phải
      `rank`, `numerator/denominator` khi không phải `ratio`, `candidate_table_ids` rỗng/quá dài,
      table_id không thuộc release, operation ngoài whitelist, `expected_unit` mâu thuẫn operation.
- [ ] **DoD:** mọi file valid parse được và mọi file invalid bị từ chối, sau khi 15.4/15.5 xong.

### Task 15.4 — Schema và error codes

- [ ] `planning/plan_contracts.py`, dùng lại `_FrozenModel` (`extra="forbid"`, `frozen=True`),
      `NonEmptyString`, `_canonical_tuple` từ `retrieval/contracts.py` — **không** tạo base model
      mới.
- [ ] `tuple[...]` thay `list[...]` cho mọi trường tập hợp, để plan hashable và so sánh byte được
      như mọi contract khác trong repo.
- [ ] `PlanErrorCode = Literal[...]` theo đúng tiền lệ `AmbiguityCode` (Day 10): mã ổn định,
      sắp xếp, unique — không dùng chuỗi tự do.
- [ ] Chốt **ngữ pháp `period`**: đề xuất năm trần `"YYYY"` là canonical duy nhất trong plan;
      validator từ chối `"YYYY-MM-DD"`; việc khớp `2024` với cell `2024-12-31` là trách nhiệm của
      compiler (Ngày 18), không phải của plan.
- [ ] `candidate_table_ids`: bắt buộc, `1 <= len <= 12`, **không giả định phần tử đầu là đúng** —
      F2@R hiện `0,4836`, tức quá nửa top-R là nhiễu. Cấu hình `retrieval.final_top_k` hiện là 6.
- [ ] `expected_unit` chỉ nhận 6 giá trị canonical đã đo được trong release.
- [ ] **DoD:** `mypy` 0 lỗi mới; mọi file trong `tests/golden/plans/valid/` parse được.

### Task 15.5 — Semantic validator

Bảng arity bắt buộc — validator ép đúng bảng này, mỗi vi phạm một error code riêng:

| Operation | companies | periods | metrics | Field riêng | `expected_unit` |
| --- | :---: | :---: | :---: | --- | --- |
| `lookup` | 1 | 1 | 1 | — | tiền tệ |
| `compare` | ≥ 1 | ≥ 1 | 1 | (xem ghi chú) | tiền tệ |
| `difference` | 1–2 | 1–2 | 1 | đúng 2 toán hạng | tiền tệ |
| `growth_rate` | 1 | **2** | 1 | — | `percent` |
| `ratio` | 1 | 1 | — | `numerator_metric` + `denominator_metric` bắt buộc | `ratio` hoặc `percent` |
| `average` | ≥ 1 | ≥ 1 | 1 | tổng toán hạng ≥ 2 | tiền tệ |
| `sum` | ≥ 1 | ≥ 1 | 1 | — | tiền tệ |
| `rank` | ≥ 2 | 1 | 1 | `top_k` bắt buộc, `1 <= top_k <= len(companies)` | tiền tệ |

- [ ] Quy tắc chung: field không thuộc operation phải **vắng mặt**, không phải để `None` — bắt
      bằng `extra="forbid"` cộng validator riêng cho `top_k` / `numerator_metric` /
      `denominator_metric`.
- [ ] Validator kiểm tra mọi `metric` (và `numerator/denominator`) thuộc 56 canonical, hoặc giải
      được qua nhánh đã chốt ở ADR 0004.
- [ ] Validator kiểm tra mọi `candidate_table_ids` tồn tại trong `tables.parquet` của release.
- [ ] **Ghi chú `compare` — câu hỏi thiết kế còn mở:** `SubmissionItem.answer` là **một** `float`,
      nhưng "so sánh A với B" tự nhiên trả về hai số. Phải chốt ở Ngày 15: hoặc `compare` sinh ra
      một đại lượng dẫn xuất (hiệu? tỷ lệ?), hoặc nó bị loại khỏi tập operation sinh đáp án nộp
      bài và chỉ dùng nội bộ. **Không để mở sang Ngày 18** — compiler không tự quyết được.
- [ ] **DoD:** mọi file trong `tests/golden/plans/invalid/` bị từ chối với **đúng** error code đã
      ghi trong file, không phải chỉ "bị từ chối".

### Task 15.6 — Property tests

- [ ] Dùng `hypothesis` (đã có trong dev deps) sinh tổ hợp field ngẫu nhiên; bất biến: **không tồn
      tại plan nào vừa qua được validator vừa vi phạm bảng arity 15.5**.
- [ ] Bất biến round-trip: `model_validate(model_dump(mode="json"))` bằng chính nó với mọi plan hợp lệ.
- [ ] Bất biến tất định: cùng input → cùng chuỗi error code, sắp xếp ổn định.
- [ ] **DoD:** `pytest -q tests/unit/planning tests/golden/plans` xanh; không giảm số test hiện có.

### Task 15.7 — Chốt phạm vi và ghi rủi ro

- [ ] Ghi rõ trong README: gold truy hồi ≠ tập câu hỏi nộp bài; câu dạng liệt kê `notes` nằm ngoài
      phạm vi `FinancialQueryPlan`.
- [ ] Ghi vào [plan.md § 14 Rủi ro](../../plan.md): 8 operation đang được đóng băng **trước khi**
      thấy `data/official/test_questions.json`; nếu tập thật lệch phân bố, chi phí mở lại schema
      rơi vào Ngày 17–18.
- [ ] Nêu rõ nợ kỹ thuật normalization từ mục 1.1 — đây là ứng viên hàng đầu cho Ngày 16
      (*"mở rộng dictionary alias"*), giờ đã có số đo cụ thể để nhắm.
- [ ] **DoD:** không còn giả định ngầm nào về phạm vi planner nằm ngoài văn bản.

---

## 3. Thứ tự thực thi

```
15.1 đo phủ ──► 15.2 ADR 0004 ──┐
                                 ├──► 15.4 schema ──► 15.5 validator ──► 15.6 property tests
15.3 JSON examples ──────────────┘                                              │
                                                                                ▼
                                                                        15.7 phạm vi + rủi ro
```

15.1 và 15.3 làm song song được. 15.4 **không** bắt đầu trước khi ADR 0004 chốt — chọn sai chiến
lược định vị metric thì toàn bộ validator phải viết lại.

## 4. Định nghĩa hoàn tất (toàn ngày)

- [ ] ADR 0004 chốt chiến lược định vị metric, có số đo hậu thuẫn.
- [ ] 8 operation có ví dụ JSON hợp lệ lấy từ câu hỏi thật, ≥ 12 case invalid có error code.
- [ ] Mọi plan không hợp lệ bị từ chối **trước** execution, với đúng error code (mục tiêu gốc của
      Ngày 15).
- [ ] Câu hỏi thiết kế `compare` → scalar đã có câu trả lời dứt khoát.
- [ ] Ngữ pháp `period` canonical đã chốt và được validator ép.
- [ ] `pytest -q` 0 fail; `ruff check .` sạch; `mypy` 0 lỗi mới (baseline hiện tại là **0**).
- [ ] `git diff --check` sạch.

## 5. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
| --- | --- | --- |
| 40 % câu không giải được metric tới canonical | 15.1 xác nhận số Ngày 14 vẫn đúng | ADR 0004 chọn phương án B/C; nợ phần còn lại cho Ngày 16 với số đo cụ thể |
| Đóng băng 8 operation khi chưa có `test_questions.json` | Tập thật xuất hiện phép toán thứ 9 | Giữ `PlanErrorCode` và bảng arity ở **một** chỗ để mở rộng rẻ; không rải luật ra nhiều file |
| `compare` không sinh được scalar đơn trị | Không viết nổi ví dụ JSON hợp lệ ở 15.3 | Đó chính là tín hiệu để loại `compare` khỏi tập sinh đáp án nộp bài — quyết định ở 15.5, không hoãn |
| Schema phình vì cố phủ mọi trường hợp | Nhiều field optional chỉ dùng cho 1 operation | Giữ nguyên tắc: field không thuộc operation phải **vắng mặt**; nếu quá 3 field riêng thì tách operation ra contract con |
| Nhầm gold truy hồi thành gold planner | Ai đó chạy planner trên đủ 70 câu rồi báo tỷ lệ lỗi cao | 15.7 ghi rõ ranh giới; ví dụ JSON chỉ lấy từ câu thật sự có đáp án số |
