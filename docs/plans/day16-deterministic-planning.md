# Kế hoạch Ngày 16 — Deterministic parsing và ontology

> Trạng thái: **đã hoàn tất** (xem [plan.md § Ngày 16](../../plan.md),
> [README.md § Day 16](../../README.md) và [ADR 0005](../decisions/0005-operation-coverage-gaps.md)
> để biết kết quả thực tế). Viết ngày 2026-08-15 sau khi Ngày 15 hoàn tất (`6c82188`), như tài
> liệu thiết kế trước khi triển khai. Mọi con số ở mục 1 đo trực tiếp trên release khóa
> `422df141c935…`, gold70 (`data/qa/retrieval-gold-v1.jsonl`) và bộ 1.400 entity case
> (`data/qa/entity-cases-v1.jsonl`) bằng chính `entity_parser.py` hiện tại. Quyết định A/B/C ở mục
> 2 chốt lại ở ADR 0005 (A1 triển khai, B2/C2 hoãn có chủ đích) — coi ADR 0005, plan.md và README
> là nguồn sự thật, không phải bảng phác thảo ở đây. Từ vựng operation ở mục 1.8 (§16.4) không được
> cài do đo được 0 lần xuất hiện trong dữ liệu thật; nhiệm vụ 16.2 (ADR) không sinh thêm code, chỉ
> ghi quyết định.

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Entity parser (Ngày 10) | `planning/entity_parser.py`, `entity_contracts.py` | ✅ exact-match 1,0 trên case set |
| Bộ case entity | `data/qa/entity-cases-v1.jsonl` — 1.400 case, 14 template | ✅ nhưng **không có nhãn operation** |
| Schema plan (Ngày 15) | `planning/plan_contracts.py` — 8 operation | ✅ đóng băng |
| Semantic validator (Ngày 15) | `planning/plan_validator.py` — 10 error code | ✅ |
| Từ điển metric | `normalization/metrics.py` — **218 alias → 56 canonical** | ⚠️ thiếu hẳn họ ngân hàng |
| Registry công ty | `normalization/companies.py` — **100 mã** | ✅ |
| Truy hồi mặc định | BM25 v4, Recall@10 `0,9143` | ✅ dùng để cấp `candidate_table_ids` |
| `data/official/test_questions.json` | — | ❌ **vẫn chưa tồn tại** |

Ngày 16 **không** đụng tới LLM (Ngày 17) và **không** thực thi số (Ngày 18). Phạm vi đúng một
thứ: một hàm thuần `QueryEntities → FinancialQueryPlan | abstain`, cộng phần từ vựng và đo lường
để biết nó đúng bao nhiêu phần trăm.

---

## 1. Chốt chặn phải đo trước khi viết luật

### 1.1. Lỗi năm trần trụi: 20 % câu gold bị mất một kỳ

`_YEAR_RE = r"năm\s+((?:19|20)\d{2})"` bắt buộc có chữ "năm" đứng trước. Nhưng cách diễn đạt phổ
biến nhất trong gold70 là *"từ năm 2019 **đến 2020**"* / *"giữa năm 2016 **và 2017**"* — năm thứ
hai trần trụi và bị bỏ qua.

| Đo trên gold70 | Số câu | Tỷ lệ |
| --- | ---: | ---: |
| Có năm trần trụi sau `đến`/`và`/`giữa` mà parser **không** bắt được | **14** | **20,0 %** |

Hậu quả không phải "thiếu dữ liệu" mà là **sai operation**: câu tăng trưởng hai kỳ chỉ còn một kỳ
sẽ được luật xếp thành `lookup` và trả về một con số hoàn toàn khác. Đây là lỗi rẻ nhất và có giá
trị đúng-sai cao nhất trong cả ngày 16.

### 1.2. Taxonomy `intent` của gold70 ≠ taxonomy `operation` của Ngày 15

Hai không gian tên khác nhau, đang bị đặt cùng chữ `compare`. Phân rã 23 câu `intent=compare`:

| Hình dạng `(company, period, metric)` | Số câu | Operation Ngày 15 tương ứng |
| --- | ---: | --- |
| `(1, 2, 1)` | 10 | **`difference`** — không phải `compare` |
| `(1, 1, 0)` | 8 | không có (không rút được metric) |
| `(1, 1, 1)` | 3 | `lookup` |
| `(1, 1, 2)` | 2 | không có (hai metric) |

**Không một câu nào** trong 23 câu khớp `compare` của Ngày 15 (`metric_a` + `metric_b`, một kỳ).
Kế hoạch phải viết bảng ánh xạ tường minh `intent → operation` và không được suy diễn từ tên.

### 1.3. Lỗ hổng operation: so sánh chéo công ty không biểu diễn được

Template `two_companies` (100/1.400 case) sinh ra *"So sánh Các khoản giảm trừ doanh thu giữa GEG
và GEX năm 2017"* — hình dạng `(2 company, 1 period, 1 metric)`.

| Operation Ngày 15 | Có khớp không? |
| --- | --- |
| `rank` | Hình dạng khớp nhưng **bắt buộc `top_k`** và ngữ nghĩa là xếp hạng top-k, không phải đặt cạnh nhau |
| `average` / `sum` | Hình dạng khớp nhưng ngữ nghĩa là gộp một con số, không phải hai con số |
| còn lại | `companies` phải đúng 1 |

Đây là lỗ hổng thật trong bộ 8 operation, cần một quyết định có văn bản chứ không phải một
`rank(top_k=2)` lách luật. → ADR 0005.

### 1.4. "Biến động đầu năm – cuối năm" là hai **hàng**, không phải hai **kỳ**

12/23 câu `intent=growth` chỉ có một kỳ. Một phần là lỗi 1.1, phần còn lại là loại câu thật sự
một-kỳ: *"Tính biến động LNST chưa phân phối **trong năm** của VSC năm 2019"*. Giá trị đầu kỳ và
cuối kỳ nằm ở hai hàng của cùng một bảng thuyết minh:

| Nhãn hàng | Số bảng chứa | Số cell |
| --- | ---: | ---: |
| `Số dư cuối năm` | 6.820 | 51.328 |
| `Số dư đầu năm` | 6.468 | 48.962 |
| `Số cuối năm` | 3.625 | 29.515 |
| `Số đầu năm` | 3.291 | 27.635 |

`FinancialQueryPlan` hiện không có cách nào diễn đạt "hai hàng cùng bảng, cùng kỳ". Phải quyết
định: mở rộng schema, hay abstain có mã lỗi. → ADR 0005.

### 1.5. 14,3 % case sinh ra kỳ mà schema plan từ chối thẳng

`FinancialQueryPlan` ràng `periods` theo `^\d{4}$`. Parser thì hợp lệ khi trả về ngày và quý:

| Template | Số case | Giá trị parser trả | Plan nhận? |
| --- | ---: | --- | --- |
| `date_lookup` | 100 | `2025-12-31` | ❌ |
| `quarter_lookup` | 100 | `2018-Q4` | ❌ |
| còn lại (sạch) | 600 | `2017` | ✅ |

Trên gold70 thì 91/91 giá trị kỳ đều là `YYYY` nên vấn đề chưa lộ ra ở Ngày 15. Ngày 16 là lúc
đầu tiên hai lớp này gặp nhau thật.

### 1.6. Từ điển metric thiếu hẳn họ ngân hàng

20/70 câu gold70 bị `metric_unknown`. Đo trên release, các khái niệm ngân hàng phổ biến nhất có
**0 %** phủ canonical:

| Nhãn hàng thô | Số bảng | Đã canonical hoá |
| --- | ---: | ---: |
| `chứng khoán đầu tư` | 4.242 | **0** |
| `cho vay khách hàng` | 4.245 | **0** |
| `tiền gửi của khách hàng` | 3.582 | **0** |
| `chứng khoán kinh doanh` | 3.563 | **0** |
| `tiền gửi có kỳ hạn` | 3.050 | **0** |
| `dự phòng rủi ro tín dụng` | 888 | **0** |

Ngoài ra là các viết tắt parser chưa biết: `LCTT`, `LNST chưa phân phối`, `chi phí thuế hiện
hành`, `chi phí thuế thu nhập doanh nghiệp`.

### 1.7. Ràng buộc cứng: mở rộng từ điển **không được** đụng vào release

`cells.row_label_canonical` được tính lúc build dataset (`normalization/service.py` →
`data/dataset_builder.py`) và đóng băng trong `cells.parquet`. Thêm alias vào `METRIC_ALIASES`
**không** hồi tố lên release đã khóa; muốn hồi tố phải build lại → đổi `dataset_fingerprint` →
vô hiệu toàn bộ baseline ghim trong `retrieval/reference.py`, mọi `question_id` trong gold70 và
mọi artifact đánh giá của Ngày 8–14.

Vì vậy Ngày 16 mở rộng từ vựng **chỉ ở phía câu hỏi** (lexicon của entity parser), rồi để nhánh
`raw_text` của `metric_selector` (ADR 0004) gánh phần định vị hàng ở Ngày 18. Đây chính là lý do
ADR 0004 chọn Option C.

### 1.8. Toàn bộ từ vựng mục tiêu của plan.md xuất hiện **0 lần** trong dữ liệu hiện có

| Cụm từ | gold70 | entity-cases |
| --- | ---: | ---: |
| `tăng bao nhiêu` | 0 | 0 |
| `gấp mấy lần` / `gấp bao nhiêu lần` | 0 | 0 |
| `biên` | 0 | 0 |
| `bình quân` / `trung bình` | 0 | 0 |
| `cao nhất` / `thấp nhất` | 0 | 0 |
| `tỷ trọng` | 0 | 0 |

Nghĩa là các checkbox từ vựng của plan.md là **dự phòng cho bộ đề thật**, không phải phản ứng với
phân phối đã quan sát. Hệ quả về phương pháp: cài chúng thì được (rẻ, không hại), nhưng **không**
được để chúng lái thiết kế, và phải đo trên một bộ case riêng, báo cáo tách khỏi số đo trên phân
phối đã quan sát — trộn hai thứ lại sẽ tạo ra một con số "độ chính xác" vô nghĩa.

### 1.9. Trần lý thuyết của "≥ 60 % câu đơn giản"

| Hình dạng trên gold70 | Số câu | Lập được plan? |
| --- | ---: | --- |
| `(1 company, 1 period, 1 metric)` | 22 | ✅ `lookup` |
| `(1 company, 2 period, 1 metric)` | 21 | ✅ `difference`/`growth_rate` |
| `(1, ·, 0)` — không rút được metric | 20 | ❌ phải abstain |
| `(1, ·, 2)` — nhiều metric | 7 | ❌ phải abstain |

Trần = **43/70 = 61,4 %**. Nếu lấy gold70 làm mẫu số thì mục tiêu "≥ 60 %" gần như trùng với
trần, tức là chỉ đạt được khi parser gần như không sai câu nào — một mục tiêu vô tình khắc
nghiệt. Do đó nhiệm vụ đầu tiên của ngày 16 là **định nghĩa mẫu số** trước khi đo.

---

## 2. Quyết định phải chốt trước khi code (ADR 0005)

| # | Câu hỏi | Phương án |
| --- | --- | --- |
| A | So sánh chéo công ty `(≥2 company, 1 period, 1 metric)` | (A1) thêm operation `compare_companies`; (A2) nới `compare` cho phép nhánh thứ hai; (A3) abstain |
| B | "Biến động đầu năm–cuối năm" `(1 company, 1 period, 1 metric)` | (B1) thêm `expected_unit`-neutral operation `period_change` với hai hàng; (B2) abstain có mã lỗi |
| C | Kỳ dạng ngày/quý (`2025-12-31`, `2018-Q4`) | (C1) nới `periods` của schema; (C2) chiếu về `YYYY` và ghi lại bằng cờ; (C3) abstain |

Khuyến nghị mở đầu (sẽ khẳng định lại bằng số trước khi chốt trong ADR): **A1**, **B2**, **C2**.
Lý do: A1 giữ ngữ nghĩa `rank` trong sạch với chi phí một operation mới; B2 tránh mở schema cho
một cơ chế định vị hàng mà compiler Ngày 18 chưa hề có; C2 giữ được câu hỏi thay vì bỏ, mà không
phá `^\d{4}$` — thứ mà `cells.period` cũng chưa thống nhất (đã ghi nhận ở Ngày 15).

Bất kỳ phương án nào sửa `plan_contracts.py` đều phải chạy lại đủ 40 test đơn vị + 3 property
test + 20 golden case của Ngày 15 và cập nhật `tests/golden/plans/manifest.json`.

---

## 3. Nhiệm vụ

### 16.1. Định nghĩa "câu đơn giản" và dựng bộ case cấp plan

- Viết định nghĩa vận hành: đúng 1 công ty, đúng 1 metric, 1–2 kỳ, không cờ ambiguity.
- Mở rộng `entity_cases.py` sinh thêm trường `expected_operation` (hoặc `null` = phải abstain),
  giữ nguyên 14 template cũ để không phá case set đã ghim, thêm template mới cho từ vựng ở 1.8.
- Ghi ra `data/qa/plan-cases-v1.jsonl` kèm SHA-256 như `entity_case_set_sha256`.
- **Đầu ra:** mẫu số được viết ra giấy trước khi có bất kỳ số đo nào.

### 16.2. ADR 0005 — độ phủ operation

- Chốt A/B/C ở mục 2, kèm số đo tần suất từng hình dạng.
- Nếu chọn A1, cập nhật `PlanOperation`, bảng arity trong `plan_validator.py`, `_expected_ok`
  trong property test và golden case.

### 16.3. Sửa lỗi kỳ trong entity parser (TDD)

- RED: test cho `"từ năm 2019 đến 2020"`, `"giữa năm 2016 và 2017"`, `"2019-2020"`.
- GREEN: cho `_YEAR_RE` bắt năm trần trụi khi đứng sau từ nối kỳ, vẫn không bắt năm nằm trong
  tên công ty hay số hiệu thuyết minh.
- Chạy lại toàn bộ 1.400 entity case: **không được** hạ exact-match của 14 template cũ.

### 16.4. Mở rộng từ vựng phía câu hỏi (không đụng release)

- Thêm họ ngân hàng ở 1.6, các viết tắt (`LCTT`, `LNST`, `DT thuần`, `TSCĐ`, `HĐKD`).
- Thêm từ vựng operation ở 1.8 vào một lexicon **riêng** (`_OPERATION_LEXICON`), không trộn vào
  `METRIC_ALIASES`.
- Test hồi quy: `dataset_fingerprint` và `retrieval/reference.py` không đổi một byte.

### 16.5. Rule planner

- `planning/rule_planner.py`: `build_plan(entities, *, candidate_table_ids) -> RulePlanResult`.
- `candidate_table_ids` được **tiêm vào** từ truy hồi, không tự gọi BM25 — giữ hàm thuần, test
  được, và giữ ranh giới module như ADR 0001.
- Thứ tự luật xác định: operation → arity → gán metric vào đúng vai (`metric` /
  `metric_a`+`metric_b` / `numerator`+`denominator`) → `expected_unit`.
- Kết quả luôn đi qua `validate_plan_semantics`; plan không sạch **không bao giờ** được trả về.

### 16.6. Abstain có mã lỗi thay vì đoán bừa

- `RulePlanResult` = `plan | None` + `tuple[PlanAbstainCode, ...]`, theo đúng khuôn
  code-và-field của `AmbiguityCode` và `PlanErrorCode`.
- Mã tối thiểu: `operation_unknown`, `metric_role_unassignable`, `multi_metric_unsupported`,
  `period_grammar_unsupported`, `entity_ambiguous`.
- Mọi câu ở 1.9 rơi vào nhóm ❌ phải ra đúng một mã, không được ra plan.

### 16.7. Harness đánh giá

- Thêm lệnh `evaluate-plans` vào `planning/cli.py`, cùng khuôn với `evaluate-entities`.
- Hai số đo **không trộn**: (a) trên `plan-cases-v1` — số chính, được phép lặp; (b) trên gold70 —
  **held-out**, chạy đúng một lần, báo cáo nguyên trạng.
- Báo cáo tách riêng: độ chính xác operation, độ chính xác từng trường, tỷ lệ abstain, tỷ lệ
  abstain-đúng (câu không lập plan được mà có abstain) và abstain-sai.

---

## 4. Thứ tự thực thi

```
16.1 ──┬─> 16.2 ──> 16.5 ──> 16.6 ──> 16.7
       └─> 16.3 ──┘
           16.4 ──┘
```

16.1 và 16.2 phải xong trước 16.5: không thể viết luật khi bộ operation còn có lỗ và mẫu số chưa
được định nghĩa. 16.3 và 16.4 độc lập nhau, chạy song song được.

## 5. Định nghĩa hoàn thành

- [ ] ADR 0005 ở trạng thái Accepted với số đo kèm theo.
- [ ] `data/qa/plan-cases-v1.jsonl` có SHA-256 ghim và nhãn `expected_operation`.
- [ ] Lỗi 1.1 có test hồi quy; 14/70 câu gold70 lấy đúng cả hai kỳ.
- [ ] `dataset_fingerprint` không đổi; mọi baseline ghim của Ngày 8–14 vẫn xanh.
- [ ] Rule planner đạt **≥ 60 %** operation-accuracy trên tập "câu đơn giản" của 16.1.
- [ ] **0** trường hợp trả về plan cho câu lẽ ra phải abstain (chỉ tiêu cứng, không đánh đổi).
- [ ] Báo cáo held-out gold70 chạy một lần, ghi nguyên vào README.
- [ ] Lint, format, type check, toàn bộ test xanh.

## 6. Rủi ro

| Rủi ro | Ảnh hưởng | Giảm thiểu |
| --- | --- | --- |
| Từ vựng 1.8 không xuất hiện trong đề thật | Công viết luật lãng phí | Đo tách bạch; giữ lexicon operation ở một file riêng để gỡ được |
| Mẫu số "câu đơn giản" bị chọn để làm đẹp số | Chỉ số vô nghĩa | Viết định nghĩa ở 16.1 **trước** khi chạy đo lần đầu |
| Sửa `plan_contracts.py` phá golden case Ngày 15 | Mất tính tái lập | Chạy lại đủ 20 golden case + property test trong cùng commit |
| Mở rộng alias vô tình kéo theo build lại release | Vỡ mọi baseline | Test hồi quy fingerprint ở 16.4 |
| gold70 chỉ có 3 intent, không phủ 8 operation | Held-out yếu | Nêu rõ giới hạn trong README; không suy rộng con số |
| `test_questions.json` vẫn chưa có | Toàn bộ ngày 16 vẫn là dự phòng | Giữ mọi luật ở dạng dữ liệu (bảng ánh xạ), sửa được không cần sửa logic |
