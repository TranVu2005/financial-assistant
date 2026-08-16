# Ngày 23 — Coverage của submission và hợp đồng bảng bằng chứng

> **Trạng thái:** §3 Bước 1 (khớp chỉ tiêu theo nhãn thô) đã cài đặt và đo xong (2026-08-16):
> **28 → 55/1.012 answered (+27, +96 %), 0 câu mất, accuracy gold120 không đổi (0,846)**. Bước 2
> (nối 5 operation) cài đặt một phần cùng ngày: sửa 2 bug thật (compare_companies mis-route,
> average/sum không biến thiên theo company) và thêm ratio/average/sum — kết quả ròng
> **55 → 53 answered** (−2 = xoá đúng 2 câu trả lời sai tự tin đã phát hiện, 0 câu mới trả lời
> được do execution vẫn trượt canonical-only ở đúng gốc Bước 1, xem chi tiết trong mục Bước 2).
> `rank` chưa cài (cần NLP mới, hoãn). Bước 3-4 chưa cài đặt. Mọi con số dưới đây đo trực tiếp trên
> `data/processed/release_v2_422df141c935` và `data/raw/ViFinQA/questions/questions.jsonl`
> (1.012 câu), không suy diễn. Giữ đúng kỷ luật "đo trước khi sửa" của dự án. Kết quả đầy đủ:
> [plan.md, mục Ngày 24](../../plan.md), tìm "Chẩn đoán và Bước 1" và "Bước 2 ngày 2026-08-16".

Hai vấn đề người dùng nêu, hoá ra có **giao nhau ở một gốc chung** (§1.1) và một gốc riêng (§2):

- **A. Vì sao không sinh được câu trả lời** — 28/1.012 câu (2,8 %).
- **B. Bảng bằng chứng** — CSV hiện là **một dòng tổng hợp**, không phải bảng đã trích xuất.

---

## 1. Vấn đề A — phễu 1.012 câu, đo thật

Nguồn: `artifacts/evaluations/day22/submission/submission-export-422df141c935.json`.

| Tầng | Số câu | % | Mã lỗi chi tiết |
|---|---:|---:|---|
| Retrieval rỗng | 42 | 4,2 % | `no_candidate_tables` |
| Planner abstain | 617 | 61,0 % | `entity_ambiguous` 434 · `multi_metric_unsupported` 152 · `period_grammar_unsupported` 23 · `operation_unknown` 8 |
| Execution lỗi | 325 | 32,1 % | `metric_not_found` 206 · `period_unresolved` 55 · `cell_ambiguous` 52 · `unit_missing` 11 · `candidate_table_ids_scope_empty` 1 |
| **Answered** | **28** | **2,8 %** | |

Tách `entity_ambiguous` (434) ra bằng cách chạy lại `parse_query_entities` trên cả 1.012 câu —
đây là số đo mới, harness cũ không ghi:

| Mã ambiguity | Số câu | % của 1.012 |
|---|---:|---:|
| `metric_unknown` | **402** | **39,7 %** |
| `period_missing` | 55 | 5,4 % |
| `company_missing` | 11 | 1,1 % |
| `period_relative_unresolved` | 6 | 0,6 % |

### 1.1. Gốc chính — ontology 56 chỉ tiêu quá nhỏ so với corpus, và nó chặn **cả hai đầu**

Đây là **một** nguyên nhân, biểu hiện ở hai chỗ cách nhau rất xa trong đường ống, nên trước
giờ bị đếm thành hai lỗi khác nhau:

**Đầu câu hỏi.** `METRIC_ALIASES` có **218 alias → 56 canonical**. `_parse_metric` chỉ phát ra
metric khi khớp lexicon; không khớp thì gắn `metric_unknown` → `rule_planner` abstain
`entity_ambiguous`. **402/1.012 câu (39,7 %) chết ở đây.**
Ví dụ thật: *"Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC)…"* — corpus
**có** dòng `Lãi tiền gửi`, lexicon thì không.

**Đầu corpus.** `locator._metric_mask` chỉ khớp **bằng đúng** `row_label_canonical` (hoặc
`row_label_raw` bằng chuỗi tuyệt đối). Nhưng canonical hầu như trống:

| Số đo trên `cells.parquet` (6.199.661 ô) | Giá trị |
|---|---:|
| Ô có `value_numeric` | 2.620.706 (42,3 %) |
| Ô số **có** `row_label_canonical` | 148.494 — **5,7 %** |
| Ô số có `row_label_raw` nhưng **không** có canonical | 2.472.212 |
| Số nhãn canonical phân biệt | 55 |
| Số nhãn raw phân biệt | **165.416** |
| Khoá chuẩn hoá raw nằm trong `METRIC_ALIASES` | **166 / 140.897** |
| **Bảng có ZERO ô số được gán canonical** | **107.717 / 130.729 = 82,4 %** |
| Coverage canonical trung vị theo bảng | **0,0 %** (p90 = 16,7 %; p99 = 50,8 %) |

**Hệ quả trực tiếp:** retrieval có thể trả về **đúng** bảng và executor vẫn báo
`metric_not_found`, vì 82,4 % bảng không có một ô nào mang nhãn canonical. Đây chính là
206 ca `metric_not_found` — chúng **không phải** lỗi retrieval.

> **Tổng quy trách nhiệm của riêng gốc này: 402 + 206 = 608/1.012 ≈ 60 % số câu.**

Đây là lần lặp **thứ sáu** của mô hình "trường có sẵn ở thượng nguồn, chết ở hạ nguồn"
(sau `llm:`, `execution:`, `timeout_seconds`/`max_rows`, `expected_unit`, `statement_scope`):
`row_label_raw` **đã có trên 86,4 % ô** và đường ống không dùng.

### 1.2. Gốc thứ hai — rule planner chỉ với tới 4/9 operation, dù compiler đã cài đủ

`rule_planner._infer_operation` chỉ có thể phát ra `lookup`, `difference`, `growth_rate`,
`compare_companies`. **`ratio`, `compare`, `average`, `sum`, `rank` không bao giờ được sinh ra**
— mặc dù `execution/compiler.py::_dispatch` **đã cài đặt đầy đủ cả 9** và
`render_pandas_query` đã render được cả 9.

Cộng thêm `len(metrics) > 1 → abstain("multi_metric_unsupported")` (rule_planner.py:106) — nhưng
`ratio` và `compare` **sinh ra để nhận đúng 2 chỉ tiêu**. Đây không phải năng lực còn thiếu, mà là
năng lực **chưa nối dây**.

Phễu chỉ-riêng-planner (bỏ qua retrieval), đo trên 1.012 câu:

| Chặng | Số câu | % |
|---|---:|---:|
| Chết vì entity ambiguity | 460 | 45,5 % |
| Chết vì > 1 chỉ tiêu | **157** | **15,5 %** |
| Không suy được operation | 9 | 0,9 % |
| Dựng được plan | **386** | **38,1 %** |

Phân bố chỉ tiêu mỗi câu: 0 → 402 câu, 1 → 420, 2 → 108, 3 → 44, ≥4 → 38.
Tức **190 câu có ≥ 2 chỉ tiêu** — đúng hình dạng của nhóm câu hỏi 4 (ROA/ROE/biên lợi nhuận)
và nhóm 5 (tổng hợp/xếp hạng) mà plan.md §1.1 **bắt buộc** phải xử lý. Hiện cả hai nhóm
**không thể chạm tới về mặt cấu trúc**.

### 1.3. Gốc thứ ba — ngữ pháp kỳ

55 `period_missing` + 23 `period_grammar_unsupported` ở planner, cộng 55 `period_unresolved`
ở execution = **133 câu**. Ví dụ thật của `period_missing`: *"…vào cuối năm tài chính 2017?"* —
năm **có mặt trong câu**, parser chỉ không nhận dạng được lối diễn đạt.

### 1.4. Tỷ lệ chết ở execution

386 plan dựng được → 28 answered. **Execution giết 92 %** số plan hợp lệ. Trong đó
206/325 = 63 % là `metric_not_found`, tức quay lại đúng §1.1.

---

## 2. Vấn đề B — bảng bằng chứng phải là bảng đã trích xuất

### 2.1. Hiện trạng, trích thẳng từ ZIP đã nộp

```
data/q000004_df1.csv:
company_code,row_label_canonical,row_label_raw,period,value
FTS,profit_after_tax,,2023,444918.0
```

Một dòng. `row_label_raw` rỗng. Đây **không phải** bảng nào tồn tại trong bước xử lý dữ liệu.

### 2.2. Gốc

`compiler._replay_row()` (compiler.py:71-80) **tổng hợp** một dict cho mỗi ô bằng chứng đã định
vị; `_replay_frame` biến danh sách đó thành DataFrame; `exporter._replay_rows_to_csv_rows`
(exporter.py:74-84) ghi đúng danh sách đó ra CSV. **Bảng trích xuất thật không bao giờ được đọc.**

Điều này có chủ đích ở Ngày 18 — `pandas_query` không có cú pháp quy đổi đơn vị, nên compiler
phải replay trên một frame đã quy đổi sẵn (ghi rõ ở pandas_query.py:11-19). Nhưng nó biến
"bằng chứng" thành thứ **tự hệ thống sinh ra**, phá đúng tinh thần truy vết
`answer → execution → cells → table → page → document` của plan.md.

**Hạ tầng cần thiết đã có sẵn và chưa dùng:** `data/table_frame.py::load_table_frame(release_dir,
table_id)` trả về cả `grid` (bảng như lúc trích xuất) lẫn `values` (dạng tidy một-ô-một-dòng).
Đo thử trên một bảng thật:

```
=== GRID ===
                    0                  1                  2
0            Chỉ tiêu           Năm 2019           Năm 2020
1     Doanh thu thuần  9.992.737.963.848  9.444.931.421.578
2  Lợi nhuận sau thuế  (192.609.972.666)  (271.803.303.342)
```

### 2.3. Ba ràng buộc mà bản sửa bắt buộc phải thoả — không phải "ghi CSV to hơn"

**(a) Đa bảng — 18/28 câu (64 %) lấy bằng chứng từ 2–4 bảng khác nhau**, thường khác cả tài liệu
và năm. Phân bố: 1 bảng → 10 câu, 2 bảng → 14, 3 bảng → 3, 4 bảng → 1.
Một `df1` **không thể** là "bảng đã trích xuất".
Hợp đồng §2.4 **đã lường trước** điều này (quy tắc 6, ví dụ `q000002_df1.csv` + `q000002_df2.csv`),
nhưng cài đặt gộp hết về một `df1`, **và** trình thông dịch AST chỉ biết đúng tên `df1`
(pandas_query.py:213-216). Cần mở rộng cả hai.

**(b) Đơn vị.** `replay_rows.value` **đã quy đổi** (`_reconvert`); `value_numeric` của bảng thật
thì **thô**. Đo được: trong cùng một bảng, đơn vị nhất quán **99,5 %** (116/23.012 bảng lệch) —
nên dump nguyên một bảng là an toàn. Rủi ro nằm **giữa** `df1` và `df2` (VND so với VND_million).
Phân bố đơn vị trên ô canonical: VND 91.508 · NULL 27.945 (18,8 %) · VND_million 26.360 ·
VND_thousand 2.253 · percent 266 · VND_billion 162.

**(c) Chọn dòng.** `.iloc[0]` trên bảng đầy đủ có thể trúng dòng **khác** với ô đã định vị —
nhãn canonical trùng trong một bảng là chuyện thường (đó chính là bản chất của `cell_ambiguous`).

> Nói cách khác: sửa bảng bằng chứng **là sửa hợp đồng ngữ pháp `pandas_query`**, không phải
> sửa mỗi exporter. Validator (`validator.py:149-172`) nạp CSV rồi replay `pandas_query` và so
> với `answer` — sai một trong ba ràng buộc trên là `replay_failed`/`answer_mismatch`.

---

## 3. Chiến lược khắc phục — xếp theo đòn bẩy đo được trên mỗi đơn vị công sức

### Bước 1 — Khớp chỉ tiêu theo nhãn thô (gỡ nút thắt 60 %) — ĐÃ CÀI ĐẶT (2026-08-16)

**Ý tưởng cốt lõi:** thôi bắt **cả hai** đầu phải đi qua ontology 56 từ. Câu hỏi viết
"Lãi tiền gửi", corpus có dòng "Lãi tiền gửi" — khớp **thẳng** qua `normalized_key`, không cần
alias thủ công nào.

> **Điều chỉnh thiết kế so với dự thảo ban đầu, quyết định trước khi viết code:** dự thảo 1b bên
> dưới định để `entity_parser` tự khớp một "cụm danh từ" từ câu hỏi với tần suất toàn corpus — đây
> chính xác là **Option B** mà [ADR 0004](../decisions/0004-metric-locator-strategy.md) đã xét và
> loại bỏ ("khi nhiều hàng cùng gần khớp, không có quy tắc tất định để chọn đúng một hàng — vi phạm
> nguyên tắc không đoán"). Đo thử tần suất nhãn thô toàn corpus xác nhận rủi ro đó có thật: các nhãn
> ≥3 từ tần suất cao nhất là chú thích cấu trúc chung chung ("Số dư cuối năm" 36.742 lần, "Số dư đầu
> năm" 34.803 lần) chứ không phải tên chỉ tiêu độc lập. Cài đặt thật vì vậy **không** đặt logic này
> vào `entity_parser` (giữ đúng ràng buộc "pure, không phụ thuộc corpus" của `rule_planner.py`), mà
> tách thành module mới `planning/raw_metric_grounding.py`: chỉ khớp trong phạm vi **bảng ứng viên
> mà retrieval đã trả về cho chính câu hỏi đó**, hẹp hơn nhiều so với Option B (toàn corpus) và có
> phạm vi thu hẹp theo từng câu hỏi.

- 1a. `locator._metric_mask`: thêm nhánh khớp `normalized_key(row_label_raw) ==
  normalized_key(selector.raw_text)`, bên cạnh nhánh canonical hiện có. **Cài xong** — TDD,
  `tests/unit/execution/test_execution_locator.py`.
- 1b. **Thay bằng** `planning/raw_metric_grounding.py::ground_raw_metric` (xem hộp thoại thiết kế
  ở trên): chỉ khớp nhãn `row_label_raw` (ngưỡng an toàn ≥ 3 từ, theo tiền lệ ngưỡng alias công ty
  Ngày 22) là con chuỗi đã chuẩn hoá của câu hỏi, trong phạm vi bảng ứng viên đã retrieve.
  **Cài xong** — TDD, `tests/unit/planning/test_raw_metric_grounding.py` (13 test).
- 1c. **Cổng an toàn, giữ bất biến "không đoán":** chỉ chấp nhận khi đúng **một** nhãn (nhóm theo
  `normalized_key`, không phân biệt hoa/thường) khớp — 0 khớp hoặc ≥2 khớp đều bỏ qua. Chỉ kích
  hoạt khi `entities.ambiguity` **đúng bằng** `("metric_unknown",)` (hàm
  `plan_with_raw_grounding_fallback`). `plan_source = "rule_raw_grounded"` tách riêng để đo —
  **cài xong**, đo được qua `submission-export-*.json`.
- 1d. **Không cần** — đo thật cho thấy khớp theo văn bản câu hỏi (1b/1c) đã đủ tín hiệu; không cần
  bảng tần suất tĩnh toàn corpus.

**Đã đo (2026-08-16):** 28 → 55/1.012 answered (+27, +96 %), 0 câu mất. gold120 accuracy không đổi
0,846 → 0,846 (fallback không kích hoạt trên 10 câu `metric_unknown` của gold120 — cả 10 đều là câu
hỏi cấu trúc/định tính ngoài phạm vi operation, không phải câu tra cứu chỉ tiêu, nên không đại diện
cho hình dạng câu Bước 1 nhắm tới). `submission validate`: `valid=True items=55`. Soát tay xác nhận
rủi ro nhãn chung chung (đã lo ở hộp thoại thiết kế trên) được giảm nhẹ mạnh trên thực tế: grounding
chỉ chạy sau khi retrieval đã thu hẹp bảng ứng viên bằng chính văn bản câu hỏi, nên một nhãn như "Số
dư cuối năm" thường chỉ còn xuất hiện trong đúng một bảng có tiêu đề khớp chủ thể câu hỏi. Chi tiết
đầy đủ: [plan.md](../../plan.md), mục Ngày 24, "Chẩn đoán và Bước 1 ngày 2026-08-16".

**Cách đo:** chạy lại `submission export` trên 1.012 câu, so answered và — quan trọng hơn —
so trên gold120 để chắc accuracy **không tụt**. Đây là chỗ dễ đổi coverage lấy độ đúng nhất;
áp đúng bài học Ngày 21 (`default_consolidated` tăng phủ 1,8× mà accuracy tệ đi → không chọn).

### Bước 2 — Nối dây 5 operation đã cài sẵn (ceiling 157 câu) — ĐÃ CÀI ĐẶT MỘT PHẦN (2026-08-16)

`_infer_operation` nhận thêm số chỉ tiêu, thay vì abstain khi > 1:

| Hình dạng | Operation | Đã có ở compiler? |
|---|---|---|
| 2 chỉ tiêu + từ khoá tỷ lệ (`tỷ lệ`, `biên`, `hệ số`, `ROA`, `ROE`) | `ratio` | ✅ |
| 2 chỉ tiêu, không từ khoá tỷ lệ | `compare` | ✅ |
| 1 chỉ tiêu + ≥ 3 kỳ | `average` / `sum` | ✅ |
| ≥ 3 công ty + 1 kỳ + từ khoá xếp hạng | `rank` | ✅ |

Rẻ vì compiler, renderer và sandbox **đã hỗ trợ đủ**; chỉ thiếu suy luận ở planner.

> **Bug thật tìm thấy trước khi cài** (không phải giả thuyết): `_infer_operation` khớp
> `compare_companies` với `n_companies >= 2`, nhưng `compile_compare_companies` chỉ đọc
> `companies[0]`/`companies[1]`. 35/1.012 câu thật nêu 3–10 công ty bị khớp nhầm; **2 câu (931, 973)
> đã trả lời sai một cách tự tin trong bản ZIP đã nộp của Bước 1** (931: hỏi trung bình một tỷ lệ
> qua 5 công ty, trả lời hiệu số 2 công ty đầu bằng VND thô; 973: hỏi đếm số công ty thoả điều kiện,
> trả lời một hiệu số). Sửa `== 2` ở cả rule planner **và** `_validate_compare_companies` (phòng thủ
> hai lớp, vì LLM planner cũng có thể sinh plan này). Chi tiết đầy đủ + số đo:
> [plan.md](../../plan.md), mục Ngày 24, "Bước 2 ngày 2026-08-16".

**Đã cài, đã đo (2026-08-16):**

- `ratio`: chỉ suy luận khi có từ khoá tỷ lệ (đo 25/26 câu 2-chỉ-tiêu thật khớp) — `compare` **cố
  tình không cài** (0/26 câu thật là hình dạng đó, không có bằng chứng để định tuyến).
  `entity_parser.ordered_metric_canonicals` khôi phục thứ tự numerator/denominator từ
  `entities.spans` (bị mất do `entities.metrics` sắp xếp alphabet).
- `average`/`sum`: cả hai hướng biến thiên (theo kỳ, theo công ty), có cổng loại trừ từ khoá xếp
  hạng/đếm để không lẫn với câu hỏi phức hợp.
- **Bug thứ hai cùng họ, bắt bằng TDD trước khi nối dây**: dispatch `average`/`sum` ở cả
  `execution/compiler.py` và `execution/pandas_query.py` luôn lặp qua kỳ tại `companies[0]` cố
  định, dù validator đã cho phép biến thiên theo company. Sửa cả hai lớp.
- `rank`: **chưa cài** — cần trích `top_k` từ câu hỏi tự nhiên, một năng lực NLP mới chưa tồn tại
  (rủi ro đoán cao, hoãn).

**Kết quả đo trên 1.012 câu thật:** 55 → **53** answered (đúng −2, xoá 931/973 — hạ coverage để xoá
sai tự tin, đúng tinh thần Ngày 21). **0 câu mới trả lời được** dù `build_plan` dựng đúng plan cho
20/26 câu ratio và 18/35 câu trung bình/tổng — cả 38 đều dừng ở execution (`metric_not_found` 22,
`cell_ambiguous` 13, `period_unresolved` 2, `unit_missing` 1).

**Phát hiện quan trọng nhất của Bước 2 — cùng gốc với Bước 1, nhưng ở tầng khác:** chỉ tiêu trong
câu hỏi khớp alias canonical thành công (không phải `metric_unknown`), nhưng bảng cụ thể của
công ty/kỳ được hỏi có thể chỉ mang `row_label_raw`, không có `row_label_canonical` (82,4 % bảng,
đã đo ở §1.1) — `locate()` chỉ so khớp canonical nên vẫn trượt. Grounding thô của Bước 1 chỉ kích
hoạt khi **planner** abstain `metric_unknown`; không chạm tới trường hợp này vì planner đã khớp
canonical thành công, chỉ hỏng ở **compiler**. Ứng viên bước tiếp theo (chưa làm): mở rộng cơ chế
grounding xuống tầng `locate()` — khi canonical trượt, thử khớp `row_label_raw` trong đúng bảng ứng
viên đó theo alias đã biết của chỉ tiêu, với cùng cổng an toàn "chỉ một khớp".

`submission validate` trên ZIP mới: `valid=True items=53`. gold120: accuracy không đổi 0,846 → 0,846,
0 hồi quy. Verification đầy đủ: 1.235 test qua, `ruff`/`mypy` sạch.

### Bước 3 — Ngữ pháp kỳ (ceiling 133 câu)

Nhận `cuối/đầu năm tài chính YYYY`, `cuối kỳ`, quý và ngày. Rẻ và độc lập với Bước 1-2.

### Bước 4 — Hợp đồng bảng bằng chứng (vấn đề B)

Làm **sau** Bước 1-3, vì mỗi câu trả lời mới sinh ra thêm CSV, và làm trước thì phải sửa hai lần.

- 4a. Mở rộng trình thông dịch AST: chấp nhận `df1..dfN` (hiện hard-code đúng `df1`),
  cả ở `pandas_query._eval_node` lẫn `sandbox`/`validator`.
- 4b. `CompiledQuery` mang thêm ánh xạ `biến → table_id`. Exporter gọi
  `load_table_frame(release_dir, table_id)` cho **từng** bảng nguồn và ghi **nguyên bảng tidy**
  (giữ `cell_id`, `row_label_raw`, `row_label_canonical`, `column_label_raw`, `period`, `unit`,
  `value_numeric`, `source_line_start`) — truy vết được đến từng ô, đúng plan.md §Global Constraints.
- 4c. Xử lý ràng buộc (b): thêm cột `unit` vào CSV và **quy đổi ở tầng render** — hoặc, đơn giản
  hơn và giữ nguyên ngữ pháp hiện tại, ghi thêm cột `value_canonical` đã quy đổi bên cạnh
  `value_numeric` thô, rồi cho `pandas_query` đọc `value_canonical`. Cần một ADR chốt lựa chọn.
- 4d. Xử lý ràng buộc (c): thay `.iloc[0]` bằng bộ lọc đủ chặt để chọn đúng dòng (thêm điều kiện
  `cell_id`), hoặc giữ `.iloc[0]` nhưng chứng minh bộ lọc chỉ còn đúng một dòng — quyết định
  bằng đo, không bằng phỏng đoán.
- 4e. Bổ sung contract test: CSV trong ZIP phải khớp **byte-for-byte** với
  `load_table_frame(table_id)` của release — bảo đảm bằng chứng là bảng **đã trích xuất**,
  không phải bảng do exporter dựng lại.

### Việc **không** làm

- **Không** fine-tune planner. Ngày 21 đã đo: retrieval chỉ chiếm 5 % lỗi; nút thắt là
  ontology và arity — hai thứ fine-tune không sửa được.
- **Không** đổi chính sách scope để nâng số answered (Ngày 21 đã đo là bẫy).
- **Không** đụng LLM planner. Ngày 22 đã đo: 0 câu thêm, 73,5 % sai schema. Bước 1 xử lý đúng
  nguyên nhân mà ADR 0006 B1 gọi là "thiếu vocabulary trong prompt" — nhưng bằng dữ liệu, không
  bằng token.

---

## 4. Trần coverage ước tính

Không cộng dồn được vì các nguyên nhân chồng nhau (ví dụ một câu vừa `metric_unknown` vừa 2 chỉ
tiêu). Trần **rời** từng bước, để chọn thứ tự chứ không phải để hứa kết quả:

| Bước | Trần ước tính | Đo thật | Ghi chú |
|---|---:|---:|---|
| 1. Khớp nhãn thô | ~608 | **+27** | Trần cộng 402 planner + 206 execution như hai góc độc lập — đo thật
cho thấy chồng lấn đáng kể: chỉ 109/391 câu `metric_unknown` khớp được một nhãn duy nhất trong bảng
đã retrieve, và chỉ 27/109 sống sót hết execution (phần còn lại đúng đắn dừng ở `period_unresolved`/
`cell_ambiguous` — không phải lỗi). Xem chi tiết ở mục Bước 1 phía trên. |
| 2. Nối operation | ~157 | **−2** (net) | 38 plan hợp lệ dựng được (20 ratio + 18 avg/sum) nhưng **0**
sống sót execution — trượt đúng bức tường canonical-only của Bước 1 (§1.1), chỉ ở tầng compiler
thay vì planner. −2 đến từ sửa 2 bug thật (compare_companies mis-route) đã trả lời sai tự tin.
`rank` chưa cài. |
| 3. Ngữ pháp kỳ | ~133 | chưa làm | |
| 4. Bảng bằng chứng | 0 | chưa làm | Đúng đắn/truy vết, không phải coverage |

Cổng phải giữ nguyên trong suốt: **answer accuracy ≥ 0,85** và **sai tự tin < 5 %**. Coverage tăng
mà accuracy tụt là **hỏng**, không phải tiến bộ — đúng cách Ngày 21 đã phán quyết.
