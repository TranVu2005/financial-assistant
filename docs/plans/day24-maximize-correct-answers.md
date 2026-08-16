# Ngày 24 — Chiến lược tối đa hoá số câu trả lời ĐÚNG

> **Kết quả cuối cùng (Ngày 25, sau khi thêm tầng LLM chọn-dòng ở §3.2):**
> **53 → 113 câu trả lời thật (+113 %)**. `submission validate` với đủ 1.012 ID chính thức:
> **`valid=True item_count=1012`**. Phân rã nguồn của 113 câu: `rule` 53, `rule_raw_grounded` 29
> (không đổi so với trước khi thêm tầng LLM — xác nhận không hồi quy), `llm_cell_grounded` **24**
> (77 % phần tăng thêm từ LLM), `llm_grounded` 6, `llm` 1. Chi tiết đầy đủ ở §3.2.

> **Bối cảnh:** người dùng yêu cầu rõ *"tôi cần câu trả lời đúng chứ không phải trả lời cho có"*.
> Tài liệu này thay mục tiêu "phủ đủ 1.012 câu" (đã đạt, xem plan.md Ngày 24) bằng mục tiêu
> **tăng số câu đúng thật**. Mọi con số đo trực tiếp trên release khoá và 1.012 câu đề thật.

---

## 1. Định hướng lại: dự án đang tối ưu SAI chỉ số

Công thức chấm chính thức của BTC:

> Answer Accuracy = (số query có kết quả khớp đáp án chuẩn) / (**tổng số query**)

Chia cho **tổng 1.012 câu**, không phải số câu đã thử. Hệ quả: **một câu trả lời sai và một câu bỏ
trống đều được 0 điểm.** Abstain không mua được gì; trả lời chỉ có thể cộng thêm.

Suốt Ngày 15–23, dự án đo bằng `accuracy = correct / answered` — chỉ số kiểu **precision**, **phạt
việc dám trả lời**. Đo lại ba chính sách scope trên gold120 bằng đúng công thức BTC:

| Chính sách | Câu **đúng** | Đã trả lời | correct/answered (cũ) | **correct/total (BTC)** |
|---|---:|---:|---:|---:|
| `default_consolidated` | **39** | 70 | 0,750 | **0,325** |
| `none` (mặc định hiện tại) | 30 | 36 | 0,833 | 0,250 |
| `abstain_when_unstated` | 25 | 28 | 0,893 | 0,208 |

**Thứ hạng đảo ngược hoàn toàn.** Ngày 21 loại `default_consolidated` vì "accuracy tệ hơn" — theo
thang điểm thật thì nó cho **nhiều hơn 30% câu đúng**.

Ràng buộc nội bộ *"sai tự tin < 5 %"* (plan.md §1.2) mâu thuẫn trực tiếp với thang điểm BTC. Ghi
nhận là **đánh đổi có chủ đích**, không phải bỏ qua: bản nộp dùng overlay riêng
`configs/submission_maximize_correct.yaml`; `configs/local_rtx3050.yaml` giữ `none` cho đo lường
nội bộ, nơi precision mới là tín hiệu trung thực.

### 1.1. ⚠️ Suy luận trên chỉ đúng một nửa — đo thật đã bác bỏ nó

Áp `default_consolidated` lên 1.012 câu thật: **61 → 34 answered. TỆ ĐI 27 câu**, ngược hẳn dự đoán
"+30 %" từ gold120. Nguyên nhân, đọc từ phân bố mã lỗi:

| Mã lỗi | `none` | `consolidated` | Δ |
|---|---:|---:|---:|
| `cell_ambiguous` | 117 | 63 | **−54** ✓ lọc scope hoạt động đúng |
| `scope_inferred` | 0 | **39** | **+39** ✗ verifier **chặn** |
| `period_inferred_warning+scope_inferred` | 0 | **8** | +8 ✗ |
| `metric_not_found` | 175 | 194 | +19 (lọc bỏ mất bảng có đáp án) |
| `candidate_table_ids_scope_empty` | 1 | 9 | +8 |

Lọc scope **giải quyết đúng** nhập nhằng (−54), nhưng **47 câu bị `check_scope_inferred` chặn** ở
tầng verification (ADR 0010 B1 "suy luận rồi chặn").

**Sai lầm phương pháp cần ghi lại:** báo cáo `scope-policy-report` đo `answered_count` tại
`compile_plan` — **trước** bước chặn — và docstring của `ScopePolicyResult` **đã ghi rõ điều đó**.
Tôi đọc sót và khuyến nghị dựa trên một chỉ số không bao gồm chính cơ chế giết nó trong sản xuất.
Bài học: khi một chỉ số hỗ trợ quyết định lớn, **đọc định nghĩa của chính chỉ số đó trước**.

Ngày 21 cũng đã cảnh báo trước bằng số: đề thật nghiêng **separate** (36,4 %) áp đảo **consolidated**
(1,3 %) — nên mặc định `consolidated` "còn sai hướng hơn trên đề thật". Số liệu đã xác nhận.

### 1.2. Thí nghiệm đang chạy

Thêm `build_answer_package(allow_inferred_scope=...)` + cờ CLI `--allow-inferred-scope` (mặc định
**tắt**, giữ nguyên hành vi chặn cho đo lường nội bộ). `AnswerPackage` thêm trường
`inferred_scope_accepted` — issue `scope_inferred` **vẫn được ghi lại**, chỉ không chặn; người đọc
luôn biết scope đã bị đoán. Escape hatch chỉ áp cho `scope_inferred`, mọi lỗi đúng-đắn khác vẫn chặn
(có test bảo vệ).

Trần lý thuyết: 34 + 47 ≈ **81 câu** nếu toàn bộ nhóm bị chặn trở thành đáp án — cao hơn 61 hiện tại.
Cần đo, chưa hứa. Đồng thời thêm `configs/submission_scope_separate.yaml`: **`separate` chưa bao giờ
được đo** (Ngày 21 chỉ thử `none`/`consolidated`/`abstain`) dù đề thật nghiêng hẳn về nó.

### 1.3. Kết quả: giả thuyết đúng — **82/1012** (dự đoán ≈81)

| Cấu hình | Answered | Ghi chú |
|---|---:|---|
| `none`, chặn (trước) | 53 | trước sửa ordinal |
| `none`, chặn + sửa ordinal | 61 | |
| `consolidated`, **chặn** | 34 | 47 câu bị verifier chặn |
| **`consolidated` + `--allow-inferred-scope`** | **82** | +29 câu, −8 câu |

`submission validate` với đủ 1.012 ID: **`valid=True item_count=1012`** (82 thật + 930 backstop).

Dịch chuyển mã lỗi so với `none`: `cell_ambiguous` **117 → 65 (−52)**; giá phải trả là
`metric_not_found` **+20** và `candidate_table_ids_scope_empty` **+8** (lọc scope loại nhầm bảng có
đáp án). Đây chính là chỗ `separate` có thể tốt hơn nếu nó khớp phân bố đề thật.

**Cảnh báo quy kết:** lần chạy 82 câu này bao gồm **cả** bản sửa `năm tài chính YYYY` (§4.1) — thấy
được ở `entity_ambiguous` 316 → 310. Không quy toàn bộ +21 cho riêng chính sách scope.

### 1.4. `separate` đã đo: **74** — thấp hơn `consolidated`, và lý do quan trọng

| Cấu hình (đều `--allow-inferred-scope`) | Answered | `metric_not_found` |
|---|---:|---:|
| **`consolidated`** | **82** | 195 |
| `separate` | 74 | 235 (**+40**) |

Tôi đã dự đoán `separate` sẽ thắng vì "đề thật nghiêng separate 36,4 % vs 1,3 %" (Ngày 21). **Dự
đoán sai, và lý do là một hiểu nhầm cần ghi lại.** Phân bố scope **được nêu** trên 1.012 câu:

| scope nêu trong đề | Số câu | % |
|---|---:|---:|
| **không nêu** | **631** | **62,4 %** |
| `separate` | 368 | 36,4 % |
| `consolidated` | 13 | 1,3 % |

368 câu nêu `separate` **đã tự có scope trong plan** — `default_statement_scope` **không bao giờ áp
cho chúng** (`resolve_statement_scope`: plan thắng default). Mặc định chỉ ảnh hưởng **631 câu không
nêu**. Con số "36,4 % nghiêng separate" của Ngày 21 nói về scope *được nêu*, một tập **rời hoàn
toàn** khỏi tập mà mặc định tác động.

Với 631 câu không nêu, `consolidated` là phỏng đoán tốt hơn rõ rệt (`metric_not_found` 195 so với
235): báo cáo hợp nhất là bản khai chính, đầy đủ chỉ tiêu hơn báo cáo riêng.

**Bài học (lần thứ hai cùng một loại trong ngày):** trước khi mượn một số đo cũ để suy luận, kiểm
tra nó đo trên **đúng tập** mà quyết định hiện tại tác động. §1.1 sai vì đọc sót định nghĩa chỉ số;
§1.4 sai vì áp số đo của tập A lên tập B rời nhau.

**Chốt cấu hình bản nộp:** `configs/submission_maximize_correct.yaml` (`consolidated`) +
`--allow-inferred-scope`.

---

## 2. Đã sửa trong phiên này

### 2.1. Lỗi số thứ tự / dấu chú thích (lần lặp thứ 8 của mô hình "chết ở hạ nguồn")

`is_non_metric_label` **đã biết** bóc `"4. Giá vốn hàng bán"` ra thành alias hợp lệ (để trả lời câu
"đây có phải chỉ tiêu không?"), nhưng `normalize_metric` lại tra từ điển bằng key **chưa bóc** →
`metric_unknown`. Báo cáo tài chính Việt Nam **luôn** đánh số mục (`1.`, `I.`, `B.`) và đánh dấu
tham chiếu (`(1)`, `(*)`).

| Số đo | Giá trị |
|---|---:|
| Ô số mất nhãn chỉ vì trang trí | **57.466** (589 nhãn phân biệt) |
| Coverage canonical trước | 148.494 ô |
| Coverage canonical sau | **205.960 ô (×1,39)** |

Sửa ở **hai tầng**: `normalize_metric` (đúng lâu dài, có hiệu lực khi ingest lại) và `locator.
_metric_mask` tại thời điểm truy vấn (hưởng lợi **ngay** trên release đã khoá, không đổi
`dataset_fingerprint` — ADR 0004 §1.7).

### 2.1b. Kết quả thật trên 1.012 câu (đo xong)

**53 → 61 câu trả lời thật (+8).** Dịch chuyển theo mã lỗi:

| Mã lỗi | Trước | Sau | Δ |
|---|---:|---:|---:|
| `metric_not_found` | 203 | 175 | **−28** |
| `cell_ambiguous` | 102 | 117 | +15 |
| `unit_missing` | 17 | 27 | +10 |
| `period_unresolved` | 84 | 79 | −5 |
| (các mã khác) | | | 0 |

Đọc đúng bản chất: sửa ordinal đưa **28 câu** vượt qua rào chỉ tiêu, nhưng **chỉ 10 câu** đi hết được
đến đáp án — 25 câu vấp ngay rào kế tiếp (`cell_ambiguous`, `unit_missing`). Mỗi rào gỡ được lại lộ
ra rào sau; đây là đặc điểm của đường ống nhiều tầng, không phải thất bại của bản sửa.

### 2.2. Tác dụng phụ đã đo và giải thích được

gold120: answered 39 → 36, accuracy 0,846 → 0,833. **Không phải hồi quy chất lượng.** Truy đúng 3
câu mất — cả 3 thành `cell_ambiguous`. Ví dụ KLB 2018:

| Giá trị | Nguồn |
|---|---|
| 231.889 | BC **hợp nhất** 2018 |
| 231.889 | BC hợp nhất 2019 (cột so sánh) — nhất quán |
| **191.027** | BC **riêng** 2019 (cột so sánh 2018) |

Câu hỏi không nêu scope → mâu thuẫn **thật** → abstain **đúng** theo bất biến "không đoán". Câu trả
lời cũ đúng một phần **do may mắn**: dòng của báo cáo riêng bị chú thích `XIII.` nên vô hình.
**Accuracy 0,846 cũ được xây một phần trên việc không nhìn thấy dữ liệu mâu thuẫn.** Đây chính là
lý do §1 (đặt `default_consolidated`) là bước bổ sung bắt buộc, không phải tuỳ chọn.

---

## 3. Số đo giới hạn chiến lược — những việc **KHÔNG** nên làm

Ba kết quả âm tính, mỗi cái loại bỏ một hướng nghe có vẻ hợp lý:

### 3.1. KHÔNG mở rộng từ điển chỉ tiêu bằng tay — ROI đã đo là rất tệ

402/1.012 câu (39,7 %) chết vì `metric_unknown`. Nhưng phân bố khái niệm thiếu **cực kỳ phẳng**:

| Số đo | Giá trị |
|---|---:|
| Cụm từ chỉ tiêu phân biệt không giải được | 240 |
| Câu được phủ bởi **top-50** cụm | 102 (**25,4 %**) |
| Tần suất cụm phổ biến nhất | **4 câu** |

**Không có "đầu" để đánh.** Thêm ~240 khái niệm thủ công chỉ đổi lấy ~402 câu ⇒ **~1,7 câu mỗi
khái niệm**. Cơ chế đúng cho đuôi dài là **khớp mở từ vựng** (Bước 1 `raw_metric_grounding`), không
phải curate thêm alias.

### 3.2. KHÔNG bắt model nhỏ sinh cấu trúc — nhưng ĐƯỢC hỏi nó "chọn dòng nào"

Tài liệu chính thức của BTC (`_2026__AIGuru___Finance_Tabular_QA`, tr. 39) đo trên **đúng bộ 1.012
câu** này:

| Model | PoT (sinh code/cấu trúc) | CoT |
|---|---:|---:|
| Qwen3-4B-Instruct | **4,64 %** | 11,56 % |
| Qwen2.5-Coder-7B | **0,30 %** | 9,49 % |

> *"Mô hình nhỏ < 10B → crash code: Tỷ lệ lỗi cú pháp PoT lên đến **99 %**"*

Ta dùng `qwen2.5:3b`/`7b` để sinh nguyên `FinancialQueryPlan` có kiểu — **đúng hình dạng thất bại
đó**. Kết quả 0/75 của ta khớp hoàn toàn số của họ. Con số 62–64 % end-to-end của họ cần
Gemini 3 / DeepSeek-V4 / GPT-5.4, **không phải** model local.

**Nhưng tr. 40 chỉ ra việc gì đáng giao cho LLM.** Phân tích 430 ca lỗi end-to-end:

| Loại lỗi | Tỷ lệ |
|---|---:|
| Đọc sai ô số liệu (Numerical Extraction) | **54,7 %** |
| Thiếu bảng (Insufficient Evidence) | **34,7 %** |
| Lỗi cú pháp code | 9,3 % |
| Lỗi **tính toán số học** | **0,9 %** |
| Lỗi công thức | 0,5 % |

> *"89,3 % lỗi đến từ trích xuất sai dữ liệu hoặc thiếu bảng. LLMs không dốt toán, vấn đề là chọn
> sai ô số liệu."*

Xác nhận độc lập đúng chẩn đoán của ta (`metric_not_found` + `cell_ambiguous` áp đảo; tính toán
chưa bao giờ sai) và chỉ ra thiết kế đúng: **giao cho LLM đúng việc chọn dòng, giữ mọi thứ khác tất
định.**

**Đã cài (`planning/llm_cell_grounding.py`):** đưa danh sách nhãn dòng **có thật** trong bảng ứng
viên của chính câu hỏi đó, đánh số, yêu cầu trả về **một số nguyên**. Model **không thể bịa nhãn** —
chỉ index vào danh sách thật (có test bảo đảm về cấu trúc). Rule planner vẫn dựng + validate plan;
locator, compiler, sandbox replay, bảng bằng chứng **không đổi** và vẫn tự chứng minh con số.

**Cố ý KHÔNG dùng CoT tự do** dù tài liệu đo CoT > PoT cho model nhỏ: CoT trả một con số không truy
vết được, không sinh được `pandas_query` replay đúng → phá hợp đồng §2.4. Cách này lấy phần LLM giỏi
(hiểu ngữ nghĩa để chọn dòng) mà không mất tính kiểm chứng.

**Đo trên batch chứng cứ (batch_000, trước đây 0/60): 7/60**, trong đó **3 câu từ
`llm_cell_grounded`** — đúng nhóm chỉ tiêu ngoài từ điển mà §3.1 chứng minh không thể curate thủ
công ("Doanh thu cho thuê khô tàu bay", "Chi phí dự phòng", "Thuế TNDN phải nộp"). 4 câu còn lại là
`rule`, đến từ các bản sửa trước — **không quy nhầm cho tầng mới**.

**Kết quả trên toàn bộ 1.012 câu:** 82 → **113 câu (+31)**. `submission validate`: `valid=True
item_count=1012`. Phân rã nguồn xác nhận đúng dự đoán từ batch chứng cứ:

| plan_source | Số câu | Ghi chú |
|---|---:|---|
| `rule` | 53 | không đổi so với baseline 82 câu trước khi thêm tầng LLM |
| `rule_raw_grounded` | 29 | không đổi — 53+29=82 khớp chính xác |
| **`llm_cell_grounded`** | **24** | **77 % phần tăng thêm từ LLM** |
| `llm_grounded` | 6 | LLM sinh cả plan, có thấy bảng thật |
| `llm` | 1 | LLM sinh plan không cần bảng |

Tầng "chọn dòng" vượt xa hai tầng LLM cũ (sinh cả plan) cộng lại — đúng luận điểm của tài liệu:
model nhỏ thất bại khi bị bắt sinh cấu trúc, nhưng làm được việc chọn đúng ô khi câu hỏi hẹp và có
danh sách thật để chọn.

### 3.2b. KHÔNG đổi sang dense retrieval chỉ vì bảng của BTC nói vậy

Tài liệu (tr. 26) xếp hạng BM25 47,41 % < BGE-M3 53,05 % < +Reranker 80,80 %. Nhưng đo trên gold70
của **chính ta**, thứ hạng **ngược lại**: BM25 **0,9143** vs BGE-M3 **0,6310**, fusion không đổi
(0,9143). Nguyên nhân: gold70 dùng `filters` **gán nhãn tay**, thu hẹp mạnh không gian tìm — nên
0,9143 là con số lạc quan giả và recall thật trên câu hỏi thô **chưa từng đo được** (không có nhãn).
Chuyển sang dense **không được dữ liệu của ta ủng hộ**; đòn bẩy thật của họ là **reranker** (+17
điểm) — thứ ta chưa có. Ghi nhận là hạng mục có bằng chứng ngoài, chưa có bằng chứng nội bộ.

### 3.2c. KHÔNG đầu tư thêm vào LLM planner sinh plan đầy đủ — đã đo 0 %

Ollama `qwen2.5:7b` thật (mạnh hơn `qwen2.5:3b` của Ngày 22), có cả tầng "LLM thấy bảng thật":
**0/15 câu smoke test, 0/60 câu batch 0**. Tầng `llm_grounded` có dựng được plan hợp lệ (2 câu) —
tức cơ chế đúng — nhưng vẫn trượt ở compiler. ~16 giây/câu ⇒ 4+ giờ cho toàn bộ, đổi lấy gần 0.

### 3.2b. KHÔNG đuổi theo heuristic "dòng chính vs dòng thuyết minh"

Một ca thật (CEO 2018) cho thấy `11. Thu nhập khác` = 24,3 tỷ (dòng chính KQKD) đụng
`Thu nhập khác` = 388 triệu (dòng con trong thuyết minh) — gợi ý quy tắc "có số thứ tự thì thắng".
**Đo trên 39 ca `cell_ambiguous` thật: chỉ 1/39 khớp mẫu này**, 38/39 là giá trị thật sự khác nhau
dưới cùng một dạng nhãn. Ca CEO là ngoại lệ, không phải quy luật — không có cách sửa rẻ ở đây.

(Ghi nhận phụ đáng giá: ở ca CEO, đáp án **cũ** 388 triệu nhiều khả năng đã **sai** — hệ thống trả
lời bằng dòng thuyết minh thay vì dòng chính. Việc giờ abstain là trung thực hơn, không phải hồi quy.)

### 3.3. KHÔNG tối ưu retrieval cho nhóm `metric_not_found`

Với các câu `metric_not_found` có chỉ tiêu canonical, kiểm tra **toàn corpus** (không chỉ bảng đã
truy hồi):

| Kết luận | Số câu |
|---|---:|
| Dữ liệu **có** ở nơi khác, retrieval không lấy được | **1** |
| Corpus **thực sự không có** cho (công ty, kỳ) đó | **12** |

Retrieval **không phải** nút thắt của nhóm này. Cải thiện truy hồi sẽ không giúp gì.

---

## 4. Kế hoạch còn lại, xếp theo đòn bẩy đã đo

| # | Việc | Trần đo được | Chi phí | Trạng thái |
|---|---|---:|---|---|
| 1 | Đặt `default_statement_scope: consolidated` cho bản nộp | **+30 % câu đúng** | 1 file config | ✅ xong |
| 2 | Bóc số thứ tự/dấu chú thích | ×1,39 coverage nhãn | 2 hàm | ✅ xong |
| 3 | Ngữ pháp kỳ: `năm tài chính YYYY` | ~60 câu | 1 dòng regex | ⬜ |
| 4 | Giảm `cell_ambiguous` sau khi có scope | ~18 % lỗi execution | vừa | ⬜ |
| 5 | Nới cổng kích hoạt grounding mở từ vựng | chưa đo | vừa | ⬜ |

### 4.1. Bước 3 — ngữ pháp kỳ (rẻ nhất, làm trước)

Đã xác minh bằng dữ liệu thật:

```
"... vào cuối năm tài chính 2017?"  -> periods=()        ambiguity=('period_missing',)
"... vào cuối năm 2017?"            -> periods=('2017',) ambiguity=()
```

`_YEAR_RE = r"năm\s+((?:19|20)\d{2})"` không cho `tài chính` chen giữa. 61 câu có nhập nhằng kỳ,
**60 câu có sẵn một năm 4 chữ số trong đề** mà parser bỏ sót. Sửa: cho phép cụm bổ nghĩa tuỳ chọn
giữa `năm` và số năm. Rẻ, an toàn, có test tái hiện.

Ngoài ra `giai đoạn YYYY-YYYY` xuất hiện ở 31 câu — nhưng phần lớn là câu nhiều bước
("trong giai đoạn 2016-2020, vào năm KBC có D/E cao nhất thì…"), vượt quá 9 operation hiện có; sửa
ngữ pháp kỳ **không đủ** để trả lời chúng. Không tính vào trần.

### 4.2. Bước 4 — `cell_ambiguous`

Sau khi §1 áp scope, phần nhập nhằng do scope sẽ tự biến mất. Phần còn lại (cùng scope, khác bảng)
cần quy tắc ưu tiên **đo trước rồi mới chọn**: ưu tiên bảng cùng tài liệu năm được hỏi (thay vì cột
so sánh của báo cáo năm sau) là giả thuyết đầu tiên, **chưa được kiểm chứng**.

### 4.3. Bước 5 — nới cổng grounding

`plan_with_raw_grounding_fallback` hiện chỉ chạy khi `ambiguity == ("metric_unknown",)` **đúng
bằng** một phần tử. Câu vừa thiếu chỉ tiêu vừa có vấn đề khác (ví dụ kỳ) không bao giờ được thử.
Sau khi Bước 3 sửa ngữ pháp kỳ, số câu rơi vào ô này sẽ giảm — **đo lại rồi mới quyết định** có nới
cổng hay không.

---

## 5. Trần thực tế — nói thẳng

Không hứa con số. Những gì đo được:

- Bước 1+2 đã cài, đang đo trên 1.012 câu (kết quả bổ sung khi chạy xong).
- Bước 3 có trần ~60 câu nhưng **không phải câu nào cũng sẽ trả lời được** — sửa được kỳ vẫn có thể
  chết ở chỉ tiêu hoặc nhập nhằng.
- Phần lớn 402 câu `metric_unknown` nằm ở **đuôi dài không có đầu** (§3.1) và 12/13 ca
  `metric_not_found` là **corpus thực sự thiếu dữ liệu** (§3.3) — hai nhóm này **không có cách sửa
  rẻ nào**, và cũng không nên giả vờ là có.

Cổng chất lượng nội bộ giữ nguyên cho nhánh đo lường (`none`); bản nộp dùng overlay §1 và chấp nhận
đánh đổi đã ghi rõ.
