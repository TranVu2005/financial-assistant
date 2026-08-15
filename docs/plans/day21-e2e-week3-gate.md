# Kế hoạch Ngày 21 — E2E và review cổng tuần 3

> **Trạng thái:** đã đo, **đã thực hiện xong** (2026-08-15). ADR:
> [0010](../decisions/0010-statement-scope-contract.md).
> **Ngày đo:** 2026-08-15. **Release khoá:** `data/processed/release_v2_422df141c935`,
> `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
> **Mọi con số trong tài liệu này là kết quả chạy thật trên release đã khoá và trên mã Ngày 20 vừa
> commit (`f74c01b`)**, không phải ước lượng.
>
> **Kết quả sau khi cài đặt xong:** xem blockquote kết quả trong [plan.md](../../plan.md) và mục
> "Day 21" trong [README.md](../../README.md) — accuracy 0,846 (chưa đạt 0,85, chính sách `none`),
> invalid plan 0 %, 100% có nguồn; bộ QA 70→120, answer-gold 30→58.

Mục Ngày 21 trong [plan.md](../../plan.md) yêu cầu ba việc: nâng bộ QA lên ≥ 120 câu; chạy câu hỏi
từ text → retrieval → plan → execution → answer package; phân lỗi thành retrieval, planning,
normalization, execution, verification. Cổng: answer và execution accuracy ≥ 0,85; invalid plan
< 5 %; 100 % answered có source.

**Phép đo tìm ra một thứ nghiêm trọng hơn cả ba gạch đầu dòng đó: đường ống chưa bao giờ được chạy
end-to-end.** Cái gọi là "E2E" của Ngày 20 nạp thẳng `gold_table_ids` vào planner làm
`candidate_table_ids`, rồi lại truyền chính tập đó vào `retrieved_table_ids` — nên **retriever bị
bỏ qua hoàn toàn** và phép kiểm `check_evidence_outside_retrieval` được thoả mãn một cách tầm
thường. Thay bằng top-10 thật của BM25 v4: **số câu trả lời được rơi 30 → 9 (−70 %)**, accuracy
0,900 → 0,667.

Và nguyên nhân **không phải retrieval kém** (§ 1.2): trong mọi ca hỏng đã soi, bảng gold **đều nằm
trong** top-10. Thủ phạm là một chiều dữ liệu mà `FinancialQueryPlan` **không hề mô hình hoá**:
báo cáo **riêng** (`separate`) và **hợp nhất** (`consolidated`) của cùng doanh nghiệp, cùng kỳ, cùng
chỉ tiêu — **giá trị khác nhau thật**. Đây là lần lặp **thứ năm** của mô hình "trường có sẵn ở
thượng nguồn, chết ở hạ nguồn" (sau `llm:`, `execution:`, `timeout_seconds`/`max_rows`,
`expected_unit`): `documents.parquet.statement_scope` đã tồn tại trong release khoá và **không được
tham chiếu ở bất kỳ đâu** trong `planning/`, `execution/`, `verification/`.

Nghiêm trọng hơn nữa (§ 1.5): **không có chính sách scope đơn giản nào vượt được cổng.** Đo thật cả
ba lựa chọn — mở rộng phủ và giữ đúng đắn **đi ngược chiều nhau**. Ngày 21 vì vậy phải là một ngày
**đo và review cổng** theo đúng khuôn mẫu Ngày 14, không phải một ngày vá lỗi.

---

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Bảng xếp hạng BM25 v4 đã lưu cho cả 70 câu | `artifacts/evaluations/day14/v2/retrieval-v2-bm25-v4-422df141c935.json` | ✅ `per_question[*].predicted_table_ids`, top-10 |
| Compiler + sandbox (Ngày 18–19) | `execution/` | ✅ đã commit `e052e22` |
| Answer package + 5 phép kiểm (Ngày 20) | `verification/` | ✅ đã commit `f74c01b` |
| Nhãn đáp án tay | `data/qa/answer-gold-v1.jsonl` | ✅ 30 nhãn |
| Bộ QA retrieval | `data/qa/retrieval-gold-v1.jsonl` | ⚠️ **70 câu — thiếu 50 so với cổng ≥ 120** |
| `statement_scope` của tài liệu | `documents.parquet` | ⚠️ **có sẵn, 0 tham chiếu ở hạ nguồn** — § 1.3 |
| Đường chạy E2E thật (retrieval trong vòng lặp) | — | ❌ **không tồn tại** — § 1.1 |
| Tỷ lệ invalid plan (chỉ số cổng) | `verification/evaluation.py` | ❌ **không tính được** — harness nuốt abstain, § 1.8 |
| Phân lỗi theo tầng | — | ❌ **chưa tồn tại** |
| Latency | đo ở § 1.9 | ✅ p95 = 0,73 s, dư sức so với cổng 15 s |

---

## 1. Chốt chặn phải đo trước khi viết code

### 1.1. Đường ống **chưa bao giờ** chạy end-to-end — retriever bị bỏ qua

`verification/evaluation.py::evaluate_answer_packages_on_gold` hiện làm:

```python
known_table_ids = frozenset(question.gold_table_ids)
plan_result = build_plan(entities, candidate_table_ids=question.gold_table_ids, ...)
...
package = build_answer_package(..., retrieved_table_ids=known_table_ids, ...)
```

`candidate_table_ids` **là** gold, và `retrieved_table_ids` **cũng là** gold. Nên con số "30/30
verified, accuracy 0,9" của Ngày 20 là một phép đo **compiler + verifier với retrieval hoàn hảo**,
không phải phép đo hệ thống. Thay `gold_table_ids` bằng `predicted_table_ids` (BM25 v4 top-10, đã
lưu sẵn từ Ngày 14 — không cần chạy lại index):

| | plannable | answered | verified | rejected | scored | correct | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold tables (Ngày 20 báo cáo) | 51 | **30** | 30 | 0 | 30 | 27 | **0,900** |
| **BM25 v4 top-10 (thật)** | 51 | **9** | 9 | 0 | 9 | 6 | **0,667** |

**Phủ rơi từ 30/70 (42,9 %) xuống 9/70 (12,9 %).** Không câu nào đổi từ đúng sang sai
(`verdict flipped gold→bm25: 0`) — hệ thống hỏng **an toàn**, đúng thiết kế. Nhưng cổng Ngày 21
("answer accuracy ≥ 0,85") đang được báo cáo trên một đường ống không có retrieval.

### 1.2. Thủ phạm là `cell_ambiguous`, và **không phải** do retrieval kém

Phân rã lỗi của đúng cùng 70 câu, chỉ đổi nguồn bảng ứng viên:

| mã lỗi | gold tables | BM25 top-10 | Δ |
| --- | ---: | ---: | ---: |
| abstain `entity_ambiguous` | 10 | 10 | 0 |
| abstain `multi_metric_unsupported` | 9 | 9 | 0 |
| `metric_not_found` | 11 | 10 | −1 |
| `period_unresolved` | 8 | 8 | 0 |
| **`cell_ambiguous`** | **2** | **24** | **+22** |

21 câu mất đi thì **21/21 rơi vào `cell_ambiguous`**. Planner không hề đổi hành vi (abstain y hệt).

Soi 6 ca đầu: **cả 6 đều có `gold_subset_of_retrieved = True`** — retriever tìm đúng bảng gold, mỗi
lần. Cái nó **thêm vào** mới là vấn đề. Ví dụ `retq_00888e79366b911` (CTG, lưu chuyển tiền thuần HĐKD
2022), gold có 2 bảng, retrieval trả 10, và 4 bảng cùng mang chỉ tiêu đó ở kỳ đó:

```
GOLD [separate]     CTG_..._2022_separate      -> 84.420.878 VND_million
GOLD [separate]     CTG_..._2023_separate      -> 84.420.878 VND_million
     [consolidated] CTG_..._2022_consolidated  -> 84.463.729 VND_million
     [consolidated] CTG_..._2023_consolidated  -> 84.463.729 VND_million
```

Hai giá trị **khác nhau thật**, và cái phân biệt chúng nằm ngay trong đường dẫn tài liệu:
`separate` với `consolidated`. Locator đúng khi từ chối — nó không có thông tin để chọn.
`conflicts confined to one document? {False: 6}` — 6/6 xung đột trải trên nhiều tài liệu.

### 1.3. `statement_scope` **đã có sẵn** trong release khoá, và chết ở hạ nguồn

```
documents.parquet.statement_scope:  consolidated 957 | separate 954 | other 53 | aggregated 7
tables by scope:                    consolidated 75.801 | separate 66.885 | other 2.887 | aggregated 438
grep statement_scope src/…/{planning,execution,verification}/  ->  0 kết quả
```

Trường phân biệt chính xác hai nhánh xung đột ở § 1.2 **nằm sẵn trong release**, đã được ingestion
gán cho từng tài liệu, và **không một dòng code planning/execution/verification nào đọc nó**.
`entity_parser.py` cũng **không** bắt từ khoá scope trong câu hỏi (`riêng`, `công ty mẹ`, `hợp nhất`).

Đây là lần lặp thứ năm của đúng một mô hình lỗi. Ba lần trước đều được phát hiện bằng cách đo trước
khi viết code, không phải bằng đọc code.

### 1.4. Scope là **cấu trúc chủ đạo** của corpus, không phải ca biên

```
(company, period, metric) nhóm, toàn corpus        : 22.680
  có mặt ở CẢ HAI scope                            : 14.717  (64,9 %)
  ...và giá trị BẤT ĐỒNG                           : 13.656  (92,8 % của nhóm hai-scope)
  ...trong đó tách sạch 1 giá trị / 1 scope        : 11.553  (84,6 % của nhóm bất đồng)
```

Gần hai phần ba chỉ tiêu tồn tại song song ở hai scope; **93 %** trong số đó bất đồng giá trị. Nhưng
**84,6 % nhóm bất đồng tách sạch**: mỗi scope đúng một giá trị. Nghĩa là **riêng chiều scope đã đủ
khử nhập nhằng cho phần lớn ca** — nếu biết được scope nào được hỏi.

### 1.5. Nhưng **không chính sách scope nào vượt cổng** — đây là biên đánh đổi thật

Đo end-to-end với retrieval thật, cả ba chính sách, trên đúng 70 câu:

| chính sách | answered/70 | scored | correct | **wrong** | accuracy | scope suy diễn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` (hiện tại) | 9 | 9 | 6 | **3** | 0,667 | 0 |
| `default_consolidated` | **25** | 24 | 14 | **10** | **0,583** | 22 |
| `abstain_when_unstated` | 3 | 3 | 3 | **0** | **1,000** | 0 |

Đọc bảng này cho kỹ:

- **Mặc định `consolidated` gần gấp ba phủ (9 → 25) nhưng làm accuracy TỆ ĐI (0,667 → 0,583) và
  nhân ba số câu sai tự tin (3 → 10).** 10/25 = **40 % câu trả lời sai mà vẫn trình bày tự tin** —
  vi phạm thẳng ràng buộc cứng nhất của dự án (`< 5 %`, mục 1.2 plan.md). Đây là cái bẫy: chỉ số
  "answered" đẹp lên trong khi hệ thống xấu đi.
- **Từ chối khi câu hỏi không nêu scope cho accuracy tuyệt đối 1,000, 0 câu sai — và 3/70 = 4,3 %
  phủ.** An toàn tuyệt đối, vô dụng tuyệt đối.

**Kết luận cho cổng:** answer accuracy ≥ 0,85 **không thể đạt trong một ngày** bằng cách thêm một
chiều dữ liệu. Ngày 21 phải đo, dựng harness, phân lỗi, và **ra phán quyết cổng có nhánh xử lý**
đúng như Ngày 14 đã làm với F2@R — không được chọn `default_consolidated` chỉ vì nó làm con số
"answered" đẹp lên.

### 1.6. gold70 **không đại diện** cho tập đề thật, đúng ngay ở chiều scope này

Quét 1.012 câu hỏi ViFinQA chính thức (`data/raw/ViFinQA/questions/questions.jsonl`):

| | không nêu scope | `separate` | `consolidated` | cả hai |
| --- | ---: | ---: | ---: | ---: |
| **Đề chính thức (1.012)** | 630 (62,3 %) | **368 (36,4 %)** | 13 (1,3 %) | 1 |
| **gold70** | 54 (77,1 %) | 5 (7,1 %) | 8 (11,4 %) | 3 |
| bảng gold của gold70 thực tế nằm ở | — | 21 | 45 | 3 |

Hai lệch nghiêm trọng:

1. **Đề chính thức nêu scope trong 37,7 % câu** (chủ yếu "công ty mẹ" → `separate`), gold70 chỉ
   22,9 %. Bộ QA đang **đánh giá thấp** tầm quan trọng của chiều này.
2. **Đề chính thức nghiêng hẳn về `separate` (36,4 % so với 1,3 %); bảng gold của gold70 nghiêng
   ngược lại về `consolidated` (45 so với 21).** Nghĩa là mặc định `consolidated` ở § 1.5 —
   vốn đã tệ trên gold70 — sẽ còn **sai hướng hơn nữa** trên đề thật.

Nên việc "nâng bộ QA lên ≥ 120 câu" **không phải là thêm số lượng**: nó phải sửa chính lệch phân bố
này, nếu không mọi con số cổng sau đó vẫn tiếp tục nói dối.

### 1.7. Một bug locator thật, đã đo được bán kính ảnh hưởng

Trong 4 ca vẫn `cell_ambiguous` **sau khi** đã áp đúng scope, ca `retq_e2059ca4277998a` (OCB,
`profit_after_tax` 2019) có thông điệp lỗi tự tố cáo:

```
conflicting values: 2582236224358.0 None , 2582236224358.0 VND
```

**Giá trị giống hệt nhau.** Chỉ khác: một ô không ghi đơn vị. `locator.py:76` dùng
`drop_duplicates(subset=["value", "unit"])`, nên `(X, None)` và `(X, "VND")` bị đếm là hai cặp phân
biệt → báo xung đột ở nơi **không có xung đột**. Bán kính đo được:

```
(company, scope, metric, period) nhóm                     : 38.459
  có đúng MỘT giá trị phân biệt                           : 34.534
  ...nhưng vẫn >1 cặp (value, unit) -> cell_ambiguous oan : 868
     do một ô thiếu đơn vị                                : 859  (99,0 %)
     do hai đơn vị thật khác nhau (xung đột thật)         :   9  ( 1,0 %)
```

**859 nhóm bị từ chối oan.** Đây là lỗi thật, nhỏ, sửa được, khác với `unit_missing` của Ngày 20
(ở đó ô *duy nhất* thiếu đơn vị; ở đây ô thiếu đơn vị *đồng hành* với ô có đơn vị và cùng giá trị).

### 1.8. Ba nguyên nhân nhập nhằng còn lại **không** phải scope

4 ca còn `cell_ambiguous` sau khi áp scope tách thành ba nhóm khác hẳn nhau:

| ca | nguyên nhân | ghi chú |
| --- | --- | --- |
| `retq_e2059ca4…` (OCB) | đơn vị NULL — § 1.7 | **sửa được ngay** |
| `retq_4ed36641…`, `retq_5a293140…` (NVL) | **bảng báo cáo bộ phận** ("Báo cáo bộ phận theo lĩnh vực kinh doanh") mang cùng chỉ tiêu canonical cho từng bộ phận | chiều *loại bảng*, không phải scope |
| `retq_15f35797…` (VSC) | **bảng biến động vốn chủ**: cùng `retained_earnings` xuất hiện với số dư đầu kỳ và cuối kỳ | chiều *cột/kỳ*, không phải scope |

Hai nhóm sau cần một chiều "loại bảng". Kiểm tra xem `tables.statement_type` có dùng được không:

```
statement_type:  NULL 115.120 (78,8 %) | cash_flow_statement 13.712 | income_statement 7.862
                 balance_sheet 7.745   | notes 1.479 | equity_changes 93
```

**NULL 78,8 %** — không dùng được làm bộ lọc. Nên đây **không** phải món ăn liền; phải ghi thành nợ
có mã lỗi riêng và có số đo, không được cố sửa trong Ngày 21.

### 1.9. Harness nuốt abstain, nên chỉ số cổng "invalid plan < 5 %" không tính được

`evaluate_answer_packages_on_gold` gặp `plan_result.plan is None` thì `continue` **không ghi
`failures`**. Nên báo cáo Ngày 20 liệt kê 21 "failures" đều là lỗi execution; **19 câu abstain hoàn
toàn vô hình**. Đo trực tiếp: 19/70 abstain (10 `entity_ambiguous`, 9 `multi_metric_unsupported`).
Không sửa harness thì cổng Ngày 21 có một chỉ số **không thể báo cáo**.

### 1.10. Nhãn gold **sạch** ở chiều scope, và latency dư sức

Hai tin tốt, đo để khỏi đi sửa nhầm:

- **0 câu có scope nêu trong đề mâu thuẫn với scope của bảng gold.** (Bộ đếm thô ban đầu báo "3 mâu
  thuẫn"; soi từng ca thì cả 3 là câu nêu **cả hai** scope — lỗi phân loại của chính bộ đếm, không
  phải lỗi nhãn. Ghi lại đây để không ai đi "sửa" nhãn đang đúng.)
- **Latency p50 = 0,563 s, p95 = 0,728 s, max = 0,866 s** trên 30 câu (plan → execution → package,
  không tính retrieval vì dùng ranking đã lưu). Cổng là 15 s. **Không phải vấn đề**, không tối ưu gì
  trong Ngày 21.

---

## 2. Quyết định thiết kế

### A. Nơi ở của scope — **A1: trường `statement_scope` trên `FinancialQueryPlan`**

Không lọc ngầm ở tầng retrieval (A2) và không "phá hoà" ở tầng locator (A3). Scope là **thuộc tính
ngữ nghĩa của câu hỏi**, đúng hạng với `companies` và `periods`: phải nhìn thấy được trong plan, phải
qua `validate_plan_semantics`, phải xuất hiện trong artifact audit. Lọc ngầm ở tầng dưới sẽ tái tạo
đúng cái bệnh § 1.3 — một chiều dữ liệu quyết định đáp án mà plan không khai báo.

### B. Khi câu hỏi không nêu scope — **B1: suy diễn có mặc định, và `scope_inferred` là lỗi CHẶN**

§ 1.5 đã loại hai lựa chọn thuần: mặc định im lặng cho 40 % sai tự tin; từ chối thẳng cho 4,3 % phủ.
Lựa chọn thứ ba: **vẫn suy diễn và vẫn tính, nhưng đánh dấu `scope_inferred` và để nó CHẶN**
`verification_status`. Hệ quả: câu đó **không** được trình bày như đáp án chắc chắn, nhưng answer
package vẫn mang đầy đủ số, evidence và lý do — dùng được cho audit, cho UI Ngày 22–23 ("thiếu thông
tin: câu hỏi không nêu riêng hay hợp nhất"), và cho việc đo tiến bộ.

Khác với `period_inferred` (Ngày 20, **không** chặn): kỳ suy diễn sai lệch kỳ, còn scope suy diễn sai
**đổi hẳn con số** — § 1.4 đo được 92,8 % nhóm hai-scope bất đồng giá trị. Mức chặn khác nhau vì bán
kính thiệt hại khác nhau, và cả hai đều có số đo đứng sau.

Mặc định khi suy diễn: **`consolidated`**, vì đó là hình thái phổ biến của gold70 (45/21) — nhưng
ghi rõ trong ADR rằng § 1.6 đo được đề chính thức nghiêng ngược lại, nên mặc định này **phải được đo
lại** khi bộ QA đã sửa lệch phân bố ở nhiệm vụ 21.9. Không khoá nó bằng hằng số trong code; đặt trong
`ExecutionSettings` để đổi được bằng config và đo được bằng ablation.

### C. Nhập nhằng đơn vị NULL — **C1: khử trùng theo giá trị, giữ đơn vị đã biết**

§ 1.7: 859/868 ca nhập nhằng oan chỉ vì `(X, None)` ≠ `(X, "VND")`. Sửa: khi tập ứng viên rút gọn
về **đúng một giá trị phân biệt**, và các đơn vị khác nhau **chỉ ở chỗ có NULL hay không**, thì đó
không phải xung đột — lấy giá trị đó với đơn vị đã biết duy nhất. Nếu có **hai đơn vị thật khác
nhau** (9 ca) thì vẫn `cell_ambiguous` như cũ. Không đụng tới `unit_missing` của Ngày 20: khi
**mọi** ô đều thiếu đơn vị thì vẫn là `unit_missing`.

### D. Nhập nhằng do loại bảng — **D1: mã lỗi riêng + đo, KHÔNG sửa trong Ngày 21**

§ 1.8: `statement_type` NULL 78,8 % nên không có bộ lọc rẻ tiền. Ngày 21 chỉ **tách mã lỗi** để phân
lỗi cho đúng tầng và để đếm được, rồi ghi thành nợ có số. Cố sửa trong hôm nay là đổi một chiều dữ
liệu chưa đo lấy một con số cổng đẹp — đúng cái bẫy § 1.5.

### E. Harness E2E — **E1: module `pipeline/` mới, retrieval nằm TRONG vòng lặp, abstain được ghi**

Không nhét thêm vào `verification/evaluation.py`: nó đang đo compiler+verifier với retrieval hoàn hảo
và **vẫn nên tiếp tục đo đúng thứ đó** (nó là ablation "retrieval hoàn hảo" hữu ích, § 1.1). Đường
E2E thật là một người dùng khác, có một câu hỏi khác. Nó phải:

- nhận bảng ứng viên từ **ranking đã lưu** của BM25 v4 (không rebuild index — ranking Ngày 14 đã
  content-addressed theo đúng fingerprint), và ghi rõ nguồn ranking + sha256 vào báo cáo;
- **ghi lại abstain** thành một hạng mục lỗi hạng nhất (§ 1.9);
- **quy mỗi lỗi về đúng một tầng**: `retrieval` (gold ⊄ retrieved), `planning` (abstain),
  `normalization` (metric/period không khớp được), `execution` (compile/sandbox), `verification`
  (bị verifier chặn). Quy tắc quy tầng phải kiểm được, vì § 1.2 cho thấy trực giác "cell_ambiguous
  = lỗi retrieval" là **sai**.

### F. Bộ QA ≥ 120 — **F1: mở rộng có sửa lệch phân bố, giữ nguyên quy tắc chống rò rỉ của gold70**

Giữ nguyên năm quy tắc trong `retrieval-gold-v1.provenance.md` (chọn tài liệu/bảng **trước** khi
viết câu hỏi; **không** mở bất kỳ ranked list nào; neo `relative_path` + `line_start/end`;
`stable_question_id`; sắp theo `question_id`). Thêm một ràng buộc mới từ § 1.6: **50 câu bổ sung
phải kéo tỷ lệ câu nêu scope của bộ QA về gần 37,7 % của đề chính thức, và nghiêng về `separate`**
đúng như đề chính thức. Nếu không, bộ 120 câu vẫn không đo được cái mà nó cần đo.

### G. Phán quyết cổng — **G1: báo cáo biên đánh đổi, không báo cáo một con số**

Cổng Ngày 21 hỏi một con số accuracy. § 1.5 cho thấy con số đó **là hàm của một chính sách** có thể
chọn tuỳ ý trong khoảng phủ 4 %–36 %. Báo cáo một con số mà không kèm chính sách là tự lừa. Artifact
cổng phải là **bảng biên** (phủ × accuracy × số câu sai tự tin) cho cả ba chính sách, cộng phán quyết
đạt/không đạt kèm nhánh xử lý — đúng khuôn mẫu Ngày 14.

---

## 3. Nhiệm vụ

| # | Việc | Kiểm chứng |
| --- | --- | --- |
| 21.1 | Viết [ADR 0010](../decisions/0010-statement-scope-contract.md) (A1/B1/C1/D1/E1/F1/G1) | ADR tồn tại, 7 quyết định có số đo đứng sau |
| 21.2 | Sửa nhập nhằng oan do đơn vị NULL trong `locator.py` (TDD) | test đỏ tái hiện đúng ca OCB `2582236224358.0 None` vs `… VND`; 9 ca hai-đơn-vị-thật vẫn `cell_ambiguous` |
| 21.3 | Thêm `statement_scope` vào `FinancialQueryPlan` + `plan_validator` (TDD) | plan khai scope không hợp lệ bị chặn; plan cũ không khai vẫn hợp lệ |
| 21.4 | Bắt scope từ câu hỏi trong `entity_parser` (TDD) | `riêng`/`công ty mẹ` → `separate`; `hợp nhất`/`toàn tập đoàn` → `consolidated`; nêu cả hai → nhập nhằng; đo lại đúng 37,7 % trên 1.012 câu đề chính thức |
| 21.5 | Lọc bảng ứng viên theo scope trong `execution/` + `ExecutionSettings.default_statement_scope` | 19 ca § 1 được giải; test kiến trúc: mọi trường `ExecutionSettings` đều được tham chiếu |
| 21.6 | `scope_inferred` — mã lỗi verification **chặn** + trường trên `AnswerPackage` (TDD) | `is_blocking_issue("scope_inferred") is True`; `period_inferred_warning` vẫn không chặn |
| 21.7 | Dựng `pipeline/` + CLI `run-e2e` (retrieval trong vòng lặp, ghi abstain) | chạy thật trên gold70, số khớp § 1.1/§ 1.5 |
| 21.8 | Phân lỗi theo 5 tầng + báo cáo biên đánh đổi 3 chính sách | test cho quy tắc quy tầng, gồm ca `cell_ambiguous` **không** bị quy về retrieval |
| 21.9 | Nâng bộ QA lên ≥ 120 câu, chống rò rỉ, sửa lệch phân bố scope (§ 1.6) | ≥ 120 bản ghi; tỷ lệ nêu scope tiến về 37,7 %; nghiêng `separate`; provenance ghi rõ |
| 21.10 | Mở rộng `answer-gold` cho các câu mới trả lời được | nhãn gán tay từ dòng nguồn, đọc độc lập rồi mới đối chiếu (ADR 0009 A2) |
| 21.11 | Verification đầy đủ + phán quyết cổng + cập nhật `plan.md`/`README` | pytest/ruff/mypy sạch; `git status --short data/processed/ src/…/normalization/` rỗng |

### Chi tiết 21.2 — sửa cho đúng chỗ

Ranh giới phải tách bạch, cả ba nhánh đều cần test riêng:

| tình huống | kết quả đúng |
| --- | --- |
| một giá trị, một ô có `VND`, một ô `NULL` | ✅ trả giá trị đó, đơn vị `VND` (**859 ca**) |
| một giá trị, **mọi** ô đều `NULL` | `unit_missing` (giữ nguyên Ngày 20) |
| một giá trị, hai đơn vị thật khác nhau | `cell_ambiguous` (**9 ca**, giữ nguyên) |
| hai giá trị phân biệt | `cell_ambiguous` (giữ nguyên) |

### Chi tiết 21.8 — quy tầng thế nào cho khỏi đổ oan

§ 1.2 là bài học: 22 ca `cell_ambiguous` mới **trông như** lỗi retrieval (chỉ xuất hiện khi bật
retrieval thật) nhưng **không phải** — bảng gold vẫn nằm trong top-10, cái thiếu là một chiều trong
plan contract. Quy tắc quy tầng phải phân biệt được:

- `retrieval` ⇔ **gold ⊄ retrieved**, và chỉ khi đó;
- `cell_ambiguous` **có** gold trong tập ứng viên ⇒ quy về `planning` (plan thiếu chiều phân biệt),
  **không** phải `retrieval`;
- báo cáo phải in cả hai cột (gold có trong top-10 hay không) để người đọc tự kiểm được kết luận này.

---

## 4. Thứ tự thực hiện

21.1 → 21.2 (sửa lỗi độc lập, giảm nhiễu cho mọi phép đo sau) → 21.3 → 21.4 → 21.5 → 21.6 → 21.7 →
21.8 → 21.9 → 21.10 → 21.11.

Lý do 21.2 đứng trước mọi thứ dính scope: nó là lỗi **không** liên quan scope. Sửa trước thì mọi con
số scope đo sau đó không bị 859 ca oan làm bẩn.

Lý do 21.9/21.10 đứng cuối: chỉ sau khi harness E2E (21.7) và phân lỗi (21.8) chạy được thì mới biết
**câu nào đáng gán nhãn** — gán nhãn đáp án cho câu mà đường ống chắc chắn abstain là phí công.

---

## 5. Definition of Done

- [ ] ADR 0010 có đủ 7 quyết định, mỗi quyết định neo vào một số đo trong § 1.
- [ ] `locator.py` không còn báo `cell_ambiguous` cho 859 nhóm một-giá-trị; 9 nhóm hai-đơn-vị vẫn báo.
- [ ] `FinancialQueryPlan.statement_scope` tồn tại, được validator kiểm, và **được `execution/` đọc**
      (test kiến trúc chặn tái phát mô hình § 1.3).
- [ ] `scope_inferred` chặn `verification_status`; `period_inferred_warning` vẫn không chặn.
- [ ] CLI `run-e2e` chạy được trên release khoá, retrieval **trong** vòng lặp, và **ghi abstain**.
- [ ] Báo cáo có: phân lỗi 5 tầng, cột "gold có trong top-10 không", và **bảng biên 3 chính sách**
      (phủ × accuracy × số câu sai tự tin).
- [ ] Bộ QA ≥ 120 câu, provenance ghi rõ quy tắc chống rò rỉ **và** phân bố scope trước/sau.
- [ ] Phán quyết cổng ghi trong `plan.md`: đạt hay không, theo chính sách nào, nợ gì, chuyển sang ngày nào.
- [ ] `pytest`, `ruff check`, `ruff format --check`, `mypy` sạch.
- [ ] `git status --short data/processed/ src/financial_report_qa/normalization/` rỗng.

---

## 6. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
| --- | --- | --- |
| **Chọn `default_consolidated` vì con số "answered" đẹp** | báo cáo khoe phủ 25/70 mà không in cột "sai tự tin" | § 1.5 đo sẵn: 10/25 = 40 % sai. Bảng biên ở 21.8 **bắt buộc** có cột này |
| Bộ 120 câu chỉ thêm số lượng, giữ nguyên lệch § 1.6 | tỷ lệ nêu scope vẫn ~23 % | 21.9 có tiêu chí phân bố định lượng, kiểm được bằng script |
| Rò rỉ khi mở rộng QA | câu mới trùng khớp bất thường với top-10 | giữ nguyên 5 quy tắc `retrieval-gold-v1.provenance.md`, không mở ranked list |
| Cố sửa nhập nhằng loại bảng (§ 1.8) trong hôm nay | `statement_type` NULL 78,8 % | ADR quyết định D1: chỉ tách mã lỗi và đo, không sửa |
| Gán nhãn đáp án bằng chính output executor | accuracy nhảy lên ~1,0 một cách khả nghi | giữ ADR 0009 A2: đọc nguồn độc lập **trước**, đối chiếu **sau** |
| Cổng không đạt và bị lờ đi | `plan.md` chỉ ghi số đẹp | khuôn mẫu Ngày 14: ghi rõ không đạt + nhánh xử lý + ngày trả nợ |
