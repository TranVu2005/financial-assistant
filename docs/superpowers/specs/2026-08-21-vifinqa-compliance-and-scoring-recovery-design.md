# ViFinQA — Khắc phục vi phạm thể lệ & thu hồi điểm số

- **Ngày:** 2026-08-21
- **Deadline Vòng 1:** 31/08/2026 (10 ngày)
- **Trạng thái:** Design — chờ duyệt

---

## 1. Bối cảnh

Pipeline hiện tại (~29.5k LOC) có kiến trúc symbolic đầy đủ: ingestion → normalization →
retrieval → planning → execution → verification → submission. Lần export gần nhất
(`artifacts/submissions/v2gaps_full.zip`, 10,877s) trả lời được 186/1012 câu.

Ba tài liệu chi phối thiết kế này:

| Nguồn | Nội dung quyết định |
|---|---|
| `THỂ LỆ STAGE 2_2026.pdf` mục VII | Truy hồi = **50%** điểm, chấm F2 macro, độc lập với Answer Accuracy |
| `THỂ LỆ` bổ sung (evidence CSV) | Kết quả phải tính trực tiếp từ CSV; cấm hardcode; vi phạm → **không tính điểm** |
| Dashboard chấm điểm công khai (10 cột, xem §2.3bis) | Truy hồi tách riêng DOCS/TABLES, mỗi cái đủ Precision/Recall/F2/MRR5 — không chỉ F2 |
| `_2026__AIGuru___Finance_Tabular_QA 1.pdf` | Baseline đo trên chính ViFinQA: Recall@10 80.8%; QA retrieved 62–64%; 89.3% lỗi là extraction/evidence |

### Ràng buộc

- **Mọi model < 14B** (BTC công bố; áp dụng cả embedding và reranker)
- Compute: local RTX 3050 Laptop **6GB VRAM** + Colab/Kaggle **T4/P100 16GB**
- 14B-q4 (~9GB) **không chạy được local**; chỉ chạy trên Colab dạng batch offline

---

## 2. Bằng chứng đo được

### 2.1 Vi phạm quy định evidence CSV (P0)

Đo trực tiếp trên `v2gaps_full.zip`:

```
Tổng số câu                              1012
CSV chỉ có 1 dòng dữ liệu                 964   (95.3%)
pandas_query dùng .iloc[0]               1009   (99.7%)
answer == giá trị duy nhất trong CSV      964   ← VI PHẠM
```

Ví dụ câu `id=1`:

```csv
company_code,row_label_canonical,row_label_raw,column_label,period,value
VJC,,Lãi tiền gửi,,2018,208253.2012980
```
```python
df1[(df1.company_code=="VJC") & (df1.row_label_raw=="Lãi tiền gửi") & (df1.period==2018)]["value"].iloc[0]
```

CSV chứa đúng một ô và ô đó **là** đáp án. Về chức năng đây là
`result = df["answer"].iloc[0]` — mẫu thể lệ nêu đích danh là không hợp lệ.
`backstop_answer.py` tự mô tả hành vi này trong docstring:
*"builds a trivially self-consistent lookup around it, so the packaged CSV +
pandas_query always replay to the declared answer."*

Ở nhánh backstop, nhãn dòng còn không liên quan tới câu hỏi:

| id | Câu hỏi về | `row_label_raw` trong CSV |
|---|---|---|
| 2 | Số dư cho vay khách hàng ngành Thương mại | `Số dư đầu năm` |
| 3 | Chi phí dự phòng | `Lợi nhuận sau thuế (triệu đồng)` |

**Rủi ro:** không chỉ mất điểm 964 câu. Mục VIII liệt kê *"Hardcode đáp án benchmark"*
là hành vi không hợp lệ, BTC có quyền *"Loại đội thi vi phạm thể lệ"*.

### 2.2 Nguyên nhân gốc của 2.1

`_real_table_evidence_rows()` (exporter.py:129) làm đúng việc — xuất bảng thật —
nhưng chỉ được gọi trên nhánh **answered** (exporter.py:528), không bao giờ trên nhánh
backstop; và nó tự huỷ kết quả khi replay không khớp (exporter.py:167).

Đo trên toàn corpus (`release_v2_422df141c935`), tỷ lệ predicate khớp >1 giá trị khác nhau:

| Predicate | Số nhóm | Nhập nhằng |
|---|---:|---:|
| `company_code + row_label_raw + period` ← **đang dùng** | 378,133 | **31.58%** |
| `+ column_label_raw` | 451,701 | 26.79% |
| `+ table_id` (phạm vi 1 bảng) | 766,360 | **4.37%** |
| `table_id + row_idx + col_idx` | 822,615 | 0.00% |

Predicate hiện tại nhập nhằng 31.6% → replay trên bảng thật lấy trúng giá trị khác →
`return None` → rơi về CSV một dòng. Đây là cơ chế sinh ra 964 file vi phạm.

### 2.3 Mất điểm truy hồi (P1)

Thể lệ chấm truy hồi **độc lập** với việc trả lời được hay không. Nhưng:

- `_relevant_docs_and_tables()` (exporter.py:172) chỉ lấy bảng từ `compiled.evidence`
  — tức chỉ những bảng **đã dùng để tính**, không phải tập bảng đã truy hồi.
- `build_backstop_item()` nhận `candidate_table_ids` (10 bảng) nhưng emit đúng **1 bảng**
  suy ra từ một ô tuỳ ý, cho **826/1012** câu.

Retriever chỉ trượt 42/1012 câu (4.2%) ở mức `no_candidate_tables`, nhưng submission chỉ
báo cáo bảng đúng cho ~18% câu. Điểm truy hồi đang bị buộc chặt vào nhánh answering.

**F2 nghiêng recall 4×, nhưng precision giảm theo `1/k`.** Ước lượng với 1 gold table/câu:

| k | Recall | Precision | F2 |
|---|---|---|---|
| 1 | 0.45 | 0.45 | 0.450 |
| 3 | 0.65 | 0.22 | **0.465** |
| 5 | 0.72 | 0.14 | 0.400 |
| 10 | 0.80 | 0.08 | 0.286 |

`k=10` hiện tại có thể thấp hơn `k=3` tới 40%. Cần sweep thực nghiệm.

### 2.3bis Dashboard chấm điểm thật: 10 cột, không chỉ "F2 macro"

Cột dashboard công khai của cuộc thi:

```
EXECUTION ACCURACY | TABLES F2-MACRO | DOCS F2-MACRO | TABLES PRECISION |
TABLES RECALL | TABLES MRR5 | DOCS PRECISION | DOCS RECALL | DOCS MRR5 |
ANSWER ACCURACY
```

Điều này chi tiết hơn tóm tắt "Precision/Recall/F2 macro" của mục VII trong PDF thể lệ.
Hai điểm mới có tác động đến thiết kế:

1. **DOCS và TABLES được chấm riêng, không gộp làm một.** `relevant_docs` và
   `relevant_tables` phải đúng độc lập — tối ưu một cái không tự động tối ưu cái kia.
   Trong dữ liệu này, một tài liệu (`doc_id`) chứa nhiều bảng, nên `relevant_docs` gần
   như luôn dễ đạt độ chính xác cao hơn `relevant_tables`; không được coi nhẹ tables vì
   docs "coi như đã đúng theo".

2. **MRR5 đo thứ hạng, không phải tập hợp.** `MRR5 = 1/rank_của_kết_quả_đúng_đầu_tiên_trong_top_5`
   (0 nếu không có kết quả đúng nào lọt top-5). Khác hẳn F2/Precision/Recall — những chỉ
   số coi `relevant_tables` là một **tập** không thứ tự. Hệ quả trực tiếp:

   > **Thứ tự các phần tử trong mảng `relevant_docs`/`relevant_tables` phải là thứ tự
   > retrieval-rank (điểm cao nhất trước), không phải thứ tự bất kỳ khác** (alphabet
   > theo `table_id`, thứ tự xuất hiện trong DB, v.v.). Một implementation đúng bảng
   > (đúng tập F2) nhưng sai thứ tự (bảng đúng nằm ở vị trí 8 thay vì vị trí 1) mất gần
   > hết điểm MRR5 dù F2 không đổi.

   `retrieve_candidate_table_ids()` (`retrieval/live_query.py:21`) đã trả về đúng thứ tự
   này — docstring ghi rõ *"in retrieval-rank order"*. Nhưng `build_cell_frame()`
   (`execution/cell_frame.py:123`) sắp lại theo `ORDER BY c.table_id, c.row_idx, c.col_idx`
   — **alphabet theo table_id**, không phải theo điểm. Bất kỳ implementation nào của
   `_relevant_docs_and_tables()` lấy dữ liệu qua `build_cell_frame()` rồi duyệt theo thứ
   tự nó trả về sẽ làm hỏng MRR5 một cách âm thầm — F2 vẫn đúng, kiểm tra "câu nào cũng
   đủ bảng" vẫn pass, nhưng điểm dashboard tụt mà không ai biết vì sao. Xem §6.1 để biết
   cách tránh.

### 2.4 Phân bố lỗi trả lời (P2)

```
491  execution: metric_not_found          48.5%
186  answered                             18.4%
109  planning: llm_plan_invalid           10.8%
 50  execution: cell_ambiguous             4.9%
 49  execution: period_unresolved          4.8%
 42  retrieval: no_candidate_tables        4.2%
 31  execution: unit_missing               3.1%
 30  execution: plan_rejected              3.0%
 24  còn lại                               2.4%
```

`metric_not_found` **tăng** từ 304 → 491 so với lần đo trước: sửa được planning khiến
nhiều câu đi xa hơn, rồi chết ở cổng metric dictionary. Retrieval không phải nút thắt.

---

## 3. Nguyên tắc thiết kế

**N1 — Tách hai nhánh theo cấu trúc chấm điểm.** Thể lệ chấm truy hồi và trả lời độc lập,
nên hai nhánh phải độc lập trong code. Nhánh answering thất bại không được phép ghi đè
đầu ra của nhánh retrieval.

**N2 — Evidence CSV luôn là lát cắt của bảng nguồn, không bao giờ tổng hợp ngược từ đáp án.**

**N3 — Metric dictionary là tín hiệu scoring, không phải cổng chặn.** (Đã kết luận trong
`ViFinQA_Design_V2_Recommendations.md`, chưa thi hành.)

**N4 — Giữ compiler, thay đầu vào grounding.** Compiler sinh pandas deterministic là
điểm mạnh: nó miễn nhiễm với tỷ lệ crash cú pháp 99% mà tài liệu Day 1 đo được ở model
<10B. Không thay bằng free-form code generation. Thay vào đó, đổi *nguồn quyết định dòng*
từ rule-based dictionary sang LLM chọn trong danh sách ứng viên hữu hạn — một tác vụ
lựa chọn có ràng buộc mà 14B làm tốt, khác hẳn sinh code tự do mà nó làm kém.

**N5 — Không quantize embedding/reranker.** Đầu ra là toạ độ/điểm số liên tục dùng để
so sánh; nhiễu lượng tử hoá làm lệch thứ hạng ứng viên gần nhau, và không Critic nào bắt
được lỗi này (tài liệu Day 2, trang 27).

---

## 4. Kiến trúc mục tiêu

```
                 1012 câu hỏi
                      │
        ┌─────────────┴──────────────┐
        ▼                            ▼
┌──────────────────────┐   (top-k table_ids)
│ NHÁNH 1 — RETRIEVAL  │────────────┐
│ 50% điểm, KHÔNG LLM  │            │
│                      │            ▼
│ BM25 + Qwen3-Emb-4B  │   ┌──────────────────────────┐
│   → RRF → Reranker-4B│   │ NHÁNH 2 — ANSWER + EXEC  │
│   → top-k* (sweep F2)│   │                          │
│                      │   │ row retrieval (đã có)    │
│ relevant_docs        │   │      ↓                   │
│ relevant_tables      │   │ LLM 14B chọn dòng        │
│ cho CẢ 1012 câu      │   │      ↓                   │
└──────────┬───────────┘   │ compiler → sandbox       │
           │               │      ↓                   │
           │               │ verification             │
           │               └────────────┬─────────────┘
           │                            │
           │        answer / pandas_query / csv_path
           ▼                            ▼
      ┌──────────────────────────────────────┐
      │  SUBMISSION LINTER (chốt chặn cứng)  │
      │  fail build nếu dính mẫu hardcode    │
      └──────────────────┬───────────────────┘
                         ▼
                   submission.zip
```

Điểm mấu chốt: mũi tên từ Nhánh 1 sang submission **không đi qua** Nhánh 2.

---

## 5. P0 — Evidence CSV hợp lệ (ưu tiên tuyệt đối)

### 5.1 Hợp đồng evidence — 4 bất biến

**BI-1. CSV là lát cắt của bảng nguồn.** Nguồn duy nhất được phép là
`build_cell_frame(release_dir, table_ids)`. Cấm mọi đường sinh dòng tổng hợp.
`build_backstop_item()` bị cấm dựng `row` thủ công.

**BI-2. Giữ trọn bộ bảng nguồn.** Mọi `row_label` × mọi `period` × mọi `column_label`
của bảng đã grounding. Không lọc xuống dòng đáp án. (Quyết định của người dùng:
"toàn bộ bảng nguồn" — an toàn nhất khi giải trình, dễ chứng minh truy vết.)

**BI-3. Phép tính nằm trong query, không nằm trong CSV.** Câu hỏi dẫn xuất
(`growth_rate`, `ratio`, `difference`, `average`) phải có CSV chứa **đầu vào** và
`pandas_query` chứa phép toán. Cần audit lại toàn bộ nhóm này — hiện chỉ 17/1012 file
có 2 dòng, con số quá thấp so với tỷ lệ câu hỏi Medium/Intermediate (43%).

**BI-4. Backstop không tổng hợp gì cả.** Nó xuất bảng top-1 đã retrieve nguyên vẹn +
một lookup best-effort. Đáp án sai → mất điểm Answer (chấp nhận được). Đáp án bịa từ
CSV bịa → mất tư cách dự thi (không chấp nhận được).

### 5.2 Sửa predicate để bảng thật replay đúng

Đổi predicate sinh bởi compiler từ:

```python
(company_code == C) & (row_label_raw == R) & (period == P)          # 31.58% nhập nhằng
```

thành, với CSV thu hẹp về **một** `table_id`:

```python
(row_label_raw == R) & (column_label_raw == L) & (period == P)      # 4.37% nhập nhằng
```

Với 4.37% còn lại (dòng trùng nhãn nhưng khác giá trị — lỗi OCR lặp dòng), thêm
`row_idx` làm **tie-break vị trí cuối cùng**, luôn đi kèm predicate ngữ nghĩa:

```python
(row_label_raw == R) & (column_label_raw == L) & (period == P) & (row_idx == I)
```

Phần ngữ nghĩa mang ý nghĩa, `row_idx` chỉ phá thế hoà giữa các nhãn giống hệt nhau.
Đây là lựa chọn có thể giải trình, khác với lookup thuần vị trí.

### 5.3 Submission linter — chốt chặn cứng

Module mới `src/financial_report_qa/submission/compliance.py`. Chạy trước khi zip,
**fail build** (exit code khác 0) nếu bất kỳ câu nào dính:

| Mã | Điều kiện fail |
|---|---|
| `C1` | CSV có < 2 dòng dữ liệu |
| `C2` | `answer` trùng khít giá trị duy nhất trong CSV |
| `C3` | CSV có cột tên `answer` / `result` |
| `C4` | `pandas_query` chứa literal số trùng `answer` |
| `C5` | `pandas_query` không tham chiếu cột nào của CSV |
| `C6` | `row_label` trong query không tồn tại trong CSV |
| `C7` | replay CSV + query trong sandbox ≠ `answer` |

`C7` là bất biến quan trọng nhất: nó chứng minh đáp án **thực sự** được tính từ CSV.
Linter phải chạy trong CI và trong `submission export`, không phải là script tuỳ chọn.

### 5.4 Kiểm chứng

- Đo lại phân bố số dòng/CSV: kỳ vọng **0 file có 1 dòng**.
- Chạy linter trên `v2gaps_full.zip` cũ để xác nhận nó bắt được 964 vi phạm (test hồi quy).

---

## 6. P1 — Tách nhánh truy hồi (50% điểm)

### 6.1 Tách đường đi — giữ nguyên thứ tự retrieval-rank (bắt buộc cho MRR5)

`relevant_docs` / `relevant_tables` lấy từ **đầu ra retrieval stage**, cho cả 1012 câu,
không phụ thuộc `compiled.evidence`. Nhánh answering chỉ đóng góp `answer`,
`pandas_query`, `csv_path`.

**Ràng buộc bổ sung do §2.3bis:** nguồn dữ liệu duy nhất hợp lệ cho danh sách này là
`retrieve_candidate_table_ids()` (`retrieval/live_query.py:21`), vì nó là nơi duy nhất
giữ đúng thứ tự retrieval-rank. **Không được** dựng lại danh sách bằng cách duyệt
`build_cell_frame()` — hàm đó `ORDER BY table_id` (alphabet), phá thứ tự và làm hỏng
MRR5 một cách âm thầm (F2 vẫn đúng nên không có test nào tự nhiên bắt được lỗi này).

Quy tắc dựng: với `retrieved_table_ids` đã ở đúng thứ tự rank —

1. Khử trùng lặp bằng `dict.fromkeys(retrieved_table_ids)` (giữ lần xuất hiện đầu, giữ thứ tự).
2. Với mỗi `table_id` theo đúng thứ tự đó, tra `doc_id`/`source_line_start` của nó — dùng
   **bất kỳ một** cell thuộc bảng đó làm khoá tra cứu citation, không cần toàn bộ cell.
3. Build `docs`/`tables` bằng cách duyệt **theo thứ tự bước 1**, không theo thứ tự trả về
   của bất kỳ truy vấn DB nào khác.

Với 42 câu `no_candidate_tables`: vẫn phải emit danh sách rỗng hoặc best-effort —
không được emit bảng tuỳ ý.

### 6.2 Sweep k theo F2 và MRR5

Cần một tập gold table để đo. Nguồn: `data/qa/week1_pilot_422df141c935/expected-tables.csv`
(cần xác minh độ phủ). Sweep k ∈ {1,2,3,5,8,10,15}, đo cả F2 macro **và** MRR5 (§2.3bis) —
hai chỉ số có thể tối ưu ở k khác nhau: F2 phạt precision khi k lớn, MRR5 chỉ quan tâm vị
trí kết quả đúng đầu tiên trong top-5 nên gần như không đổi khi k > 5. Chọn k là điểm cân
bằng giữa hai chỉ số, không chỉ tối đa hoá F2 một mình.
Đây là thay đổi tham số, không phải code mới — lợi ích/công sức cao nhất trong toàn dự án.

### 6.3 Nâng retriever

Thay dense encoder hiện tại bằng stack tài liệu Day 1 đã đo trên chính ViFinQA:

| Cấu hình | Recall@10 | Hợp lệ <14B | Vừa T4 16GB |
|---|---|---|---|
| BM25 | 47.41% | ✓ | ✓ |
| BGE-M3 | 53.05% | ✓ | ✓ |
| Qwen3-Embedding-4B | 63.90% | ✓ | ✓ (fp16 ≈ 8GB) |
| Qwen3-Embedding-8B | 67.48% | ✓ | ~ (cần q8) |
| **Qwen3-Embedding-4B + Reranker** | **80.19%** | ✓ | ✓ (chạy tuần tự) |
| Qwen3-Embedding-8B + Reranker | 80.80% | ✓ | ✗ |

Chọn **Qwen3-Embedding-4B + Reranker-4B**: chênh 8B chỉ 0.61% nhưng vừa VRAM.
Index toàn corpus offline trên Colab, tải embeddings về, phục vụ local bằng FAISS/numpy —
không cần GPU khi chạy submission.

Chạy tuần tự (embed toàn bộ trước, rerank sau) để tránh tranh chấp VRAM. Không quantize (N5).

---

## 7. P2 — Nâng Answer Accuracy dưới ràng buộc <14B

### 7.1 Gỡ cổng metric dictionary (491 câu)

`locator.locate()` hiện yêu cầu `MetricSelector` khớp một canonical metric. Đổi thành:
metric dictionary chỉ đóng góp **điểm cộng** cho row retrieval, không quyết định
đi tiếp hay dừng. Câu hỏi không khớp dictionary vẫn phải chảy qua row retrieval → LLM chọn dòng.

### 7.2 LLM chọn dòng thay vì rule-based (theo N4) — TableRAG-lite

**Vì sao không phải TableRAG nguyên bản.** TableRAG (tài liệu Day 2 trang 15) giải
quyết bảng đơn lẻ dài hàng nghìn dòng bằng 2 tầng: schema retrieval (chọn cột) → cell-
targeted pointer (chỉ lấy ô giao điểm). Đo trên corpus thật
(`release_v2_422df141c935`, 130,729 bảng có ô số):

| Số dòng/bảng | Giá trị |
|---|---:|
| Trung vị | 6 |
| p95 | 23 |
| p99 | 33 |
| Max toàn corpus | 240 |
| Bảng >100 dòng | 6 / 130,729 |

Vấn đề gốc của TableRAG — bảng vượt context — không tồn tại ở cấp **một bảng** trong
dữ liệu này. Áp nguyên bộ máy 2 tầng ở cấp đó là giải quyết vấn đề không có.

**Vấn đề thật nằm ở cấp khác: gộp k=10 bảng lại cho LLM.** Trung bình 10 bảng "thường"
cộng lại ~200 ô; 10 bảng lớn nhất corpus cộng lại 3,655 ô. 200 ô thì 14B nhét vừa một
lượt gọi, nhưng đổ thẳng cả 10 bảng thô vào prompt vẫn lãng phí context và tăng nhiễu
cho bước chọn dòng. Đây là chỗ nguyên lý TableRAG — *lọc bằng tín hiệu rẻ trước khi
đưa vào LLM* — vẫn đúng, chỉ dịch từ cấp "trong 1 bảng" sang cấp "giữa các bảng đã
retrieve".

Hạ tầng cho tầng lọc đó **đã tồn tại và đã được wire**: `retrieval/row_fusion.py` +
`row_service.py` (BM25 + dense + fuzzy + alias trên dòng, `submission/cli.py:220-276`)
nhưng đang là tuỳ chọn tắt mặc định. Biến nó thành đường đi chính:

```
câu hỏi + top-k bảng (Nhánh 1, §6)
   → row_fusion.retrieve_rows(query, candidate_table_ids=<10 bảng>, k=20)
     — tầng lọc rẻ, không cần LLM: BM25+dense+fuzzy+alias trên toàn bộ dòng
       của 10 bảng, thu về ~20 dòng ứng viên (có nhãn, kỳ, giá trị)
   → LLM 14B: "dòng nào trả lời câu hỏi này?" (chọn 1 trong 20, output = index)
     — tầng ngữ nghĩa, chỉ chạy trên tập đã lọc hẹp
   → MetricSelector theo vị trí (position-based, locator.py:55 đã hỗ trợ)
   → compiler → sandbox
```

Việc LLM chọn dòng là tác vụ **phân loại có ràng buộc**, không phải sinh code — dạng
bài 14B làm tốt. `_position_mask()` trong locator.py đã hỗ trợ chọn dòng theo vị trí,
không cần viết mới.

### 7.3 Cho phép đoán khi nhập nhằng (~130 câu)

`cell_ambiguous` (50) + `period_unresolved` (49) + `unit_missing` (31): hệ thống đã tìm
thấy dữ liệu nhưng từ chối cam kết. Answer Accuracy = correct/**total**, nên bỏ trống và
sai đều bằng 0 — abstain không mua được gì. Lập luận này đã được ghi trong
`configs/submission_maximize_correct.yaml` cho `statement_scope` (0.325 vs 0.250)
nhưng chưa áp dụng cho locator.

Chọn ứng viên theo thứ tự ưu tiên xác định (ưu tiên dòng statutory, kỳ gần nhất, đơn vị
phổ biến nhất trong bảng) thay vì `.iloc[0]` tuỳ ý.

**Lưu ý:** thay đổi này không được vi phạm P0 — CSV vẫn phải là bảng thật, đáp án vẫn
phải replay đúng từ CSV.

### 7.4 Model & vận hành

- **Model:** Qwen3-14B hoặc Qwen2.5-14B-Instruct, q4_K_M (~9GB), chạy trên Colab T4.
- **Chế độ:** batch offline toàn bộ 1012 câu → xuất file quyết định (question_id → row_index)
  → mang về máy local, submission chạy hoàn toàn không cần GPU.
- Tách biệt này giữ vòng lặp local nhanh và không phụ thuộc session Colab.
- **Bỏ** `qwen3:4b` khỏi đường chính. Tài liệu Day 1 đo Qwen3-4B-Instruct: CoT 11.56%,
  PoT 4.64% — dưới trần hữu dụng.

---

## 8. Thứ tự thực hiện

| Ưu tiên | Hạng mục | Tính chất | Phụ thuộc |
|---|---|---|---|
| **P0** | Evidence CSV + linter (§5) | Tồn vong | — |
| **P1a** | Tách relevant_tables khỏi answering (§6.1) | 50% điểm | — |
| **P1b** | Sweep k theo F2 (§6.2) | 50% điểm | P1a |
| **P1c** | Qwen3-Embedding-4B + Reranker (§6.3) | 50% điểm | P1a |
| **P2a** | Gỡ metric gate (§7.1) | Answer | P0 |
| **P2b** | LLM chọn dòng (§7.2) | Answer | P2a, P1c |
| **P2c** | Cho phép đoán khi nhập nhằng (§7.3) | Answer | P0, P2a |

P0 và P1a độc lập nhau, có thể làm song song. P0 phải xong trước mọi lần nộp.

---

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Sửa predicate làm hỏng các câu đang answered | Linter `C7` replay mọi câu; so sánh answer trước/sau trên tập gold |
| CSV toàn bảng làm zip quá lớn | Đo trước; nếu vượt giới hạn nộp, lùi về "nhóm dòng + mọi kỳ" (phương án 2 đã bàn) |
| Index lại corpus bằng Qwen3-Emb-4B tốn nhiều giờ Colab | Chạy nền sớm, song song với P0; giữ BM25+dense cũ làm fallback |
| Session Colab timeout giữa chừng | Checkpoint theo shard; embeddings ghi ra file tăng dần |
| `expected-tables.csv` không đủ phủ để sweep k | Xác minh ngay ở bước đầu P1b; nếu thiếu, gán nhãn tay ~100 câu |
| Vòng chạy full 3 giờ quá chậm để lặp | Dựng tập con 120 câu để đo nhanh; chỉ chạy full trước khi nộp |

---

## 10. Ngoài phạm vi

- Multi-agent + reflection loop (tài liệu Day 2) — để dành Vòng 2 nếu còn thời gian
- Numeric masking / de-lexicalization — không cần, vì compiler đã sinh code deterministic
- Serving engine (vLLM/SGLang) — batch offline không cần throughput cao
- Sản phẩm demo (chatbot/dashboard, mục VI.2 thể lệ) — hạng mục riêng, không thuộc design này
- Re-ingest corpus (sẽ đổi `dataset_fingerprint`, làm mất mọi baseline đã pin)

---

## 11. Tiêu chí thành công

1. Linter báo **0 vi phạm** trên submission mới; 0 file CSV có 1 dòng.
2. `relevant_tables`/`relevant_docs` được emit từ retrieval cho cả 1012 câu, **đúng thứ tự
   retrieval-rank** (kiểm bằng test: thứ tự khớp `retrieve_candidate_table_ids()`, không
   khớp thứ tự `build_cell_frame()`).
3. Cả 8 chỉ số truy hồi đo được trên tập gold (TABLES/DOCS × Precision/Recall/F2/MRR5),
   không chỉ F2, với k đã cân bằng giữa F2 và MRR5.
4. Answered count > 186, và mỗi câu answered replay đúng từ CSV trong sandbox.
