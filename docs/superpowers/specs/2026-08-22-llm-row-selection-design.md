# ViFinQA — LLM chọn dòng thay rule-based grounding

- **Ngày:** 2026-08-22
- **Deadline Vòng 1:** 31/08/2026 (9 ngày)
- **Trạng thái:** Design — chờ duyệt
- **Tiền đề:** P0 (compliance) và P1a (tách retrieval) đã land trên `main`
  (commits `5d1b317..51bfdca`, xem
  `2026-08-21-vifinqa-compliance-and-scoring-recovery-design.md`)

---

## 1. Bối cảnh

Sau khi P0/P1a land, submission đã hợp lệ theo thể lệ (0 vi phạm C1–C7) và điểm
truy hồi đã tách khỏi nhánh trả lời. Nút thắt còn lại là **Answer Accuracy**.

Đo trên lần export đầy đủ gần nhất (1012 câu):

| Chỉ số | Giá trị |
|---|---:|
| answered | **53** |
| backstopped | 959 |
| `stage=execution` thất bại | 804 |
| `stage=planning` thất bại | 109 |
| `stage=retrieval` thất bại | 42 |

Phân bố mã lỗi đầy đủ, đo trên chính lần export này
(`artifacts/evaluations/v2gaps_full/submission-export-422df141c935.json`):

```
491  execution: metric_not_found                 48.5%   ← mục tiêu §5
133  execution: evidence_frame_replay_mismatch   13.1%   ← xem 2.1
109  planning:  llm_plan_invalid                 10.8%
 50  execution: cell_ambiguous                    4.9%   ← mục tiêu §6
 49  execution: period_unresolved                 4.8%   ← mục tiêu §6
 42  retrieval: no_candidate_tables               4.2%
 31  execution: unit_missing                      3.1%   ← mục tiêu §6
 30  execution: plan_rejected                     3.0%
 17  execution: candidate_table_ids_scope_empty   1.7%
```

`metric_not_found` là nhóm lớn nhất và là mục tiêu chính của thiết kế này.
Ba nhóm `cell_ambiguous`/`period_unresolved`/`unit_missing` (130 câu) là mục
tiêu của §6.

### 1.1 Về 133 câu `evidence_frame_replay_mismatch`

Mã này do P0 Task 3 sinh ra: nhánh answered tìm được đáp án nhưng bảng nguồn
thật không replay ra đúng giá trị đó, nên câu bị đẩy sang backstop thay vì
đóng gói một CSV dựng ngược từ đáp án. Đây là hành vi **đúng theo thiết kế** —
nó là cái giá của việc chặn hardcode, không phải hồi quy.

Thiết kế này **không nhắm trực tiếp** vào nhóm đó, nhưng nhiều khả năng làm nó
co lại như tác dụng phụ: khi LLM chọn dòng, `MetricSelector` trở thành
position-bound (`table_id` + `row_idx`), và theo bảng đo trong spec trước
(§2.2), predicate theo vị trí có tỷ lệ nhập nhằng **0.00%** so với **31.58%**
của predicate ngữ nghĩa đang dùng. Nhập nhằng chính là nguyên nhân khiến replay
lệch. Cần đo lại sau khi land — không cam kết trước.

### Ràng buộc

- **Mọi model < 14B tham số** (BTC công bố). Xem §3 — ràng buộc này loại bỏ
  đề xuất model cũ trong spec trước.
- Compute: local RTX 3050 Laptop 6GB VRAM + Colab/Kaggle T4/P100 16GB.
- **Không re-ingest corpus.** Giữ nguyên
  `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
- `SubmissionItem` giữ đúng 7 trường, `extra="forbid"`.
- Compliance linter (C1–C7) là chốt chặn cứng — mọi thay đổi ở đây vẫn phải
  đi qua nó, không có đường vòng.

---

## 2. Nguyên nhân gốc của `metric_not_found`

`locate()` (`execution/locator.py:174`) khớp dòng qua `_metric_mask`, vốn có ba
đường: position-bound, canonical dictionary, hoặc raw-text normalized. Đường
đang dùng chủ yếu là dictionary/raw-text — tức **so khớp nhãn**. Câu hỏi tiếng
Việt diễn đạt tự do ("Tổng số dư phải thu khác ngắn hạn") hiếm khi trùng khít
`row_label_raw` trong báo cáo ("Phải thu ngắn hạn khác"), nên `metric_rows`
rỗng → `metric_not_found`.

`ground_with_recovery` (`planning/cell_grounding.py:133`) đã có thang phục hồi
3 tầng (raw-metric rule → LLM chọn nhãn → candidate switching → context
expansion), nhưng tầng đầu vẫn là rule-based dictionary, và LLM chỉ được gọi
**sau khi rule thất bại**. Log thực tế cho thấy tầng phục hồi chấp nhận những
dòng có `fused_score` chỉ 0.018–0.02 — gần như tạp nhiễu.

**Kết luận:** so khớp nhãn bằng luật là sai dạng bài. Việc "dòng nào trong 20
dòng ứng viên trả lời câu hỏi này" là **phân loại có ràng buộc** — đúng dạng
bài model 7–8B làm tốt, khác hẳn sinh code tự do mà nó làm kém (N4 trong spec
trước).

---

## 3. Chọn model — sửa một mâu thuẫn trong tài liệu cũ

Spec trước (§7.4) đề xuất **Qwen3-14B hoặc Qwen2.5-14B-Instruct**. Cả hai đều
có **~14.7B tham số thật**, tức **vi phạm chính ràng buộc `< 14B`** mà cùng tài
liệu đó đặt ra ở Global Constraints. Mục VIII thể lệ liệt kê vi phạm giới hạn
model là căn cứ loại đội thi. Đây là lỗi sót trong tài liệu cũ.

**Quyết định: Qwen3-8B** (8.2B tham số).

| Model | Tham số | Hợp lệ <14B | Ghi chú |
|---|---:|:---:|---|
| Qwen3-14B / Qwen2.5-14B-Instruct | ~14.7B | ✗ | Đề xuất cũ — **loại** |
| **Qwen3-8B** | 8.2B | ✓ | **Chọn.** Cùng dòng Qwen đang dùng (`qwen3:4b` planning, Qwen3-Embedding-4B ở §6.3 spec cũ) |
| Qwen2.5-7B-Instruct | 7.6B | ✓ | Dự phòng nếu Qwen3-8B không đạt |
| Mistral-Nemo-12B-Instruct | 12B | ✓ | Dự phòng nếu cần model lớn hơn |

q4_K_M (~5GB) vừa T4 16GB thoải mái, chạy batch offline.

---

## 4. Kiến trúc mục tiêu

```
câu hỏi + top-k bảng (Nhánh 1 retrieval — KHÔNG đổi)
   │
   ▼
row_fusion.retrieve_rows(question, candidate_table_ids, k=20)
   │  BM25+dense+fuzzy+alias trên toàn bộ dòng của top-k bảng
   │  (đã wire sẵn tại submission/cli.py:220-276, đang tắt mặc định)
   ▼
row_choice_batch.py  ──► batch_*.jsonl  ──► [COLAB: Qwen3-8B q4]
   │                                              │
   │                                              ▼
   │                                    decisions.jsonl
   │                                    {question_id, chosen_index}
   ▼                                              │
row_choice_decision.py ◄───────────────────────────┘
   │  map chosen_index → cell thật (table_id, row_idx)
   ▼
MetricSelector position-bound  (locator.py:55 `_position_mask` — đã có)
   │
   ▼
locate()  ── nếu period_unresolved / cell_ambiguous / unit_missing:
   │            áp tie-break P2c thay vì abstain (§6)
   ▼
compiler → sandbox → verification → compliance linter → submission
   (toàn bộ phần này KHÔNG đổi)
```

**Bất biến then chốt:** mũi tên từ LLM chỉ mang **một chỉ số nguyên**
(`chosen_index`), không bao giờ mang giá trị đáp án. Đáp án vẫn được tính từ
CSV bằng pandas trong sandbox. Điều này giữ nguyên tinh thần N4 và bảo đảm
compliance linter vẫn có ý nghĩa.

---

## 5. P2a/P2b — LLM chọn dòng

### 5.1 Phạm vi thay đổi

**Thay:** bước chọn dòng/metric. Rule-based dictionary matching
(`ground_raw_metric`, `planning/raw_metric_grounding.py`) bị **xóa khỏi đường
chính**; LLM là nguồn quyết định dòng duy nhất cho **cả 1012 câu**.

**Giữ nguyên:**
- Retrieval Nhánh 1 (BM25 hiện tại) — nâng retriever là spec riêng
- `plan_router`/rule planner cho **loại phép tính** (`growth_rate`, `ratio`,
  `statement_scope`…) — đây là "làm phép gì", khác với "dòng nào"
- Compiler deterministic, sandbox, verification, submission, compliance linter

### 5.2 Chế độ batch offline

Chạy 1012 câu một lượt trên Colab, xuất file quyết định, mang về máy local.
Submission chạy hoàn toàn không cần GPU.

Lý do (giữ nguyên lập luận §7.4 spec cũ): tách vòng lặp local khỏi session
Colab. Một lần chạy full export mất ~3 giờ; không được để nó phụ thuộc vào một
session Colab còn sống hay không.

### 5.3 Định dạng file

**Batch (`data/qa/row_choice_batches/batch_NNN.jsonl`)** — một câu mỗi dòng:

```json
{
  "question_id": 795,
  "question": "Chi phí trả trước ngắn hạn khác của X cuối năm 2023 là bao nhiêu?",
  "candidates": [
    {"index": 0, "row_label": "Chi phí trả trước ngắn hạn khác",
     "row_group_context": "IV. Tài sản ngắn hạn khác",
     "statement_type": "balance_sheet", "periods": ["2023", "2022"],
     "units": ["VND"], "table_title": "BẢNG CÂN ĐỐI KẾ TOÁN"}
  ]
}
```

Trường lấy từ `RowFusedCandidate.metadata` (`retrieval/row_documents.py:20`):
`row_label_raw`, `row_group_context_raw`, `statement_type`, `title`, `periods`,
`units`. Không đưa `value` vào prompt — model không cần thấy giá trị để chọn
dòng, và đưa vào sẽ tạo cám dỗ model "chọn theo con số trông đúng".

**Quyết định (`data/qa/row_choice_decisions.jsonl`)**:

```json
{"question_id": 795, "chosen_index": 0}
```

Chỉ hai trường. Mọi thông tin khác được tra lại từ candidate list ở local —
file quyết định không được phép mang dữ liệu có thể mâu thuẫn với corpus.

### 5.4 Prompt

Constrained JSON output (`json_schema_constrained: true` như config Ollama hiện
tại), schema `{"chosen_index": integer}`. Temperature 0.0. Nếu model trả index
ngoài `[0, len(candidates))` → coi như quyết định không hợp lệ (§7).

---

## 6. P2c — Tie-break thay vì abstain

`locate()` có đúng ba điểm abstain sau khi dòng đã được định vị. Mỗi điểm nhận
một luật ưu tiên riêng, đúng thứ tự spec cũ §7.3 nêu (statutory → kỳ gần nhất →
đơn vị phổ biến):

| Điểm abstain | Vị trí | Luật tie-break |
|---|---|---|
| `cell_ambiguous` | `locator.py:209` | **Ưu tiên dòng statutory.** `_prefer_statutory_rows` (`locator.py:116`) **đã tồn tại** nhưng bị khóa sau `prefer_statutory_rows: bool = False` mà không caller nào bật. Bật lên. Nếu vẫn còn xung đột: chọn `(value, unit)` xuất hiện nhiều nhất; hòa tiếp thì chọn theo `(table_id, row_idx, col_idx)` nhỏ nhất để tất định. |
| `period_unresolved` | `locator.py:186` | **Kỳ gần nhất.** Chọn cell có `period` gần `period` yêu cầu nhất; hòa (cách đều hai bên) thì ưu tiên kỳ **muộn hơn**. |
| `unit_missing` | `locator.py:224` | **Đơn vị phổ biến nhất trong bảng.** Suy đơn vị từ các cell khác cùng `table_id`; nếu bảng không có cell nào có đơn vị thì mới abstain thật. |

**Kích hoạt qua config, không sửa mặc định.** Theo đúng khuôn mẫu đã có trong
`configs/submission_maximize_correct.yaml`: file đó bật
`default_statement_scope: consolidated` cho bản nộp, trong khi
`configs/local_rtx3050.yaml` giữ mặc định thận trọng để đo chất lượng nội bộ.
Thêm cờ `execution.resolve_ambiguity_by_priority: true` vào overlay nộp bài.

Lập luận giữ nguyên từ spec cũ §7.3 và từ chính comment trong
`submission_maximize_correct.yaml`: Answer Accuracy = correct/**total**, nên
sai và bỏ trống đều bằng 0 — abstain không mua được gì.

**Ràng buộc bắt buộc:** thay đổi này **không được** vi phạm P0. CSV vẫn là lát
cắt bảng thật, đáp án vẫn phải replay đúng từ CSV. Compliance linter C7 canh
gác điều này tự động.

---

## 7. Xử lý lỗi

| Tình huống | Xử lý |
|---|---|
| `row_fusion` trả 0 ứng viên | Rơi xuống backstop (Task 4 P0, không đổi) |
| Thiếu `question_id` trong file quyết định | Tie-break tất định trên candidate list (§7.1), **không** gọi LLM lần 2 |
| `chosen_index` ngoài phạm vi / không parse được | Như trên |
| Dòng đã chọn nhưng `locate()` vẫn thất bại | Áp tie-break §6; nếu vẫn thất bại → backstop |
| File quyết định thiếu hoàn toàn | Fail nhanh với thông báo rõ, **không** âm thầm rơi về rule-based đã xóa |

### 7.1 Tie-break khi thiếu quyết định LLM

Chọn ứng viên `rank=1` từ `row_fusion` (điểm fused cao nhất). Đây là quyết định
tất định, giải trình được, và tận dụng được retrieval — khác hẳn `.iloc[0]` tùy
ý. Ghi log rõ ràng để phân biệt câu nào do LLM chọn, câu nào do fallback.

---

## 8. Cấu trúc file

| File | Trách nhiệm | Thao tác |
|---|---|---|
| `src/financial_report_qa/planning/row_choice_batch.py` | Sinh batch JSONL từ candidate list | **Tạo** |
| `src/financial_report_qa/planning/row_choice_decision.py` | Đọc quyết định, map index → cell, dựng `MetricSelector` position-bound | **Tạo** |
| `notebooks/colab_row_choice_qwen3_8b.ipynb` | Notebook Colab: load Qwen3-8B q4, suy luận batch, ghi quyết định | **Tạo** |
| `src/financial_report_qa/execution/locator.py` | Tie-break tại 3 điểm abstain (§6) | **Sửa** |
| `src/financial_report_qa/core/config.py` | Thêm `resolve_ambiguity_by_priority` | **Sửa** |
| `configs/submission_maximize_correct.yaml` | Bật cờ tie-break cho bản nộp | **Sửa** |
| `src/financial_report_qa/planning/cell_grounding.py` | Attempt 0 giờ là đọc quyết định LLM; xóa nhánh rule | **Sửa** |
| `src/financial_report_qa/planning/raw_metric_grounding.py` | Xóa `ground_raw_metric` khỏi đường chính | **Sửa/xóa** |
| `src/financial_report_qa/submission/cli.py` | Bật `row_fusion` mặc định; nhận `--row-choice-decisions` | **Sửa** |

---

## 9. Kiểm thử

Toàn bộ test chạy **không cần LLM thật và không cần Colab**:

- `row_choice_batch.py`: unit test định dạng batch — trường đầy đủ, index liên
  tục từ 0, không rò rỉ `value` vào prompt.
- `row_choice_decision.py`: unit test parse + map index→cell; các ca lỗi
  (index ngoài phạm vi, thiếu question_id, file rỗng).
- Tie-break §6: unit test riêng cho từng luật, dùng frame giả lập — bao gồm ca
  hòa (để khẳng định tính tất định).
- Tích hợp: file quyết định fixture tĩnh → chạy hết compiler/sandbox/compliance,
  xác nhận đường dây nối đúng.
- Hồi quy P0: compliance linter phải vẫn báo **0 vi phạm** trên bundle mới.

---

## 10. Vận hành — quy trình bạn chạy

1. **Local:** sinh batch (nhanh, không cần GPU)
2. **Colab:** upload notebook + batch, chạy Qwen3-8B, tải `decisions.jsonl` về
3. **Local:** đặt file quyết định vào `data/qa/`, chạy `_run_full_export.py`
4. **Local:** đối chiếu `answered_count` trước/sau

Hướng dẫn chi tiết từng lệnh nằm trong plan thực thi.

---

## 11. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| **LLM chọn sai dòng ở những câu rule-based đang đúng** (53 câu answered hiện tại) | Đây là rủi ro đã được chấp nhận có ý thức khi chọn "toàn bộ 1012 câu qua LLM". Giảm thiểu: so sánh `answered_count` và tập id answered trước/sau; nếu tụt, đã có sẵn dữ liệu để quyết định có cần đường lai hay không |
| Qwen3-8B yếu hơn 14B ở tác vụ chọn dòng | Tác vụ là phân loại có ràng buộc, không phải sinh code — dạng bài 8B làm tốt. Dự phòng: Mistral-Nemo-12B (vẫn <14B) |
| Session Colab timeout giữa chừng | Batch chia nhỏ, ghi quyết định tăng dần theo từng batch; chạy lại chỉ tốn phần còn thiếu |
| Tie-break đoán bừa làm giảm chất lượng | Answer Accuracy = correct/total nên không lỗ; và C7 vẫn canh gác tính hợp lệ |
| Xóa rule-based làm mất đường lui | Rule-based vẫn nằm trong git history; và fallback `rank=1` (§7.1) bảo đảm luôn có quyết định |
| Không đủ 9 ngày | P2c (§6) độc lập với Colab, land được ngay. P2a/P2b cần Colab — nếu kẹt, P2c một mình vẫn có giá trị đo được |

---

## 12. Ngoài phạm vi

- **P1c** — nâng retriever lên Qwen3-Embedding-4B + Reranker-4B (spec riêng,
  cần phiên Colab riêng để index lại corpus)
- Sweep `k` theo F2/MRR5 — bị chặn vì `expected-tables.csv` không phải mapping
  câu hỏi→bảng gold; `sweep_k` đã sẵn sàng khi có gold thật
- Multi-agent / reflection loop
- Sản phẩm demo (chatbot/dashboard, mục VI.2 thể lệ)
- Re-ingest corpus

---

## 13. Tiêu chí thành công

1. Compliance linter vẫn báo **0 vi phạm** trên bundle mới (không được hồi quy P0).
2. `metric_not_found` giảm mạnh so với **491** — mục tiêu trực tiếp của §5.
3. `answered_count` **> 53**, và mỗi câu answered replay đúng từ CSV trong sandbox.
4. Ba mã `cell_ambiguous` (50) / `period_unresolved` (49) / `unit_missing` (31)
   — tổng 130 câu — giảm về gần 0 sau khi bật tie-break §6.
5. Log phân biệt rõ câu nào do LLM chọn, câu nào do fallback `rank=1`.
6. **Quan sát, không cam kết:** ghi lại `evidence_frame_replay_mismatch` (hiện
   133) trước/sau, để kiểm chứng giả thuyết ở §1.1.
