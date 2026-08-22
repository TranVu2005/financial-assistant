# ViFinQA — Thiết kế lại đúng kiến trúc mục tiêu

> Thay thế: `2026-08-21-vifinqa-compliance-and-scoring-recovery-design.md` (§4 của nó
> là nguồn của kiến trúc dưới đây), `2026-08-22-llm-row-selection-design.md`.
> Ngày: 2026-08-23. Deadline vòng 1: 31/08/2026.

**Mục tiêu:** hệ thống chỉ còn đúng hai nhánh trong kiến trúc mục tiêu, không còn tầng
dự phòng nào khác. Mọi thứ không nằm trên sơ đồ bị xoá khỏi repo.

---

## 1. Bối cảnh

Bản chạy full ngày 2026-08-22 (`artifacts/evaluations/v2gaps_full/`):
`answered 60/1012`, `backstopped 952`. Đối chiếu với spec 2026-08-21:

| Hạng mục | Trạng thái |
|---|---|
| P0 — Evidence CSV + linter C1–C7 | ✅ xong (0/1012 file CSV có 1 dòng) |
| P1a — Tách `relevant_tables` khỏi answering | ✅ xong (có test ghim thứ tự MRR5) |
| P1b — Sweep k | ❌ chưa làm, còn hardcode `k=10` |
| P1c — Qwen3-Emb-4B + Reranker | ❌ chưa làm, `dense_weight=0.0` |
| P2a — Gỡ cổng metric dictionary | ⚠️ gỡ ở `locate()`, nhưng cổng dời lên `build_plan` |
| P2b — LLM chọn dòng | ⚠️ có, nhưng bị `build_plan` chặn 41% câu |
| §5.2 — Predicate ngữ nghĩa + `row_idx` tie-break | ❌ code làm ngược: thuần vị trí |

Vấn đề gốc không phải chất lượng model, mà là **hình dạng hệ thống**: answering hiện là
một thang 7 tầng (rule → evidence planner → plan router → llm_planner grounded → LLM
chọn dòng → candidate switching → context expansion), mỗi tầng có cổng từ chối riêng.
Câu hỏi rơi qua thang này chết ở tầng nào cũng được, và không tầng nào chịu trách nhiệm.

---

## 2. Bằng chứng đo được

Đo trong phiên 2026-08-22/23 trên chính bản chạy full.

### 2.1 Thang tầng: tầng lớn nhất có tỷ lệ trả lời 0%

| `plan_source` | Số câu | Answered |
|---|---|---|
| `llm` (plan router + llm_planner) | 409 | **0** |
| `rule` | 280 | 12 |
| `rule_raw_grounded` | 85 | 15 |
| `llm_row_choice` | 73 | 7 |
| `llm_cell_grounded_recovered` | 49 | 10 |
| `llm_evidence_planner` | 42 | 11 |
| `no_candidate_tables` | 42 | 0 |
| `llm_grounded` | 26 | 1 |
| `llm_cell_grounded_context_expanded` | 6 | 4 |

409 câu (40%) đi qua tầng sinh plan bằng LLM và **không câu nào** ra đáp án — 269 chết ở
`llm_plan_invalid`. Đây là tầng đắt nhất và vô dụng nhất.

### 2.2 `build_plan` chặn 41% câu vì lý do không liên quan đến dòng

Chạy `build_plan` cho cả 1012 câu với nhãn dòng đã được thay sẵn (mô phỏng đúng nhánh
LLM chọn dòng):

```
598  dựng được plan
232  operation_unknown
 97  period_grammar_unsupported
 61  entity_ambiguous
 24  multi_metric_unsupported
```

**414 câu chọn dòng đúng cỡ nào cũng vô ích.** Đó là lý do 970 quyết định LLM chỉ cho ra
73 câu `llm_row_choice`.

Bóc tách 232 câu `operation_unknown`:

- **84 câu 1 công ty** — thực chất là `lookup` một ô có sẵn ("Tỷ lệ sở hữu … là bao nhiêu
  phần trăm?"). Bộ suy luận operation thấy chữ "tỷ lệ" nên tưởng là phép chia.
- **141 câu ≥3 công ty** — `rank` / `compare_companies` / đếm ("Xét nhóm cổ phiếu CEO,
  HPX, KBC, SNZ, VIC, VPI và VRE…"). Một `MetricSelector` không phục vụ được nhiều công ty.

97 câu `period_grammar_unsupported` là do `_PERIOD_PATTERN = ^\d{4}$` — câu hỏi viết
"vào ngày 31/12/2015" thay vì "năm 2015".

### 2.3 Cổng evidence vứt 173 câu đã có đáp án đúng

Đo 87 câu đường tất định:

```
62  lệch đúng hệ số đơn vị (47 × 1e9, 15 × 1e6)  — đã sửa ở 435e9a3
 5  lệch float Decimal↔float64                   — đã sửa ở 435e9a3
18  query_rejected — predicate nhãn không khớp corpus thật
 2  lệch thật
```

18 ca `query_rejected` là hệ quả trực tiếp của việc §5.2 chưa thi hành: query lọc
`row_label_raw == "cho vay khách hàng"` (chữ thường, cách nói của câu hỏi) trong khi
corpus ghi "Cho vay khách hàng".

### 2.4 Retrieval đang chạy ở cấu hình yếu nhất

`dense_weight=0.0`, không reranker. Số đo trên chính ViFinQA (tài liệu Day 1):

| Cấu hình | Recall@10 |
|---|---|
| BM25 (đang dùng) | 47.41% |
| Qwen3-Embedding-4B | 63.90% |
| **Qwen3-Emb-4B + Reranker-4B** | **80.19%** |

Retrieval chiếm 50% điểm và đang bỏ lại 33 điểm phần trăm recall.

---

## 3. Nguyên tắc thiết kế

**N1 — Hai nhánh độc lập trong code.** Nhánh answering thất bại không được ghi đè đầu ra
nhánh retrieval.

**N2 — Evidence CSV luôn là lát cắt bảng nguồn**, không bao giờ tổng hợp ngược từ đáp án.

**N3 — Metric dictionary là tín hiệu scoring, không phải cổng chặn.**

**N4 — Giữ compiler, thay đầu vào.** Compiler sinh pandas deterministic là điểm mạnh;
không thay bằng free-form code generation.

**N5 — Không quantize embedding/reranker.** Đầu ra là điểm số liên tục dùng để xếp hạng;
nhiễu lượng tử hoá làm lệch thứ hạng ứng viên gần nhau và không có Critic nào bắt được.

**N6 — Một đường đi duy nhất cho answering (mới).** Không thang tầng, không tầng dự
phòng ngoài backstop. Một câu hỏi đi qua đúng một chuỗi bước; hỏng ở đâu thì hỏng rõ ở
đó. Lý do: §2.1 cho thấy thang tầng che giấu tầng 0% và khiến không tầng nào chịu trách
nhiệm.

**N7 — Quyết định của LLM là offline và không bao giờ mang giá trị số (mới).** File quyết
định chỉ chứa chỉ số vào danh sách ứng viên dựng lại được ở local. Một file quyết định cũ
hay bị sửa không thể bơm số liệu mâu thuẫn với corpus vào bài nộp — đây là điều giữ cho
cam kết chống hardcode còn đứng vững.

---

## 4. Kiến trúc mục tiêu

```
                          1012 câu hỏi
                               │
        ┌──────────────────────┴───────────────────────┐
        ▼                                              ▼
┌────────────────────────┐                  (top-k* table_ids)
│  NHÁNH 1 — RETRIEVAL   │──────────────────────────┐
│  50% điểm, KHÔNG LLM   │                          │
│                        │                          ▼
│  BM25 ─┐               │        ┌─────────────────────────────────┐
│        ├→ RRF → Rerank │        │   NHÁNH 2 — ANSWER + EXEC       │
│  Emb-4B┘      -er-4B   │        │                                 │
│           ↓            │        │  row retrieval (bm25+fuzzy+     │
│      top-k* (sweep     │        │       alias, k=20)              │
│      F2 ∧ MRR5)        │        │            ↓                    │
│                        │        │  LLM Qwen3-8B (batch offline)   │
│  relevant_docs         │        │  → operation + dòng/công ty     │
│  relevant_tables       │        │            ↓                    │
│  cho CẢ 1012 câu       │        │  lắp plan tất định              │
└───────────┬────────────┘        │            ↓                    │
            │                     │  compiler → sandbox             │
            │                     │            ↓                    │
            │                     │  verification                   │
            │                     └───────────────┬─────────────────┘
            │                                     │
            │            answer / pandas_query / csv_path
            ▼                                     ▼
      ┌───────────────────────────────────────────────────┐
      │      SUBMISSION LINTER (chốt chặn cứng C1–C7)     │
      │      fail build nếu dính mẫu hardcode             │
      └───────────────────────┬───────────────────────────┘
                              ▼
                        submission.zip
```

Hai bất biến của sơ đồ:

1. Mũi tên từ Nhánh 1 sang submission **không đi qua** Nhánh 2.
2. Nhánh 2 là một đường thẳng. Không có mũi tên quay lui, không có tầng thứ hai.

---

## 5. Nhánh 1 — Retrieval

### 5.1 Stack

```
BM25 (đã có)         ─┐
                      ├→ RRF (retrieval/fusion.py, đã có) ─→ Qwen3-Reranker-4B ─→ top-k*
Qwen3-Embedding-4B   ─┘
```

`retrieval/fusion.py::FusionService` đã hiện thực RRF với trọng số bm25/dense và đã có
test — hiện chỉ chưa được nối vào đường chạy live. Nối vào, không viết lại.

### 5.2 Thêm Qwen3-Embedding-4B vào allowlist encoder

`retrieval/dense_contracts.py` hiện chốt cứng:

```python
EncoderName = Literal["bge-m3", "multilingual-e5-small"]
max_sequence_length: Literal[512] = 512
```

Cần: thêm `"qwen3-embedding-4b"` vào `EncoderName`, và nới `max_sequence_length` thành
`Literal[512, 1024, 2048, 8192]` (mặc định giữ 512 cho hai encoder cũ). `revision` vẫn
bắt buộc là SHA 40 ký tự — index phải pin được model.

Không quantize (N5): `dtype: Literal["float32"]` giữ nguyên.

### 5.3 Reranker — module mới

`retrieval/reranker.py`, cùng khuôn với `dense_encoder.py`:

- `RerankerSpec` (model_id + revision pinned, `dtype="float32"`, batch_size)
- `Reranker` Protocol: `score(query: str, documents: Sequence[str]) -> np.ndarray`
- `Qwen3CrossEncoderReranker` — cài đặt thật
- `rerank(query, candidates, scores) -> tuple[str, ...]` — nhận đầu ra RRF, trả thứ tự mới

Chạy **tuần tự** với encoder (embed toàn bộ trước, rerank sau) để tránh tranh chấp VRAM.

Reranker chỉ xếp lại top-N của RRF (N = 50, không phải toàn corpus) — đây là điều làm nó
chạy được trên máy local không GPU trong thời gian chấp nhận được.

### 5.4 Index offline

Colab (notebook mới `notebooks/colab_index_qwen3_emb_4b.ipynb`):

1. Embed toàn bộ corpus bằng Qwen3-Embedding-4B, fp16 trên T4 (≈8GB).
2. Checkpoint theo shard, ghi tăng dần — session timeout không mất việc đã làm.
3. Tải embeddings về, phục vụ local bằng `dense_index.py` đã có (numpy/FAISS).

**Không cần GPU khi chạy submission** — embeddings đã tính sẵn; chỉ reranker cần chạy
lúc query, và nó chạy trên top-50 nên CPU chịu được.

### 5.5 Sweep k

`retrieval/retrieval_scoring.py` đã có bộ chấm 8 chỉ số (TABLES/DOCS × Precision/Recall/
F2/MRR5) nhưng chưa có caller. Thêm CLI `retrieval sweep-k`:

- Gold: `data/qa/week1_pilot_422df141c935/expected-tables.csv` — **phải xác minh độ phủ
  trước**; nếu không đủ, gán nhãn tay ~100 câu.
- k ∈ {1, 2, 3, 5, 8, 10, 15}, đo cả F2 macro **và** MRR5.
- Chọn k cân bằng hai chỉ số, không tối đa hoá F2 một mình: F2 phạt precision khi k lớn,
  MRR5 gần như không đổi khi k > 5.

### 5.6 `relevant_docs`/`relevant_tables`

Giữ nguyên `submission/citation_summary.py` — đã đúng: lấy từ `retrieve_candidate_table_ids()`
theo đúng thứ tự retrieval-rank, không sắp lại theo `build_cell_frame()` (hàm đó
`ORDER BY table_id`, phá MRR5 âm thầm). Đã có test ghim.

42 câu `no_candidate_tables`: vẫn emit danh sách rỗng, không emit bảng tuỳ ý.

---

## 6. Nhánh 2 — Answer

Một đường thẳng, năm bước. Không có bước thứ sáu.

```
row retrieval → LLM batch (operation + dòng) → lắp plan → compiler+sandbox → verification
```

### 6.1 Row retrieval — giữ nguyên

`retrieval/row_fusion.py` (bm25 + fuzzy + alias, `DEFAULT_ROW_CANDIDATE_COUNT = 20`).
Sau khi P1c xong, cân nhắc cho row retrieval dùng chung reranker — nhưng đó là việc sau,
không thuộc spec này.

### 6.2 LLM batch — quyết cả operation lẫn dòng

Model: **Qwen3-8B** (8.2B < 14B). Sơ đồ gốc ghi "14B" nhưng Qwen3-14B thực đo ~14.7B
tham số, vi phạm chính ràng buộc <14B của thể lệ.

Đây là **điểm quyết định duy nhất** trong nhánh 2. Nó thay cả `rule_planner._infer_operation`
lẫn `raw_metric_grounding`.

**File batch** (`data/qa/row_choice_batches/batch_XXX.jsonl`), một dòng một câu:

```json
{
  "question_id": 42,
  "question": "Xét nhóm VIC, VHM, VRE năm 2023, công ty nào có doanh thu thuần cao nhất?",
  "companies": ["VIC", "VHM", "VRE"],
  "periods": ["2023"],
  "candidates": [
    {"index": 0, "company_code": "VIC", "row_label": "Doanh thu thuần",
     "row_group_context": null, "table_title": "Báo cáo KQKD", "periods": ["2023", "2022"]},
    {"index": 1, "company_code": "VHM", "row_label": "Doanh thu thuần", "...": "..."}
  ]
}
```

**Bất biến N7:** `candidates` **không bao giờ** chứa `value` hay `fused_score`. LLM chỉ
thấy nhãn và ngữ cảnh, không thấy số.

**File quyết định** (`data/qa/row_choice_decisions.jsonl`):

```json
{"question_id": 42, "operation": "rank", "chosen": [{"company_code": "VIC", "index": 0},
                                                     {"company_code": "VHM", "index": 1},
                                                     {"company_code": "VRE", "index": 2}]}
```

- `operation` ∈ 9 giá trị `PlanOperation` đã có: `lookup`, `compare`, `compare_companies`,
  `difference`, `growth_rate`, `ratio`, `average`, `sum`, `rank`.
- `chosen` là danh sách; câu một công ty có đúng một phần tử.
- `index` trỏ vào `candidates` dựng lại ở local — không mang nhãn, không mang giá trị.

**Xử lý quyết định hỏng** (tất định, không hỏi lại model):

| Tình huống | Xử lý |
|---|---|
| Thiếu `question_id` trong file | `operation="lookup"`, `chosen=[rank-1]` |
| `operation` ngoài 9 giá trị | `lookup` |
| `index` ngoài phạm vi | dùng rank-1 cho công ty đó |
| Số `chosen` không khớp arity của `operation` | hạ về `lookup` với rank-1 |
| Nhãn dòng > 512 ký tự / có ký tự điều khiển | cắt về 512, lọc ký tự điều khiển |

Không có nhánh nào ném exception ra ngoài — một quyết định hỏng làm hỏng đúng một câu,
không giết cả lần chạy 3 tiếng.

### 6.3 Lắp plan tất định

Từ `entities` (deterministic) + `decision`:

| Trường plan | Nguồn |
|---|---|
| `operation` | decision |
| `companies` | entity parser |
| `periods` | entity parser (§6.4) |
| `metric` / `metric_a`/`metric_b` / `numerator`/`denominator` | decision → `MetricSelector` position-bound |
| `candidate_table_ids` | Nhánh 1 |
| `expected_unit` | entity parser (`map_requested_unit`) |
| `statement_scope` | entity parser |

Selector luôn position-bound (`table_id` + `row_index` cùng có mặt).

**Bước lắp này không được phép từ chối vì lý do ngữ nghĩa.** Nó chỉ từ chối khi arity
thật sự không thoả (`plan_validator` — ví dụ `rank` mà chỉ có 1 công ty), và khi từ chối
thì hạ về `lookup` chứ không abstain. Đây là điều thay thế `rule_planner.build_plan` và
là lý do 414 câu ở §2.2 được gỡ.

### 6.4 Chuẩn hoá kỳ — gỡ 97 câu

`_PERIOD_PATTERN = ^\d{4}$` giữ nguyên (plan vẫn chỉ nhận năm trần — compiler dựa vào
đó). Sửa ở **entity parser**: "31/12/2015", "ngày 31 tháng 12 năm 2015", "cuối năm 2015"
→ `periods = ("2015",)`.

Với báo cáo tài chính, ngày kết thúc kỳ và năm tài chính là một; đây là chuẩn hoá đúng
về mặt kế toán, không phải đoán bừa.

### 6.5 compiler → sandbox → verification — giữ nguyên

Không đụng vào. Đây là phần mạnh nhất của hệ thống.

### 6.6 Không còn tầng nào khác

Xoá hẳn: plan router, llm_planner, evidence planner, column refinement, raw metric
grounding, candidate switching, context expansion. Xem §8.

**Backstop giữ lại** — nó thuộc §5.1 BI-4 của spec gốc, không phải một tầng suy luận:
xuất bảng top-1 đã retrieve nguyên vẹn + một lookup best-effort, không tổng hợp gì cả.
Nó tồn tại để bài nộp đủ 1012 dòng, không phải để cứu câu trả lời.

---

## 7. Evidence & linter

### 7.1 Predicate — thi hành §5.2 (chưa làm)

Hiện `pandas_query._position_clauses` cố ý bỏ hẳn nhãn dòng:

```python
(df1.table_id == T) & (df1.row_idx == I) & (df1.period == P)      # thuần vị trí
```

Docstring viện dẫn "plan.md §14", nhưng §14 của `plan.md` hiện tại là *"Rủi ro lịch trình
và phương án cắt giảm"* — tham chiếu treo, quyết định lệch spec không còn tài liệu biện
minh nào sống.

Đổi thành dạng §5.2 yêu cầu — ngữ nghĩa mang ý nghĩa, vị trí chỉ phá thế hoà:

```python
(df1.row_label_raw == R) & (df1.column_label == L) & (df1.period == P) & (df1.row_idx == I)
```

`R` phải là **nhãn thật của corpus** tại `(table_id, row_idx)`, không phải cách nói trong
câu hỏi. Đây là thứ gỡ 18 ca `query_rejected` ở §2.3: query hiện lọc theo text của
selector nên không khớp gì trong bảng thật.

Đồng thời sửa `compiler._replay_row`: nó đang dựng dòng replay từ `selector.raw_text`
thay vì nhãn thật, khiến replay nội bộ tự khớp một cách giả tạo.

Lý do phải giữ phần ngữ nghĩa: bài nộp phải **giải trình được**. Một query thuần toạ độ
không chứng minh được nó lấy đúng chỉ tiêu nào.

### 7.2 Linter C1–C7 — giữ nguyên

`submission/compliance.py` đã đủ 7 mã và đã chạy chặn cứng trước khi zip. Không đụng.

### 7.3 Đổi đơn vị nằm trong query — đã sửa

Commit `435e9a3`. Ghi lại ở đây vì nó là bất biến của kiến trúc: `pandas_query` phải tự
tái tạo được `answer` từ CSV đóng gói, vì `validate_submission_zip` replay đúng như vậy.
Grammar sandbox cho `Add/Sub/Div/BitAnd` nhưng **không cho `Mult`** — phép đổi đơn vị
render bằng phép chia.

---

## 8. Những gì bị xoá

Mọi thứ đã commit ở `435e9a3`, khôi phục được bằng git.

### 8.1 Tầng answering ngoài kiến trúc

```
planning/plan_router.py              planning/llm_planner.py
planning/llm_prompt.py               planning/llm_contracts.py
planning/evidence_planner.py         planning/evidence_plan_contracts.py
planning/llm_evidence_planner.py     planning/evidence_facts.py
planning/column_refinement.py        planning/raw_metric_grounding.py
planning/llm_cell_grounding.py       planning/rule_planner.py
planning/llm_evaluation.py           planning/plan_evaluation.py
planning/plan_cases.py
```

`planning/cell_grounding.py` bị viết lại thành đường thẳng §6, không xoá.
`planning/entity_parser.py`, `entity_contracts.py`, `plan_contracts.py`,
`plan_validator.py`, `fact_grounding.py`, `grounding_contracts.py`,
`row_choice_batch.py`, `row_choice_decision.py`, `llm_client.py`,
`table_context_rendering.py`, `evidence_rendering.py` **giữ lại**.

### 8.2 Nhánh nghiên cứu chết

```
retrieval/graph.py                   retrieval/graph_contracts.py
retrieval/graph_service.py           retrieval/graph_evaluation.py
retrieval/expansion.py               retrieval/expansion_contracts.py
retrieval/expansion_evaluation.py
retrieval/failure_evaluation.py      retrieval/system_evaluation.py
```

Kiểm tra phụ thuộc ngược: cụm graph và expansion chỉ được import bởi chính chúng và
`retrieval/cli.py`; `failure_evaluation`/`system_evaluation` chỉ bởi `retrieval/cli.py`.
Cắt các subcommand tương ứng khỏi `retrieval/cli.py`.

**Giữ lại** dù hiện chưa có caller: `retrieval/fusion.py`, `fusion_contracts.py` (RRF —
kiến trúc mục tiêu cần), `retrieval/retrieval_scoring.py` (8 chỉ số — §11 tiêu chí 3),
`retrieval/gold.py`, `retrieval/evaluation.py`, `dense_*` (Nhánh 1 cần).

### 8.3 Tài liệu đã bị thay thế

```
docs/superpowers/specs/2026-08-22-llm-row-selection-design.md
docs/superpowers/plans/2026-08-22-llm-row-selection.md
docs/superpowers/plans/2026-08-21-submission-compliance-and-retrieval-recovery.md
ViFinQA_Design_V2_Recommendations.md
```

Spec 2026-08-21 **giữ lại** — nó là nguồn của kiến trúc mục tiêu và của các bất biến
BI-1…BI-4, C1–C7. Thêm một dòng ở đầu trỏ sang spec này.

### 8.4 Test đi kèm

Xoá test của module bị xoá. Test của module giữ lại nhưng đổi hành vi (`cell_grounding`,
`pandas_query`, `compiler`, `exporter`) phải được viết lại, không xoá.

---

## 9. Thứ tự thực hiện

| # | Hạng mục | Phụ thuộc | Vì sao thứ tự này |
|---|---|---|---|
| **T1** | Xoá §8 + cắt CLI + rút `exporter` về đường thẳng | — | Làm trước để mọi bước sau không phải giữ tương thích với thứ sắp chết |
| **T2** | Predicate §7.1 + `_replay_row` nhãn thật | T1 | Gỡ 18 `query_rejected`; sửa trước khi đo lại |
| **T3** | Chuẩn hoá kỳ §6.4 | T1 | Gỡ 97 câu, độc lập với LLM |
| **T4** | Batch/decision format mới §6.2 + lắp plan §6.3 | T1, T3 | Gỡ 414 câu |
| **T5** | Chạy batch Colab + full export, đo lại | T2, T3, T4 | Chốt Answer Accuracy trước khi động vào retrieval |
| **T6** | Encoder allowlist + index Qwen3-Emb-4B trên Colab | T1 | Chạy nền, song song T2–T5 |
| **T7** | Reranker + nối RRF vào đường live | T6 | |
| **T8** | Sweep k, chọn k* | T7 | Phải có stack retrieval cuối mới sweep đúng |
| **T9** | Full export cuối + linter + nộp | T5, T8 | |

T6 chạy song song từ đầu vì index toàn corpus tốn nhiều giờ Colab.

---

## 10. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Xoá nhầm module còn cần | Mọi thứ ở `435e9a3`; chạy full test suite sau T1; bản đồ phụ thuộc ngược đã dựng ở §8.2 |
| Rút thang tầng làm mất số câu đang answered | 60 câu hiện tại chủ yếu từ `rule`/`rule_raw_grounded`/`llm_evidence_planner` — các tầng sắp xoá. Đo trước/sau trên cùng tập; nếu T5 < 60, dừng lại phân tích chứ không nộp |
| LLM 8B quyết operation kém hơn rule | Đo riêng độ chính xác operation trên tập có nhãn; nếu tệ hơn, hạ về `lookup` mặc định vẫn gỡ được 84/232 câu |
| Predicate ngữ nghĩa làm hỏng câu đang answered | `C7` replay mọi câu; so answer trước/sau |
| Index Qwen3-Emb-4B tốn nhiều giờ Colab | Chạy nền sớm (T6 song song), checkpoint theo shard |
| `expected-tables.csv` không đủ phủ để sweep k | Xác minh ngay đầu T8; thiếu thì gán nhãn tay ~100 câu |
| Vòng full 3 giờ quá chậm để lặp | Dựng tập con 120 câu để đo nhanh; chỉ chạy full trước khi nộp |

---

## 11. Ngoài phạm vi

- Multi-agent / reflection loop
- Fine-tune (QLoRA) bất kỳ model nào
- Reranker cho row retrieval (chỉ dùng cho table retrieval ở vòng này)
- Re-ingest corpus (đổi `dataset_fingerprint`, mất mọi baseline đã pin)
- Sản phẩm demo (chatbot/dashboard, mục VI.2 thể lệ)
- Serving engine (vLLM/SGLang) cho chạy local — batch offline không cần throughput

---

## 12. Tiêu chí thành công

1. Repo không còn module nào ngoài kiến trúc §4; toàn bộ test suite xanh sau T1.
2. Linter báo **0 vi phạm**; 0 file CSV có 1 dòng.
3. `relevant_tables`/`relevant_docs` emit cho cả 1012 câu, đúng thứ tự retrieval-rank.
4. Đo được cả 8 chỉ số truy hồi trên tập gold, với k* đã cân bằng F2 ∧ MRR5.
5. `plan_source` chỉ còn hai giá trị: `llm_decision` và `backstop`. Không còn tầng nào khác.
6. **Answered > 186** (tiêu chí gốc), và mỗi câu answered replay đúng từ CSV trong sandbox.
7. Recall@10 của Nhánh 1 ≥ 75% trên tập gold (BM25 hiện tại: 47.41%).
