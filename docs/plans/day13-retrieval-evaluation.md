# Kế hoạch Ngày 13 — Retrieval evaluation

> Trạng thái: **đã hoàn tất** (xem [plan.md § Ngày 13](../../plan.md) để biết kết quả thực tế).
> Viết ngày 2026-08-14 trên release khóa
> `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a` như tài liệu thiết kế
> trước khi triển khai; giữ lại làm hồ sơ vì sao gold 70 câu và metric được thiết kế như vậy.
> Số liệu cụ thể (F2@R, breakdown, root_cause) đã đổi so với ước tính ở đây — coi bản trong
> `plan.md`/`docs/decisions/0002-retrieval-metric-definition.md` là nguồn sự thật.

## 0. Đầu vào đã sẵn sàng (không phải làm lại)

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Release khóa | `data/processed/release_v2_422df141c935/` | ✅ đã qua cổng tuần 1 |
| Release lock | `data/qa/week1_pilot_422df141c935/dataset-pilot-v1.json` | ✅ |
| Gold 30 câu | `data/qa/retrieval-gold-v1.jsonl` | ✅ đã restamp theo fingerprint hiện tại |
| BM25 | `data/indexes/bm25-v3/422df141c935…/` | ✅ 146.011 bảng |
| Dense | `data/indexes/dense-day9-a/422df141c935…/` (bge-m3, e5-small) | ✅ build trên RTX 3050/WSL2 |
| Graph | `data/indexes/graph-day11-a/422df141c935…/` | ✅ 5 quan hệ |
| Báo cáo hiện hành | `artifacts/evaluations/day13/{bm25,dense-*,fusion-*,graph,expansion}` | ✅ đã chạy lại |

**Kết luận:** Ngày 13 **không cần build lại index nào**. Toàn bộ khối lượng nằm ở gold, metric,
failure analysis và re-baseline.

---

## 1. Chốt chặn phải xử lý trước tiên: cổng F2 ≥ 0,80 hiện không thể đạt được

`retrieval/evaluation.py::score_at_10` dùng **mẫu số cố định 10** cho precision:

```
precision = TP / 10          # không phải TP / |gold|
recall    = TP / |gold|
F2        = 5PR / (4P + R)
```

Với câu hỏi hoàn hảo (mọi bảng gold nằm trong top-10), precision bị chặn trên bởi `|gold| / 10`.
Trần lý thuyết của F2 trên tập gold 30 câu hiện tại:

| Nhóm | n | P trần | R trần | F2 trần |
| --- | ---: | ---: | ---: | ---: |
| gold = 1 bảng | 12 | 0,10 | 1,00 | 0,357143 |
| gold = 2 bảng | 18 | 0,20 | 1,00 | 0,555556 |
| **Macro** | **30** | | | **0,476190** |

Theo intent: `lookup` 0,357143 · `compare` 0,535714 · `growth` 0,535714.

BM25 v3 hiện đạt macro F2 **0,431217** — tức **90,6 % trần lý thuyết**, với Recall@10 0,883333.

> **Phát hiện:** cổng "retrieval F2 ≥ 0,80" ở [plan.md dòng 68 và 821](../../plan.md) là
> **bất khả thi về mặt toán học** dưới định nghĩa metric hiện tại, sai lệch ~1,7×. F2 thấp
> **không** phản ánh chất lượng retrieval kém; nó phản ánh precision bị chuẩn hóa trên mẫu số 10
> trong khi mỗi câu chỉ có 1–2 bảng đúng. Nếu không chốt lại định nghĩa trước, mọi số Ngày 13 và
> mọi quyết định ablation Ngày 14 đều dựa trên một ngưỡng vô nghĩa.

### Ba phương án

| | Phương án | Ưu | Nhược |
| --- | --- | --- | --- |
| **A** | Giữ Precision@10, hạ cổng F2 xuống ngưỡng tương đối (vd. ≥ 0,92 × trần) | Không đụng code đã công bố | Ngưỡng phụ thuộc phân bố cardinality gold, đổi gold là đổi trần → khó truyền đạt |
| **B** | Thêm **Precision@R** (R-precision, mẫu số là số bảng gold) và **F2@R**; cổng đặt trên F2@R | P, R cùng thang, F2@R = 1,0 khi hoàn hảo → ngưỡng 0,80 có nghĩa thật; so sánh được giữa các câu khác cardinality | Phải thêm model/report mới, không tái dùng được số Day 8–12 trực tiếp |
| **C** | Bỏ F2 khỏi cổng, chỉ dùng Recall@10 + MRR | Đơn giản nhất | Mất tín hiệu về nhiễu trong top-k |

**Khuyến nghị: B**, giữ Precision@10/F2@10 song song làm chỉ số lịch sử để mọi báo cáo Day 8–12
vẫn so sánh được. Không sửa `score_at_10` — thêm hàm cạnh nó.

### Task 13.1 — ADR chốt định nghĩa metric

- [ ] Viết `docs/decisions/0002-retrieval-metric-definition.md`: nêu phép tính trần ở trên, ba
      phương án, quyết định, và hệ quả lên cổng Ngày 14.
- [ ] Cập nhật `plan.md` dòng 68 (bảng KPI) và dòng 821 (cổng tuần 2) theo quyết định.
- [ ] **DoD:** không còn ngưỡng nào trong `plan.md` nằm ngoài miền giá trị khả thi của metric
      tương ứng.

---

## 2. Task 13.2 — Nâng gold từ 30 lên 70 câu

### 2.1. Lỗ hổng bao phủ của gold hiện tại

Đo trực tiếp trên `data/qa/retrieval-gold-v1.jsonl` (30 câu) so với release (1.971 tài liệu,
100 công ty, niên độ 2015–2025):

| Chiều | Hiện tại | Vấn đề |
| --- | --- | --- |
| Công ty | 10/100, mỗi công ty đúng 3 câu | Chỉ 10 % công ty; không biết hệ thống có over-fit alias của 10 mã này |
| Niên độ | **chỉ 2022 và 2023** | 9/11 năm chưa từng được đánh giá; báo cáo 2015–2019 khác hẳn về layout OCR |
| `statement_types` filter | 5/30 câu | Đường filter theo loại báo cáo gần như chưa được kiểm |
| Cardinality gold | 12 câu 1 bảng, 18 câu 2 bảng, **0 câu ≥ 3** | Không đo được hành vi multi-table thật sự |
| Bằng chứng là bảng `notes` | **0/30** | Quan hệ `explained_by_note` của graph (Day 11/12) chưa từng có câu nào cần tới nó — đây chính là lý do Day 12 chỉ tìm được 4/30 câu còn headroom |
| Intent | lookup 10 / compare 10 / growth 10 | Cân bằng, giữ nguyên tỷ lệ |

### 2.2. Thiết kế mẫu cho 40 câu bổ sung

**Nguyên tắc bất di bất dịch: giữ nguyên 30 câu cũ, không sửa một ký tự nào.** Mọi so sánh với
Day 8–12 phải còn tái lập được trên tập con 30 câu.

Chỉ tiêu cho 40 câu mới (tổng 70):

- **Intent:** +14 lookup, +13 compare, +13 growth → tổng 24/23/23.
- **Công ty:** ít nhất 30 mã **chưa** xuất hiện trong gold hiện tại; mỗi mã tối đa 2 câu mới.
- **Niên độ:** ≥ 8 câu thuộc 2015–2019, ≥ 8 câu thuộc 2024–2025; tổng gold phủ ≥ 6 năm khác nhau.
- **`statement_types`:** ≥ 12 câu mới có filter khác rỗng, phủ cả `balance_sheet`,
  `income_statement`, `cash_flow_statement`, `notes`.
- **Cardinality:** ≥ 8 câu có `|gold| = 3`, ≥ 4 câu có `|gold| ≥ 4`.
- **Notes:** ≥ 10 câu mà bằng chứng bắt buộc gồm ít nhất một bảng `notes` (để `explained_by_note`
  và `shared_metric` có cơ hội đóng góp thật ở Ngày 14).
- **Câu khó có chủ đích:** ≥ 5 câu dùng alias/viết tắt tiếng Việt không khớp chuỗi với tiêu đề
  bảng (kiểm tra đường `metric_aliases`), ≥ 3 câu mà công ty có cả báo cáo `separate` và
  `consolidated` cùng năm (kiểm tra `statement_scope`).

### 2.3. Quy trình soạn — chống rò rỉ

`stable_question_id` cố ý **không** phụ thuộc vào prediction. Phải giữ đúng tinh thần đó:

1. Chọn tài liệu theo chỉ tiêu 2.2 **trước**, đọc file `_extracted.txt` trong release.
2. Viết câu hỏi **từ nội dung báo cáo**, không bao giờ từ ranked list của BM25/dense.
3. Ghi bằng chứng thủ công: `relative_path`, `line_start`, `line_end`, `table_id`, `verified: true`.
4. Sinh `question_id` bằng `stable_question_id(question, filters, gold_table_ids, FINGERPRINT)`
   với fingerprint hiện tại.
5. Ghép vào file, **sort theo `question_id`**, chạy `validate-gold`.

> Nếu bất kỳ câu nào được soạn bằng cách nhìn kết quả retrieval rồi chọn bảng nào trông hợp lý,
> toàn bộ số Ngày 14 mất giá trị. Ghi rõ quy trình đã dùng vào `reviewed_by`/README.

### 2.4. Ràng buộc kỹ thuật `load_gold_questions` sẽ ép

Tất cả đã có sẵn trong [`retrieval/gold.py`](../../src/financial_report_qa/retrieval/gold.py) —
liệt kê ở đây để soạn cho đúng ngay lần đầu:

- `question_id` duy nhất, **sắp xếp tăng dần**, không dòng trống.
- `dataset_fingerprint` khớp `EXPECTED_FINGERPRINT`.
- `sorted(gold_table_ids) == sorted(e.table_id for e in gold_evidence)`.
- `question_id` phải tái tính đúng từ `stable_question_id`.
- Nội dung `question` không được trùng nhau.
- Mọi `table_id` phải tồn tại trong `tables.parquet`; `relative_path` khớp `documents.parquet`;
  `periods` trong filter phải tồn tại trong `cells.parquet` của bảng gold.

### 2.5. Thay đổi code kèm theo

- [ ] `retrieval/gold.py:17` — `REQUIRED_GOLD_QUESTION_COUNT = 30` → `70`.
- [ ] Rà test có hằng số 30 trong `tests/unit/retrieval/` và `tests/integration/retrieval/`.
- [ ] **DoD:** `validate-gold` báo `validated 70 reviewed retrieval questions`; chạy hai lần cho
      cùng byte output.

---

## 3. Task 13.3 — Bộ metric mở rộng

### 3.1. Metric cần thêm

| Metric | Định nghĩa | Vì sao |
| --- | --- | --- |
| `MRR` | `1 / rank` của bảng gold **đầu tiên** trong top-10, 0 nếu không có | Đo hệ thống có đặt bằng chứng lên đầu không — quan trọng cho Day 17+ khi planner chỉ nhìn vài kết quả đầu |
| `Recall@3/5/10` | số bảng gold trong top-k, chia cho số bảng gold | Ngân sách context của LLM ở tuần 3 nhiều khả năng là 3–5 bảng, không phải 10 |
| `Precision@R` | số bảng gold trong top-R, chia cho R, với R = số bảng gold | R-precision, cùng thang với recall |
| `F2@R` | F2 từ cặp (`Precision@R`, `Recall@10`) | Metric đặt cổng theo quyết định 13.1 |

### 3.2. Chiều phân tách (breakdown)

Ngoài `by_intent` đã có, thêm:

- `by_gold_cardinality`: `one_table` / `two_tables` / `three_or_more`
- `by_period_cardinality`: `one_period` / `multiple_periods`
- `by_statement_filter`: `filtered` / `unfiltered`
- `by_report_era`: `2015_2019` / `2020_2023` / `2024_2025`

(Hai chiều đầu đã có tiền lệ trong `expansion_evaluation.py` — tái dùng đúng tên nhãn.)

### 3.3. Ràng buộc tương thích

`RetrievalMetrics` và `RetrievalEvaluationReport` là pydantic `frozen=True, extra="forbid"`.
Thêm field bắt buộc sẽ làm **mọi JSON Day 8–12 đã lưu không đọc lại được**.

- [ ] Không sửa `RetrievalMetrics` / `RetrievalEvaluationReport` / `score_at_10`.
- [ ] Thêm model mới cạnh chúng (vd. `RetrievalMetricsExtended`, `RetrievalEvaluationReportV2`)
      và hàm `evaluate_retrieval_v2` bọc lại vòng lặp hiện có.
- [ ] `k` cố định 10 cho ranking, nhưng cho phép `evaluate_retrieval_v2` nhận `diagnostic_k`
      (mặc định 100) chỉ để ghi rank của gold nằm ngoài top-10 vào failure export — **không** dùng
      cho bất kỳ metric nào.
- [ ] **DoD:** đọc lại được toàn bộ file trong `artifacts/evaluations/day{9,10,11,13}/` bằng model
      cũ sau khi thay đổi; test hồi quy chứng minh điều đó.

---

## 4. Task 13.4 — Export failure cases

Đích: `artifacts/evaluations/day13/failures-422df141c935.{json,md}`.

Mỗi bản ghi failure gồm:

- `question_id`, `question`, `intent`, `filters`
- `gold_table_ids` và `missing_gold_table_ids`
- top-10 dự đoán kèm `score`, `matched_tokens`, `metadata` (đã có sẵn trong `RetrievalTrace`)
- **`gold_rank_beyond_10`**: rank thật của từng bảng gold bị trượt, đo ở `diagnostic_k = 100`;
  `null` nếu ngoài 100 → phân biệt được "xếp hạng kém" với "không truy hồi được"
- `failure` tự động từ `_failure_for` (`no_eligible_documents` / `no_index_tokens` /
  `zero_gold_hits` / `partial_gold_hits`)
- `root_cause` **gán tay** theo bảng phân loại cố định:
  `missing_alias` / `ocr_corruption` / `filter_too_narrow` / `filter_too_wide` /
  `gold_label_error` / `ranking_only` / `unknown`

Baseline hiện tại có 4/30 câu lỗi (3 `zero_gold_hits`, 1 `partial_gold_hits`) — quá ít để rút kết
luận. Với 70 câu và các chiều khó ở 2.2, kỳ vọng đủ mẫu để Ngày 14 quyết định dựa trên nguyên nhân
thay vì cảm tính.

- [ ] **DoD:** mọi câu có `failure != "none"` đều có `root_cause` khác `unknown` hoặc có ghi chú
      giải thích vì sao không phân loại được. Ngày 14 đọc file này để quyết định
      "ưu tiên normalization/aliases hơn đổi model".

---

## 5. Task 13.5 — Re-baseline sau khi gold thành 70 câu

Đổi gold ⇒ **phá vỡ chốt chặn tái lập** ở `dense_evaluation.py:277`:

```python
def _validate_bm25_reference(report):
    expected = (0.1466666666666667, 0.8833333333333333, 0.4312169312169313)
    ...
    if report.question_count != 30 or ...
```

Hàm này gác **cả ba** lệnh `evaluate-dense`, `evaluate-fusion` và `evaluate-expansion`
(`fusion_evaluation.py:30`, `expansion_evaluation.py:25`).

Thứ tự bắt buộc:

1. [ ] Chạy `evaluate` (BM25) trên gold 70 → lấy macro mới.
2. [ ] Cập nhật `expected` và `question_count` trong `_validate_bm25_reference`; cập nhật
       `_LOCKED_FINGERPRINT`/`_FINGERPRINT` và hằng số 30 trong
       `tests/unit/retrieval/test_dense_evaluation.py`, `test_fusion_evaluation.py`,
       `test_expansion_evaluation.py`.
3. [ ] Chạy lại `evaluate-dense` ×2 encoder (trong env WSL2 `financial-dense-gpu`,
       `--encoder-device cuda` — phải khớp device đã build vì `device` nằm trong
       `encoder_spec_sha256`; xóa `data/indexes/dense-query-cache/day9-a-*` trước để giữ
       trạng thái cold hợp lệ).
4. [ ] Chạy lại `evaluate-fusion` ×2, `evaluate-graph`, `evaluate-expansion`.
5. [ ] Ghi toàn bộ vào `artifacts/evaluations/day13/` (ghi đè bản 30 câu sau khi đã lưu bản cũ
       sang `artifacts/evaluations/day13/gold30/` để giữ so sánh).

- [ ] **DoD:** `evaluate-expansion` chạy được đến hết mà không cần bỏ qua chốt chặn nào.

---

## 6. Task 13.6 — Đồng bộ tài liệu

- [ ] `README.md` § Day 13: bảng metric mới, phân bố gold 70 câu, kết quả BM25/dense/fusion/graph,
      tóm tắt failure theo `root_cause`.
- [ ] `plan.md`: tick checkbox Ngày 13; sửa cổng Ngày 14 theo ADR 13.1.
- [ ] Commit đề xuất: `feat(retrieval): expand gold to 70 and add ranked evaluation metrics`.

---

## 7. Thứ tự thực thi

```
13.1 ADR metric ──► 13.3 metric mở rộng ──┐
                                          ├──► 13.5 re-baseline ──► 13.6 tài liệu
13.2 gold 30→70 ──► 13.4 failure export ──┘
```

13.1 và 13.2 chạy song song được. 13.5 **phải** sau cả hai.

## 8. Định nghĩa hoàn tất (toàn ngày)

- [ ] `validate-gold` pass với 70 câu; chạy hai lần byte-identical.
- [ ] Mọi cổng số trong `plan.md` nằm trong miền khả thi của metric tương ứng.
- [ ] `artifacts/evaluations/day13/` có đủ: bm25, dense ×2, fusion ×2, graph, expansion, failures.
- [ ] Báo cáo cũ (Day 8–12) vẫn đọc và tái lập được.
- [ ] `pytest -q` toàn bộ: 0 fail. `ruff check .`: 0 lỗi mới. `mypy`: đúng 33 lỗi có sẵn, 0 lỗi mới.
- [ ] `git diff --check` sạch.

## 9. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
| --- | --- | --- |
| Soạn 40 câu gold thủ công là phần tốn thời gian nhất, dễ tràn sang ngày 14 | Sau 4 giờ chưa xong 20 câu | Cắt chỉ tiêu xuống 55 câu tổng, giữ nguyên **tất cả** chỉ tiêu về niên độ/notes/cardinality — bao phủ quan trọng hơn số lượng |
| Rò rỉ khi soạn gold (nhìn ranked list) | Recall@10 mới cao bất thường (> 0,95) | Kiểm chéo ngẫu nhiên 10 câu: bằng chứng phải truy được từ file gốc mà không cần chạy index |
| Đổi định nghĩa metric làm mất khả năng so sánh với Day 8–12 | Không tái tạo được số cũ | Bắt buộc giữ `score_at_10` và báo cáo song song cả hai bộ metric |
| Gold 70 câu làm lộ ra lỗi normalization mới | Nhiều `root_cause = ocr_corruption` | Đây là kết quả hợp lệ của Ngày 13 — ghi nhận, để Ngày 14 quyết định, **không** vá gấp |
