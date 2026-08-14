# Kế hoạch Ngày 14 — Review cổng tuần 2

> Trạng thái: **đã hoàn tất** (xem [plan.md § Ngày 14](../../plan.md) và
> [ADR 0003](../decisions/0003-graph-expansion-decision.md) để biết kết quả thực tế). Viết ngày
> 2026-08-14 trên release khóa `422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`,
> sau khi Ngày 13 hoàn tất (`d4071f7`), như tài liệu thiết kế trước khi triển khai. Số liệu cụ thể
> đã đổi so với chẩn đoán ban đầu (BM25 v4 đạt Recall@10 0,9143 thay vì dự đoán chỉ cần đóng
> khoảng trần 100/100 — con số thật cao hơn ước tính vì thêm từ vựng còn cải thiện cả những câu
> ngoài top 15 lỗi ban đầu) — coi `plan.md`/ADR 0003 là nguồn sự thật.

## 1. Kết quả cổng hiện tại: **trượt cả hai chỉ tiêu**

| Chỉ tiêu | Ngưỡng | BM25 v3 | Tốt nhất toàn bảng | Thiếu |
| --- | ---: | ---: | ---: | ---: |
| F2@R | ≥ 0,80 | 0,491346 | 0,494129 | **−0,306** |
| Recall@10 | ≥ 0,90 | 0,880952 | 0,880952 | −0,019 |

Recall chỉ thiếu ~1,4 câu. F2@R mới là vấn đề thật: giữ nguyên Recall@10 hiện tại, `Precision@R`
phải đi từ **0,4226 → 0,5850** mới chạm 0,80.

Điểm "tốt nhất" trong bảng graph-expansion là `alpha = 0` — tức **graph không đóng góp gì**, lặp
lại đúng kết luận Ngày 12. Chênh 0,494129 vs 0,491346 đến từ tier mâu thuẫn metadata (Recall@10
giống hệt, chỉ MRR và P@R khác → cùng tập top-10, khác thứ tự), không phải từ mở rộng graph.

---

## 2. Chẩn đoán trước khi quyết định

### 2.1. Đây **không** phải bài toán thu hồi — là bài toán xếp hạng

Từ `artifacts/evaluations/day13/failures-422df141c935.json` (`diagnostic_k = 100`):

- 11/70 câu lỗi, tổng **15 bảng gold bị trượt khỏi top-10**.
- **Cả 15 bảng đều tìm thấy trong top-100**, rank nằm gọn trong khoảng **11–44**.
- **0 bảng nằm ngoài top-100.**

Hệ quả trực tiếp: với một reranker hoàn hảo trên top-50, **trần Recall@10 = 1,000** và cổng
Recall@10 ≥ 0,90 đạt được ngay. Pool ứng viên đã chứa toàn bộ bằng chứng cần thiết.

> Vì vậy: **đổi embedding model không giải quyết được gì.** Dense (bge-m3, e5-small) đã thua BM25
> ở mọi cấu hình trọng số suốt Day 9/10/13. Ngân sách Ngày 14 phải đổ vào nội dung document và
> xếp hạng, không phải vào model — đúng như plan.md đã định hướng.

### 2.2. Nguyên nhân gốc: **94 % nhãn dòng bị loại khỏi document BM25**

[`retrieval/documents.py`](../../src/financial_report_qa/retrieval/documents.py) gom nhãn dòng
bằng câu SQL có mệnh đề lọc:

```sql
list(DISTINCT struct_pack(canonical := c.row_label_canonical, raw := c.row_label_raw))
    FILTER (WHERE c.row_label_canonical IS NOT NULL)
```

`_metric_labels` sau đó bỏ tiếp mọi bản ghi có `canonical` rỗng. Nghĩa là **khi normalization
không canonical hoá được một nhãn, cả `raw` cũng bị vứt** — không vào dòng `metrics`, cũng không
vào dòng `metric aliases`.

Đo trên release khóa:

| Đại lượng | Giá trị |
| --- | ---: |
| Cell có `row_label_raw` | 5.353.511 |
| Cell có `row_label_canonical` | 317.594 (**5,93 %**) |
| Cell có raw nhưng **không** có canonical → bị vứt | **5.035.917** |
| Bảng có ≥ 1 nhãn trong document BM25 | 25.091 / 146.011 (17,2 %) |
| Bảng **không có nhãn nào** trong document | 120.920 / 146.011 (**82,8 %**) |
| …trong đó bảng **có sẵn raw nhưng bị vứt** | **109.499 (75,0 % corpus)** |

Ba phần tư corpus đang được index chỉ bằng `title`, `statement`, `group context`, `company`,
`periods`, `units` — **không có một nhãn dòng nào**.

Điều này khớp từng chữ với ghi chú trong failure report:

- STB 2024: *"…contains the exact labels 'Cho vay khách hàng' and 'Chứng khoán đầu tư', but the
  BM25 document retains neither label; the eligible gold table ranks 20."*
- VSF / SJG / HSG / MSN (`notes`): *"…the BM25 document has a generic notes title and no metric
  labels."*
- SHB 2020: *"…the BM25 document retains only other_income; the eligible gold ranks 12."*

6/11 câu lỗi được gán `missing_alias`, nhưng nhãn **không hề thiếu trong nguồn** — nó bị pipeline
xây document loại bỏ. Đây là lỗi một dòng SQL, không phải thiếu từ điển alias.

### 2.3. Chi phí sửa nhỏ hơn dự tưởng

Lo ngại hiển nhiên là phình document. Đo thực tế:

| Đại lượng | Giá trị |
| --- | ---: |
| Nhãn raw **distinct** trên mỗi bảng: p50 / p90 / p99 / max | 6 / 18 / 32 / **56** |
| Tổng cặp (bảng, nhãn raw) distinct | 1.130.604 |
| Độ dài trung bình một nhãn | 27,8 ký tự |
| Ước tính text thêm vào | ~31 MB trên corpus 126,5 MB → **+25 %** |

Max 56 nhãn/bảng nên **không cần cắt ngưỡng**. Dense re-encode ước tính bge-m3 ~55 → ~70 phút,
e5-small ~6 → ~8 phút trên RTX 3050.

### 2.4. Hai lỗi dữ liệu lẻ đã xác định

- **MML 2017** (`gold_label_error`): câu hỏi yêu cầu "công ty con" nhưng nguồn ghi
  "Công ty liên kết sở hữu gián tiếp" — nhãn gold sai, độc lập với xếp hạng.
- **DBC 2022** (ghi trong note của một câu `ranking_only`): `statement_type` của bảng bị phân loại
  sai; truy vấn này không filter nên chưa lộ, nhưng sẽ lộ ở câu có `statement_types`.

### 2.5. Câu hỏi còn mở — không được đoán

Phân khúc **2015–2019 có `Precision@R` 0,2659** so với 0,6250 của 2024–2025 (kém 2,35×). Giả
thuyết đầu tiên là normalization kém trên OCR báo cáo cũ, nhưng **tỷ lệ canonical hoá phẳng
~5,5–6,4 % ở mọi năm** (2015: 5,77 %; 2019: 6,08 %; 2024: 5,78 %) → giả thuyết đó **không đứng
vững**. Nguyên nhân chưa biết; phải đo trước khi đề xuất sửa (xem Task 14.6).

---

## 3. Các task

### Task 14.1 — Sửa nhãn gold sai (làm trước, rẻ)

- [ ] Đọc lại nguồn của câu MML 2017; hoặc sửa `gold_table_ids` cho khớp "công ty con", hoặc sửa
      câu hỏi thành "công ty liên kết sở hữu gián tiếp".
- [ ] Tính lại `question_id` (đổi nội dung ⇒ đổi hash), sort lại file, chạy `validate-gold`.
- [ ] Ghi vào README vì sao sửa — đây là lần đầu một câu gold bị sửa sau khi đã review.
- **DoD:** `validate-gold` EXIT=0 với 70 câu; failure report mới không còn `gold_label_error`.

### Task 14.2 — Giữ nhãn `raw` trong document BM25 (đòn bẩy chính)

- [ ] Bỏ mệnh đề `FILTER (WHERE c.row_label_canonical IS NOT NULL)`; đổi sang lọc
      `WHERE c.row_label_canonical IS NOT NULL OR c.row_label_raw IS NOT NULL`.
- [ ] Sửa `_metric_labels` để chấp nhận bản ghi có `canonical` rỗng nhưng `raw` khác rỗng
      (`MetricLabelObservation` cần cho phép `canonical` rỗng, hoặc dùng một trường riêng).
- [ ] **Giữ nguyên hai dòng tách biệt** `metrics:` (canonical) và `metric aliases:` (raw) — không
      trộn, để sau này còn đánh trọng số theo trường được.
- [ ] Test hồi quy: một bảng chỉ có `row_label_raw` phải xuất hiện nhãn đó trong `text`.
- [ ] Đo lại kích thước corpus thực tế và so với ước tính +25 %.
- **DoD:** `build-documents` chạy hai lần cho sha256 giống nhau; số bảng không có nhãn giảm từ
  120.920 xuống ≤ 11.500 (chỉ còn bảng thật sự không có nhãn nào).

### Task 14.3 — Rebuild và re-baseline

Đổi nội dung document ⇒ đổi sha256 corpus ⇒ **mọi index và baseline phải làm lại**. Thứ tự bắt buộc:

1. [ ] `build-index` (BM25 v4) trên corpus mới.
2. [ ] `build-dense-corpus` + `build-dense-index` ×2 encoder trong WSL2 env `financial-dense-gpu`,
       `--encoder-device cuda` (device nằm trong `encoder_spec_sha256`).
3. [ ] `build-graph` — **bắt buộc**, vì quan hệ `shared_metric` tính Jaccard trên metric labels;
       thêm 1,13 triệu nhãn sẽ đổi hoàn toàn bucket này.
4. [ ] Xóa `data/indexes/dense-query-cache/*` trước khi đánh giá (giữ trạng thái cold hợp lệ).
5. [ ] Chạy `evaluate` (BM25) → cập nhật `_validate_bm25_reference` → chạy `evaluate-dense` ×2,
       `evaluate-fusion` ×2, `evaluate-graph`, `evaluate-expansion`.
6. [ ] Ghi vào `artifacts/evaluations/day14/`; **giữ nguyên** `day13/` để so sánh trước/sau.

- **DoD:** toàn bộ 6 hệ thống chạy hết không phải bỏ qua chốt chặn nào; `day13/` không bị ghi đè.

> ⚠️ Đây là lần thứ ba baseline bị phá trong hai tuần. Ước tính ~1,5 giờ GPU + re-baseline. Chốt
> quyết định làm/không làm **trước** khi chạy, đừng làm nửa chừng.

### Task 14.4 — Ablation đầy đủ

Bảng ablation trên **cùng gold 70, cùng corpus mới**, báo cáo cả 4 metric (F2@R, Recall@10,
Recall@3, MRR) và cả 4 chiều breakdown:

| Hệ thống | Có sẵn ở `day13/v2` |
| --- | --- |
| BM25 v3 (baseline cũ) | ✅ |
| BM25 v4 (corpus mới) | ✗ |
| dense bge-m3 / e5-small | ✅ |
| fusion (bge / e5) | ✅ |
| fusion + graph expansion | ✅ |

- [ ] Cột bắt buộc: **delta so với BM25 v3 cũ** — để tách đóng góp của 14.2 khỏi mọi thứ khác.
- **DoD:** một bảng duy nhất trong README trả lời được "thay đổi nào mang lại bao nhiêu".

### Task 14.5 — Quyết định giữ hay bỏ graph

Bằng chứng hiện có nói **bỏ**: điểm tốt nhất luôn là `alpha = 0` ở cả Day 12 và Day 13.

Nhưng bằng chứng đó **chưa đủ để kết luận**, vì hai lý do:

1. Gold 70 câu có **0 câu cần bảng `notes`** (chỉ tiêu Ngày 13 đặt ≥ 10, đã trượt) → quan hệ
   `explained_by_note` chưa từng được đánh giá một lần nào.
2. `shared_metric` được tính trên metric labels đang thiếu 94 % → sau 14.2 quan hệ này sẽ khác hẳn.

- [ ] Chạy lại lưới 13 điểm sau khi có corpus mới.
- [ ] **Quy tắc quyết định chốt trước:** giữ graph **chỉ khi** tồn tại điểm `alpha > 0` thắng
      `alpha = 0` ở F2@R **hoặc** ở Recall@10 của phân khúc `three_or_more` (hiện 0,7639 — yếu nhất),
      với biên ≥ 0,01. Ngược lại: bỏ graph khỏi đường chính, giữ code + báo cáo làm phụ lục.
- **DoD:** quyết định ghi thành ADR `docs/decisions/0003-graph-expansion-decision.md`, kèm số đo.

### Task 14.6 — Điều tra khoảng cách 2015–2019

- [ ] Tách `Precision@R` theo (era × intent) và (era × có/không `statement_types`) để tìm biến gây
      nhiễu — hiện chưa rõ là do era hay do phân bố intent trong các câu era cũ.
- [ ] Kiểm tra chất lượng OCR: so `quality_score` trung bình của bảng gold theo era.
- [ ] Nếu sau 14.2 khoảng cách biến mất → ghi nhận và đóng. Nếu còn → mở task cho Tuần 3.
- **DoD:** một kết luận có số đo, hoặc một câu "chưa xác định được, giả thuyết X bị bác bởi Y".

### Task 14.7 — Chốt cổng và xử lý trường hợp trượt

Kịch bản nhiều khả năng nhất: 14.2 đưa Recall@10 vượt 0,90 nhưng F2@R vẫn dưới 0,80.

Chốt trước ba nhánh xử lý, **không quyết định lúc đang nhìn số**:

| Nếu | Thì |
| --- | --- |
| Cả hai đạt | Qua cổng, sang Tuần 3 |
| Recall@10 ≥ 0,90 nhưng F2@R < 0,80 | Sang Tuần 3 **có nợ kỹ thuật ghi rõ**; mở task reranker vào Tuần 4 (`Ngày 27`); planner Tuần 3 phải chịu được top-10 nhiễu |
| Recall@10 < 0,90 | **Không sang Tuần 3.** Ưu tiên normalization/aliases theo plan.md; cắt phạm vi Ngày 15–16 |

- [ ] Cập nhật `plan.md` § Ngày 14 với kết quả thật và nhánh đã chọn.
- [ ] Nếu chọn nhánh 2, ghi nợ vào `plan.md` § 14 (Rủi ro lịch trình).

### Task 14.8 — Dọn mục lỗi thời trong plan.md

Checkbox *"Hoàn thành dataset Representative khoảng 60 báo cáo nếu collector ổn"* đã lỗi thời:
release khóa hiện có **1.971 tài liệu / 100 công ty**, vượt xa 60 báo cáo, và từ "Representative"
không được định nghĩa ở bất kỳ chỗ nào khác trong repo.

- [ ] Đổi thành ghi nhận đã vượt chỉ tiêu, hoặc xóa hẳn.

---

## 4. Thứ tự thực thi

```
14.1 sửa gold ──┐
                ├──► 14.2 giữ raw label ──► 14.3 rebuild + re-baseline ──┬──► 14.4 ablation ──► 14.7 chốt cổng
14.8 dọn plan ──┘                                                        ├──► 14.5 quyết định graph
                                                                         └──► 14.6 điều tra era
```

14.1 và 14.8 làm được ngay. 14.2 là điểm không quay lại — cân nhắc kỹ trước khi bắt đầu 14.3.

## 5. Định nghĩa hoàn tất (toàn ngày)

- [ ] `validate-gold` EXIT=0, 70 câu, không còn `gold_label_error`.
- [ ] Số bảng không có nhãn trong document giảm từ 120.920 xuống ≤ 11.500.
- [ ] `artifacts/evaluations/day14/` đủ 6 hệ thống + failure report; `day13/` còn nguyên.
- [ ] Bảng ablation có cột delta so với BM25 v3 cũ.
- [ ] ADR 0003 chốt giữ/bỏ graph kèm số đo.
- [ ] Nhánh xử lý cổng đã chọn và ghi vào `plan.md`.
- [ ] `pytest -q` 0 fail; `ruff check .` sạch; `mypy` 0 lỗi (baseline hiện tại là 0, không phải 33).
- [ ] `git diff --check` sạch.

## 6. Rủi ro

| Rủi ro | Dấu hiệu | Xử lý |
| --- | --- | --- |
| 14.2 phá baseline lần thứ ba, tốn ~1,5 h GPU + re-baseline | — | Chốt quyết định trước khi chạy; không bắt đầu 14.3 nếu chưa chắc làm 14.2 |
| Thêm 1,13 triệu nhãn làm nhiễu BM25, F2@R **giảm** | BM25 v4 kém v3 ở bảng ablation | Đã tách sẵn dòng `metrics` / `metric aliases` → hạ trọng số dòng alias thay vì quay lui toàn bộ |
| Sửa gold MML bị coi là chỉnh gold theo kết quả | — | Sửa **trước** khi chạy bất kỳ đánh giá mới nào; lý do dựa trên văn bản nguồn, không dựa trên rank |
| Gold vẫn 0 câu `notes` ⇒ quyết định graph thiếu căn cứ | 14.5 lại ra `alpha = 0` | Ghi rõ trong ADR 0003 là "bỏ vì **chưa có bằng chứng**", không phải "đã chứng minh vô dụng"; mở lại nếu Tuần 3 bổ sung câu `notes` |
| Trượt cổng làm trễ Tuần 3 | Recall@10 < 0,90 sau 14.3 | Nhánh 3 ở Task 14.7 đã định sẵn; cắt phạm vi Ngày 15–16 thay vì kéo dài Ngày 14 |
