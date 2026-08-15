# ADR 0010: Hợp đồng statement scope và đường ống E2E thật

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A1 (`statement_scope` là trường trên `FinancialQueryPlan`, không lọc ngầm ở tầng
  retrieval hay locator); B1 (suy diễn có mặc định khi câu hỏi không nêu, và `scope_inferred` là mã
  lỗi verification CHẶN); C1 (khử nhập nhằng do đơn vị NULL — giữ đơn vị đã biết khi chỉ có một giá
  trị phân biệt); D1 (nhập nhằng do loại bảng nhận mã lỗi riêng, không sửa trong Ngày 21); E1 (harness
  E2E mới trong `pipeline/`, retrieval nằm trong vòng lặp, ghi lại abstain); F1 (mở rộng bộ QA sửa
  lệch phân bố scope, giữ nguyên 5 quy tắc chống rò rỉ của gold70); G1 (cổng báo cáo bảng biên đánh
  đổi 3 chính sách, không báo cáo một con số)

## Bối cảnh

[Kế hoạch Ngày 21](../plans/day21-e2e-week3-gate.md) đo trên release đã khoá
(`data/processed/release_v2_422df141c935`) và mã Ngày 20 vừa commit (`f74c01b`), phát hiện: đường
ống "E2E" của Ngày 20 nạp thẳng `gold_table_ids` vào cả `candidate_table_ids` lẫn
`retrieved_table_ids`, nên retriever bị bỏ qua hoàn toàn. Thay bằng ranking BM25 v4 thật đã lưu
(Ngày 14): answered rơi từ 30/70 xuống 9/70, accuracy 0,900 → 0,667. Nguyên nhân không phải retrieval
kém — 6/6 ca đã soi có bảng gold nằm trong top-10 — mà là `statement_scope` (báo cáo riêng/hợp nhất)
chưa được `FinancialQueryPlan` mô hình hoá, dù trường này đã có sẵn trong release khoá. ADR này chốt
7 quyết định thiết kế bắt buộc trước khi viết code.

## Quyết định A: scope nên sống ở đâu trong hệ thống?

**Đã chọn: A1 — trường `statement_scope` trên `FinancialQueryPlan`, cùng hạng với `companies` và
`periods`. Không lọc ngầm ở tầng retrieval (A2) và không "phá hoà" âm thầm ở tầng locator (A3).**

Đo được retrieval **không** phải nơi mất thông tin — gold luôn nằm trong top-10 — nên lọc ở tầng đó
sẽ không giải quyết gì và giấu vấn đề thật. Lọc ngầm trong locator (không đi qua plan) sẽ tái tạo
đúng cái bệnh đã đo: một chiều dữ liệu quyết định đáp án mà không plan nào khai báo được nó, không
audit nào nhìn thấy nó, và validator không kiểm được nó. `statement_scope` phải đi qua
`validate_plan_semantics` và xuất hiện trong mọi artifact audit, đúng vai trò `companies`/`periods`
đang có.

## Quyết định B: khi câu hỏi không nêu scope thì làm gì?

**Đã chọn: B1 — vẫn suy diễn scope (mặc định `consolidated`) và vẫn tính ra đáp án, nhưng đánh dấu
`scope_inferred` và để mã đó CHẶN `verification_status`.**

Đo được cả hai lựa chọn thuần đều thất bại: mặc định im lặng (`default_consolidated`) đưa answered từ
9/70 lên 25/70 nhưng accuracy **tệ đi** 0,667 → 0,583 và 10/25 = 40% câu trả lời **sai mà vẫn trình
bày tự tin** — vi phạm thẳng ràng buộc cứng nhất của dự án (< 5%, mục 1.2 plan.md); từ chối thẳng khi
không nêu scope (`abstain_when_unstated`) cho accuracy tuyệt đối 1,000 nhưng chỉ 3/70 = 4,3% phủ.

Lựa chọn B1 tách hai việc ra: **tính** (để không bỏ phí thông tin đã có) và **trình bày là chắc
chắn** (để không vi phạm ràng buộc sai-tự-tin). Answer package vẫn mang đầy đủ số, evidence, lý do —
dùng được cho audit và cho UI Ngày 22-23 ("thiếu thông tin: câu hỏi không nêu riêng hay hợp nhất") —
nhưng không được trình bày như một đáp án chắc chắn.

Khác với `period_inferred` (Ngày 20, không chặn): kỳ suy diễn sai lệch kỳ báo cáo, còn scope suy diễn
sai **đổi hẳn con số** — đo được 92,8% nhóm hai-scope bất đồng giá trị thật, không phải sai số nhỏ.
Mức chặn khác nhau vì bán kính thiệt hại khác nhau, cả hai đều có số đo đứng sau.

Mặc định `consolidated` khi suy diễn dựa trên hình thái gold70 (bảng gold nghiêng 45 consolidated so
với 21 separate), nhưng đo được đề chính thức 1.012 câu nghiêng ngược lại (`separate` 36,4% so với
`consolidated` 1,3%) — nên mặc định này phải đo lại khi bộ QA sửa lệch phân bố (nhiệm vụ 21.9), và
không khoá cứng bằng hằng số: đặt trong `ExecutionSettings.default_statement_scope` để đổi được bằng
config và đo được bằng ablation.

## Quyết định C: nhập nhằng do đơn vị NULL xử lý thế nào?

**Đã chọn: C1 — khi tập ứng viên rút gọn về đúng một giá trị phân biệt, và các đơn vị khác nhau chỉ
ở chỗ có NULL hay không, đó không phải xung đột: lấy giá trị đó với đơn vị đã biết duy nhất.**

Đo được `locator.py` dùng `drop_duplicates(subset=["value", "unit"])`, nên `(X, None)` và
`(X, "VND")` bị đếm là hai cặp phân biệt, gây `cell_ambiguous` ở nơi giá trị **giống hệt nhau** (ca
OCB: `2582236224358.0 None` vs `2582236224358.0 VND`). Đo bán kính: 868 nhóm
(company, scope, metric, period) có đúng một giá trị nhưng vẫn báo `cell_ambiguous`; **859/868
(99,0%)** là do một ô thiếu đơn vị đồng hành với ô có đơn vị và cùng giá trị; chỉ 9/868 (1,0%) là hai
đơn vị thật khác nhau. Ranh giới rõ ràng: khi **mọi** ô đều thiếu đơn vị vẫn giữ nguyên `unit_missing`
(Ngày 20); khi có ≥2 đơn vị đã biết khác nhau vẫn giữ nguyên `cell_ambiguous`.

## Quyết định D: nhập nhằng do loại bảng xử lý thế nào?

**Đã chọn: D1 — mã lỗi riêng để phân lỗi cho đúng tầng và đo được, KHÔNG sửa trong Ngày 21.**

Sau khi áp đúng scope, 4 ca vẫn `cell_ambiguous`: 1 ca là bug đơn vị NULL (Quyết định C), 3 ca còn
lại (NVL báo cáo bộ phận theo lĩnh vực; VSC bảng biến động vốn chủ với số dư đầu/cuối kỳ) là một
chiều khác — *loại bảng*, không phải scope. Kiểm tra `tables.statement_type` làm bộ lọc: **NULL
78,8%** (115.120/146.011 bảng) — không dùng được. Cố sửa hôm nay là đổi một chiều dữ liệu chưa đo lấy
một con số cổng đẹp, đúng cái bẫy đã đo ở Quyết định B. Ghi thành nợ có mã lỗi và số đo, không đoán
giải pháp.

## Quyết định E: harness E2E thật dựng ở đâu?

**Đã chọn: E1 — module `pipeline/` mới, retrieval nằm TRONG vòng lặp (dùng ranking BM25 v4 đã lưu,
không rebuild index), và abstain được ghi lại thành hạng mục lỗi hạng nhất.**

Không nhét vào `verification/evaluation.py`: nó đang đo compiler+verifier với retrieval hoàn hảo và
nên tiếp tục đo đúng thứ đó (một ablation "retrieval hoàn hảo" hữu ích). Đo được
`evaluate_answer_packages_on_gold` gặp `plan is None` thì `continue` không ghi `failures` — 19/70 câu
abstain hiện vô hình, nên chỉ số cổng "invalid plan < 5%" không tính được. Harness mới phải quy mỗi
lỗi về đúng một tầng (`retrieval`/`planning`/`normalization`/`execution`/`verification`) theo quy tắc
kiểm được: `cell_ambiguous` **có** gold trong tập ứng viên phải quy về `planning` (thiếu chiều phân
biệt trong plan), không phải `retrieval` — đo được trực giác ngược lại (§ 1.2 kế hoạch) là sai.

## Quyết định F: mở rộng bộ QA lên ≥120 câu như thế nào?

**Đã chọn: F1 — giữ nguyên 5 quy tắc chống rò rỉ của `retrieval-gold-v1.provenance.md`; 50 câu bổ
sung phải kéo tỷ lệ nêu scope và độ nghiêng `separate`/`consolidated` của bộ QA về gần đúng đề chính
thức.**

Đo được gold70 không đại diện cho đề thật đúng ở chiều này: đề chính thức 1.012 câu nêu scope trong
37,7% và nghiêng hẳn `separate` (36,4% so với 1,3% `consolidated`); gold70 chỉ nêu 22,9% và bảng gold
nghiêng ngược lại (45 `consolidated` so với 21 `separate`). Thêm số lượng mà không sửa lệch này thì
mọi con số cổng đo sau đó (kể cả mặc định ở Quyết định B) vẫn tiếp tục đo sai thứ mà hệ thống sẽ gặp
trên đề thật.

## Quyết định G: cổng báo cáo cái gì?

**Đã chọn: G1 — bảng biên đánh đổi (phủ × accuracy × số câu sai tự tin) cho cả ba chính sách scope,
kèm phán quyết đạt/không đạt và nhánh xử lý. Không báo cáo một con số accuracy duy nhất.**

Cổng Ngày 21 hỏi "answer accuracy ≥ 0,85" như một con số, nhưng đo được con số đó là hàm của một
chính sách scope có thể chọn tuỳ ý trong khoảng phủ 4,3%–35,7% (9 đến 25 câu answered trên 70) — báo
cáo một con số mà không kèm chính sách là tự lừa. Đúng khuôn mẫu Ngày 14 (F2@R không đạt, ghi rõ +
nhánh xử lý + nợ kỹ thuật), không phải chọn chính sách nào làm con số "đẹp" hơn.

## Số đo hỗ trợ quyết định

| Số đo | Giá trị | Nguồn |
| --- | ---: | --- |
| answered với gold tables (báo cáo Ngày 20) | 30/70 | § 1.1 kế hoạch Ngày 21 |
| answered với retrieval thật (BM25 v4) | 9/70 | § 1.1 |
| accuracy: gold tables vs retrieval thật | 0,900 vs 0,667 | § 1.1 |
| câu verdict đổi từ đúng sang sai khi bật retrieval thật | 0 | § 1.1 |
| `cell_ambiguous`: gold tables vs retrieval thật | 2 vs 24 | § 1.2 |
| ca `cell_ambiguous` mới có gold nằm trong top-10 (đã soi) | 6/6 | § 1.2 |
| `statement_scope` tham chiếu trong `planning/`,`execution/`,`verification/` | 0 | § 1.3 |
| nhóm (company,period,metric) có mặt ở cả hai scope | 14.717/22.680 (64,9%) | § 1.4 |
| ...trong đó bất đồng giá trị | 13.656 (92,8%) | § 1.4 |
| ...trong đó tách sạch 1 giá trị/1 scope | 11.553 (84,6% của bất đồng) | § 1.4 |
| chính sách `none`: answered / accuracy / sai tự tin | 9 / 0,667 / 3 | § 1.5 |
| chính sách `default_consolidated`: answered / accuracy / sai tự tin | 25 / 0,583 / 10 (40%) | § 1.5 |
| chính sách `abstain_when_unstated`: answered / accuracy / sai tự tin | 3 / 1,000 / 0 | § 1.5 |
| đề chính thức (1.012) nêu scope / nghiêng `separate` | 37,7% / 36,4% vs 1,3% | § 1.6 |
| gold70 nêu scope / bảng gold nghiêng | 22,9% / 45 consolidated vs 21 separate | § 1.6 |
| nhóm một-giá-trị bị báo `cell_ambiguous` oan do đơn vị NULL | 859/868 (99,0%) | § 1.7 |
| ...do hai đơn vị thật khác nhau | 9/868 (1,0%) | § 1.7 |
| `tables.statement_type` NULL | 115.120/146.011 (78,8%) | § 1.8 |
| câu abstain bị harness Ngày 20 nuốt (không ghi `failures`) | 19/70 | § 1.9 |
| câu có scope nêu mâu thuẫn với bảng gold | 0/70 | § 1.10 |
| latency p50 / p95 (plan→execution→package, không tính retrieval) | 0,563s / 0,728s | § 1.10 |
