# ADR 0009: Hợp đồng answer package và verifier

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A2 (gán nhãn tay đáp án từ dòng nguồn, không tự sinh từ executor); B1 (`ratio` là
  đơn vị tính, `percent` là dạng trình bày — validator chấp nhận cả hai cho `growth_rate`); C1 (mã
  lỗi riêng `unit_missing`, cấm chuỗi `'nan'` lọt vào `CellMatch`); D1 (hai phép so dung sai khác
  nhau — tính lại thì bằng tuyệt đối, hiển thị thì trong độ chính xác đã khai báo); E1 (answer
  package tự chứa, tự kiểm được — mang cả `retrieved_table_ids`); F1 (template trước, LLM tuỳ chọn,
  bị kiểm bằng diff token số); G1 (package `verification/` mới + CLI `verify-answers`)

## Bối cảnh

[Kế hoạch Ngày 20](../plans/day20-answer-verifier-citation.md) đo trên release đã khoá
(`data/processed/release_v2_422df141c935`) và mã Ngày 19 vừa commit (`e052e22`), phát hiện: bộ QA
hiện không có một đáp án số nào được gán nhãn (`GoldRetrievalQuestion` không có trường "answer"),
nên cổng Ngày 21 ("answer accuracy ≥ 0,85") hiện không tính được. Đo thêm ba chốt chặn: `expected_unit`
NULL 30/30 plan và mâu thuẫn không thể thoả mãn giữa validator (`growth_rate` chỉ cho `percent`) và
executor (hardcode trả `ratio`); 20,7% ô số toàn corpus (532.901/2.579.383) không có đơn vị, và NULL
biến thành chuỗi bịa `'nan'` lọt vào `CellMatch.unit`, bị quy oan thành `unit_incompatible`; 15/30
đáp án gold70 có > 15 chữ số có nghĩa (13 `growth_rate` trả `ratio` 28 chữ số) nên cần hợp đồng hiển
thị. ADR này chốt 7 quyết định thiết kế bắt buộc trước khi viết code verifier.

## Quyết định A: đáp án chuẩn lấy từ đâu?

**Đã chọn: A2 — gán nhãn tay 30 đáp án từ dòng nguồn thật, không tự sinh từ output executor.**

Không chọn A1 ("lấy output executor làm nhãn"): nó tự chứng minh chính mình, accuracy luôn bằng 1,0
và cổng Ngày 21 thành vô nghĩa. Gán nhãn tay khả thi vì đo được **33/33 ô evidence trên gold70 truy
ngược được 100%** tới `documents.relative_path` + `cells.source_line_start/end` + `tables.title_raw`
— khác hẳn tình trạng Ngày 18 (`tables.csv_path` NULL cho cả 146.011 bảng). Người gán nhãn đọc dòng
nguồn, ghi giá trị và đơn vị đọc được, **không nhìn số executor tính ra trước khi ghi** — nếu hai bên
lệch, đó là phát hiện cần điều tra, không phải lỗi quy trình cần sửa nhãn cho khớp máy.

## Quyết định B: mâu thuẫn `growth_rate` xử lý thế nào?

**Đã chọn: B1 — `ratio` là đơn vị *tính* (bản chất số học, không thứ nguyên), `percent` là dạng
*trình bày* (nhân 100, thêm ký hiệu %). `_validate_growth_rate` chấp nhận cả hai giá trị.**

Đo được `_validate_growth_rate` chỉ cho phép `expected_unit = "percent"`, trong khi
`compile_growth_rate` hardcode trả về `"ratio"` — một plan `growth_rate` hợp lệ không bao giờ có thể
khai đúng đơn vị mà executor sẽ trả về. Mâu thuẫn này vô hình trên gold70 vì rule planner không đặt
`expected_unit` (30/30 NULL), nhưng prompt LLM Ngày 17 dạy model đặt `"percent"`
([llm_prompt.py:50](../../src/financial_report_qa/planning/llm_prompt.py#L50)) — nên nhánh LLM sẽ va
phải mâu thuẫn 100% ngay khi kích hoạt. Giữ executor trả `ratio` nguyên trạng (đúng bản chất số học);
đưa quan hệ trình bày `ratio` ↔ `percent` (hệ số 100) vào verifier như một bảng tương đương tường
minh, không phải một phép nhân rải rác trong code kết xuất.

## Quyết định C: thiếu đơn vị xử lý thế nào?

**Đã chọn: C1 — mã lỗi riêng `unit_missing`; `cell_frame` giữ NULL là NULL; `CellMatch.unit` ràng
buộc đúng 6 giá trị `CanonicalUnit`, cấm chuỗi bịa `'nan'` lọt vào evidence.**

Đo được đường đi cụ thể của lỗi: parquet NULL → pandas `NaN` → `str(...)` → chuỗi `"nan"`, lọt qua
ràng buộc `NonEmptyString` vì nó *là* một chuỗi không rỗng. `locate` sau đó gọi `convert_scale` với
đơn vị `"nan"`, nhận `ValueError`, và `compile_plan` quy oan thành `unit_incompatible` — mã đó nghĩa
là "trộn đơn vị không quy đổi được", trong khi sự thật là "ô này không ghi đơn vị nào cả". Hai
nguyên nhân khác nhau (chất lượng trích xuất vs. logic normalization) cần hai mã lỗi khác nhau để
Ngày 21 phân loại đúng. 416 ô như vậy nằm trong 11 bảng ứng viên gold70 — chưa kích hoạt trên 51 plan
hiện tại (phân rã lỗi Ngày 19 không có lỗi unit nào) nhưng chỉ cần một metric selector khác là chạm.

## Quyết định D: dung sai kiểm tra là gì?

**Đã chọn: D1 — hai phép so khác nhau, không gộp làm một.**

| Phép so | Dung sai | Lý do |
| --- | --- | --- |
| verifier tính lại vs `CompiledQuery.answer` | bằng tuyệt đối | cùng số học Decimal; lệch là bug, đúng tinh thần `ExecutionReplayMismatchError` Ngày 18 |
| chuỗi hiển thị vs số đã khoá | trong đúng độ chính xác hiển thị đã khai báo | làm tròn 4 chữ số một đáp án đo được mất 2,8 × 10⁻⁵ — mất mát này phải được khai báo (`display_precision`), không được giấu |

Đo được 15/30 đáp án gold70 có > 15 chữ số có nghĩa (`Decimal / Decimal` chạy ở precision mặc định
28), ví dụ `-0.01932846513079090948136258551` — chứng minh một hợp đồng hiển thị + dung sai tường
minh là bắt buộc, không phải tinh chỉnh thẩm mỹ.

## Quyết định E: answer package có hình dạng gì?

**Đã chọn: E1 — model đóng băng, tự chứa, tự kiểm được. Mang cả `retrieved_table_ids`.**

`CompiledQuery` (Ngày 18) không mang `candidate_table_ids`, nên một answer package dựng từ nó không
tự chứng minh được rằng mọi ô evidence thuộc bảng đã retrieve — phải cầm thêm plan mới kiểm được. Đo
được bất biến này hiện đúng 0 vi phạm trên gold70 nhưng không có assertion nào canh giữ — đúng vì
tình cờ đúng đường đi (`build_cell_frame` lọc sẵn theo `table_ids`), không vì có ai kiểm. `E1` sửa
hợp đồng để package tự mang bằng chứng: `retrieved_table_ids`, mỗi mục evidence có đủ
`doc_relative_path` + `source_line_start/end` + `table_title` để trích dẫn độc lập, và
`period_inferred` là cờ tổng hợp hiện ra ngoài thay vì bị nuốt (6/30 đáp án hiện dựa vào kỳ suy diễn
n = 10 của ADR 0007 quyết định C2).

## Quyết định F: vai trò của LLM trong việc diễn đạt là gì?

**Đã chọn: F1 — kết xuất mặc định là template thuần, không cần LLM. Nếu bật LLM diễn đạt, output đi
qua bộ so token số bắt buộc; token số ngoài danh sách trắng → từ chối bản diễn đạt, rơi về template.**

Đo được mọi token số trong 70 câu hỏi gold70 đều là năm — nên danh sách trắng (số đáp án đã khoá,
`plan.periods`, giá trị ô evidence) là đủ để phân biệt số hợp lệ với số LLM tự thêm. Yêu cầu của
plan.md ("không có số mới do LLM tự thêm") là một phép ép thực thi được (reject-and-fallback), không
phải một lời hứa trong prompt — đúng tinh thần deny-by-default đã dùng ở Ngày 19. Toàn bộ đường kết
xuất phải chạy được offline (không cần LLM), đúng cách Ngày 17 đã làm với eval harness rule-planner.

## Quyết định G: package đặt ở đâu?

**Đã chọn: G1 — package `verification/` mới (`contracts.py`, `checks.py`, `templates.py`,
`numeric_guard.py`, `builder.py`, `cli.py`), đúng mô hình `planning/`/`execution/` đã có. Subcommand
`verification` gắn vào `cli.py` bên cạnh 6 subcommand hiện có.**

## Số đo hỗ trợ quyết định

| Số đo | Giá trị | Nguồn |
| --- | ---: | --- |
| Trường "answer" trong `GoldRetrievalQuestion` | 0 | § 1.1 kế hoạch Ngày 20 |
| Ô evidence trên gold70 truy ngược được đầy đủ (path + dòng + tiêu đề bảng) | 33/33 (100%) | § 1.6 |
| Plan gold70 có `expected_unit` khác NULL | 0/30 | § 1.2 |
| `_validate_growth_rate` cho phép `expected_unit` nào | chỉ `"percent"` | § 1.2 |
| `compile_growth_rate` trả về đơn vị nào | hardcode `"ratio"` | § 1.2 |
| Ô số toàn corpus có `unit IS NULL` | 532.901/2.579.383 (20,7%) | § 1.3 |
| Ô thiếu unit nằm trong bảng ứng viên gold70 | 416 (11 bảng) | § 1.3 |
| Đáp án gold70 có > 15 chữ số có nghĩa | 15/30 | § 1.4 |
| Đáp án dựa trên ≥1 ô evidence kỳ suy diễn | 6/30 (10/51 ô) | § 1.5 |
| Ô evidence nằm ngoài `candidate_table_ids` (vi phạm đo được) | 0/51, nhưng không có assertion canh giữ | § 1.7 |
| Token số trong câu hỏi gold70 không phải năm | 0 | § 1.8 |
| Đáp án có evidence trải > 1 đơn vị | 0/30 | § 1.9 |

## Hệ quả

- `execution/cell_frame.py` không còn để NULL biến thành chuỗi `"nan"`; `CellMatch.unit` chỉ nhận 6
  giá trị `CanonicalUnit`.
- `ExecutionIssueCode` có thêm `unit_missing` → 11 mã.
- `planning/plan_validator.py`: `_validate_growth_rate` chấp nhận `{"percent", "ratio"}` thay vì chỉ
  `{"percent"}`.
- `verification/` là package mới, độc lập với LLM ở đường mặc định.
- `data/qa/answer-gold-v1.jsonl` là bộ nhãn mới, gán tay, có provenance người gán — không tự sinh từ
  executor.
- **Không đổi `dataset_fingerprint`**; không đụng `data/processed/` hay `normalization/`.
- **gold70 phải vẫn giải được đúng 30/51 (58,8%)** với cùng phân rã lỗi Ngày 19 — `unit_missing` kỳ
  vọng đóng góp đúng 0 trường hợp mới trên gold70 (không có lỗi unit nào trong phân rã hiện tại); nếu
  khác, đó là dấu hiệu cần giải thích, không phải cải thiện tự động được hoan nghênh.
