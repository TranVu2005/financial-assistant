# Kế hoạch Ngày 20 — Answer verifier và citation

> **Trạng thái:** đã đo, **đã thực hiện xong** (2026-08-15). ADR:
> [0009](../decisions/0009-answer-package-contract.md).
> **Ngày đo:** 2026-08-15. **Release khoá:** `data/processed/release_v2_422df141c935`,
> `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
> **Mọi con số trong tài liệu này là kết quả chạy thật trên release đã khoá và trên mã Ngày 19 vừa
> commit (`e052e22`)**, không phải ước lượng.
>
> **Kết quả sau khi cài đặt xong:** `data/qa/answer-gold-v1.jsonl` (30 nhãn tay) lần đầu tồn tại; CLI
> `verify-answers` trên gold70 cho **30/30 `answered` đều `verified`**, accuracy 0,9 so với nhãn tay
> (27/30 khớp tuyệt đối, 3/30 lệch ~2×10⁻³ VND do artifact float ingestion đã ghi rõ, không sửa nhãn
> theo máy); gold70 vẫn đúng 30/51 không đổi. Một bug thật (regex chỉ đọc nhóm dấu phẩy đầu của số
> hàng nghìn, oan 17/30 đáp án đúng) bị chính vòng chạy end-to-end bắt được, sửa bằng TDD. Chi tiết
> đầy đủ trong [ADR 0009](../decisions/0009-answer-package-contract.md) và blockquote kết quả trong
> [plan.md](../../plan.md).

Mục Ngày 20 trong [plan.md](../../plan.md) yêu cầu bốn việc: kiểm tra answer numeric với kết quả
executor và tolerance; chuẩn hoá scale/unit và phát hiện trộn nghìn/triệu/tỷ, phần trăm/tỷ lệ; xác
nhận mọi input cell thuộc bảng đã retrieve; sinh câu trả lời theo template, LLM chỉ diễn đạt khi số
đã khoá.

**Phép đo dưới đây tìm ra một thứ quan trọng hơn cả bốn gạch đầu dòng đó: bộ QA hiện tại
không có một đáp án số nào được gán nhãn.** `GoldRetrievalQuestion` có `gold_table_ids` và
`gold_evidence`, **không có trường nào chứa "answer"**, và `data/qa/` không có file đáp án. Nghĩa là
"kiểm tra answer numeric" ở Ngày 20 **không thể** là so với đáp án người gán — nó chỉ có thể là kiểm
tra tính nhất quán nội bộ. Nghiêm trọng hơn: **cổng tuần 3 ở Ngày 21 yêu cầu "answer accuracy ≥ 0,85"
— chỉ số đó hiện không tính được**. Ngày 20 phải tạo ra bộ nhãn đáp án, nếu không Ngày 21 bị chặn.

Ngoài ra phép đo tìm được **một mâu thuẫn hợp đồng đang ngủ** giữa planner và executor (§ 1.2) và
**một đơn vị bịa** len vào evidence trên 20,7 % ô số toàn corpus (§ 1.3).

---

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Compiler + sandbox (Ngày 18–19) | `execution/` — `compile_plan`, `sandbox.py` | ✅ 30/51 gold70, đã commit `e052e22` |
| Evidence có `cell_id` đầy đủ | `execution/contracts.py` — `CellMatch` | ✅ 33/33 ô truy ngược được (§ 1.6) |
| Cờ `period_inferred` | `CellMatch.period_inferred` | ✅ **đã lưu sẵn từ Ngày 18 đúng cho mục đích này** |
| Neo nguồn (dòng, đường dẫn, tiêu đề bảng) | `cells/tables/documents.parquet` | ✅ **100 % đầy đủ**, không cần khảo cổ dữ liệu |
| Quy đổi đơn vị | `normalization/units.py` — 6 `CanonicalUnit` | ✅ tái dùng nguyên |
| Hạ tầng LLM (Ngày 17) | `planning/llm_client.py`, `llm_prompt.py` | ✅ mẫu để nhân bản cho vai trò diễn đạt |
| Bộ QA có **đáp án số** | — | ❌ **không tồn tại** — xem § 1.1, đây là chặn Ngày 21 |
| `expected_unit` trên plan | `plan_contracts.py` | ❌ **NULL 30/30 plan**, và mâu thuẫn với executor — § 1.2 |
| Đơn vị của ô | `CellMatch.unit` | ⚠️ **nhận giá trị bịa `'nan'`** trên ô thiếu unit — § 1.3 |
| Package `verification/` | — | ❌ **chưa tồn tại** |

---

## 1. Chốt chặn phải đo trước khi viết code

### 1.1. Không có đáp án số nào được gán nhãn — cổng Ngày 21 đang treo

```
GoldRetrievalQuestion fields: dataset_fingerprint, filters, gold_evidence, gold_table_ids,
                              intent, question, question_id, reviewed_at, reviewed_by
fields containing 'answer': []
data/qa files: entity-cases-v1.jsonl, malicious-plan-cases-v1.jsonl, plan-cases-v1.jsonl,
               retrieval-failure-annotations-v1.jsonl, retrieval-gold-v1.jsonl, ...
```

gold70 là bộ vàng **retrieval**, `plan-cases-v1` là bộ vàng **operation/abstain**. Không bộ nào chứa
một con số đúng. Hệ quả trực tiếp:

- "Kiểm tra answer numeric với kết quả executor" ở Ngày 20 **chỉ có thể** là *nhất quán nội bộ*
  (câu trả lời hiển thị ≡ số đã khoá ≡ replay `pandas_query`), **không phải** *đúng so với báo cáo*.
  Phải nói thẳng điều này, không được để người đọc hiểu nhầm là đã đo độ chính xác.
- Cổng Ngày 21 ("answer và execution accuracy ≥ 0,85") **hiện không tính được**. Ngày 20 phải sinh
  `data/qa/answer-gold-v1.jsonl`, nếu không Ngày 21 không có gì để chấm.

May mắn là việc gán nhãn khả thi và rẻ: § 1.6 đo được **mọi** ô evidence đều truy ngược được tới
đường dẫn tài liệu + số dòng nguồn, nên người gán nhãn mở đúng dòng đó ra đọc là xong.

### 1.2. `expected_unit`: vừa là code chết, vừa **mâu thuẫn không thể thoả mãn**

Đo trên 30 plan cho ra đáp án: **`expected_unit` là `None` ở cả 30/30**. `rule_planner.py` không
bao giờ đặt trường này (0 tham chiếu). Đây là lần thứ **tư** lặp lại mô hình "trường/khối cấu hình
chết": `llm:` trước Ngày 17, `execution:` trước Ngày 18, `timeout_seconds`/`max_rows` trước Ngày 19.

Nhưng lần này tệ hơn code chết. Chạy validator thật:

```
growth_rate plan, expected_unit='percent'  -> validator issues: ()          # hợp lệ
growth_rate plan, expected_unit='ratio'    -> expected_unit_mismatch        # bị từ chối
```

`_validate_growth_rate` cho phép **duy nhất** `"percent"`
([plan_validator.py:222](../../src/financial_report_qa/planning/plan_validator.py#L222)), trong khi
`compile_growth_rate` **hardcode trả về `"ratio"`**
([operations.py:47](../../src/financial_report_qa/execution/operations.py#L47)). Nghĩa là:

> **Một plan `growth_rate` hợp lệ không bao giờ có thể khai báo đúng đơn vị mà executor sẽ trả về.**

Hiện tại mâu thuẫn này vô hình vì rule planner không đặt `expected_unit`. Nhưng **prompt LLM Ngày 17
dạy model đặt đúng `"expected_unit": "percent"`**
([llm_prompt.py:50](../../src/financial_report_qa/planning/llm_prompt.py#L50)) — nên ngay khi nhánh
LLM sinh ra một plan `growth_rate`, plan và executor bất đồng **100 %**. Ngày 20 không thể "kiểm tra
scale/unit" mà bỏ qua chuyện hai đầu đang định nghĩa đơn vị khác nhau.

### 1.3. 20,7 % ô số **không có đơn vị**, và nó biến thành chuỗi bịa `'nan'`

| Số đo | Giá trị |
| --- | ---: |
| Ô số toàn corpus (`col_idx > 0`, `value_numeric NOT NULL`) | 2.579.383 |
| ...trong đó `unit IS NULL` | **532.901 (20,7 %)** |
| Số bảng chứa ô số thiếu unit | **36.677** |
| Ô số thiếu unit **nằm trong bảng ứng viên gold70** | **416** trên **11 bảng** |

Chạy `locate` thật trên một ô như vậy (DXG, 2021, nhãn `Thu nhập khác`):

```
locate       -> CellMatch unit='nan'
compile_plan -> status=error code=unit_incompatible
```

Hai vấn đề riêng biệt:

1. **`CellMatch.unit` nhận giá trị `'nan'`** — NULL của parquet thành `NaN` của pandas, rồi
   `str(...)` thành chuỗi `"nan"`, lọt qua `NonEmptyString` vì nó *là* chuỗi không rỗng. Đây là một
   **đơn vị bịa** nằm trong bằng chứng, không phải một giá trị hợp lệ nào trong 6 `CanonicalUnit`.
2. **Mã lỗi sai nghĩa.** `unit_incompatible` nghĩa là "trộn đơn vị không quy đổi được". Sự thật là
   "ô này không ghi đơn vị nào cả". Người đọc lỗi — hoặc Ngày 21 phân loại lỗi — sẽ quy sai nguyên
   nhân, sang normalization thay vì sang chất lượng trích xuất.

Trên 51 plan gold70 hiện tại chưa có lỗi unit nào (phân rã Ngày 19: `metric_not_found` 11,
`period_unresolved` 8, `cell_ambiguous` 2), nên đây là lỗi **tiềm ẩn chứ chưa kích hoạt** — nhưng
416 ô trong đúng bảng ứng viên gold70 nghĩa là chỉ cần một metric selector khác là chạm.

### 1.4. Hình dạng đáp án thật: 13/30 là `ratio` 28 chữ số

30 kết quả `answered`:

| Chiều | Phân bố |
| --- | --- |
| Operation | `growth_rate` 13, `lookup` 9, `difference` 8 |
| Đơn vị tính được | `VND` 14, `ratio` 13, `VND_million` 3 |
| `expected_unit` khai báo | `None` 30 |

Số chữ số có nghĩa của đáp án: `9→2, 10→1, 12→1, 13→5, 14→5, 15→1, 16→1, 17→1, **28→13**`.
**15/30 đáp án có hơn 15 chữ số có nghĩa**, vì `Decimal / Decimal` chạy ở precision mặc định 28.
Ví dụ thật:

```
growth_rate  ratio  -0.01932846513079090948136258551
growth_rate  ratio  -0.5727677612443358238108669005
```

Đây là thứ người dùng sẽ nhìn thấy nếu không có hợp đồng hiển thị. Và nó cho thấy **"tolerance"
trong plan.md là hai việc khác nhau, không được gộp**:

```
raw                            : -0.01932846513079090948136258551
hiển thị dạng phần trăm, 2 số  : -1.93
làm tròn 4 số rồi đọc ngược    : -0.0193   (lệch 0.0000284651307909...)
```

- So *executor* với *verifier tính lại*: phải **khớp tuyệt đối** (cùng số học Decimal, sai một chữ
  số là bug — đúng tinh thần `ExecutionReplayMismatchError` của Ngày 18).
- So *số đã khoá* với *chuỗi hiển thị*: phải khớp **trong dung sai của chính độ chính xác hiển thị
  đã khai báo**, không phải một epsilon tuỳ ý.

### 1.5. 6/30 đáp án dựa trên kỳ **suy diễn** — món nợ n = 10 của ADR 0007

| Số đo | Giá trị |
| --- | ---: |
| Đáp án có ≥1 ô evidence dùng kỳ suy diễn | **6/30 (20 %)** |
| Ô evidence dùng kỳ suy diễn | **10/51 (19,6 %)** |

Quy tắc suy diễn (`Số cuối năm` → `report_year`, `Số đầu năm` → `report_year − 1`) là quy tắc yếu
nhất trong toàn hệ thống — ADR 0007 quyết định C2 ghi rõ nó chỉ có **n = 10 ô đối chứng**. Một phần
năm số đáp án đang tựa lên nó. `period_inferred` đã được Ngày 18 lưu sẵn **đúng cho tình huống này**;
Ngày 20 là nơi cờ đó phải trồi lên thành tín hiệu độ tin cậy cho người đọc, không được nuốt im.

### 1.6. Chuỗi trích dẫn **truy ngược được 100 %** — không cần khảo cổ dữ liệu

33 ô evidence phân biệt trên 30 đáp án. Kiểm tra độ đầy đủ của từng trường cần cho trích dẫn:

| Trường | Đầy đủ |
| --- | ---: |
| `cells.source_line_start` | 33/33 |
| `cells.source_line_end` | 33/33 |
| `documents.relative_path` | 33/33 |
| `tables.title_raw` | 33/33 |
| `tables.statement_type` | 33/33 |

Khác hẳn Ngày 18 (`tables.csv_path` NULL 146.011/146.011): ở đây dữ liệu đã sẵn sàng. Nhiệm vụ trích
dẫn của Ngày 20 là **hợp đồng và kết xuất**, không phải đi tìm dữ liệu — và đây cũng chính là thứ
làm việc gán nhãn ở § 1.1 trở nên khả thi.

### 1.7. Provenance đúng nhưng **không được kiểm**, và answer package không tự kiểm được

Đo: **0** ô evidence nằm ngoài `plan.candidate_table_ids` trên cả 30 đáp án — đúng theo cấu trúc, vì
`build_cell_frame` lọc sẵn theo `table_ids`. Nhưng:

- Bất biến đó **không có assertion nào** canh giữ. Nó đúng vì tình cờ đúng đường đi, không vì có ai
  kiểm.
- `CompiledQuery` **không mang `candidate_table_ids`** (các trường: `operation`, `status`, `answer`,
  `unit`, `evidence`, `pandas_query`, `error_code`, `error_message`). Nên một answer package dựng từ
  `CompiledQuery` **không thể tự chứng minh** rằng mọi ô của nó thuộc bảng đã retrieve — muốn kiểm
  phải cầm thêm plan. Yêu cầu "xác nhận mọi input cell đều thuộc bảng đã retrieve" của plan.md vì
  thế cần sửa hợp đồng, không chỉ thêm một vòng `for`.
- Lưu ý phạm vi: harness gold70 truyền `gold_table_ids` làm ứng viên. Trong luồng thật, ứng viên đến
  từ retrieval. Verifier phải kiểm evidence ⊆ **tập đã retrieve**, nên tập đó phải đi theo package.

### 1.8. "Không có số mới do LLM thêm": bộ đếm token số nhìn thấy gì

| Số đo | Giá trị |
| --- | --- |
| Token số / câu hỏi gold70 (toàn bộ 70) | 1 token: 35 câu; 2 token: 35 câu |
| Token số / câu hỏi (30 câu ra đáp án) | 1 token: 9; 2 token: 21 |
| Ví dụ | `So sánh ... của CTG giữa năm 2022 và năm 2023.` → `['2022', '2023.']` |

**Mọi** token số trong câu hỏi gold70 đều là **năm**. Nên bộ phát hiện "số lạ" phải có danh sách
trắng gồm: số đáp án đã khoá, các `periods` của plan, và các giá trị ô evidence — mọi token số ngoài
tập đó là vi phạm. Lưu ý kỹ thuật đã đo được: tách token thô bắt cả dấu chấm câu (`'2023.'`), nên
chuẩn hoá token là một phần của bài toán chứ không phải chi tiết vặt.

### 1.9. Trộn đơn vị: hiếm ở đáp án, phổ biến ở corpus

- Đáp án có evidence trải > 1 đơn vị: **0/30**. Đơn vị của 51 ô evidence: `VND` 43, `VND_million` 8.
- Toàn corpus (ô số): `VND` 1.320.588 · `VND_million` 586.726 · **NULL 532.901** · `percent` 109.980
  · `VND_thousand` 28.455 · `VND_billion` 723 · `ratio` 10.

Nên phép kiểm "trộn nghìn/triệu/tỷ" là **phòng thủ cho tương lai** trên dữ liệu hiện tại, còn phép
kiểm thật sự cấp bách là **thiếu đơn vị** (§ 1.3) — lớn hơn `VND_thousand` + `VND_billion` +
`percent` cộng lại.

---

## 2. Quyết định thiết kế

### A. Đáp án chuẩn — **A2: gán nhãn tay 30 đáp án từ dòng nguồn, không tự sinh từ executor**

Không chọn A1 ("lấy output executor làm nhãn"): nó tự chứng minh chính mình, accuracy luôn = 1,0 và
cổng Ngày 21 thành vô nghĩa. Gán nhãn tay dựa trên § 1.6 (mọi ô truy được về `relative_path` +
`source_line_start`), ghi ra `data/qa/answer-gold-v1.jsonl` với **provenance của người gán**
(`reviewed_by`, `reviewed_at`) đúng mẫu `GoldRetrievalQuestion` đã có. Nhãn ghi *giá trị đọc được từ
báo cáo*, độc lập với con số executor tính ra — nếu hai bên lệch, đó là phát hiện, không phải lỗi
quy trình.

### B. Mâu thuẫn `growth_rate` — **B1: `ratio` là đơn vị *tính*, `percent` là dạng *trình bày***

Executor giữ nguyên `ratio` (nó trả về một phân số không thứ nguyên — đúng về bản chất). Sửa
`_validate_growth_rate` để chấp nhận **cả** `ratio` và `percent`, và đưa quan hệ trình bày
(`ratio` ↔ `percent`, hệ số 100) vào verifier như một **bảng tương đương tường minh**, không phải
một phép nhân rải rác trong code kết xuất. `expected_unit` từ đó có nghĩa rõ: *dạng người dùng muốn
đọc*, chứ không phải *đơn vị executor phải trả về*.

Đây là cách duy nhất giữ được cả ba thứ đang đúng: bản chất số học của executor, prompt Ngày 17 đang
dạy LLM, và yêu cầu "chuẩn hoá scale/unit" của plan.md.

### C. Thiếu đơn vị — **C1: mã lỗi riêng `unit_missing`, cấm chuỗi `'nan'` lọt vào `CellMatch`**

`cell_frame` phải giữ NULL là NULL (không để pandas biến thành `NaN` rồi thành `"nan"`), và `locate`
trả `unit_missing` thay vì để `convert_scale` ném `ValueError` rồi bị quy oan thành
`unit_incompatible`. Thêm `unit_missing` vào `ExecutionIssueCode` → **11 mã**. Ràng buộc
`CellMatch.unit` thành đúng 6 `CanonicalUnit` để chuỗi bịa không bao giờ vào được evidence nữa.

Đây vẫn nằm trong phạm vi Ngày 20 vì plan.md ghi rõ "**chuẩn hoá scale/unit**" — và một đơn vị bịa
là lỗi chuẩn hoá nghiêm trọng hơn mọi trường hợp trộn scale đo được (§ 1.9).

### D. Dung sai — **D1: hai phép so khác nhau, không gộp làm một**

| Phép so | Dung sai | Lý do |
| --- | --- | --- |
| verifier tính lại vs `CompiledQuery.answer` | **bằng tuyệt đối** | cùng số học Decimal; lệch = bug, đúng tinh thần `ExecutionReplayMismatchError` |
| chuỗi hiển thị vs số đã khoá | **trong đúng độ chính xác hiển thị đã khai báo** | § 1.4: làm tròn 4 số mất 2,8 × 10⁻⁵ — mất mát này *phải* được khai báo, không được giấu |

Hợp đồng hiển thị: tiền tệ giữ nguyên số nguyên VND; `ratio` hiển thị dạng `percent` 2 chữ số thập
phân. Package mang **cả** số thô (`answer`) lẫn chuỗi hiển thị (`display`) lẫn `display_precision`,
để phép so thứ hai kiểm được thay vì phải tin.

### E. Answer package — **E1: model đóng băng, tự chứa, tự kiểm được**

Trường bắt buộc: `question_id`, `question`, `operation`, `answer` (Decimal đã khoá), `unit`,
`display`, `display_precision`, `evidence` (mỗi mục: `cell_id`, `table_id`, `doc_relative_path`,
`source_line_start/end`, `table_title`, `value`, `unit`), `retrieved_table_ids`, `pandas_query`,
`period_inferred` (cờ tổng hợp), `verification_status`, `verification_issues`.

`retrieved_table_ids` đi kèm chính là thứ sửa § 1.7: package tự chứng minh được evidence ⊆ tập đã
retrieve mà không cần cầm thêm plan.

### F. Vai trò LLM — **F1: template trước, LLM tuỳ chọn, và bị kiểm bằng diff token số**

Kết xuất mặc định là **template thuần, không cần LLM** — toàn bộ test và CLI phải chạy được offline,
đúng như Ngày 17 đã làm với eval harness. Nếu bật LLM diễn đạt, output của nó đi qua bộ so token số
(§ 1.8): danh sách trắng = {số đáp án, các `periods`, giá trị ô evidence}; token số ngoài tập đó →
**từ chối bản diễn đạt và rơi về template**, không phải cảnh báo rồi vẫn dùng. DoD của plan.md
("không có số mới do LLM tự thêm") là một *phép ép*, không phải một *lời hứa*.

### G. Nơi ở — **G1: package `verification/` mới + CLI `verify-answers`**

Đúng mô hình `planning/` (Ngày 15–17) và `execution/` (Ngày 18–19): contracts, logic, evaluation,
cli. Subcommand `verification` gắn vào `cli.py` bên cạnh 6 subcommand hiện có.

---

## 3. Nhiệm vụ

| # | Việc | Đầu ra |
| --- | --- | --- |
| 20.1 | Viết ADR 0009 chốt A2/B1/C1/D1/E1/F1/G1 | `docs/decisions/0009-answer-package-contract.md` |
| 20.2 | Sửa `'nan'` unit: `cell_frame` giữ NULL, `CellMatch.unit` ràng 6 `CanonicalUnit`, thêm mã `unit_missing` (TDD) | `execution/{cell_frame,contracts,locator}.py` |
| 20.3 | Sửa mâu thuẫn `growth_rate`: validator chấp nhận `ratio` + `percent` (TDD) | `planning/plan_validator.py` |
| 20.4 | `verification/contracts.py`: `AnswerPackage`, `Citation`, `VerificationIssue` (TDD) | package mới |
| 20.5 | `verification/checks.py`: 5 phép kiểm — tính lại, đơn vị/scale, provenance, hiển thị, kỳ suy diễn (TDD) | module + test |
| 20.6 | `verification/templates.py`: template tiếng Việt cho 9 operation, không cần LLM (TDD) | module + test |
| 20.7 | `verification/numeric_guard.py`: tách + chuẩn hoá token số, diff với danh sách trắng (TDD) | module + test |
| 20.8 | `verification/builder.py`: `CompiledQuery` + plan + retrieved ids → `AnswerPackage` đã kiểm (TDD) | module + test |
| 20.9 | **Gán nhãn `data/qa/answer-gold-v1.jsonl`** cho 30 đáp án, đọc từ dòng nguồn thật | bộ nhãn + tài liệu provenance |
| 20.10 | CLI `verify-answers` + báo cáo; verification đầy đủ; cập nhật `plan.md`/README | báo cáo + tài liệu |

### Chi tiết 20.5 — năm phép kiểm, mỗi phép gắn với một số đo

| Phép kiểm | Nội dung | Neo |
| --- | --- | --- |
| `recompute_mismatch` | verifier tính lại từ evidence, so **bằng tuyệt đối** với `answer` | § 1.4 |
| `unit_not_presentable` | `unit` tính được không trình bày được thành `expected_unit` theo bảng tương đương | § 1.2 |
| `evidence_outside_retrieval` | có ô evidence ngoài `retrieved_table_ids` | § 1.7 |
| `display_roundtrip_mismatch` | đọc ngược `display` không khớp `answer` trong `display_precision` | § 1.4 |
| `period_inferred_warning` | ≥1 ô evidence dùng kỳ suy diễn (cảnh báo, **không** chặn) | § 1.5 |

Bốn mã đầu chặn (`verification_status = "rejected"`); mã cuối chỉ hạ độ tin cậy — vì 6/30 đáp án
hiện tại dựa vào kỳ suy diễn, biến nó thành lỗi chặn sẽ xoá 20 % kết quả đang đúng.

### Chi tiết 20.9 — gán nhãn thế nào cho khỏi tự lừa mình

Người gán nhãn nhận: câu hỏi, `relative_path`, `source_line_start/end`, tiêu đề bảng — **không nhận
con số executor tính ra**. Ghi lại giá trị đọc được và đơn vị đọc được. Chỉ sau khi ghi xong mới đối
chiếu. Mọi lệch phải được ghi vào tài liệu provenance kèm nguyên nhân (lỗi executor / lỗi
normalization / nhãn sai), không được sửa nhãn cho khớp máy.

---

## 4. Thứ tự thực hiện

```
20.1 (ADR)
  ├─> 20.2 (unit_missing) ──┐
  ├─> 20.3 (growth_rate)  ──┤
  └─> 20.4 (contracts) ─────┴─> 20.5 (checks) ──┐
                            ├─> 20.6 (templates)┤
                            └─> 20.7 (numeric)  ┴─> 20.8 (builder) ─> 20.10 (CLI + verify)
  └─> 20.9 (gán nhãn — chạy song song, không chặn ai) ──────────────────┘
```

20.9 là việc tay, tốn thời gian nhất, và **không phụ thuộc code** — bắt đầu sớm, chạy song song.

---

## 5. Definition of Done

| # | Điều kiện | Cách kiểm chứng |
| --- | --- | --- |
| D1 | 30/30 đáp án gold70 dựng được `AnswerPackage` với trích dẫn đầy đủ | CLI `verify-answers` |
| D2 | 100 % đáp án có ≥1 citation truy tới `relative_path` + số dòng | assertion trong builder |
| D3 | Ô thiếu unit trả `unit_missing`, **không** trả `unit_incompatible`, và `'nan'` không bao giờ vào `CellMatch` | test dùng ô DXG thật ở § 1.3 |
| D4 | Plan `growth_rate` khai `percent` hoặc `ratio` đều hợp lệ và đều kiểm được | test validator + verifier |
| D5 | Bộ diễn đạt thêm một số lạ → bị **từ chối**, rơi về template | test `numeric_guard` với output LLM giả |
| D6 | Toàn bộ đường kết xuất chạy **không cần LLM** | test offline, không monkeypatch mạng |
| D7 | `data/qa/answer-gold-v1.jsonl` tồn tại, ≥30 nhãn, có provenance người gán | file + tài liệu |
| D8 | **gold70 vẫn đúng 30/51 (58,8 %)**, phân rã lỗi không đổi | chạy lại `compile-plans`, so với báo cáo Ngày 19 |
| D9 | `pytest` xanh (≥ 1.020), `ruff`/`mypy` sạch | lệnh verification chuẩn |
| D10 | `dataset_fingerprint` và baseline Ngày 8–14 không đổi | `git status --short data/processed/ src/financial_report_qa/normalization/` rỗng |

D8 lại là mục quan trọng nhất: 20.2 và 20.3 **sửa vào đường thực thi**, nên phải chứng minh chúng
không đổi kết quả đang đúng. Trên gold70 hiện không có lỗi unit nào (phân rã Ngày 19), nên
`unit_missing` **phải** cho ra đúng 0 trường hợp ở đó — nếu khác, một trong hai phía đã sai.

---

## 6. Rủi ro

| # | Rủi ro | Bằng chứng | Giảm thiểu |
| --- | --- | --- | --- |
| R1 | Gán nhãn tay 30 đáp án là việc tốn thời gian nhất và mang tính chủ quan | không có nhãn nào tồn tại (§ 1.1) | 20.9 chạy song song từ đầu; người gán không nhìn số máy trước khi ghi; mọi lệch phải ghi nguyên nhân |
| R2 | Sửa hợp đồng `growth_rate` đụng validator Ngày 15–17 | `_validate_growth_rate` đang chỉ cho `percent` | D8: chạy lại `evaluate-plans` **và** `compile-plans`, cả hai phải không đổi |
| R3 | `unit_missing` đổi phân rã lỗi gold70 | 416 ô thiếu unit trong 11 bảng ứng viên gold70 (§ 1.3) | gold70 hiện có **0** lỗi unit → kỳ vọng 0 thay đổi; nếu khác phải giải thích được |
| R4 | Bộ đếm token số báo động giả trên năm | mọi token số ở gold70 đều là năm (§ 1.8) | danh sách trắng gồm `plan.periods`; test riêng cho câu 2 token năm |
| R5 | 6/30 đáp án dựa kỳ suy diễn n = 10 — độ tin cậy thật thấp hơn vẻ ngoài | § 1.5 | `period_inferred_warning` là cảnh báo bắt buộc hiện trong package, không được ẩn |
| R6 | "Answer accuracy" sau Ngày 20 vẫn chỉ đo trên 30 câu | 30/51 plannable, 70 câu gold | ghi rõ mẫu số trong báo cáo; Ngày 21 yêu cầu nâng bộ QA lên ≥120 câu — đó mới là chỗ mở rộng |
| R7 | Nhãn tay có thể **sai** và làm accuracy trông tệ oan | — | tài liệu provenance ghi từng lệch kèm phân loại nguyên nhân; nhãn sai được sửa **sau khi** ghi nhận, không sửa lặng |
