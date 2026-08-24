# ViFinQA — Nhánh 2 viết lại theo Masked PAL

> Thay thế §6 của `docs/superpowers/specs/2026-08-23-target-architecture.md`.
> §1–§5 (bối cảnh, bằng chứng đo được, Nhánh 1 retrieval) và §7 (evidence &
> linter) của spec đó **vẫn còn hiệu lực**. Spec này chỉ viết lại nhánh answer.

## 1. Vì sao thay

Spec 2026-08-23 chốt nguyên tắc **N4 — Giữ compiler, thay đầu vào**: LLM chỉ
chọn `operation` từ một enum đóng, compiler tất định lắp pandas. Lý do là tính
kiểm chứng: code do LLM sinh ra không chứng minh được là tính từ evidence, và
linter C1–C7 không phân biệt nổi `ans = 4500` do LLM bịa với `ans = 4500` đọc
đúng từ bảng.

Cái giá phải trả đã đo được ở §2.2 của spec cũ: `build_plan` chặn **41%** số
câu vì lý do không liên quan đến dòng — chủ yếu vì câu hỏi không ánh xạ được
vào enum operation. Enum đóng là một cổng chặn, và nó đang chặn nhầm.

**Numeric Masking gỡ được thế lưỡng nan này.** Nếu LLM không bao giờ thấy giá
trị số và không bao giờ được phép viết một literal số, thì chương trình nó
sinh ra **về mặt cấu trúc** không thể chứa đáp án bịa. Khi đó cho phép biểu
thức tự do không còn làm mất tính kiểm chứng — nó chỉ mở rộng tập câu trả lời
được, mà không mở thêm đường nào cho hardcode.

## 2. Nguyên tắc

Giữ nguyên **N1** (hai nhánh độc lập), **N2** (evidence CSV luôn là lát cắt
bảng nguồn), **N3** (metric dictionary là tín hiệu, không phải cổng chặn),
**N5** (không quantize embedding/reranker).

**N4′ (thay N4) — Chương trình do LLM sinh ra không được chứa literal số.**

LLM chỉ sinh biểu thức trên placeholder `[NUM_i]`. Mọi giá trị vào runtime
bằng binding tất định từ CSV evidence. Không có ngoại lệ: không `* 100` để
đổi ra phần trăm, không `/ 1000` để đổi đơn vị, không `round(x, 2)`. Kiểm
được bằng AST — bất kỳ node `ast.Constant` kiểu số nào cũng là vi phạm.

Đây là siết chứ không phải nới. N4 cũ bảo vệ tính kiểm chứng bằng cách cấm
LLM viết code; N4′ bảo vệ đúng thứ đó bằng cách cấm LLM viết **số** — mạnh
hơn, vì nó cho phép biểu thức tuỳ ý mà vẫn chứng minh được mọi con số đến từ
bảng.

**N6 (giữ, làm rõ) — Một đường đi duy nhất.** Sinh lại khi verification lệch
là **retry của đúng một bước**, tối đa **1 lần**, cùng prompt, chỉ khác nhiệt
độ. Vẫn lệch thì đánh dấu low-confidence và đi tiếp. Không có đường thứ hai,
không có tầng dự phòng nào ngoài backstop đã có.

**N7 (giữ, tổng quát hoá) — Quyết định LLM không mang giá trị số.** Trước là
chỉ số vào danh sách **dòng**; giờ là chỉ số vào danh sách **ô**. `[NUM_i]`
trỏ tới `cells[i]`, mà `cells` là các chỉ số vào một danh sách ứng viên dựng
lại được ở local từ release. Một file quyết định cũ hoặc bị sửa vẫn không bơm
được số liệu vào bài nộp.

## 3. Kiến trúc

```
Nhánh 1 (retrieval) ── không đổi ──► top-k* table_ids ──► relevant_docs/tables
        │
        └──► (cùng top-k* table_ids)
                     │
                     ▼
   [1] Hierarchical Linearization      c_ij = (RowPath(i), ColPath(j))
                     │
                     ▼
   [2] Cell candidate list             ô đánh số 0..N-1, KHÔNG kèm giá trị
                     │
                     ▼
   [3] LLM batch offline (masked)  ──► {cells, program, uses, scale}
                     │
                     ▼
   [4] Program AST guard               cấm literal số; chỉ + - * / unary- abs()
                     │
                     ▼
   [5] Deterministic binding           render 2 lần: float, và lookup CSV
                     │
                     ▼
   [6] Sandboxed arithmetic eval       interpreter riêng, không pandas
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   [7a] Verify B              [7b] Verify A
   `uses` ↔ ô đã bind         giải thích lượt 2 ↔ numeric_guard
        └────────────┬────────────┘
                     │ lệch → sinh lại [3], tối đa 1 lần
                     │ vẫn lệch → low_confidence = true, đi tiếp
                     ▼
   answer / pandas_query / csv_path ──► linter C1–C7 + C8 ──► submission.zip
```

Hai bất biến của spec cũ giữ nguyên: mũi tên từ Nhánh 1 sang submission
**không đi qua** Nhánh 2; Nhánh 2 là một đường thẳng.

## 4. Từng bước

### 4.1 Hierarchical Linearization

`c_ij = (RowPath(i), ColPath(j), Value)`. Repo **đã sinh ra cả ba** ở bước
export CSV (spec `2026-08-23-normalized-table-csv-export-design.md`):

- **ColPath** = header đã làm phẳng theo đường dẫn, ví dụ `Tổng_cộng_31/12/2022`.
- **RowPath** = nhãn dòng đã gắn tiền tố nhóm, ví dụ `Nhóm > Nhãn con`, sinh
  từ `_banner_level` của normalization.
- **Value** = `value_numeric` trong `cells.parquet`.

Không cần code mới cho bước này. Phương án Graph/Tree (heterogeneous graph,
cạnh Parent-Child / Sibling / Semantic Cross-reference) **bị loại**: repo đã
thử ở `data/indexes/graph-day11-a` và §8.2 của spec cũ đã xếp nó vào nhánh
nghiên cứu chết.

### 4.2 Danh sách ô ứng viên

Đầu vào: top-k\* `table_id` từ Nhánh 1, cộng dòng ứng viên từ `row_fusion.py`
(bm25 + fuzzy + alias, `DEFAULT_ROW_CANDIDATE_COUNT = 20`), nhân với các kỳ
đã tách bởi `entity_parser.py`.

Mỗi ứng viên là một dict, **không có `value`**:

```python
{"index": 17, "company_code": "VCB", "row_path": "Doanh thu > Doanh thu thuần",
 "col_path": "Năm_2023", "period": 2023, "statement_type": "income_statement",
 "table_title": "...", "unit": "triệu VND", "table_id": "tbl_..."}
```

Thứ tự phần tử **là** hợp đồng, y như `row_choice_batch.py` hiện nay: quyết
định của LLM map ngược về ứng viên bằng chính vị trí này.

Ô không có `value_numeric` (ô rỗng, ô chữ) **không bao giờ** vào danh sách —
lọc ở bước này, không để LLM chọn rồi mới hỏng lúc bind.

### 4.3 LLM batch offline (masked)

Một lần gọi cho mỗi câu, chạy offline theo batch. Đầu ra JSON:

```json
{"cells": [42, 17],
 "program": "([NUM_0] - [NUM_1]) / [NUM_1]",
 "uses": [{"num": 0, "row": "Doanh thu thuần", "col": "Năm 2023"},
          {"num": 1, "row": "Doanh thu thuần", "col": "Năm 2022"}],
 "scale": "percent"}
```

- `cells[i]` là chỉ số vào danh sách §4.2. `[NUM_i]` trỏ tới `cells[i]`.
- `program` là **một biểu thức**, không phải câu lệnh, không gán, không import.
- `uses[i]` là LLM **tự thuật lại** nó nghĩ ô `[NUM_i]` là gì. Dùng cho §4.7a.
- `scale` là enum đóng `{"none", "percent", "thousand", "million", "billion"}`.
  Đây là chỗ duy nhất xử lý đổi thang, vì N4′ cấm `* 100` trong program. Phép
  nhân do renderer tất định của ta làm, không do LLM viết.

**Câu tra cứu thuần** (không phép tính nào) là trường hợp hợp lệ và phổ biến
nhất: `{"cells": [42], "program": "[NUM_0]", "scale": "none"}`. Guard §4.4 cho
qua vì `ast.Expression` có body là một `ast.Name` hợp lệ. Không có "operation
lookup" đặc biệt nào — nó chỉ là biểu thức ngắn nhất.

**File quyết định.** Ghi JSONL, một dòng một câu, khoá theo `question_id`, tại
đường dẫn truyền qua cờ mới `--program-decisions`. Cờ này **thay** cờ hiện có
`--row-choice-decisions` ở [submission/cli.py:125](src/financial_report_qa/submission/cli.py:125);
`question_plan.load_decisions` xoá theo §5.4. File chỉ chứa chỉ số và chuỗi
biểu thức, không chứa giá trị số nào (N7).

### 4.4 Program AST guard

Parse bằng `ast.parse(program, mode="eval")`. Deny-by-default, chỉ cho:

| Cho phép | Không cho phép |
|---|---|
| `ast.Expression`, `ast.BinOp` với `Add/Sub/Mult/Div` | mọi `ast.Constant` kiểu `int`/`float`/`complex` |
| `ast.UnaryOp` với `USub` | `Attribute`, `Subscript`, `Call` ngoài `abs` |
| `ast.Call` với `func.id == "abs"`, đúng 1 tham số | `Name` không khớp `^NUM_\d+$` |
| `ast.Name` dạng `NUM_<i>` với `0 <= i < len(cells)` | comprehension, lambda, so sánh, boolean op |

Placeholder `[NUM_0]` được thay thành định danh hợp lệ `NUM_0` trước khi parse
(dấu ngoặc vuông không parse được trong Python). Vi phạm bất kỳ →
`ProgramGuardError`, tính là một lần lệch, kích hoạt sinh lại.

Cấm `ast.Constant` số là cách thi hành N4′ bằng máy. `round()` bị cấm vì tham
số thứ hai của nó là literal số; làm tròn là việc của trình bày
(`verification/templates.py`), không phải của chương trình.

### 4.5 Binding tất định — render hai lần

Đây là điểm bắt buộc, phát hiện khi đối chiếu với `submission/compliance.py`:

- **C5** đòi `pandas_query` phải tham chiếu ít nhất một cột của CSV.
- **C7** đòi `pandas_query` replay được ra đúng `answer`.

Biểu thức `([NUM_0] - [NUM_1]) / [NUM_1]` không thoả cả hai. Nên mỗi `[NUM_i]`
được render **hai cách khác nhau từ cùng một cây AST**:

1. **Ra float** — `values[i]`, đọc từ `cells.parquet` qua `cell_frame.py`.
   Dùng để tính `answer`.
2. **Ra biểu thức tra cứu CSV** — `_cell_expr()` ở
   [pandas_query.py:111](src/financial_report_qa/execution/pandas_query.py:111)
   sinh sẵn dạng
   `df1[(df1.row_label_canonical == "...") & (df1.period == 2023)]["value"].iloc[0]`.
   Ghép vào cùng khung biểu thức → `pandas_query` của bài nộp.

`scale` được áp ở **cả hai** đường: nhân vào float, và nối `* 100` vào chuỗi
pandas. `* 100` là do renderer của ta viết, không phải LLM — N4′ không bị vi
phạm.

Hai đường phải cho cùng kết quả. C7 đã kiểm điều đó sẵn, nên đây là lớp kiểm
tra thứ ba gần như miễn phí.

### 4.6 Sandbox số học

Module mới `execution/masked_program.py`. Interpreter đệ quy trên cây AST đã
qua guard, chỉ biết bốn phép tính, `abs`, và tra `values[i]`. Không
`eval`/`exec`, không pandas, không I/O — cùng tư thế deny-by-default như
`pandas_query.py` hiện nay.

Lỗi được phân loại rõ, không nuốt: `ZeroDivisionError` → `division_by_zero`;
kết quả không hữu hạn → `non_finite_result`; cả hai là lệch, kích hoạt sinh
lại.

Ước lượng ~120 dòng kể cả guard. Nhỏ hơn `pandas_query.py` (355 dòng) một
bậc, vì ngôn ngữ nó phải chạy nhỏ hơn hẳn.

### 4.7 Dual Verification

Hai lớp bắt hai lỗi khác nhau, không thay thế nhau.

**(a) Verify B — `uses` đối chiếu ô đã bind.**

Với mỗi `i`, so `uses[i].row`/`uses[i].col` với `row_path`/`col_path` thật của
`cells[i]`. Quy tắc so, theo đúng thứ tự, dừng ở bước đầu tiên khớp:

1. Bằng nhau sau khi chuẩn hoá bằng `normalization` (bỏ dấu thừa, gộp khoảng
   trắng, lowercase).
2. Bằng nhau sau khi quy về canonical qua `metric_aliases.py` — đây là chỗ
   "doanh thu thuần" khớp với "doanh thu bán hàng và cung cấp dịch vụ".
3. `uses[i].row` là **hậu tố** của `row_path` sau chuẩn hoá — LLM thường chỉ
   thuật lại nhãn con mà bỏ tiền tố nhóm.

Không khớp ở cả ba bước → `use_binding_mismatch`. **Không dùng fuzzy match có
ngưỡng**: ngưỡng là một tham số phải chỉnh tay, và sai ở đây là sai âm thầm.
Ba luật trên đều tất định và giải thích được.

So `col`: chỉ so `period` đã parse từ `uses[i].col` với `period` của ô, không
so toàn chuỗi — `col_path` thật (`Tổng_cộng_31/12/2022`) và cách LLM thuật lại
(`Năm 2022`) khác nhau về hình thức nhưng cùng một kỳ.

Cái này bắt **trượt chỉ số**: LLM mô tả đúng ô nó muốn nhưng ghi sai số thứ tự
trong danh sách đánh số — lỗi kinh điển của prompt dạng numbered list, và là
lỗi mà **không lớp nào khác bắt được**, vì đáp án vẫn là một con số hợp lệ đọc
từ bảng.

Nó **không** bắt được trường hợp LLM thật sự tin rằng dòng sai là dòng đúng.
Nói thẳng giới hạn này để không tự huyễn hoặc về độ phủ.

**(b) Verify A — giải thích lượt hai, unmasked.**

Sau khi đã bind và chạy ra số thật, gọi LLM lượt hai với câu hỏi + các ô đã
bind kèm **giá trị thật** + đáp án đã tính, yêu cầu viết một câu giải thích.
Quét mọi token số trong giải thích; số nào không nằm trong whitelist `{đáp án,
giá trị các ô đã bind, các kỳ}` → lệch.

[numeric_guard.py:1](src/financial_report_qa/verification/numeric_guard.py:1)
đã cài đúng nguyên văn logic này cho đường paraphrase cũ. Dùng lại, không viết
lại.

Masking vẫn nguyên vẹn: lượt sinh **chương trình** vẫn mù số; chỉ lượt kiểm
mới thấy số, và lượt kiểm **không có quyền sửa đáp án** — nó chỉ có quyền báo
lệch.

**(c) Sinh lại.** Bất kỳ lệch nào ở §4.4, §4.6, §4.7a hoặc §4.7b → sinh lại
bước §4.3 đúng **1 lần**, cùng prompt, nhiệt độ khác. Vẫn lệch →
`low_confidence = true` trên `QuestionOutcome`, đáp án vẫn được nộp (bỏ trống
chắc chắn 0 điểm; sai thì cũng chỉ 0 điểm). Không có lần thứ ba.

## 5. Kiểm kê — dùng lại, sửa, xoá

### 5.1 Dùng lại nguyên vẹn

| File | Dòng | Vai trò trong pipeline mới |
|---|---|---|
| toàn bộ `retrieval/` | — | Nhánh 1 không đổi; plan `2026-08-23-retrieval-rerank-pipeline.md` vẫn hiệu lực |
| `retrieval/row_fusion.py` | — | sinh dòng ứng viên cho §4.2 |
| `planning/entity_parser.py` | 653 | tách công ty/kỳ; nguồn của `period` trong §4.2 |
| `planning/entity_contracts.py` | 124 | `QueryEntities` |
| `planning/llm_client.py` | 141 | client batch, dùng cho cả hai lượt LLM |
| `planning/evidence_rendering.py` | 172 | row fusion → đầu vào planner |
| `execution/cell_frame.py` | 231 | dựng long-format frame; nguồn `values[i]` và của CSV evidence |
| `execution/scope_filter.py` | 69 | lọc riêng/hợp nhất khi dựng ứng viên |
| `verification/numeric_guard.py` | 70 | **là** §4.7b |
| `verification/templates.py` | 139 | render `display`, làm tròn, đổi thang trình bày |
| `verification/fact_checks.py` | 100 | đối chiếu từng ô ngược về parquet |
| `submission/compliance.py` | 226 | C1–C7 giữ nguyên, thêm C8 |
| `submission/citation_summary.py` | 75 | `relevant_docs`/`relevant_tables` theo đúng thứ tự rank |
| `submission/backstop_answer.py` | 313 | backstop là tầng dự phòng **duy nhất** được phép (N6) |
| `submission/validator.py` | 207 | validate bài nộp |

### 5.2 Sửa

| File | Sửa gì |
|---|---|
| `execution/pandas_query.py` | **Giữ** `_cell_expr`, `_lit`, `_metric_column_and_value`, `replay_pandas_query` (C7 cần). **Bỏ** `render_pandas_query(plan)` và các nhánh theo `operation` — không còn `FinancialQueryPlan` để render. |
| `planning/row_choice_batch.py` | Mở rộng từ ứng viên **dòng** sang ứng viên **ô** (§4.2); payload thêm `col_path`, `period`, `unit`. Docstring "không bao giờ thấy giá trị số" giữ nguyên — nó vốn đã đúng. |
| `planning/fact_grounding.py` | Giữ phần nhãn → vị trí; bỏ phần phục vụ `MetricSelector` của compiler. |
| `verification/checks.py` | `check_recompute_mismatch`, `check_unit_not_presentable`, `check_display_roundtrip_mismatch`, `check_evidence_outside_retrieval` đổi từ nhận `CompiledQuery` sang nhận `ExecutedProgram`. `check_period_inferred_warning` và `check_scope_inferred` phụ thuộc `FinancialQueryPlan` → xoá cùng plan. |
| `verification/builder.py` | `build_answer_package` nhận `ExecutedProgram` thay `CompiledQuery`. |
| `verification/contracts.py` | `AnswerPackage` thêm `low_confidence: bool`, `program: str`, `regenerated: bool`. |
| `submission/exporter.py` | Thay chuỗi `build_plan → compile_plan → replay` bằng `build_cell_candidates → llm_batch → guard → bind → eval → verify`. Đây là file thay đổi nhiều nhất. |
| `submission/contracts.py` | `QuestionOutcome` thêm `low_confidence`, `regenerated`. |
| `submission/compliance.py` | Thêm **C8 — no numeric literal**: parse `program` đã lưu, fail nếu có `ast.Constant` số. Đây là thi hành N4′ ở chốt chặn cuối. |

### 5.3 Xoá — ngay

| File | Dòng | Vì sao |
|---|---|---|
| `planning/table_context_rendering.py` | 37 | Docstring tự khai "Day 23 last-resort tier… grounded LLM fallback" — đúng định nghĩa tầng dự phòng mà N6 cấm. §8.1 của spec cũ đã liệt kê vào diện xoá nhưng chưa xoá. |

### 5.4 Xoá — sau khi đo được PAL ≥ compiler trên gold

Không xoá trước. `compiler.py` hiện là **baseline duy nhất đang có**; xoá
trước khi có số so sánh là vứt mất thước đo. Cổng: chạy cả hai trên
`data/qa/answer-gold-v1.jsonl`, PAL phải ≥ compiler về Answer Accuracy.

| File | Dòng | Vì sao xoá |
|---|---|---|
| `execution/compiler.py` | 501 | `FinancialQueryPlan → CompiledQuery`; vai trò của nó là §4.4–§4.6 mới |
| `execution/operations.py` | 87 | một hàm cho mỗi `PlanOperation`; enum operation không còn tồn tại |
| `execution/locator.py` | 263 | `(MetricSelector, period) → CellMatch`; binding giờ theo chỉ số tường minh, không cần dò |
| `execution/tiebreak.py` | 73 | phá thế nhập nhằng của `locate()`; không còn `locate()` |
| `execution/contracts.py` | 128 | `CellMatch`, `CompiledQuery` — thay bằng contract mới |
| `planning/plan_contracts.py` | 172 | `FinancialQueryPlan` |
| `planning/plan_validator.py` | 347 | bảng arity cho mỗi operation |
| `planning/question_plan.py` | 286 | quyết định LLM → `FinancialQueryPlan` |
| `planning/cell_grounding.py` | 153 | orchestrator "quyết định → plan → compile" của đường cũ |

Tổng: khoảng **2.010 dòng** bị xoá, đổi lấy khoảng **400 dòng** mới
(`masked_program.py` ~120, contract mới ~80, batch payload mở rộng ~60,
verify B ~80, orchestration ~60). Test đi kèm các file bị xoá cũng xoá theo.

### 5.5 Contract mới

`execution/program_contracts.py`, pydantic v2 frozen, `extra="forbid"`:

- `CellCandidate` — một ô ứng viên §4.2, **không có trường value**
- `ProgramDecision` — `cells`, `program`, `uses`, `scale` (§4.3)
- `BoundValue` — `num_index`, `candidate_index`, `table_id`, `row_path`,
  `col_path`, `period`, `value: Decimal`, `unit`
- `ExecutedProgram` — `question_id`, `program`, `bindings`, `answer: float`,
  `pandas_query: str`, `csv_path`, `scale`, `regenerated: bool`,
  `low_confidence: bool`

## 6. Xử lý lỗi

Mọi lỗi đều **fail rõ tại chỗ**, không có nhánh im lặng:

| Tình huống | Mã | Hành vi |
|---|---|---|
| JSON của LLM không parse được | `decision_unparseable` | sinh lại 1 lần |
| `cells[i]` ngoài phạm vi danh sách | `candidate_index_out_of_range` | sinh lại 1 lần |
| `program` có literal số | `numeric_literal_in_program` | sinh lại 1 lần |
| `program` dùng node ngoài whitelist | `program_node_not_allowed` | sinh lại 1 lần |
| chia cho 0 | `division_by_zero` | sinh lại 1 lần |
| kết quả không hữu hạn | `non_finite_result` | sinh lại 1 lần |
| `uses` lệch ô đã bind | `use_binding_mismatch` | sinh lại 1 lần |
| giải thích chứa số ngoài whitelist | `explanation_number_not_grounded` | sinh lại 1 lần |
| sinh lại xong vẫn lệch | — | `low_confidence = true`, vẫn nộp |
| không có ô ứng viên nào | `no_cell_candidates` | backstop |

## 7. Test

Theo TDD, mỗi bước có test trước khi có code.

- **`masked_program.py`** — test thuần, không LLM: guard nhận biểu thức hợp
  lệ; guard từ chối literal số ở mọi vị trí (kể cả trong `abs()`, kể cả số âm,
  kể cả `1e3`); từ chối `Attribute`/`Subscript`/`Call` lạ/lambda; eval ra đúng
  số; chia 0 báo đúng mã.
- **Binding hai đường** — cùng một AST, float và chuỗi pandas cho cùng kết quả
  trên fixture nhỏ; chuỗi pandas replay được qua `replay_pandas_query`.
- **Verify B** — ô đã bind khớp `uses` → pass; hoán vị `cells` mà giữ nguyên
  `uses` → phải bắt được (đây chính là kịch bản trượt chỉ số).
- **Verify A** — giải thích chỉ chứa số trong whitelist → pass; chèn một số
  bịa → fail. Dùng fake LLM, không gọi mạng.
- **Sinh lại** — fake LLM sai lần một, đúng lần hai → kết quả đúng,
  `regenerated = true`. Sai cả hai lần → `low_confidence = true` và **vẫn có
  đáp án**.
- **C8** — bundle có program chứa literal số → linter fail build.
- **Hồi quy** — `tests/unit/submission/`, `tests/integration/` phải xanh;
  Nhánh 1 không được đổi hành vi (bất biến N1).

Không test nào được tải model thật hay gọi mạng.

## 8. Tiêu chí thành công

1. **Answer Accuracy trên `data/qa/answer-gold-v1.jsonl` ≥ compiler hiện tại.**
   Đây là cổng để xoá §5.4. Không đạt thì không xoá, và phải phân tích trước
   khi đi tiếp.
2. **Tỷ lệ câu không trả lời được giảm dưới 41%** (§2.2 spec cũ). Đây là lý do
   tồn tại của cả spec này; không cải thiện được số này thì thay đổi là vô
   nghĩa.
3. **C8 xanh trên toàn bộ 1012 câu** — không một program nào chứa literal số.
4. **C1–C7 vẫn xanh** — không đánh đổi compliance lấy accuracy.
5. Nhánh 1 giữ nguyên số đo: bất biến N1 có test ghim.

## 9. Rủi ro

- **Trượt chỉ số là lỗi chính, và Verify B chỉ bắt được một nửa.** Nếu LLM vừa
  mô tả sai vừa chọn sai một cách nhất quán, không lớp nào bắt được. Chấp nhận
  có ý thức: đây là giới hạn của việc bỏ dual-verification kiểu đối chiếu
  compiler (phương án đã cân nhắc và loại ở giai đoạn thiết kế).
- **`scale` là enum do LLM chọn** — chọn sai `percent` vs `none` làm đáp án
  lệch 100 lần. `check_unit_not_presentable` bắt được một phần qua bảng tương
  đương đơn vị, không bắt hết.
- **`* 100` do renderer nối vào có thể chạm C4** nếu đáp án tình cờ bằng 100.
  Cần test ghim riêng cho trường hợp này.
- **Khối lượng xoá lớn (~2.010 dòng).** Xoá theo hai đợt (§5.3 ngay, §5.4 sau
  khi đo) chính là để giảm rủi ro này.
- **Thời gian.** Hạn nộp 31/08/2026. Nhánh 1 chiếm 50% điểm và đang dở ở Task
  5–8 của `2026-08-23-retrieval-rerank-pipeline.md`, với đường đi đã đo được
  từ Recall@10 47.41% lên 80.19%. Spec này là 50% còn lại nhưng khối lượng lớn
  hơn nhiều. Thứ tự ở §10 phản ánh điều đó.

## 10. Thứ tự thực hiện

1. **Xong Nhánh 1 trước** — Task 5–8 của plan retrieval. Đây là phần có số đo
   sẵn, chi phí thấp, chiếm 50% điểm.
2. `execution/masked_program.py` + contract mới (§4.4, §4.6, §5.5) — thuần
   tuý, test được không cần LLM, không đụng gì đang chạy.
3. Binding hai đường (§4.5) — tái dùng `_cell_expr`.
4. Mở rộng `row_choice_batch.py` sang ô (§4.2, §4.3).
5. Verify B (§4.7a), rồi Verify A (§4.7b) — A gần như chỉ là nối
   `numeric_guard`.
6. Vòng sinh lại (§4.7c).
7. Nối vào `exporter.py`, thêm C8, **đo trên gold**.
8. Xoá §5.4 nếu và chỉ nếu tiêu chí §8.1 đạt.

Bước 2–6 độc lập với bước 1 về mặt file, nên chạy song song được nếu có người
làm. Bước 7 phụ thuộc cả hai.

## 11. Ngoài phạm vi

Giữ nguyên §11 của spec cũ, cộng thêm:

- **Binder** (LLM sinh pandas trên bảng đã mask) — đã cân nhắc và loại ở giai
  đoạn thiết kế: sandbox phải chạy pandas tuỳ ý do LLM viết, và bất biến
  "không literal số" không còn kiểm được đơn giản bằng AST.
- **Self-consistency** (sinh N chương trình, bỏ phiếu theo giá trị) — đã cân
  nhắc và loại: tốn N lần chi phí LLM trên 1012 câu.
- **LLM critic** (batch thứ hai phán đúng/sai) — §11 spec cũ đã loại
  multi-agent/reflection.
- **Graph/Tree table representation** — xem §4.1.
- **Reranker cho row retrieval** — vẫn ngoài phạm vi.
- **Fine-tune bất kỳ model nào.**
