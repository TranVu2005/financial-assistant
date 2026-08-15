# Kế hoạch Ngày 19 — Sandbox executor

> **Trạng thái:** đã đo, **đã thực hiện xong** (2026-08-15). ADR:
> [0008](../decisions/0008-execution-sandbox-contract.md).
> **Ngày đo:** 2026-08-15. **Release khoá:** `data/processed/release_v2_422df141c935`,
> `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
> **Mọi con số trong tài liệu này là kết quả truy vấn/chạy thật trên release đã khoá và trên mã
> Ngày 18 vừa commit (`4695aaa`)**, không phải ước lượng.
>
> **Kết quả sau khi cài đặt xong:** nhãn corpus thật có `"` compile ra `answered` đúng (trước đây
> `SyntaxError`); CLI `compile-plans` chạy lại trên gold70 vẫn đúng **30/51 (58,8 %)**, phân rã lỗi
> không đổi; 8/8 test bảo mật xanh; 1.020 test qua, `ruff`/`mypy` sạch. Một hiệu chỉnh phát hiện giữa
> chừng bởi TDD: `enable_external_access=false` (dự định ban đầu ở Quyết định F) phá luôn
> `read_parquet` cục bộ — sửa thành chỉ tắt autoinstall/autoload extension. Chi tiết đầy đủ trong
> [ADR 0008](../decisions/0008-execution-sandbox-contract.md) và blockquote kết quả trong
> [plan.md](../../plan.md).

Mục Ngày 19 trong [plan.md](../../plan.md) yêu cầu bốn việc: không `eval`/`exec` trên chuỗi LLM;
giới hạn dòng/thời gian/bộ nhớ và operation whitelist; cấm filesystem ngoài data path, network,
subprocess và import tuỳ ý; security tests với prompt injection và plan độc hại.

**Ba trong bốn việc đó Ngày 18 đã làm một nửa, và phép đo dưới đây cho thấy nửa còn lại đang hỏng
theo cách không ai đoán trước được:** `pandas_query` được sinh bằng f-string nội suy thẳng chuỗi tự
do vào literal Python. **1.988 nhãn dòng có thật trong corpus** (trên 1.790 bảng, 9.944 ô số) chứa
dấu `"`. Đưa một nhãn như thế vào `MetricSelector.raw_text` — **đúng thứ ADR 0004 phương án C bảo
planner phải làm** — khiến `compile_plan` **ném `SyntaxError` không ai bắt**. Đây không phải lỗ hổng
giả định cần kẻ tấn công: đây là **crash trên dữ liệu hợp lệ**, tái hiện 3/3 lần ở § 1.1.

Nói cách khác Ngày 19 không phải là "bọc thêm một lớp bảo mật quanh thứ đang chạy tốt". Nó là
**đóng đúng cái biên tin cậy mà Ngày 18 để hở**, và cái biên đó đang rò rỉ ra cả lỗi correctness lẫn
lỗi bảo mật từ cùng một nguyên nhân gốc.

---

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Compiler (Ngày 18) | `execution/compiler.py` — `compile_plan` | ✅ 30/51 gold70, đã commit `4695aaa` |
| Renderer + replayer | `execution/pandas_query.py` | ⚠️ **có lỗ hổng nội suy — xem § 1.1** |
| Whitelist AST interpreter | `pandas_query._eval_node` | ✅ deny-by-default đúng nguyên tắc, ⚠️ thiếu ngân sách — § 1.8 |
| Hình chiếu ô số | `execution/cell_frame.py` — DuckDB tham số hoá `?` | ✅ **không có SQL injection** (§ 1.7) |
| Error code thực thi | `execution/contracts.py` — `ExecutionIssueCode`, 6 mã | ✅ mẫu để mở rộng |
| Semantic validator | `planning/plan_validator.py` — 10 mã, chặn `top_k` | ⚠️ **`compile_plan` không gọi** (§ 1.10) |
| `ExecutionSettings` | `core/config.py` | ⚠️ **`timeout_seconds` + `max_rows` là code chết** (§ 1.9) |
| Gold70 + plan-cases | `data/qa/retrieval-gold-v1.jsonl`, `plan-cases-v1.jsonl` | ✅ dùng làm regression baseline |
| Bộ case độc hại | — | ❌ **chưa tồn tại**, nhiệm vụ 19.8 |
| Dependency mới | — | ✅ **không cần thêm gì**: `ast`, `time`, `duckdb` đã có |

---

## 1. Chốt chặn phải đo trước khi viết code

### 1.1. `pandas_query` vỡ trên **dữ liệu thật**, không cần kẻ tấn công

`render_pandas_query` sinh điều kiện bằng f-string: `f'(df1.{column} == "{value}")'`
([pandas_query.py:51](../../src/financial_report_qa/execution/pandas_query.py#L51)). `value` là
`MetricSelector.raw_text` hoặc `companies[i]` — kiểu `NonEmptyString`, tức **chỉ strip khoảng trắng,
không ràng buộc ký tự nào cả**.

Đếm trên `cells.parquet`:

| Đối tượng | Số lượng |
| --- | --- |
| Ô có `row_label_raw` | 5.353.511 |
| Ô có `row_label_raw` chứa `"` | 23.900 |
| **Nhãn phân biệt** chứa `"` | **2.270** / 165.416 (1,37 %) |
| **Nhãn phân biệt chứa `"` nằm trên ô số** (`col_idx > 0`, `value_numeric NOT NULL`) | **1.988** |
| Bảng chứa nhãn đó | **1.790** |
| Ô số bị ảnh hưởng | **9.944** |

Đây là nhãn tiếng Việt hoàn toàn bình thường — dấu ngoặc kép là quy ước viết tắt trong báo cáo tài
chính:

```
Tiền gửi tại và cho vay các tổ chức tín dụng ("TCTD") khác
Khấu hao tài sản cố định ("TSCĐ")
Ngân hàng Thương mại Cổ phần Đông Á ("DAB")
```

Chạy thật `compile_plan` với ba nhãn đầu tiên lấy trực tiếp từ corpus (công ty MBB/PNJ, kỳ có thật):

```
label='Tiền gửi và cho vay các tổ chức tín dụng ("TCTD") khác' company=MBB period=2016
rendered: df1[(df1.company_code == "MBB") & (df1.row_label_raw == "Tiền gửi ... ("TCTD") khác") & ...
>>> compile_plan RAISED UNCAUGHT SyntaxError: invalid syntax. Perhaps you forgot a comma?
```

**3/3 crash.** `SyntaxError` không nằm trong tuple `except` của `compile_plan`
([compiler.py:99-102](../../src/financial_report_qa/execution/compiler.py#L99) chỉ bắt `ValueError`
và `ZeroDivisionError`), và lời gọi `replay_pandas_query` lại nằm **ngoài** khối `try`
([compiler.py:104](../../src/financial_report_qa/execution/compiler.py#L104)) nên kể cả `ValueError`
cũng thoát ra.

Hệ quả cho thiết kế: **chặn ký tự `"` là phương án chết**, vì nó loại bỏ 1.988 nhãn hợp lệ. Phải
**escape đúng** ở renderer. Xem quyết định A.

### 1.2. Census ký tự và độ dài: biên an toàn đo được, không đoán

`row_label_raw` (5.353.511 ô):

| Ký tự | Số ô chứa | Ghi chú |
| --- | --- | --- |
| `"` | 23.900 | phải escape, không được chặn |
| `'` | 8.355 | vô hại với literal `"` |
| `\` | 8.139 | **escape sequence — nguy hiểm ngang `"`** |
| `(` `)` | 526.841 | vô hại trong literal đã escape |
| `=` | 44.062 | |
| `&` | 10.707 | |
| `[` `]` | 4.298 | |
| **LF (`\n`)** | **1** | |
| **CR (`\r`)** | **0** | |
| **TAB** | **0** | |

Độ dài `row_label_raw`: p50 = **24**, p99 = **97**, p99,9 = **328**, **max = 33.011**.

| Ngưỡng | Số ô vượt | Số **ô số** vượt |
| --- | --- | --- |
| > 300 | 6.207 (0,12 %) | — |
| > 512 | 1.057 (0,02 %) | **433** |
| > 1.000 | 75 | — |
| > 2.000 | 2 | — |

→ **Cấm ký tự điều khiển** loại đúng **1** ô thật trên 5,35 triệu. **Cắt ở 512 ký tự** loại **433 ô
số** (0,008 % corpus ô số). Cả hai đều là biên có giá được đo, không phải con số tròn cho đẹp.

`company_code` (`documents.parquet`): **1.971/1.971 tài liệu** khớp `^[A-Z0-9]+$`, độ dài tối đa
**3**, **100 mã phân biệt**. → với `companies`, allowlist ký tự nghiêm ngặt là khả thi **100 %**,
khác hẳn `raw_text`.

### 1.3. Năm loại exception thoát ra khỏi replayer, không phải một

Docstring `replay_pandas_query` viết "deny by default… raises ValueError". Đo thật với payload độc hại:

| Payload | Exception thoát ra |
| --- | --- |
| `Tiền mặt") \| (df1.period == 1900) & (df1.value == "` | **`KeyError`** |
| `x") ; __import__("os").system("calc") #` | **`SyntaxError`** |
| `x" ) ]. __class__ . __mro__ [1] . __subclasses__ ( ) [0]` | `ValueError` ✅ (whitelist chặn đúng) |
| `"A" * 50_000` | `KeyError`, chuỗi render dài **50.100** ký tự |
| biểu thức lồng sâu 1.000 mức | **`RecursionError`** |
| `df1.period.isin([...200.000 literal])` | **`decimal.InvalidOperation`** |

**Whitelist ngữ nghĩa hoạt động đúng** (`__subclasses__` bị chặn bằng `ValueError`) — không có
đường thoát ra sandbox ngữ nghĩa. Nhưng **hợp đồng exception thì sai**: 5 loại thoát ra, chỉ 1 loại
được compiler xử lý, và điểm gọi lại nằm ngoài `try`. Đây là thứ Ngày 19 phải sửa, không phải
interpreter.

### 1.4. Hỏng im lặng còn tệ hơn crash: nội suy đổi độ ưu tiên toán tử

Plan hợp lệ với `companies = ('ACB") | (df1.company_code == "VCB',)` vượt qua **cả**
`FinancialQueryPlan` **và** `validate_plan_semantics`, rồi render ra:

```python
df1[(df1.company_code == "ACB") | (df1.company_code == "VCB") & (df1.row_label_canonical == "cash_and_cash_equivalents")]
```

Chuỗi này **đúng cú pháp Python**, chạy lọt qua interpreter, và **nới rộng bộ lọc một cách âm thầm**
(`|` có độ ưu tiên thấp hơn `&`). Không có exception nào để bắt. Chỉ có escape đúng ở tầng renderer
mới chặn được lớp lỗi này — kiểm tra exception thì không.

### 1.5. Ngân sách tài nguyên: cấu hình hiện tại gấp **25 lần** trường hợp xấu nhất toàn corpus

Chạy thật 51 plan gold70 (rule planner → `build_cell_frame` → `compile_plan`):

| Chỉ số | min | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| Số dòng frame | 4 | 54 | 140 | **146** |
| `build_cell_frame` (giây) | 0,176 | 0,430 | 0,609 | 0,745 |
| `compile_plan` tổng (giây) | 0,359 | 0,823 | **1,216** | **1,424** |

Trường hợp xấu nhất **toàn corpus** (130.518 bảng có ô số): 1 bảng lớn nhất = **1.130** ô số,
p99 = 88, p50 = 13, **tổng 12 bảng lớn nhất = 4.002 ô**. Vì `candidate_table_ids` bị schema chặn ở
tối đa 12, **4.002 dòng là trần tuyệt đối** của mọi frame hợp lệ.

Cấu hình đang ghi `max_rows: 100000` — **gấp 25 lần trần tuyệt đối**. Đó không phải giới hạn, đó là
trang trí. `timeout_seconds: 5` thì hiệu chỉnh hợp lý (3,5× so với max đo được 1,424 s) — nhưng
xem § 1.9.

### 1.6. Trên Windows **không tồn tại** cơ chế timeout/memory preemptive in-process

Đo trực tiếp trên môi trường đích (`win32`, CPython 3.11.15):

```
SIGALRM:   False
setitimer: False
resource module: ABSENT -> No module named 'resource'
```

`signal.SIGALRM`, `signal.setitimer` và module `resource` (`RLIMIT_AS`) **đều không có**. Mọi công
thức "đặt alarm rồi ngắt" hay "setrlimit bộ nhớ" trong sách vở đều không áp dụng được ở đây. Đây là
ràng buộc nền tảng, phải thiết kế vòng qua nó chứ không giả vờ giải quyết. Xem quyết định D.

### 1.7. DuckDB mặc định mở toang; SQL thì **không** có lỗ

Kiểm tra connection mà `build_cell_frame` đang tạo (`duckdb.connect(":memory:")`, không cấu hình gì):

```
enable_external_access       = True
autoinstall_known_extensions = True
autoload_known_extensions    = True
memory_limit                 = 12.5 GiB
threads                      = 12
```

Engine đọc release **có quyền truy cập mạng và tự cài extension**. Mặt tích cực: `_QUERY` dùng
placeholder `?` cho cả ba đường parquet lẫn danh sách `table_id`, và `TableId` bị ràng buộc
`^tbl_[0-9a-f]{64}$` → **không có đường SQL injection nào từ plan**. Nên đây là hardening chiều sâu,
không phải vá lỗ đang chảy máu. Đã kiểm chứng là hardening có tác dụng thật:

```
SET enable_external_access=false; autoinstall_known_extensions=false; autoload_known_extensions=false
→ SELECT * FROM read_csv_auto('https://example.com/x.csv')
→ PermissionException: Cannot access file ... - file system operations
```

### 1.8. Interpreter không có ngân sách cấu trúc

| Phép thử | Kết quả |
| --- | --- |
| BinOp lồng 200 mức | OK, 0,0005 s |
| BinOp lồng **1.000** mức | **`RecursionError`** (`sys.getrecursionlimit() = 1000`) |
| `isin` với **200.000** literal | parse + chạy hết **0,63 s**, chuỗi nguồn ~1,4 MB |

Không có giới hạn độ dài chuỗi, số node AST, hay độ sâu đệ quy. `_eval_node` đệ quy thẳng nên độ sâu
biểu thức ánh xạ 1:1 vào ngăn xếp Python.

### 1.9. `timeout_seconds` và `max_rows` là code chết — **lần thứ ba**

`grep` toàn `src/`: `allow_operations` được đọc đúng **một** chỗ
([compiler.py:84](../../src/financial_report_qa/execution/compiler.py#L84)). `timeout_seconds` và
`max_rows` **không có tham chiếu nào ngoài định nghĩa trong `core/config.py`**.

Đây là lần thứ ba đúng một mô hình lặp lại: khối `llm:` là code chết trước Ngày 17, khối `execution:`
là code chết trước Ngày 18, và giờ **hai phần ba khối `execution:` vẫn là code chết sau Ngày 18**.
DoD Ngày 19 phải có một mục kiểm tra máy móc cho việc này, không dựa vào mắt người.

### 1.10. Biên tin cậy không có người gác, nhưng kênh vào hẹp hơn tưởng

- `compile_plan` **không gọi** `validate_plan_semantics`. `llm_planner` có gọi
  ([llm_planner.py:61](../../src/financial_report_qa/planning/llm_planner.py#L61)), nhưng đó là
  thiện chí của caller chứ không phải hợp đồng được ép buộc ở biên.
- `top_k` **đã** được validator chặn (`1 <= top_k < len(companies)`,
  [plan_validator.py:299](../../src/financial_report_qa/planning/plan_validator.py#L299)) — nhưng
  chỉ khi có ai đó gọi validator.
- Prompt injection **quy về plan injection**: `build_user_prompt` chỉ nhận `question`, và nhãn dòng
  của bảng ứng viên **đã bị loại khỏi prompt có chủ đích** (ADR 0006 quyết định B1). Output LLM bị
  ràng JSON schema + `LLMPlanOutput` + validator. Sau tất cả các lớp đó, **đúng hai trường tự do**
  sống sót tới executor: **`companies` và `MetricSelector.raw_text`**. Bộ security test phải nhắm
  vào đúng hai trường đó, không phải bắn đại vào prompt.
- Audit import toàn `execution/`: **không có** `os`, `subprocess`, `socket`, `pickle`, `httpx`,
  `requests`. Phụ thuộc ngoài duy nhất có khả năng chạm hệ thống là `duckdb` (§ 1.7).

---

## 2. Quyết định thiết kế

### A. Chuỗi tự do trong `pandas_query` — **A2: escape đúng ở renderer + chặn ở contract**

Không chọn A1 ("cấm ký tự lạ") vì § 1.1 đo được nó phá **1.988 nhãn thật**. Làm **cả hai lớp**:

1. **Renderer escape đúng**: literal chuỗi sinh bằng `json.dumps(value, ensure_ascii=False)` thay vì
   `f'"{value}"'`. JSON string literal là tập con hợp lệ của Python string literal cho mọi ký tự
   không điều khiển, nên chuỗi render vẫn đọc được và vẫn parse được bằng `ast`. Việc này một mình
   đã đóng cả § 1.1 (crash) lẫn § 1.4 (hỏng im lặng).
2. **Contract chặn phần thừa**: `raw_text` và `companies` được ràng buộc **cấm ký tự điều khiển**
   (giá đo được: 1 ô/5,35 triệu) và **độ dài ≤ 512** (giá đo được: 433 ô số). `companies` thêm
   allowlist `^[A-Za-z0-9]{1,16}$` (giá đo được: **0** — cả 1.971 tài liệu đều khớp).

Lý do làm cả hai: escape sửa **correctness**, contract giới hạn **bề mặt tấn công**. Chỉ làm một
trong hai thì DoD "payload độc hại bị từ chối **có error code**" không đạt — escape làm payload trở
thành vô hại nhưng vẫn *chạy*, không *bị từ chối*.

### B. Sandbox ở đâu — **B2: module `execution/sandbox.py` mới, là cổng duy nhất tới replay**

Không nhét thêm vào `pandas_query.py` (B1). `pandas_query.py` giữ đúng hai việc thuần khiết: render
và diễn giải. `sandbox.py` sở hữu **hợp đồng giới hạn**: ngân sách cấu trúc, chuyển đổi exception,
đo thời gian, và trả về kiểu kết quả có mã lỗi. `compile_plan` **chỉ được** gọi replay qua sandbox.
Có test khẳng định điều đó (không module nào ngoài `sandbox.py` import `replay_pandas_query`).

### C. Hợp đồng exception — **C1: sandbox bắt `Exception` rộng, đổi thành mã lỗi có kiểu**

§ 1.3 đo được 5 loại exception thoát ra và danh sách đó **không đóng được bằng liệt kê** — thêm một
node AST mới là thêm một loại mới. Deny-by-default nghĩa là: **mọi** exception phát sinh khi replay
là **từ chối**, không phải sập.

Hai rào chắn để việc bắt rộng này không che lỗi lập trình (bài học từ `execution/cli.py` Ngày 18):

- `KeyboardInterrupt`/`SystemExit` không bị bắt (chúng là `BaseException`).
- Sandbox **ghi lại tên lớp exception gốc** vào `error_message`, và `ExecutionReplayMismatchError`
  **giữ nguyên là lỗi cứng** — *sai số* là bug phải làm vỡ build; *từ chối* là dữ liệu.

### D. Timeout và bộ nhớ — **D3: chặn theo cấu trúc trước khi chạy, đo thời gian sau khi chạy**

§ 1.6 chứng minh timeout preemptive in-process **không khả thi** trên nền tảng đích. Không giả vờ
giải quyết. Thay vào đó **chặn đầu vào để công việc bị chặn theo cấu trúc**:

| Ngân sách | Giá trị đề xuất | Căn cứ đo |
| --- | --- | --- |
| Độ dài chuỗi query | 4.096 ký tự | payload 50.100 ký tự ở § 1.3; query hợp lệ dài nhất trên gold70 ≪ 1.000 |
| Số node AST | 2.000 | `isin(200k)` = ~200.000 node (§ 1.8) |
| Độ sâu AST | 50 | `RecursionError` ở 1.000; biểu thức thật sâu < 15 |
| Số dòng frame | `max_rows`, **hạ 100.000 → 20.000** | trần tuyệt đối toàn corpus = **4.002** (§ 1.5), giữ 5× headroom |

Sau khi chạy, sandbox đo `time.perf_counter()` và **trả về mã lỗi `budget_exceeded` nếu vượt
`timeout_seconds`** — đây là *phát hiện hậu nghiệm*, không phải *ngắt*, và tài liệu phải nói thẳng
như vậy. Cách ly thật (subprocess/container) là việc của deployment, **ghi rõ ngoài phạm vi Ngày 19**.

Giữ `timeout_seconds: 5` (3,5× max đo được 1,424 s). Hạ `max_rows` xuống 20.000 trong
`configs/local_rtx3050.yaml`.

### E. Gác biên tin cậy — **E1: `compile_plan` tự validate, không tin caller**

`compile_plan` gọi `validate_plan_semantics` trước khi chạm dữ liệu, trả `CompiledQuery` lỗi mã
`plan_rejected` thay vì tin rằng caller đã validate. Chi phí: một lần gọi hàm thuần trên plan đã có
sẵn trong bộ nhớ (không I/O). Lợi: § 1.10 hết là lỗ hổng, và `top_k` được chặn ở mọi đường vào.

### F. DuckDB — **F1: hardening mọi connection, không ngoại lệ**

Mọi `duckdb.connect` trong `execution/` chạy ngay hai lệnh `SET`: `autoinstall_known_extensions` và
`autoload_known_extensions` = false. **Đã hiệu chỉnh khi thực hiện (19.7):** § 1.7 đo ban đầu bật cả
`enable_external_access=false` và thấy `read_csv_auto('https://…')` bị chặn — đúng, nhưng chưa kiểm
tra `enable_external_access=false` cũng chặn luôn `read_parquet` cục bộ mà `cell_frame._QUERY` phụ
thuộc. Test TDD viết cho 19.7 lộ ra điều đó ngay (8/8 test cell_frame đỏ với
`file system operations are disabled`) — chỉ tắt autoinstall/autoload là đủ để chặn mạng (thiếu
`httpfs` thì không đọc được `http(s)://`), không cần và không được tắt `enable_external_access`. Xem
chi tiết trong [ADR 0008 Quyết định F](../decisions/0008-execution-sandbox-contract.md).

### G. Mã lỗi mới — **G1: bốn mã, thêm vào `ExecutionIssueCode`**

| Mã | Nghĩa |
| --- | --- |
| `plan_rejected` | plan không qua charset/length guard hoặc `validate_plan_semantics` (E1, A2) |
| `query_rejected` | replay ném exception bất kỳ — payload không thực thi được (C1) |
| `budget_exceeded` | vượt ngân sách độ dài/node/độ sâu/thời gian (D3) |
| `row_limit_exceeded` | frame vượt `max_rows` (D3) |

Giữ nguyên 6 mã Ngày 18 → tổng **10 mã**. `operation_not_allowed` đã có sẵn, phủ đúng yêu cầu
"operation whitelist" của plan.md.

---

## 3. Nhiệm vụ

| # | Việc | Đầu ra |
| --- | --- | --- |
| 19.1 | Viết ADR 0008 chốt A2/B2/C1/D3/E1/F1/G1 | `docs/decisions/0008-execution-sandbox-contract.md` |
| 19.2 | Ràng buộc charset/length cho `companies` + `raw_text` (TDD) | `planning/plan_contracts.py` + test |
| 19.3 | **Sửa lỗi § 1.1**: renderer escape bằng `json.dumps` (TDD, test đỏ tái hiện `SyntaxError` từ nhãn corpus thật) | `execution/pandas_query.py` |
| 19.4 | Ngân sách cấu trúc trong interpreter: độ dài, số node, độ sâu (TDD) | `execution/pandas_query.py` |
| 19.5 | `execution/sandbox.py`: kiểu kết quả, bắt exception rộng, đo thời gian (TDD) | module mới + test |
| 19.6 | Bốn mã lỗi mới + `compile_plan` tự validate + gọi replay qua sandbox (TDD) | `contracts.py`, `compiler.py` |
| 19.7 | Ép `max_rows` trong `cell_frame` + hardening DuckDB (TDD) | `cell_frame.py`, `configs/local_rtx3050.yaml` |
| 19.8 | **Bộ case độc hại** + security tests | `data/qa/malicious-plan-cases-v1.jsonl`, `tests/security/` |
| 19.9 | Test kiến trúc: không nơi nào ngoài `sandbox.py` gọi `replay_pandas_query`; mọi trường `ExecutionSettings` đều có tham chiếu | `tests/unit/execution/` |
| 19.10 | Chạy lại `compile-plans` trên gold70, **khẳng định vẫn 30/51**, verification đầy đủ, cập nhật `plan.md`/README | báo cáo + tài liệu |

### Chi tiết 19.8 — bộ case độc hại phải nhắm đúng hai trường tự do

§ 1.10 đo được: chỉ `companies` và `raw_text` đi từ đầu vào không tin cậy tới executor. Bộ case gồm
**tám lớp**, mỗi lớp gắn với một phép đo cụ thể ở trên:

| Lớp | Payload mẫu | Mã lỗi kỳ vọng | Neo |
| --- | --- | --- | --- |
| Thoát literal | `x") \| (df1.value == "` | `plan_rejected` *(hoặc render an toàn)* | § 1.1 |
| Đổi độ ưu tiên | `ACB") \| (df1.company_code == "VCB` | `plan_rejected` | § 1.4 |
| Thực thi mã | `x") ; __import__("os").system("calc") #` | `plan_rejected` | § 1.3 |
| Leo thang thuộc tính | `__class__.__mro__[1].__subclasses__()` | `query_rejected` | § 1.3 |
| Bom độ dài | `"A" * 50_000` | `plan_rejected` | § 1.2 |
| Bom độ sâu | 1.000 mức lồng | `budget_exceeded` | § 1.8 |
| Ký tự điều khiển | `a"\nimport os\n"b` | `plan_rejected` | § 1.2 |
| Nhãn hợp lệ có `"` | `Khấu hao tài sản cố định ("TSCĐ")` | **`answered`** — *phải chạy được* | § 1.1 |

Lớp cuối cùng quan trọng ngang bảy lớp trên: nó là bằng chứng hardening **không** đánh đổi bằng việc
vứt bỏ 1.988 nhãn thật.

---

## 4. Thứ tự thực hiện

```
19.1 (ADR)
  └─> 19.2 (contract guard) ──┐
  └─> 19.3 (escape) ──────────┤
  └─> 19.4 (ngân sách AST) ───┴─> 19.5 (sandbox) ─> 19.6 (mã lỗi + compiler)
  └─> 19.7 (max_rows + duckdb) ──────────────────────────┘
                                                          └─> 19.8 (security tests)
                                                                └─> 19.9 (test kiến trúc)
                                                                      └─> 19.10 (verify + docs)
```

19.3 làm **trước** 19.2 về mặt ưu tiên logic (nó sửa lỗi đang sống), nhưng cả hai độc lập nhau nên
thứ tự thực thi không quan trọng. 19.7 độc lập hoàn toàn với nhánh chuỗi.

---

## 5. Definition of Done

| # | Điều kiện | Cách kiểm chứng |
| --- | --- | --- |
| D1 | Nhãn corpus thật chứa `"` compile được ra `answered` | test dùng nhãn lấy từ `cells.parquet`, không phải fixture bịa |
| D2 | Cả 7 payload độc hại trả **mã lỗi có kiểu**, không exception thoát ra | `tests/security/` |
| D3 | **Không có** `eval`/`exec`/`pd.eval`/`DataFrame.query` trong `src/` | test grep toàn repo |
| D4 | Cả 3 trường `ExecutionSettings` đều được đọc trong `src/` | test kiến trúc (19.9) |
| D5 | `replay_pandas_query` chỉ được gọi từ `sandbox.py` | test kiến trúc (19.9) |
| D6 | Connection DuckDB của `cell_frame` chặn được truy cập mạng | test khẳng định `PermissionException` |
| D7 | **gold70 vẫn đúng 30/51 (58,8 %)**, phân rã lỗi không đổi | chạy lại CLI `compile-plans`, so với báo cáo Ngày 18 |
| D8 | `pytest` xanh (≥ 987 pass), `ruff check`/`format --check`/`mypy` sạch | lệnh verification chuẩn |
| D9 | `dataset_fingerprint` và baseline Ngày 8–14 không đổi | `git status --short data/processed/ src/financial_report_qa/normalization/` rỗng |

D7 là mục quan trọng nhất: Ngày 19 là **hardening**, không phải cải thiện độ chính xác. Nếu con số
đổi thì hoặc đã sửa một bug (phải giải thích được), hoặc đã làm hỏng thứ đang chạy.

---

## 6. Rủi ro

| # | Rủi ro | Bằng chứng | Giảm thiểu |
| --- | --- | --- | --- |
| R1 | Đổi cách escape làm **đổi chuỗi `pandas_query` đã nộp** ở Ngày 18 | 30/51 plan đang sinh chuỗi theo format cũ | D7: chạy lại CLI, khẳng định 30/51 và phân rã lỗi y hệt; chuỗi chỉ đổi ở nhãn có ký tự đặc biệt |
| R2 | `except Exception` rộng che lỗi lập trình | đúng bài học `execution/cli.py` Ngày 18 | C1: chỉ trong `sandbox.py`, ghi lại tên lớp gốc, `ExecutionReplayMismatchError` vẫn là lỗi cứng |
| R3 | Cắt độ dài 512 loại nhãn hợp lệ | **433 ô số** (0,008 %) bị loại, đo được | ghi vào ADR như chi phí đã biết; mã lỗi `plan_rejected` phân biệt được với `metric_not_found` |
| R4 | `budget_exceeded` theo thời gian là **hậu nghiệm**, không ngắt được | § 1.6: không có SIGALRM/setitimer/resource trên win32 | ghi thẳng vào ADR + README là giới hạn đã biết; ngân sách cấu trúc (D3) mới là hàng rào thật |
| R5 | Hạ `max_rows` 100.000 → 20.000 chặn nhầm truy vấn hợp lệ tương lai | trần tuyệt đối đo được = **4.002** (12 bảng lớn nhất corpus) | 5× headroom; giá trị nằm trong config nên chỉnh được không cần sửa code |
| R6 | Nhãn dòng từ PDF độc hại chảy vào `evidence` → câu trả lời | nhãn corpus **không** vào prompt (ADR 0006 B1) nhưng **có** vào `CompiledQuery.evidence` | **ngoài phạm vi Ngày 19**, ghi rõ là ràng buộc của Ngày 20 (verifier + citation) |
