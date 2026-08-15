# ADR 0008: Hợp đồng sandbox thực thi

- **Trạng thái:** Accepted
- **Ngày:** 2026-08-15
- **Quyết định:** A2 (escape chuỗi đúng bằng `json.dumps` ở renderer + chặn charset/length ở
  contract, không cấm ký tự); B2 (`execution/sandbox.py` mới là cổng duy nhất tới replay); C1
  (sandbox bắt `Exception` rộng, đổi thành mã lỗi có kiểu, giữ nguyên tên lớp gốc); D3 (chặn theo
  cấu trúc — độ dài/số node/độ sâu/số dòng — trước khi chạy; đo thời gian sau khi chạy, không ngắt
  preemptive); E1 (`compile_plan` tự gọi `validate_plan_semantics`, không tin caller); F1 (hardening
  mọi connection DuckDB trong `execution/` bằng cách tắt autoinstall/autoload extension — **không**
  tắt `enable_external_access`, xem hiệu chỉnh trong Quyết định F); G1 (bốn mã lỗi mới:
  `plan_rejected`, `query_rejected`, `budget_exceeded`, `row_limit_exceeded`)

## Bối cảnh

[Kế hoạch Ngày 19](../plans/day19-sandbox-executor.md) đo trên release đã khoá
(`data/processed/release_v2_422df141c935`) và mã Ngày 18 vừa commit (`4695aaa`), phát hiện: biên tin
cậy mà Ngày 18 để hở không chỉ là rủi ro bảo mật giả định — nó là **bug correctness đang sống**.
`render_pandas_query` nội suy chuỗi tự do bằng f-string; **1.988 nhãn dòng có thật** trong corpus
(1.790 bảng, 9.944 ô số) chứa dấu `"` theo đúng quy ước viết tắt tiếng Việt
(`Khấu hao tài sản cố định ("TSCĐ")`). Đưa một nhãn như thế vào `MetricSelector.raw_text` — đúng
thứ ADR 0004 phương án C bảo planner phải làm — khiến `compile_plan` ném `SyntaxError` không ai bắt,
tái hiện 3/3 lần trên dữ liệu MBB/PNJ thật. Đo thêm: replay thoát ra 5 loại exception khác nhau
(`ValueError`, `SyntaxError`, `KeyError`, `RecursionError`, `decimal.InvalidOperation`) trong khi
compiler chỉ bắt một; nội suy vào `companies` sinh được query đúng cú pháp nhưng sai ngữ nghĩa (đổi
độ ưu tiên `|`/`&`) mà không có exception nào để bắt; trần tuyệt đối của mọi frame hợp lệ đo được là
4.002 dòng trong khi `max_rows` cấu hình 100.000 (gấp 25 lần); trên `win32` không tồn tại
`SIGALRM`/`setitimer`/module `resource`, nên timeout preemptive in-process không khả thi. ADR này
chốt 7 quyết định thiết kế bắt buộc trước khi viết code sandbox.

## Quyết định A: chuỗi tự do trong `pandas_query` xử lý thế nào?

**Đã chọn: A2 — escape đúng ở renderer bằng `json.dumps(value, ensure_ascii=False)` thay cho
f-string; đồng thời chặn charset/length ở tầng contract (`plan_contracts.py`), không cấm ký tự.**

Không chọn "cấm dấu `\"`" vì đo được nó loại bỏ 1.988 nhãn hợp lệ (1,37 % nhãn phân biệt trên ô số).
JSON string literal là tập con hợp lệ của Python string literal cho mọi ký tự không điều khiển, nên
chuỗi render vẫn `ast.parse` được sau khi escape. `raw_text` và `companies` bị ràng buộc cấm ký tự
điều khiển (đo được: loại đúng 1 ô/5.353.511) và độ dài ≤ 512 (đo được: loại 433 ô số, 0,008 %).
`companies` thêm allowlist `^[A-Za-z0-9]{1,16}$` — đo được cả 1.971 tài liệu trong corpus đều khớp
`^[A-Z0-9]+$` với độ dài tối đa 3, nên chi phí ràng buộc này là 0.

Lý do làm cả hai lớp: escape sửa correctness (chuỗi thực thi được và trả đúng nghĩa), contract giới
hạn bề mặt tấn công (payload không lọt tới renderer). Chỉ escape thôi thì payload trở nên vô hại
nhưng vẫn *chạy* — không thoả yêu cầu plan.md "payload độc hại bị **từ chối** có error code".

## Quyết định B: sandbox đặt ở đâu?

**Đã chọn: B2 — module `execution/sandbox.py` mới, là cổng duy nhất gọi `replay_pandas_query`.**

`pandas_query.py` giữ đúng hai việc thuần khiết: render và diễn giải AST whitelist. `sandbox.py` sở
hữu hợp đồng giới hạn (ngân sách cấu trúc, chuyển đổi exception, đo thời gian) và trả về kiểu kết quả
có mã lỗi. `compile_plan` chỉ được gọi replay qua `sandbox.py` — có test kiến trúc khẳng định không
module nào khác import `replay_pandas_query` (nhiệm vụ 19.9).

## Quyết định C: hợp đồng exception khi replay là gì?

**Đã chọn: C1 — sandbox bắt `Exception` rộng (không phải `BaseException`), đổi thành mã lỗi
`query_rejected` có kiểu, giữ nguyên tên lớp exception gốc trong `error_message` để không mất thông
tin debug.**

Đo được 5 loại exception thoát ra và danh sách đó không đóng được bằng liệt kê — thêm một node AST
mới vào interpreter là thêm một loại lỗi mới có thể phát sinh. Deny-by-default nghĩa là mọi exception
khi replay là từ chối, không phải sập. `KeyboardInterrupt`/`SystemExit` không bị bắt vì chúng kế thừa
`BaseException`, không phải `Exception`. Hai rào chắn để việc bắt rộng này không lặp lại lỗi
`execution/cli.py` Ngày 18 (bắt nhầm `ValueError` của caller): (1) tên lớp exception gốc luôn được
ghi lại, không nuốt câm; (2) `ExecutionReplayMismatchError` — trường hợp replay chạy được nhưng ra
số khác compiler tự tính — vẫn là lỗi cứng làm vỡ build, không đi qua `sandbox.py`.

## Quyết định D: giới hạn thời gian và bộ nhớ bằng cách nào?

**Đã chọn: D3 — chặn theo cấu trúc (độ dài chuỗi, số node AST, độ sâu AST, số dòng frame) trước khi
chạy; đo `time.perf_counter()` sau khi chạy và trả `budget_exceeded` nếu vượt `timeout_seconds`,
không ngắt preemptive.**

Đo trực tiếp trên môi trường đích (`win32`, CPython 3.11.15): `signal.SIGALRM` và `signal.setitimer`
không tồn tại, module `resource` (`RLIMIT_AS`) không tồn tại. Mọi công thức "đặt alarm rồi ngắt" hay
"setrlimit bộ nhớ" không áp dụng được trên nền tảng này — ADR không giả vờ giải quyết bằng cách viết
code gọi API không tồn tại. Ngân sách cấu trúc thay vào đó chặn *đầu vào* để công việc bị giới hạn về
mặt kết cấu trước khi có cơ hội chạy lâu:

| Ngân sách | Giá trị | Căn cứ đo |
| --- | ---: | --- |
| Độ dài chuỗi query | 4.096 ký tự | payload dài nhất quan sát 50.100 ký tự; query hợp lệ trên gold70 ≪ 1.000 |
| Số node AST | 2.000 | `isin(200k literal)` tạo ra ~200.000 node |
| Độ sâu AST | 50 | `RecursionError` xảy ra ở độ sâu 1.000 (`sys.getrecursionlimit()=1000`); biểu thức thật < 15 |
| Số dòng frame (`max_rows`) | 20.000 (hạ từ 100.000) | trần tuyệt đối toàn corpus đo được = 4.002 (12 bảng lớn nhất); giữ 5× headroom |

`timeout_seconds` giữ nguyên 5 giây (3,5× so với `compile_plan` thật đo được max 1,424 s). Việc đo
sau-khi-chạy là phát hiện hậu nghiệm, không phải ngắt — tài liệu (ADR này, README) phải nói thẳng
điều đó. Cách ly thật bằng subprocess/container là việc của tầng deployment, nằm ngoài phạm vi Ngày
19.

## Quyết định E: ai gác biên tin cậy của `compile_plan`?

**Đã chọn: E1 — `compile_plan` tự gọi `validate_plan_semantics` trước khi chạm dữ liệu, không tin
rằng caller (rule planner, LLM planner, hay bất kỳ caller tương lai nào) đã validate.**

`compile_plan` trước Ngày 19 không gọi `validate_plan_semantics` — hàm này chỉ được `llm_planner.py`
gọi như một bước quy trình, không phải một hợp đồng được ép buộc ở biên compiler. Hệ quả đo được:
`top_k` (đã bị validator chặn `1 <= top_k < len(companies)`) và các ràng buộc arity khác chỉ có hiệu
lực khi có ai đó chủ động gọi validator trước. Chi phí thêm bước gọi: một hàm thuần trên plan đã có
sẵn trong bộ nhớ, không I/O, không đáng kể so với 0,4–1,4 giây của `compile_plan`.

## Quyết định F: connection DuckDB có cần hardening không?

**Đã chọn: F1 — mọi `duckdb.connect` trong `execution/` chạy hai lệnh `SET` ngay sau khi tạo:
`autoinstall_known_extensions=false`, `autoload_known_extensions=false`.**

**Hiệu chỉnh so với phép đo ban đầu ở kế hoạch Ngày 19 § 1.7 và ADR nháp đầu tiên:** phép đo đầu tiên
kiểm chứng "bật cả ba lệnh `SET` (gồm cả `enable_external_access=false`) thì `read_csv_auto('https://…')`
trả `PermissionException`" — đúng, nhưng **chưa kiểm tra `enable_external_access=false` có phá đọc file
local hay không**. Viết test TDD cho nhiệm vụ 19.7 mới lộ ra: `enable_external_access=false` chặn
**toàn bộ** truy cập filesystem, kể cả `read_parquet` cục bộ mà `_QUERY` phụ thuộc — 8/8 test
`cell_frame` đỏ với `PermissionException: file system operations are disabled by configuration` ngay
trên đường dẫn local. Đo lại: chỉ tắt `autoinstall_known_extensions` và `autoload_known_extensions`
là đủ để chặn mạng, vì extension `httpfs` (bắt buộc cho mọi đường dẫn `http(s)://`) không được đóng gói
sẵn — không autoload được thì đọc mạng thất bại thẳng với lỗi thiếu extension, không im lặng chạm mạng.
Đây là ví dụ cụ thể cho lý do bắt buộc TDD ở Ngày 19: một quyết định "hardening" tưởng vô hại đã suýt
phá chức năng chính nếu không có test đỏ bắt được trước khi merge.

Mặt khác, đo được `_QUERY` trong `cell_frame.py` đã dùng placeholder `?` cho mọi giá trị từ plan
(đường dẫn parquet lẫn danh sách `table_id`, vốn đã bị ràng buộc `^tbl_[0-9a-f]{64}$` bởi schema), nên
**không có đường SQL injection nào từ plan** — quyết định F1 là hardening chiều sâu, không phải vá lỗ
đang chảy máu. Đã kiểm chứng hai lệnh `SET` này vẫn đủ tác dụng: `read_csv_auto('https://…')` trả lỗi
thiếu extension `httpfs` thay vì đọc mạng thành công, **và** `read_parquet` cục bộ vẫn hoạt động bình
thường (10/10 test `cell_frame` xanh).

## Quyết định G: mã lỗi mới nào cần thêm?

**Đã chọn: G1 — bốn mã lỗi mới, thêm vào `ExecutionIssueCode`:**

| Mã | Nghĩa | Sinh ra bởi quyết định |
| --- | --- | --- |
| `plan_rejected` | plan không qua charset/length guard hoặc `validate_plan_semantics` | A2, E1 |
| `query_rejected` | replay ném exception bất kỳ khi diễn giải chuỗi `pandas_query` | C1 |
| `budget_exceeded` | vượt ngân sách độ dài/số node/độ sâu AST, hoặc vượt `timeout_seconds` | D3 |
| `row_limit_exceeded` | hình chiếu ô số vượt `max_rows` | D3 |

Giữ nguyên 6 mã Ngày 18 (`operation_not_allowed`, `metric_not_found`, `period_unresolved`,
`cell_ambiguous`, `unit_incompatible`, `division_by_zero`) → tổng 10 mã. `operation_not_allowed` đã
có sẵn từ Ngày 18 và phủ đúng yêu cầu "operation whitelist" của plan.md — không cần mã mới cho việc
đó.

## Số đo hỗ trợ quyết định

| Số đo | Giá trị | Nguồn |
| --- | ---: | --- |
| Nhãn dòng phân biệt chứa `"` trên ô số | 1.988 (1.790 bảng, 9.944 ô) | § 1.1 kế hoạch Ngày 19 |
| Crash `SyntaxError` không ai bắt trên nhãn corpus thật | 3/3 tái hiện | § 1.1 |
| Ô có `row_label_raw` chứa ký tự điều khiển (LF/CR/TAB) | 1/5.353.511 | § 1.2 |
| Ô số bị loại nếu cắt ở độ dài 512 | 433/(toàn bộ ô số) | § 1.2 |
| Tài liệu có `company_code` khớp `^[A-Z0-9]+$`, độ dài ≤ 3 | 1.971/1.971 (100 %) | § 1.2 |
| Loại exception khác nhau thoát ra khỏi `replay_pandas_query` khi bị tấn công | 5 (`ValueError`, `SyntaxError`, `KeyError`, `RecursionError`, `InvalidOperation`) | § 1.3 |
| `compile_plan` thật trên gold70: p95 / max thời gian | 1,216 s / 1,424 s | § 1.5 |
| Trần tuyệt đối số dòng frame (12 bảng lớn nhất toàn corpus) | 4.002 | § 1.5 |
| `max_rows` cấu hình trước Ngày 19 | 100.000 (gấp 25× trần tuyệt đối) | § 1.5 |
| `SIGALRM`/`setitimer`/module `resource` trên `win32` | không tồn tại (đo trực tiếp) | § 1.6 |
| `enable_external_access` mặc định của `duckdb.connect(":memory:")` | `True` | § 1.7 |
| Đường SQL injection từ plan vào `cell_frame._QUERY` | 0 (mọi giá trị qua placeholder `?`) | § 1.7 |
| BinOp lồng gây `RecursionError` | ở độ sâu 1.000 | § 1.8 |
| `isin(200.000 literal)` chạy hết | 0,63 s, chuỗi nguồn ~1,4 MB | § 1.8 |
| Tham chiếu `timeout_seconds`/`max_rows` trong `src/` trước Ngày 19 | 0 | § 1.9 |
| Trường tự do sống sót tới executor sau mọi lớp ràng buộc plan | 2 (`companies`, `MetricSelector.raw_text`) | § 1.10 |

## Hệ quả

- `pandas_query.py` không còn dùng f-string nội suy cho literal chuỗi — mọi giá trị chuỗi render qua
  `json.dumps(value, ensure_ascii=False)`.
- `sandbox.py` là điểm chặn bắt buộc; `compile_plan` không còn gọi `replay_pandas_query` trực tiếp.
- `ExecutionSettings.max_rows` và `.timeout_seconds` không còn là code chết — `cell_frame.py` và
  `sandbox.py` đọc chúng.
- `configs/local_rtx3050.yaml` hạ `max_rows` từ 100.000 xuống 20.000.
- **Không đổi `dataset_fingerprint`**; không đụng `data/processed/` hay `normalization/`.
- **gold70 phải vẫn giải được đúng 30/51 (58,8 %)** với cùng phân rã lỗi Ngày 18
  (`metric_not_found` 11, `period_unresolved` 8, `cell_ambiguous` 2) — nếu số đổi, đó là dấu hiệu một
  quyết định ở ADR này đã vô tình chặn nhầm một plan hợp lệ, không phải một cải thiện được hoan
  nghênh không giải thích.
- Rủi ro đã biết và ghi nhận chứ không giấu: `budget_exceeded` theo thời gian là hậu nghiệm (không
  ngắt được giữa chừng trên nền tảng đích); cắt độ dài 512 loại 433 ô số hợp lệ khỏi khả năng truy
  vấn qua `raw_text` (chấp nhận được vì tỷ lệ 0,008 %); nhãn dòng độc hại từ nguồn PDF không được
  Ngày 19 chặn khỏi `CompiledQuery.evidence` — đó là ràng buộc của Ngày 20 (verifier + citation), ghi
  rõ ngoài phạm vi.
