# Kế hoạch Ngày 18 — Deterministic compiler

> **Trạng thái:** đã đo, **đã thực hiện xong** (2026-08-15). ADR:
> [0007](../decisions/0007-deterministic-compiler-contract.md).
> **Ngày đo:** 2026-08-15. **Release khoá:** `data/processed/release_v2_422df141c935`,
> `dataset_fingerprint = 422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a`.
> **Mọi con số trong tài liệu này là kết quả truy vấn thật trên release đã khoá**, không phải ước lượng.
>
> **Kết quả sau khi cài đặt xong (CLI `compile-plans` chạy thật trên gold70):** resolved rate
> **30/51 (58,8 %)**, khớp chính xác trần lý thuyết đo được ở § 1.3 dưới. Phân rã lỗi trên 21 plan
> không giải được: `metric_not_found` 11, `period_unresolved` 8, `cell_ambiguous` 2 — không có lỗi
> `unit_incompatible` hay `division_by_zero` nào rơi vào gold70 (hai mã này được golden test bao phủ
> riêng, xem § 1.5). Báo cáo đầy đủ: `artifacts/evaluations/day18/compiled-plans-422df141c935.md`
> (gitignored, tái tạo được — xem lệnh CLI trong `plan.md`).

Mục Ngày 18 trong [plan.md](../../plan.md) yêu cầu bốn việc: một hàm compiler cho mỗi operation;
chỉ cho phép DataFrame/cột/scalar đã whitelist; sinh `pandas_query` dễ đọc; golden tests cho dấu âm,
null, duplicate row, nhiều đơn vị và chia cho 0.

Phép đo dưới đây cho thấy **phần số học không phải là chỗ khó**. Chỗ khó là **locator**: đi từ
`FinancialQueryPlan` (metric canonical + năm) xuống một ô số duy nhất trong `cells.parquet`. Ở trạng
thái hiện tại chỉ **24/51 plan (47,1 %)** mà rule planner sinh ra trên gold70 là **giải được tới ô
số**. Ngày 18 có headroom đo được — khác hẳn Ngày 17.

---

## 0. Đầu vào đã sẵn sàng

| Hạng mục | Vị trí | Trạng thái |
| --- | --- | --- |
| Schema plan (Ngày 15) | `planning/plan_contracts.py` — 9 operation | ✅ đóng băng |
| Semantic validator (Ngày 15) | `planning/plan_validator.py` — 10 error code | ✅ mẫu `PlanValidationIssue` để nhân bản |
| Rule planner (Ngày 16) | `planning/rule_planner.py` | ✅ operation accuracy 1,0 |
| Router + LLM fallback (Ngày 17) | `planning/plan_router.py` | ✅ false-plan rate 0,0 |
| Chuẩn hoá đơn vị | `normalization/units.py` — `convert_scale`, `economic_value`, `unit_multiplier` | ✅ **đã có sẵn, có test, tái dùng nguyên** |
| Corpus ô số | `release_v2_422df141c935/cells.parquet` — 6.199.661 ô / 146.011 bảng | ✅ |
| `duckdb` | `pyproject.toml` | ✅ **không cần thêm dependency nào cho Ngày 18** |
| Package `execution/` | `src/financial_report_qa/execution/` | ⚠️ **chỉ có `__init__.py` với một docstring** |
| Cấu hình `execution:` | `configs/base.yaml`, `configs/local_rtx3050.yaml` | ❌ **code chết** — xem § 1.7 |
| `tables.csv_path` | `tables.parquet` | ❌ **NULL cho cả 146.011 bảng** — xem § 1.6 |

---

## 1. Chốt chặn phải đo trước khi viết code

### 1.1. Chỉ 15,4 % ô có `period`, và 37,7 % trong số đó **không phải** dạng `YYYY`

Đếm trên toàn bộ `cells.parquet`:

| Thuộc tính | Số ô | % của 6.199.661 |
| --- | ---: | ---: |
| `value_numeric` khác NULL | 2.620.706 | 42,3 % |
| `period` khác NULL | 952.363 | 15,4 % |
| `value_numeric` **và** `period` | 822.679 | 13,3 % |
| thêm `unit` khác NULL | 726.503 | 11,7 % |
| thêm `row_label_canonical` khác NULL | **55.891** | **0,9 %** |

Và dạng giá trị của `period`:

| Dạng | Số ô | Ví dụ |
| --- | ---: | --- |
| `YYYY` | 593.164 | `2013` … `2025` |
| `YYYY-MM-DD` | **359.199** | `2013-12-31` … `2031-12-31` |

`FinancialQueryPlan` ép `periods` khớp `^\d{4}$` (`plan_contracts.py:20`). Nếu locator so sánh
chuỗi trực tiếp, nó **âm thầm bỏ 359.199 ô (37,7 %)**. Đây là lỗi tôi đã mắc ở vòng đo đầu tiên và
phải sửa: **locator bắt buộc chuẩn hoá `period` của ô về năm trước khi so khớp.**

### 1.2. 62,5 % bảng **không có `period` trên bất kỳ ô nào** — cột mở/cuối kỳ là lối ra

| Độ phủ `period` của bảng | Số bảng | Ô số bên trong |
| --- | ---: | ---: |
| **Không ô nào có `period`** | **91.266** | **1.687.213** |
| Một phần có `period` | 54.745 | 933.493 |

Đây là bố cục bảng cân đối kế toán chuẩn Việt Nam: cột không ghi năm mà ghi *`Số cuối năm`* /
*`Số đầu năm`*, năm nằm ở cấp tài liệu (`documents.report_year`). Ví dụ thật — câu gold70
*"Tính tốc độ tăng trưởng tổng tài sản riêng của GEG từ năm 2022 đến năm 2023"*: bảng gold có 355 ô,
**0 ô có `period`**, `column_label_raw` chỉ gồm `Số cuối năm VND`, `Số đầu năm VND`, `Mã số`,
`Thuyết minh`, `TÀI SẢN`.

Đếm các dấu hiệu này trên các ô số **không có** `period`:

| Dấu hiệu trong `column_label_raw` | Số ô số |
| --- | ---: |
| `Số cuối năm` | 225.371 |
| `Số đầu năm` | 217.414 |
| cả hai | 163 |
| **không dấu hiệu nào** | **1.355.079** |

Suy ra 442.785 ô (24,6 % số ô không có `period`) khôi phục được kỳ từ `report_year`.

**Kiểm chứng quy tắc suy diễn.** Với các ô *vừa* có `period` tường minh *vừa* có dấu hiệu mở/cuối kỳ
— tức có thể đối chiếu — độ lệch so với `documents.report_year`:

| Dấu hiệu | `period − report_year` | Số ô |
| --- | ---: | ---: |
| `Số cuối năm` | `0` | 5 |
| `Số đầu năm` | `−1` | 5 |

Nhất quán 100 %, **không có phản ví dụ nào**. Nhưng cỡ mẫu chỉ **n = 10**. Đây là bằng chứng yếu và
phải ghi thẳng vào rủi ro (§ 6 R1), không được trình bày như quy tắc đã chứng minh.

### 1.3. Headroom thật: 47,1 % → 58,8 % số plan giải được tới ô số

Chạy `rule_planner.build_plan` trên 70 câu gold70 với `candidate_table_ids` = bảng gold (cô lập lỗi
retrieval khỏi phép đo): **51 plan** được sinh (19 abstain — đúng con số Ngày 17), gồm
`lookup` 20, `growth_rate` 18, `difference` 13. 51 plan này mở ra **82 khe** `(plan × kỳ × selector)`.

Giải mỗi khe xuống ô số, loại `col_idx = 0` (xem § 1.4):

| Locator | 1 ô | n ô, cùng giá trị | n ô, **giá trị khác nhau** | 0 ô | Khe giải được duy nhất | **Plan giải được trọn vẹn** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chỉ `period` tường minh (đã chuẩn hoá về năm) | 33 | 8 | 2 | 39 | 41/82 — 50,0 % | **24/51 — 47,1 %** |
| **+ suy diễn kỳ từ mở/cuối kỳ + `Năm trước`** | 39 | 12 | 2 | 29 | **51/82 — 62,2 %** | **30/51 — 58,8 %** |

Quy tắc suy diễn nâng số ô `số + kỳ` toàn corpus từ **822.679 lên 1.287.719 (+56,5 %)**.

Phân rã 29 khe còn hỏng:

| Nguyên nhân | Số khe | Ngày 18 xử lý được? |
| --- | ---: | --- |
| `metric_label_absent` — nhãn canonical **không tồn tại** trong bảng ứng viên | 16 | ❌ **Không.** Chỉ 0,9 % ô có `row_label_canonical`; đây là nợ của normalization, không phải của compiler. |
| `period_unresolved` — kỳ vẫn không suy ra được | 13 | ❌ Không (bảng không có cả `period` lẫn dấu hiệu mở/cuối kỳ). |

Cộng thêm 2 khe `n ô, giá trị khác nhau`. **Cả 31 trường hợp này phải trả về error code có kiểu,
tuyệt đối không được trả về một con số.** Đó là ranh giới trung thực của Ngày 18.

### 1.4. Bốn lớp nhập nhằng đã đếm được, phải xử lý tường minh

| Lớp | Phép đo | Kết luận thiết kế |
| --- | --- | --- |
| **Ô nhãn ở `col_idx = 0`** | 11.734 ô có `period` nằm ở cột 0, chỉ **64** ô trong đó có `value_numeric` | Whitelist `col_idx > 0`. Giá phải trả: 64 ô (0,008 %). |
| **Duplicate row** | Cùng `(table, row_label_raw, period, col_idx)` xuất hiện ở ≥2 `row_idx`: **35.766 / 766.710 nhóm (4,67 %)**, trong đó **33.321 (93,2 %) có giá trị khác nhau** | Đây chính là mục "duplicate row" plan.md yêu cầu. Không được lấy dòng đầu. |
| **Một nhãn trải nhiều cột** | 5.578 / 758.561 nhóm (0,74 %) | Bảng báo cáo bộ phận: một `row_label` × 5 cột phân khúc + cột `Tổng`. |
| **Trộn đơn vị** | 501 / 47.040 bảng (1,06 %) có ≥2 `unit`; ở mức nhóm `(table, row_label)` là **23.826 / 811.261 (2,94 %)** | Đây là mục "nhiều đơn vị". |

Ví dụ thật của lớp thứ ba, bảng
`tbl_85faaa3e24eeb6e14ea42f20fe54548432142e826b66323e12aa4aaa216bf2dd`: nhãn
`Doanh thu thuần từ bán hàng ra bên ngoài` kỳ 2015 cho **5 ô** với `column_label_raw` lần lượt là
`Năm 2015\nDịch vụ vận tảiVND`, `…kho nổiVND`, `…Thương mạiVND`, `…Dịch vụ khácVND`,
`Năm 2015\nTổngVND`. Bốn ô đầu là phân khúc, ô cuối là tổng — **giá trị khác nhau, không được cộng
lại và cũng không được chọn bừa**.

### 1.5. Dấu âm và null đã được normalization xử lý; chia-0 hiếm nhưng có thật

| Trường hợp | Phép đo |
| --- | ---: |
| Ô âm | **327.743**, trong đó **326.782 (99,7 %)** viết bằng dấu ngoặc `(1.234)` trong `value_raw` |
| Ô âm viết bằng dấu `-` đứng trước | 961 |
| Ô có `value_numeric = 0` | **4.649** trên **1.257 bảng** |
| Ô `value_numeric` NULL nhưng có `value_raw` | 3.032.318 |
| Ký tự null phổ biến nhất trong `value_raw` | `-` (546.246), chuỗi rỗng (546.637), `VND` (41.095), `Số cuối năm` (26.553), `Số đầu năm` (21.719) |

Kết luận: compiler **không parse lại chuỗi**, chỉ đọc `value_numeric` — dấu âm đã đúng dấu từ Ngày 5.
Nhưng golden test vẫn phải khoá hành vi này (plan.md yêu cầu tường minh), và `value_raw` như `VND`
hay `Số cuối năm` cho thấy **ô tiêu đề lọt vào vùng dữ liệu** — thêm một lý do lọc `value_numeric
IS NOT NULL` thay vì lọc theo vị trí.

### 1.6. `pandas_query` **không phải trang trí** — nó là đường thực thi thứ hai

plan.md § 2.4 quy tắc 6–7: mỗi biến `dfN` trong `pandas_query` ánh xạ một CSV trong gói nộp, và
*"validator nạp từng CSV vào đúng tên biến rồi replay biểu thức Pandas trong sandbox whitelist"*, so
kết quả với `answer` theo tolerance.

Nhưng: **`tables.csv_path` là NULL cho cả 146.011 bảng.** Hình dạng DataFrame mà `pandas_query` trỏ
tới **chưa tồn tại**. Nếu Ngày 18 không chốt hình chiếu này, `pandas_query` là chuỗi không thể phủ
định — viết gì cũng "đúng" vì không có gì để replay.

⇒ **Ngày 18 phải định nghĩa hình chiếu DataFrame chính tắc**, và DoD phải là *replay lại chính biểu
thức đó trên hình chiếu và so bằng với kết quả compiler tự tính*.

### 1.7. Khối `execution:` trong YAML là code chết

```
$ grep -rn "ExecutionSettings\|load_execution\|allow_operations\|max_rows" --include=*.py src/ tests/
(không có kết quả)
```

`configs/local_rtx3050.yaml:14-26` khai báo `timeout_seconds: 5`, `max_rows: 100000` và 9
`allow_operations`, nhưng **không dòng Python nào đọc**. Đúng tình trạng khối `llm:` trước Ngày 17.
Ngày 18 nối dây khối này, y hệt việc 17.2 đã làm.

---

## 2. Quyết định thiết kế

Ghi thành **ADR 0007** (nhiệm vụ 18.1).

### A. Nguồn dữ liệu — **A1: compiler đọc thẳng `cells.parquet`**

Retrieval service chỉ trả về `table_id`, không mở ô. Compiler cần cấp ô. Đọc thẳng bằng DuckDB, giới
hạn trong `plan.candidate_table_ids` (tối đa 12 bảng theo schema Ngày 15).

### B. Hình chiếu DataFrame — **B1: dạng dài (long format), 8 cột cố định**

```
table_id | row_idx | col_idx | row_label | column_label | period | unit | value
```

`pandas_query` sinh ra sẽ đọc được như sau:

```python
df1[(df1.row_label == "TỔNG CỘNG TÀI SẢN") & (df1.period == 2023)]["value"].iloc[0]
```

**Sai khác có chủ ý so với ví dụ ở plan.md § 2.4** (`df1[(df1.company == 'VNM') & (df1.year == 2023)]['net_revenue']`):
ví dụ đó là dạng rộng, mỗi metric một cột. Corpus này không có hình dạng ấy — một bảng báo cáo tài
chính có **metric nằm ở dòng**, không nằm ở cột, và tên dòng là tiếng Việt tự do. Ép sang dạng rộng
sẽ phải bịa tên cột. Hợp đồng nộp bài chỉ ràng buộc *replay phải ra đúng `answer`* (quy tắc 3 và 7),
không ràng buộc hình dạng cột — nên dạng dài hợp lệ. Ghi rõ vào ADR để Ngày 21 không hiểu nhầm.

### C. Giải kỳ — **C2: `period` tường minh, sau đó suy diễn từ `report_year`**

Thứ tự ưu tiên, dừng ở bước khớp đầu tiên:

1. `period` của ô, chuẩn hoá về 4 chữ số đầu (bắt cả 359.199 ô dạng ISO).
2. `column_label` chứa `số cuối năm` → `documents.report_year`.
3. `column_label` chứa `số đầu năm` hoặc `năm trước` → `report_year − 1`.

Kỳ suy diễn phải được **đánh dấu** trong evidence (`period_inferred: bool`) để Ngày 20 hạ mức tin cậy.
Đo được: +6 plan gold70 (24 → 30).

### D. Nhập nhằng — **D1: không bao giờ đoán**

| Số ô khớp | Hành vi |
| --- | --- |
| 0 | error `metric_not_found` hoặc `period_unresolved` |
| 1 | trả giá trị |
| n, **mọi `value_numeric` bằng nhau** | trả giá trị, evidence giữ **toàn bộ** `cell_id` |
| n, **có giá trị khác nhau** | error `cell_ambiguous`, liệt kê ứng viên |

Đo được: nhánh "n ô cùng giá trị" cứu 12/82 khe; nhánh xung đột giữ lại 2 khe ở dạng lỗi thay vì
dạng số sai.

### E. Đơn vị — **E1: quy về giá trị kinh tế bằng `convert_scale`, cấm trộn nhóm**

Tái dùng `normalization/units.py` nguyên trạng. Số học tiền tệ quy về `VND`; `convert_scale` **đã**
raise khi trộn tiền tệ với `percent`/`ratio`, compiler bắt và đổi thành error `unit_incompatible`.
Không viết lại bảng hệ số.

### F. Bằng chứng `pandas_query` — **F1: replay là điều kiện DoD, không phải tuỳ chọn**

Mỗi kết quả compile chạy qua một replayer whitelist (chỉ boolean mask, `[]`, `.iloc`, số học scalar —
không `eval`, không `exec`, đúng tinh thần Ngày 19) trên chính hình chiếu B1, và **phải** bằng kết quả
compiler tự tính. Lệch một trường hợp là hỏng build.

---

## 3. Nhiệm vụ

| # | Nhiệm vụ | Tệp | TDD |
| --- | --- | --- | --- |
| 18.1 | ADR 0007: chốt A1/B1/C2/D1/E1/F1, ghi rõ sai khác dạng dài vs plan.md § 2.4 | `docs/decisions/0007-deterministic-compiler-contract.md` | — |
| 18.2 | `ExecutionSettings` + `load_execution_settings`, nối khối `execution:` đang chết | `core/config.py` | ✅ |
| 18.3 | Error code: `ExecutionError` + 6 mã con | `core/errors.py`, `execution/contracts.py` | ✅ |
| 18.4 | Hình chiếu B1: `build_cell_frame(release, table_ids)` — 8 cột, lọc `col_idx > 0` và `value_numeric IS NOT NULL`, gắn `effective_period` theo C2 | `execution/cell_frame.py` | ✅ |
| 18.5 | Locator: `(selector, period) → CellMatch`, 4 nhánh theo D1 | `execution/locator.py` | ✅ |
| 18.6 | **9 hàm compiler, mỗi operation một hàm**, đơn vị theo E1 | `execution/operations.py` | ✅ |
| 18.7 | `render_pandas_query` + replayer whitelist | `execution/pandas_query.py` | ✅ |
| 18.8 | **Golden tests**: dấu âm, null, duplicate row, nhiều đơn vị, chia cho 0 | `tests/golden/execution/` | ✅ |
| 18.9 | CLI `compile-plans` + báo cáo trên gold70 và plan-cases; test tính tất định (compile 2 lần → giống hệt) | `execution/cli.py` | ✅ |
| 18.10 | Chạy full test/lint/format/mypy, cập nhật `plan.md` + `README.md` | — | — |

### Chi tiết 18.6 — chín hàm, chín chữ ký hẹp

| Operation | Đầu vào locator | Phép tính |
| --- | --- | --- |
| `lookup` | 1 ô | trả nguyên |
| `difference` | 2 kỳ × 1 metric | `v₂ − v₁` |
| `growth_rate` | 2 kỳ × 1 metric | `(v₂ − v₁) / |v₁|`; `v₁ = 0` → `division_by_zero` |
| `compare` | 1 kỳ × 2 metric | `vₐ − v_b` |
| `compare_companies` | 2 công ty × 1 kỳ × 1 metric | `v_A − v_B` |
| `ratio` | numerator + denominator | `vₙ / v_d`; `v_d = 0` → `division_by_zero` |
| `average` | n kỳ × 1 metric | trung bình cộng, cấm mẫu rỗng |
| `sum` | n kỳ × 1 metric | tổng |
| `rank` | n công ty × 1 kỳ × 1 metric, `top_k` | sắp giảm dần, cắt `top_k` |

Mỗi hàm nhận `CellMatch` **đã giải xong**, không tự truy vấn — nên test được bằng dữ liệu dựng tay,
không cần release.

### Chi tiết 18.8 — năm golden test plan.md yêu cầu, gắn với số đo thật

| Golden test | Neo vào phép đo |
| --- | --- |
| Dấu âm | § 1.5 — 326.782 ô dùng dấu ngoặc; khẳng định compiler giữ nguyên dấu, không parse lại `value_raw` |
| Null | § 1.5 — `-` và chuỗi rỗng; khẳng định ô null bị loại khỏi hình chiếu chứ không thành `0` |
| Duplicate row | § 1.4 — 33.321 nhóm giá trị xung đột; khẳng định trả `cell_ambiguous`, **không** lấy dòng đầu |
| Nhiều đơn vị | § 1.4 — 23.826 nhóm; khẳng định quy đổi đúng qua `convert_scale`, và trộn `VND` với `percent` thì lỗi |
| Chia cho 0 | § 1.5 — 4.649 ô bằng 0; khẳng định `growth_rate`/`ratio` trả `division_by_zero`, không trả `inf`/`NaN` |

---

## 4. Thứ tự thực hiện

```
18.1 ADR
  └─> 18.2 config ──┐
      18.3 errors ──┼─> 18.4 cell_frame ─> 18.5 locator ─> 18.6 operations ─> 18.7 pandas_query
                    │                                            │                   │
                    └────────────────────────────────────────────┴───────> 18.8 golden ─> 18.9 CLI ─> 18.10
```

18.2 và 18.3 độc lập, chạy song song được. 18.4 là nút thắt: mọi thứ sau nó phụ thuộc vào hình chiếu.

---

## 5. Definition of Done

| # | Tiêu chí | Ngưỡng | Cách đo |
| --- | --- | --- | --- |
| D1 | **`pandas_query` replay khớp kết quả compiler** | **100 %, không ngoại lệ** | 18.7 replayer chạy trên mọi plan compile được |
| D2 | **Tính tất định** | compile 2 lần → `CompiledQuery` giống hệt từng byte | test trong 18.9 |
| D3 | Tỷ lệ plan gold70 giải được tới ô số | **≥ 30/51 (58,8 %)** — trần đã đo ở § 1.3 | báo cáo 18.9 |
| D4 | **Không có đáp án sai âm thầm** | 100 % khe không giải được trả error code có kiểu, **0 giá trị đoán** | báo cáo 18.9 |
| D5 | Nhập nhằng | **0** trường hợp ≥2 ô giá trị khác nhau bị thu về một số | golden test 18.8 |
| D6 | Năm golden test plan.md liệt kê | đủ 5, mỗi test neo vào một phép đo ở § 1 | 18.8 |
| D7 | Suite | toàn bộ test xanh, `ruff check`/`ruff format --check`/`mypy` sạch | 18.10 |
| D8 | Baseline cũ | `git status --short data/processed/ src/financial_report_qa/normalization/` rỗng | 18.10 |

**D3 là trần, không phải mục tiêu cần vượt.** 16 khe `metric_label_absent` và 13 khe
`period_unresolved` nằm ngoài tầm compiler; cố nâng D3 bằng cách nới locator là đổi lỗi-lộ-thiên lấy
lỗi-âm-thầm, vi phạm D4.

---

## 6. Rủi ro

| # | Rủi ro | Bằng chứng | Giảm thiểu |
| --- | --- | --- | --- |
| R1 | **Quy tắc suy diễn kỳ chỉ có n = 10 làm chứng** | § 1.2 — 5 ô `Số cuối năm` lệch 0, 5 ô `Số đầu năm` lệch −1, không phản ví dụ | Đánh dấu `period_inferred` trong evidence; Ngày 20 hạ mức tin cậy; ghi giới hạn cỡ mẫu vào ADR 0007 |
| R2 | `row_label_canonical` chỉ phủ 0,9 % ô | § 1.1 | Không sửa ở Ngày 18. Locator hỗ trợ sẵn nhánh `raw_text` (ADR 0004 Option C) để nợ normalization trả sau mà không phải đổi API |
| R3 | Bảng báo cáo bộ phận: một nhãn × nhiều cột phân khúc | § 1.4 — 5.578 nhóm | Ngày 18 trả `cell_ambiguous`. **Không** đoán cột `Tổng` — chưa đo được độ tin cậy của heuristic đó |
| R4 | Hình chiếu dạng dài lệch ví dụ ở plan.md § 2.4 | § 1.6 | Ghi thẳng lý do vào ADR 0007; hợp đồng nộp bài chỉ ràng buộc replay-khớp-`answer` |
| R5 | Chưa có `csv_path`, chưa kiểm chứng được đầu-cuối với validator thật | § 1.6 — NULL cho 146.011 bảng | Ngày 18 chốt hình chiếu và replay nội bộ; xuất CSV thật là việc của exporter Ngày 21. Ghi là **nợ đã biết**, không giấu |
| R6 | 64 ô số nằm ở `col_idx = 0` bị whitelist loại | § 1.4 | Chấp nhận: 0,008 % ô, đổi lấy việc loại 11.670 ô nhãn |
